import copy
import functools
import os
import re

import blobfile as bf
import numpy as np
import torch as th
#from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

import csv

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        data_mode,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        device_id=None,
        save_path
    ):
        self.loss_history = []
        self.data_mode = data_mode
        self.device = device_id
        self.save_path = save_path
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = resume_step
        self.global_batch = self.batch_size
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        self.model.to(self.device)
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("self.resume_step : ", self.resume_step)

            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(resume_checkpoint, map_location=self.device)
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(
                ema_checkpoint, map_location=self.device
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    # 训练函数入口
    def run_loop(self):
        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):

            img, bad_img, img_path= next(self.data)

            img = th.tensor(img).to(device=self.device)
            bad_img = th.tensor(bad_img).to(device=self.device)
            step_size = self.step

            self.run_step(img, bad_img, step_size)

            if self.step % self.log_interval == 0:
                logger.dumpkvs() 

            if self.step % self.save_interval == 0:
                self.save(self.data_mode)

            self.step += 1

        #保存训练完的模型
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.data_mode)

    def run_step(self, img, bad_img,step_size):

        self.forward_backward(img, bad_img,step_size)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, img, bad_img, step_size):
        self.mp_trainer.zero_grad()  #清理上一次反向传播产生的梯度

        for i in range(0, img.shape[0], self.microbatch):
            img_tmp = img[i: i + self.microbatch].to(self.device)
            bad_img_tmp = bad_img[i: i + self.microbatch].to(self.device)

            t, weights = self.schedule_sampler.sample(img_tmp.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                img_tmp,
                bad_img_tmp,
                t,
                step_size,
                device=self.device
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.loss_history.append(loss.item())

            if step_size % 100 == 0:
                # 将损失值和步长写入 CSV 文件
                with open("loss_data.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([step_size, loss.item()])
            # Log the loss values
            logger.logkv("loss", loss.item())

            # if len(self.loss_history) > 1:
            #         plt.figure()
            #         plt.plot(self.loss_history)
            #         plt.xlabel('Step')
            #         plt.ylabel('Loss')
            #         plt.title('Training Loss Curve')
            #         plt.savefig('loss_curve.png')
            #         plt.close()
                    
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, data_mode):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            save_path = self.save_path

            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"mode_{data_mode}_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{data_mode}_{rate}_{(self.step+self.resume_step):06d}.pt"

            with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                th.save(state_dict, f)
                print("Model saved in{save_path}".format(save_path=save_path))


        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    """
    Parse the trailing step from a checkpoint filename.
    """
    match = re.search(r"(\d+)\.pt$", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    checkpoint_name = os.path.basename(main_checkpoint)
    mode_match = re.match(r"mode_(.+)_\d+\.pt$", checkpoint_name)
    if mode_match:
        data_mode = mode_match.group(1)
        filenames = [
            f"ema_{data_mode}_{rate}_{step:06d}.pt",
            f"ema_{rate}_{step:06d}.pt",
        ]
    else:
        filenames = [f"ema_{rate}_{step:06d}.pt"]

    for filename in filenames:
        path = bf.join(bf.dirname(main_checkpoint), filename)
        if bf.exists(path):
            return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
