"""
Train a diffusion model on images.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse  #命令行参数解析；
#from options import TrainOptions
try:
    import wandb
except ImportError:
    wandb = None
try:
    import wandb_config as local_wandb_config
except ImportError:
    local_wandb_config = None
from guided_diffusion import logger #分布式训练相关工具；
from guided_diffusion.image_datasets import  load_CL_IMG_data #加载训练数据；
from guided_diffusion.resample import create_named_schedule_sampler #训练时用于 schedule sampling 的工具；
from guided_diffusion.script_util import (args_to_dict, add_dict_to_argparser, CL_IMG_create_model_and_diffusion) #模型创建、默认参数等；
import torch as th
from guided_diffusion.train_util import TrainLoop  #训练核心循环逻辑；
from local_config import apply_local_overrides


def main():

    args = create_argparser().parse_args()
    if local_wandb_config is not None:
        args.use_wandb = getattr(local_wandb_config, "USE_WANDB", args.use_wandb)
        args.wandb_project = getattr(
            local_wandb_config, "WANDB_PROJECT", args.wandb_project
        )
        args.wandb_entity = getattr(
            local_wandb_config, "WANDB_ENTITY", args.wandb_entity
        )
        args.wandb_run_name = getattr(
            local_wandb_config, "WANDB_RUN_NAME", args.wandb_run_name
        )
        args.wandb_mode = getattr(
            local_wandb_config, "WANDB_MODE", args.wandb_mode
        )
        args.wandb_dir = getattr(local_wandb_config, "WANDB_DIR", args.wandb_dir)
        args.wandb_log_interval = getattr(
            local_wandb_config, "WANDB_LOG_INTERVAL", args.wandb_log_interval
        )
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.abspath(args.save_path)

    #device = dist_util.dev(args.gpu_id)
    device = th.device(f"cuda:{args.gpu_id}" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
        th.cuda.set_device(args.gpu_id)
    #dist_util.setup_dist()
    logger.configure(args.save_path)

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError(
                "W&B is enabled but the wandb package is not installed. "
                "Install it with: pip install wandb"
            )
        wandb_api_key = ""
        if local_wandb_config is not None:
            wandb_api_key = getattr(local_wandb_config, "WANDB_API_KEY", "")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key.strip()
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name or os.path.basename(args.save_path),
            "config": vars(args),
            "dir": args.wandb_dir or args.save_path,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_mode:
            wandb_kwargs["mode"] = args.wandb_mode
        wandb_run = wandb.init(**wandb_kwargs)

    logger.log("Creating CT_IMG model and diffusion...")
    #logger.log("在创建扩散模型设置方差为可学习")

    # 改了数据的输入结构和输出结构
    #用于创建：扩散模型（一个 U-Net）扩散过程（调度、时间步等逻辑）
    model, diffusion = CL_IMG_create_model_and_diffusion(
        **args_to_dict(args, 
                       [  # 提取与模型和diffusion有关的参数名
                            "image_size", "num_channels", "num_res_blocks", "num_heads",
                            "num_heads_upsample", "num_head_channels", "attention_resolutions",
                            "channel_mult", "dropout", "use_checkpoint", "use_scale_shift_norm",
                            "resblock_updown", "use_fp16", "use_new_attention_order", "learn_sigma",
                            "diffusion_steps", "noise_schedule", "timestep_respacing", "use_kl",
                            "predict_xstart", "rescale_timesteps", "rescale_learned_sigmas",
                            "condition_channels", "use_afr", "afr_kernel_size"
                        ]
        ),
        device=device,
    )

    if th.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    
    model.to(device)
    #用于训练过程中从不同时间步 t 采样的策略（uniform、loss-aware 等）。
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating CT_IMG_data loader...")

    # 加载训练数据
    data = load_CL_IMG_data(
        data_dir1=args.data_dir1,
        data_dir2=args.data_dir2,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode='train',
        num_input_slices=args.condition_channels,
        crop_x_start=args.crop_x_start,
        crop_x_end=args.crop_x_end,
        crop_y_start=args.crop_y_start,
        crop_y_end=args.crop_y_end,
        use_mmap=args.use_mmap,
        normalization_mode=args.normalization_mode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        shuffle=args.shuffle,
        augment_condition=args.augment_condition,
        condition_aug_probability=args.condition_aug_probability,
        condition_contrast_min=args.condition_contrast_min,
        condition_contrast_max=args.condition_contrast_max,
        condition_noise_std=args.condition_noise_std,
        condition_blur_probability=args.condition_blur_probability,
    )

    logger.log("training...")

    # TrainLoop才是主要修改的地方
    try:
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            data_mode=args.data_mode,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            device_id=device,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            loss_log_interval=args.loss_log_interval,

            resume_checkpoint=args.resume_checkpoint,
            resume_step = args.resume_step,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            boundary_loss_weight=args.boundary_loss_weight,
            boundary_edge_weight=args.boundary_edge_weight,
            wandb_run=wandb_run,
            wandb_log_interval=args.wandb_log_interval,

            save_path=args.save_path,
        ).run_loop()
    finally:
        if wandb_run is not None:
            wandb_run.finish()

def create_argparser():
    defaults = dict(
        # ==== 运行相关 ====
        gpu_id=0,
        save_path="checkpoints/ct_degraded_817",

        # ==== 数据相关 ====
        data_mode='npy',
        # data_dir1 is the phantom supervision label directory.
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/ct-sart/ct_sart_degraded",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/cl_label_npy",
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
        normalization_mode="volume",
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
        augment_condition=True,
        condition_aug_probability=0.3,
        condition_contrast_min=0.6,
        condition_contrast_max=0.8,
        condition_noise_std=0.02,
        condition_blur_probability=0.1,

        # ==== 模型结构相关 ====
        image_size=768,
        condition_channels=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="24,12",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
        use_afr=False,
        afr_kernel_size=7,

        # ==== 扩散相关 ====
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,

        # ==== 训练参数 ====
        lr=1e-4,
        batch_size=2,
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=150000,
        microbatch=-1,
        ema_rate="0,0.9999",
        log_interval=1000,
        save_interval=30000,
        loss_log_interval=1,
        boundary_loss_weight=0.1,
        boundary_edge_weight=3.0,  # 3.0
        use_wandb=True,
        wandb_project="CL_DIFF",
        wandb_entity="",
        wandb_run_name="",
        wandb_mode="",
        wandb_dir="",
        wandb_log_interval=100,
        resume_checkpoint="",
        resume_step = 0,
        fp16_scale_growth=1e-3,
    )
    apply_local_overrides(defaults, __file__)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
