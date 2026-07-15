"""
Train a diffusion model on images.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse  #命令行参数解析；
#from options import TrainOptions
from guided_diffusion import logger #分布式训练相关工具；
from guided_diffusion.image_datasets import  load_CL_IMG_data #加载训练数据；
from guided_diffusion.resample import create_named_schedule_sampler #训练时用于 schedule sampling 的工具；
from guided_diffusion.script_util import (args_to_dict, add_dict_to_argparser, CL_IMG_create_model_and_diffusion) #模型创建、默认参数等；
import torch as th
from guided_diffusion.train_util import TrainLoop  #训练核心循环逻辑；


def main():

    args = create_argparser().parse_args()
    if not os.path.isabs(args.save_path):
        args.save_path = os.path.abspath(args.save_path)

    #device = dist_util.dev(args.gpu_id)
    device = th.device(f"cuda:{args.gpu_id}" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
        th.cuda.set_device(args.gpu_id)
    #dist_util.setup_dist()
    logger.configure(args.save_path)

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
                            "condition_channels"
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        shuffle=args.shuffle,
    )

    logger.log("training...")

    # TrainLoop才是主要修改的地方
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

        resume_checkpoint=args.resume_checkpoint,
        resume_step = args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

        save_path=args.save_path,
    ).run_loop()

def create_argparser():
    defaults = dict(
        # ==== 运行相关 ====
        gpu_id=0,
        save_path="checkpoints/first_test",

        # ==== 数据相关 ====
        data_mode='npy',
        # data_dir1 is the CT-FDK label directory, data_dir2 is the CL-FDK input directory.
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/ct_label_npy",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/cl_label_npy",
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,

        # ==== 模型结构相关 ====
        image_size=768,
        condition_channels=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,

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
        lr_anneal_steps=300000,
        microbatch=-1,
        ema_rate="0,0.9999",
        log_interval=1000,
        save_interval=50000,
        resume_checkpoint="",
        resume_step = 0,
        fp16_scale_growth=1e-3,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
