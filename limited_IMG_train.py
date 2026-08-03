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
from guided_diffusion.unet_v2 import load_v1_state_dict #从 v1 checkpoint 热启动；
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
                            "condition_channels",
                            # v2 骨干相关
                            "arch", "freq_resolutions", "transformer_depth", "freq_depth",
                            "mlp_ratio", "freq_expansion", "freq_patch_size",
                        ]
        ),
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"arch={args.arch}, trainable parameters: {n_params / 1e6:.2f} M")

    if args.init_from_v1:
        if args.resume_checkpoint:
            raise ValueError(
                "--init_from_v1 and --resume_checkpoint are mutually exclusive"
            )
        if args.arch == "v1":
            raise ValueError("--init_from_v1 is only meaningful for a v2 arch")
        logger.log(f"warm starting from v1 checkpoint: {args.init_from_v1}")
        load_v1_state_dict(
            model, th.load(args.init_from_v1, map_location="cpu")
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
        loss_log_interval=args.loss_log_interval,

        resume_checkpoint=args.resume_checkpoint,
        resume_step = args.resume_step,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_clip=args.grad_clip,

        save_path=args.save_path,
    ).run_loop()

def create_argparser():
    defaults = dict(
        # ==== 运行相关 ====
        gpu_id=0,
        save_path="checkpoints/phantom_label_guss",

        # ==== 数据相关 ====
        data_mode='npy',
        # data_dir1 is the CT-FDK label directory, data_dir2 is the CL-FDK input directory.
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/phantom_guss",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/cl_label_npy",
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
        # arch: v1  = 原始纯卷积 UNet（基线，可加载 v1 checkpoint）
        #       v2a = UNet + 粗尺度稠密自注意力（DiT block）
        #       v2b = v2a 再叠加中尺度频域注意力（FSAS/DFFN block）
        arch="v2b",
        image_size=768,
        condition_channels=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        # 这两项按“特征图边长”解释：768 分辨率下各层依次为
        # 768/384/192/96/48/24/12。稠密注意力放粗尺度，频域块放中尺度。
        attention_resolutions="24,12",
        freq_resolutions="96,48",
        transformer_depth=1,
        freq_depth=1,
        mlp_ratio=4.0,
        freq_expansion=2.66,
        freq_patch_size=8,
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
        # batch_size 是优化器看到的等效批量，microbatch 是单次前向的实际批量，
        # 两者配合即为梯度累积。batch_size=16/microbatch=2 的显存占用与
        # batch_size=2 相同，但等效批量提升 8 倍。
        batch_size=16,
        microbatch=2,
        schedule_sampler="uniform",
        weight_decay=0.0,
        # 与 v1 的算力预算对齐：v1 是 300000 步 × 2 = 600000 次样本前向，
        # 等效批量 16 下对应 37500 步。
        lr_anneal_steps=37500,
        lr_warmup_steps=2000,
        grad_clip=1.0,
        use_bf16=True,
        # 0.9999 的平均窗口约 10000 步，在 37500 步的预算里会占到 27%，EMA 权重
        # 会明显滞后。0.999 的窗口约 1000 步，与总步数的相对比例和 v1 一致。
        ema_rate="0,0.999",
        log_interval=200,
        # 总步数缩短后必须同步调小，否则整个训练只会落下两个 checkpoint。
        save_interval=2500,
        loss_log_interval=1,
        resume_checkpoint="",
        resume_step = 0,
        # 从 v1 checkpoint 热启动 v2（与 resume_checkpoint 互斥）。
        # adaLN 门控初值为 0，因此迁移过来的权重行为与 v1 完全一致。
        init_from_v1="",
        fp16_scale_growth=1e-3,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
