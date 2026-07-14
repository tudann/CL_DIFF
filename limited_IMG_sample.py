"""
Sample reconstructed slices from paired CT/CL npy volumes.
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch as th
from functools import partial
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

from guided_diffusion import logger
from guided_diffusion.image_datasets import load_CL_IMG_data, normalize_image
from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict, CL_IMG_create_model_and_diffusion


def indicate(img1, img2):
    if len(img1.shape) == 3:
        batch = img1.shape[0]
        psnr0 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)
        for i in range(batch):
            t1 = img1[i, ...] / np.max(img1[i, ...])
            t2 = img2[i, ...] / np.max(img2[i, ...])
            psnr0[i] = psnr(t1, t2, data_range=1)
            ssim0[i] = ssim(t1, t2, data_range=1)
            mse0[i] = mse(t1, t2)
        return psnr0, ssim0, mse0
    img1 /= img1.max()
    img2 /= img2.max()
    return psnr(img1, img2, data_range=1), ssim(img1, img2, data_range=1), mse(img1, img2)


def main():
    args = create_argparser().parse_args()
    device = th.device(f"cuda:{args.gpu_id}" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
        th.cuda.set_device(args.gpu_id)

    model, diffusion = CL_IMG_create_model_and_diffusion(
        **args_to_dict(
            args,
            [
                "image_size", "num_channels", "num_res_blocks", "num_heads",
                "num_heads_upsample", "num_head_channels", "attention_resolutions",
                "channel_mult", "dropout", "use_checkpoint", "use_scale_shift_norm",
                "resblock_updown", "use_fp16", "use_new_attention_order", "learn_sigma",
                "diffusion_steps", "noise_schedule", "timestep_respacing", "use_kl",
                "predict_xstart", "rescale_timesteps", "rescale_learned_sigmas",
                "condition_channels",
            ],
        ),
        device=device,
    )
    model.load_state_dict(th.load(args.model_path, map_location=device))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_CL_IMG_data(
        data_dir1=args.data_dir1,
        data_dir2=args.data_dir2,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode="test",
        num_input_slices=args.condition_channels,
        crop_x_start=args.crop_x_start,
        crop_x_end=args.crop_x_end,
        crop_y_start=args.crop_y_start,
        crop_y_end=args.crop_y_end,
        use_mmap=args.use_mmap,
    )

    run_sampler = partial(diffusion.CL_IMG_sample_loop_test)
    os.makedirs("./result/npy/re", exist_ok=True)

    metrics_list = []
    for data_batch in data:
        img, bad_img, sample_name = data_batch
        sample_name = sample_name[0]
        img_name = sample_name.rsplit("_z", 1)[0]
        z_idx = int(sample_name.rsplit("_z", 1)[1])

        bad_img = bad_img.to(device)
        result_img = run_sampler(
            model=model,
            bad_img=bad_img,
            shape=bad_img.shape,
            slover_data=args.slover_data,
            img_bz=bad_img,
        )
        result_img = th.mean(result_img, 0, keepdim=True)
        result_img = np.squeeze(result_img.cpu().numpy())
        gt_img = np.squeeze(img[0].numpy())

        re_path = f"./result/npy/re/{img_name}_z{z_idx:03d}.png"
        cv2.imwrite(re_path, (normalize_image(result_img) * 255).astype(np.uint8))

        result_img_norm = normalize_image(result_img)
        gt_img_norm = normalize_image(gt_img)
        p, s, m = indicate(result_img_norm[None, ...], gt_img_norm[None, ...])
        metrics_list.append([f"{img_name}_z{z_idx:03d}", float(p), float(s), float(m) * 1000])

    with open("./result/npy/image_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageName", "PSNR", "SSIM", "MSE"])
        writer.writerows(metrics_list)


def create_argparser():
    defaults = dict(
        gpu_id=0,
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/ct_label_npy",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/cl_label_npy",
        batch_size=1,
        model_path="/home/lqg/code_8T/24/lt/checkpoints/no_dab_25d/ema_npy_0.9999_000000.pt",
        slover_data="no",
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
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="ddim50",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
