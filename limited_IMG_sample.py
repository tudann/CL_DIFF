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


class SingleCLVolumeDataset:
    def __init__(
        self,
        input_npy,
        label_npy="",
        image_size=768,
        num_input_slices=3,
        crop_x=(127, 895),
        crop_y=(127, 895),
        use_mmap=True,
    ):
        if num_input_slices % 2 != 1:
            raise ValueError("num_input_slices must be odd, e.g. 3 for [z-1,z,z+1].")

        mmap_mode = "r" if use_mmap else None
        self.input_path = input_npy
        self.label_path = label_npy
        self.input_volume = np.load(input_npy, mmap_mode=mmap_mode)
        self.label_volume = np.load(label_npy, mmap_mode=mmap_mode) if label_npy else None
        self.image_size = image_size
        self.num_input_slices = num_input_slices
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.stem = os.path.splitext(os.path.basename(input_npy))[0]

        if self.input_volume.ndim != 3:
            raise ValueError(f"Expected input npy layout (x, y, z): {input_npy}")
        if self.label_volume is not None and self.label_volume.shape != self.input_volume.shape:
            raise ValueError(
                f"Label/input shape mismatch: {label_npy} {self.label_volume.shape} vs "
                f"{input_npy} {self.input_volume.shape}"
            )

        crop_h = crop_x[1] - crop_x[0]
        crop_w = crop_y[1] - crop_y[0]
        if crop_h != image_size or crop_w != image_size:
            raise ValueError(f"Crop size ({crop_h}, {crop_w}) does not match image_size={image_size}.")

    def __len__(self):
        return self.input_volume.shape[2]

    def __iter__(self):
        for z in range(len(self)):
            yield self[z]

    def __getitem__(self, z):
        z_count = self.input_volume.shape[2]
        half = self.num_input_slices // 2
        z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        cond_slices = [
            np.asarray(self.input_volume[x0:x1, y0:y1, zi], dtype=np.float32)
            for zi in z_indices
        ]
        cond_stack = np.stack([normalize_image(slice_) for slice_ in cond_slices], axis=0)
        cond_stack = th.from_numpy(cond_stack[None, ...].astype(np.float32))

        if self.label_volume is None:
            label_slice = None
        else:
            label_slice = np.asarray(self.label_volume[x0:x1, y0:y1, z], dtype=np.float32)
            label_slice = normalize_image(label_slice)[None, :, :].astype(np.float32)

        return label_slice, cond_stack, f"{self.stem}_z{z:03d}"


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

    if args.input_npy:
        data = SingleCLVolumeDataset(
            input_npy=args.input_npy,
            label_npy=args.label_npy,
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=(args.crop_x_start, args.crop_x_end),
            crop_y=(args.crop_y_start, args.crop_y_end),
            use_mmap=args.use_mmap,
        )
    else:
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
    re_dir = os.path.join(args.output_dir, "re")
    os.makedirs(re_dir, exist_ok=True)

    metrics_list = []
    volume_slices = []
    for sample_idx, data_batch in enumerate(data):
        if args.max_samples > 0 and sample_idx >= args.max_samples:
            break

        img, bad_img, sample_name = data_batch
        if isinstance(sample_name, (list, tuple)):
            sample_name = sample_name[0]
        img_name = sample_name.rsplit("_z", 1)[0]
        z_idx = int(sample_name.rsplit("_z", 1)[1])

        cond_img = bad_img.to(device, non_blocking=True)
        center_channel = cond_img.shape[1] // 2
        start_img = cond_img[:, center_channel:center_channel + 1]
        result_img = run_sampler(
            model=model,
            bad_img=start_img,
            shape=start_img.shape,
            slover_data=args.slover_data,
            img_bz=cond_img,
        )
        result_img = np.squeeze(result_img[0, 0].cpu().numpy())
        volume_slices.append(result_img.astype(np.float32))

        re_path = os.path.join(re_dir, f"{img_name}_z{z_idx:03d}.png")
        cv2.imwrite(re_path, (normalize_image(result_img) * 255).astype(np.uint8))

        if img is not None:
            gt_img = np.squeeze(img[0].numpy() if hasattr(img, "numpy") else img)
            result_img_norm = normalize_image(result_img)
            gt_img_norm = normalize_image(gt_img)
            p, s, m = indicate(result_img_norm[None, ...], gt_img_norm[None, ...])
            metrics_list.append([f"{img_name}_z{z_idx:03d}", float(p), float(s), float(m) * 1000])

    if volume_slices:
        volume = np.stack(volume_slices, axis=-1)
        np.save(os.path.join(args.output_dir, f"{img_name}_re.npy"), volume)

    if metrics_list:
        with open(os.path.join(args.output_dir, "image_metrics.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ImageName", "PSNR", "SSIM", "MSE"])
            writer.writerows(metrics_list)


def create_argparser():
    defaults = dict(
        gpu_id=0,
        input_npy="",
        label_npy="",
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/ct_label_npy",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/cl_label_npy",
        batch_size=1,
        model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/checkpoints/first_test/ema_npy_0.9999_250000.pt",
        output_dir="./result/npy",
        max_samples=0,
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
