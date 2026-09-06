"""
Sample reconstructed slices from paired CT/CL npy volumes.
"""
import argparse
import csv
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch as th
from functools import partial
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

from guided_diffusion import logger
from guided_diffusion.image_datasets import (
    load_CL_IMG_data,
    normalize_image,
    validate_normalization_mode,
    volume_normalization_range,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    load_local_config,
    local_config_path,
    CL_IMG_create_model_and_diffusion,
)


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
        normalization_mode="volume",
        percentile_low=1.0,
        percentile_high=99.0,
        independent_slices=False,
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
        self.normalization_mode = validate_normalization_mode(normalization_mode)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.independent_slices = bool(independent_slices)
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

        self.input_range = None
        self.label_range = None
        if self.independent_slices:
            print(
                "Independent slice mode: each slice is repeated across all "
                f"{self.num_input_slices} condition channels."
            )
        elif self.normalization_mode in ("volume", "percentile"):
            self.input_range = volume_normalization_range(
                self.input_volume, self.crop_x, self.crop_y,
                self.normalization_mode, self.percentile_low, self.percentile_high,
            )
            if self.label_volume is not None:
                self.label_range = volume_normalization_range(
                    self.label_volume, self.crop_x, self.crop_y,
                    self.normalization_mode, self.percentile_low, self.percentile_high,
                )

    def __len__(self):
        return self.input_volume.shape[2]

    def __iter__(self):
        for z in range(len(self)):
            yield self[z]

    def __getitem__(self, z):
        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        if self.independent_slices:
            cropped = np.asarray(self.input_volume[x0:x1, y0:y1, z], dtype=np.float32)
            normalized = normalize_image(cropped)
            cond_stack = th.from_numpy(
                np.stack([normalized] * self.num_input_slices, axis=0)[None].astype(np.float32)
            )
            label_slice = None
            if self.label_volume is not None:
                label_slice = np.asarray(
                    self.label_volume[x0:x1, y0:y1, z], dtype=np.float32
                )
                label_slice = normalize_image(label_slice)[None].astype(np.float32)
            return label_slice, cond_stack, f"{self.stem}_z{z:03d}"

        z_count = self.input_volume.shape[2]
        half = self.num_input_slices // 2
        z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

        cond_slices = [
            np.asarray(self.input_volume[x0:x1, y0:y1, zi], dtype=np.float32)
            for zi in z_indices
        ]
        cond_stack = np.stack(
            [normalize_image(slice_, self.input_range) for slice_ in cond_slices],
            axis=0,
        )
        cond_stack = th.from_numpy(cond_stack[None, ...].astype(np.float32))

        if self.label_volume is None:
            label_slice = None
        else:
            label_slice = np.asarray(self.label_volume[x0:x1, y0:y1, z], dtype=np.float32)
            label_slice = normalize_image(
                label_slice, self.label_range
            )[None, :, :].astype(np.float32)

        return label_slice, cond_stack, f"{self.stem}_z{z:03d}"


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


class SingleCLRawSliceDataset:
    def __init__(
        self,
        input_raw_dir,
        image_size=768,
        num_input_slices=3,
        crop_x=(127, 895),
        crop_y=(127, 895),
        raw_height=1024,
        raw_width=1024,
        raw_dtype="float32",
        raw_pattern="*.raw",
        raw_order="C",
        volume_name="",
        normalization_mode="volume",
        percentile_low=1.0,
        percentile_high=99.0,
        independent_raws=False,
    ):
        if num_input_slices % 2 != 1:
            raise ValueError("num_input_slices must be odd, e.g. 3 for [z-1,z,z+1].")

        self.input_raw_dir = input_raw_dir
        self.raw_files = sorted(glob.glob(os.path.join(input_raw_dir, raw_pattern)), key=natural_sort_key)
        if not self.raw_files:
            raise ValueError(f"No raw files found in {input_raw_dir} with pattern {raw_pattern}.")

        self.image_size = image_size
        self.num_input_slices = num_input_slices
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.raw_height = raw_height
        self.raw_width = raw_width
        self.raw_dtype = np.dtype(raw_dtype)
        self.raw_order = raw_order
        self.independent_raws = bool(independent_raws)
        requested_mode = validate_normalization_mode(normalization_mode)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.normalization_mode = "slice" if self.independent_raws else requested_mode
        self.expected_values = raw_height * raw_width
        self.stem = volume_name or os.path.basename(os.path.abspath(input_raw_dir))

        crop_h = crop_x[1] - crop_x[0]
        crop_w = crop_y[1] - crop_y[0]
        if crop_h != image_size or crop_w != image_size:
            raise ValueError(f"Crop size ({crop_h}, {crop_w}) does not match image_size={image_size}.")

        self.input_range = None
        self.slice_ranges = [None] * len(self.raw_files)
        if self.independent_raws:
            print(
                f"Independent RAW mode: repeating each file across "
                f"{self.num_input_slices} condition channels."
            )
        elif self.normalization_mode in ("volume", "percentile"):
            self.input_range = self._compute_volume_range()

    def __len__(self):
        return len(self.raw_files)

    def __iter__(self):
        for z in range(len(self)):
            yield self[z]

    def _read_raw_slice(self, z):
        path = self.raw_files[z]
        data = np.fromfile(path, dtype=self.raw_dtype)
        if data.size != self.expected_values:
            raise ValueError(
                f"Raw file size mismatch: {path} has {data.size} values, "
                f"expected {self.expected_values} for shape ({self.raw_height}, {self.raw_width})."
            )
        return data.reshape((self.raw_height, self.raw_width), order=self.raw_order)

    def _compute_volume_range(self):
        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        if self.normalization_mode == "percentile":
            values = np.concatenate([
                self._read_raw_slice(z)[x0:x1, y0:y1].ravel()
                for z in range(len(self))
            ])
            low = float(np.percentile(values, self.percentile_low))
            high = float(np.percentile(values, self.percentile_high))
            return (low, high) if high > low else (float(np.min(values)), float(np.max(values)))
        min_value = float("inf")
        max_value = float("-inf")
        for z in range(len(self)):
            cropped = self._read_raw_slice(z)[x0:x1, y0:y1]
            min_value = min(min_value, float(np.min(cropped)))
            max_value = max(max_value, float(np.max(cropped)))
        return min_value, max_value

    def __getitem__(self, z):
        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        if self.independent_raws:
            cropped = np.asarray(self._read_raw_slice(z)[x0:x1, y0:y1], dtype=np.float32)
            self.slice_ranges[z] = (float(np.min(cropped)), float(np.max(cropped)))
            normalized = normalize_image(cropped)
            cond_stack = th.from_numpy(
                np.stack([normalized] * self.num_input_slices, axis=0)[None].astype(np.float32)
            )
            sample_name = os.path.splitext(os.path.basename(self.raw_files[z]))[0]
            return None, cond_stack, sample_name

        z_count = len(self)
        half = self.num_input_slices // 2
        z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

        cond_slices = [
            np.asarray(self._read_raw_slice(zi)[x0:x1, y0:y1], dtype=np.float32)
            for zi in z_indices
        ]
        cond_stack = np.stack(
            [normalize_image(slice_, self.input_range) for slice_ in cond_slices],
            axis=0,
        )
        cond_stack = th.from_numpy(cond_stack[None, ...].astype(np.float32))
        return None, cond_stack, f"{self.stem}_z{z:03d}"


def indicate(img1, img2):
    if len(img1.shape) == 3:
        batch = img1.shape[0]
        psnr0 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)
        for i in range(batch):
            t1 = np.clip(img1[i, ...], 0.0, 1.0)
            t2 = np.clip(img2[i, ...], 0.0, 1.0)
            psnr0[i] = psnr(t1, t2, data_range=1)
            ssim0[i] = ssim(t1, t2, data_range=1)
            mse0[i] = mse(t1, t2)
        return psnr0, ssim0, mse0
    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)
    return psnr(img1, img2, data_range=1), ssim(img1, img2, data_range=1), mse(img1, img2)


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def save_xz_pngs(volume, output_dir, stem):
    """Save all xz views from an (x, y, z) volume."""
    os.makedirs(output_dir, exist_ok=True)
    for y_idx in range(volume.shape[1]):
        cv2.imwrite(
            os.path.join(output_dir, f"{stem}_xz_y{y_idx:03d}.png"),
            to_uint8(volume[:, y_idx, :]),
        )


def normalize_volume(volume):
    """Normalize one complete H x W x Z volume with a shared value range."""
    min_value = float(np.min(volume))
    max_value = float(np.max(volume))
    if max_value == min_value:
        return np.zeros_like(volume, dtype=np.float32), min_value, max_value
    normalized = (volume - min_value) / (max_value - min_value)
    return normalized.astype(np.float32), min_value, max_value


def affine_align_pred_to_gt(pred, gt):
    """Fit pred_aligned = scale * pred + bias to the label."""
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
    gt = np.clip(gt.astype(np.float64), 0.0, 1.0)
    x = pred.ravel()
    y = gt.ravel()
    if np.allclose(x, x[0]):
        aligned = pred + float(np.mean(y - x))
    else:
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        aligned = scale * pred + bias
    return np.clip(aligned, 0.0, 1.0).astype(np.float32)


def prepare_metric_images(pred, gt, metrics_align="none"):
    pred_metric = np.clip(pred, 0.0, 1.0).astype(np.float32)
    gt_metric = np.clip(gt, 0.0, 1.0).astype(np.float32)
    if metrics_align == "affine":
        pred_metric = affine_align_pred_to_gt(pred_metric, gt_metric)
    elif metrics_align != "none":
        raise ValueError("metrics_align must be 'none' or 'affine'.")
    return pred_metric, gt_metric


def draw_centered_text(canvas, text, x0, x1, y, font_scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = x0 + max((x1 - x0 - text_w) // 2, 0)
    cv2.putText(canvas, text, (x, y + text_h), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def save_comparison(path, cl_img, re_img, gt_img=None, metrics=None):
    cl_u8 = to_uint8(cl_img)
    re_u8 = to_uint8(re_img)
    panels = [cl_u8, re_u8]
    titles = ["CL-FDK input", "Diffusion result"]

    if gt_img is not None:
        panels.append(to_uint8(gt_img))
        titles.append("CT-FDK target")

    h, w = panels[0].shape
    top_h = 78
    gap = 12
    canvas_w = len(panels) * w + (len(panels) - 1) * gap
    canvas_h = top_h + h
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if metrics is not None:
        p, s, m = metrics
        metric_text = f"PSNR: {p:.2f}  SSIM: {s:.4f}  MSE(x1000): {m:.3f}"
    else:
        metric_text = "No CT label: metrics unavailable"
    cv2.putText(
        canvas,
        metric_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for idx, (panel, title) in enumerate(zip(panels, titles)):
        x0 = idx * (w + gap)
        x1 = x0 + w
        draw_centered_text(canvas, title, x0, x1, 44, font_scale=0.75, thickness=2)
        canvas[top_h:top_h + h, x0:x1] = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(path, canvas)


def save_slice_outputs(
    re_dir, comp_dir, img_name, z_idx, cl_img, result_img, gt_img=None,
    metrics_align="none",
):
    """Save one slice and calculate its optional paired-image metrics."""
    slice_name = img_name if z_idx is None else f"{img_name}_z{z_idx:03d}"
    re_path = os.path.join(re_dir, f"{slice_name}.png")
    cv2.imwrite(re_path, to_uint8(result_img))

    metrics = None
    metrics_row = None
    result_for_compare = result_img
    if gt_img is not None:
        result_for_compare, gt_img = prepare_metric_images(
            result_img, gt_img, metrics_align
        )
        p, s, m = indicate(result_for_compare[None, ...], gt_img[None, ...])
        metrics = (float(p), float(s), float(m) * 1000)
        metrics_row = [slice_name, metrics[0], metrics[1], metrics[2]]

    comp_path = os.path.join(comp_dir, f"{slice_name}_comparison.png")
    save_comparison(comp_path, cl_img, result_for_compare, gt_img=gt_img, metrics=metrics)
    return metrics_row


def main():
    args = create_argparser().parse_args()
    if args.independent_raws and not args.input_raw_dir and not args.input_npy:
        print("independent_raws is ignored in paired directory mode.")
        args.independent_raws = False
    if args.sampler == "ddim":
        # True conditional DDIM with the configured number of steps.
        if args.ddim_steps <= 0:
            raise ValueError("ddim_steps must be positive.")
        args.timestep_respacing = f"ddim{args.ddim_steps}"
    elif args.sampler == "p_sample":
        # Original stochastic p_sample path with the configured number of steps.
        if args.p_sample_steps <= 0:
            raise ValueError("p_sample_steps must be positive.")
        args.timestep_respacing = f"ddim{args.p_sample_steps}"
    else:
        raise ValueError("sampler must be 'ddim' or 'p_sample'.")

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
    print(f"Sampler: {args.sampler}, inference steps: {diffusion.num_timesteps}")

    if args.input_raw_dir:
        data = SingleCLRawSliceDataset(
            input_raw_dir=args.input_raw_dir,
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=(args.crop_x_start, args.crop_x_end),
            crop_y=(args.crop_y_start, args.crop_y_end),
            raw_height=args.raw_height,
            raw_width=args.raw_width,
            raw_dtype=args.raw_dtype,
            raw_pattern=args.raw_pattern,
            raw_order=args.raw_order,
            volume_name=args.raw_volume_name,
            normalization_mode=args.normalization_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            independent_raws=args.independent_raws,
        )
    elif args.input_npy:
        data = SingleCLVolumeDataset(
            input_npy=args.input_npy,
            label_npy=args.label_npy,
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=(args.crop_x_start, args.crop_x_end),
            crop_y=(args.crop_y_start, args.crop_y_end),
            use_mmap=args.use_mmap,
            normalization_mode=args.normalization_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            independent_slices=args.independent_raws,
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
            normalization_mode=args.normalization_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )

    if args.metrics_align not in ("none", "affine"):
        raise ValueError("metrics_align must be 'none' or 'affine'.")
    input_range = getattr(data, "input_range", None)
    if (args.save_input_scale_npy or args.save_input_scale_png) and not args.independent_raws:
        if input_range is None or args.normalization_mode not in ("volume", "percentile"):
            raise ValueError("Input-scale output requires volume or percentile normalization.")
        input_min, input_max = input_range
        if not np.isfinite(input_min) or not np.isfinite(input_max) or input_max <= input_min:
            raise ValueError(f"Invalid input normalization range: {input_range}")
    else:
        input_min = input_max = None

    if args.sampler == "ddim":
        run_sampler = partial(diffusion.CL_IMG_ddim_sample_loop_test, eta=0.0)
    else:
        run_sampler = partial(
            diffusion.CL_IMG_sample_loop_test,
            warm_start_strength=args.warm_start_strength,
        )
        print(
            f"P-sample warm-start strength: {args.warm_start_strength:.3f}"
        )
    re_dir = os.path.join(args.output_dir, "re")
    global_re_dir = os.path.join(args.output_dir, "re_global")
    input_scale_re_dir = os.path.join(args.output_dir, "re_input_scale")
    comp_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(re_dir, exist_ok=True)
    os.makedirs(global_re_dir, exist_ok=True)
    if args.save_input_scale_png:
        os.makedirs(input_scale_re_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    metrics_list = []
    volume_slices = []
    gt_volume_slices = []
    output_futures = []
    with ThreadPoolExecutor(max_workers=1) as output_executor, th.inference_mode():
        for sample_idx, data_batch in enumerate(data):
            if args.max_samples > 0 and sample_idx >= args.max_samples:
                break

            img, bad_img, sample_name = data_batch
            if isinstance(sample_name, (list, tuple)):
                sample_name = sample_name[0]
            per_file_raw = bool(args.input_raw_dir and args.independent_raws)
            if per_file_raw:
                img_name, z_idx = sample_name, None
            else:
                img_name = sample_name.rsplit("_z", 1)[0]
                z_idx = int(sample_name.rsplit("_z", 1)[1])

            cond_img = bad_img.to(device, non_blocking=True)
            center_channel = cond_img.shape[1] // 2
            start_img = cond_img[:, center_channel:center_channel + 1]
            cl_img = np.squeeze(start_img[0, 0].detach().cpu().numpy()).copy()
            result_img = run_sampler(
                model=model,
                bad_img=start_img,
                shape=start_img.shape,
                slover_data=args.slover_data,
                img_bz=cond_img,
            )
            result_img = np.squeeze(result_img[0, 0].cpu().numpy()).copy()
            if not per_file_raw:
                volume_slices.append(result_img.astype(np.float32))
            elif args.save_re_npy or args.save_input_scale_npy or args.save_input_scale_png:
                result_norm = np.clip(result_img, 0.0, 1.0).astype(np.float32)
                if args.save_re_npy:
                    np.save(os.path.join(args.output_dir, f"{img_name}_re.npy"), result_norm)
                if args.save_input_scale_npy or args.save_input_scale_png:
                    slice_min, slice_max = data.slice_ranges[sample_idx]
                    if slice_max <= slice_min:
                        raise ValueError(f"Invalid input range for {img_name}: {(slice_min, slice_max)}")
                    input_scale_slice = result_norm * (slice_max - slice_min) + slice_min
                    if args.save_input_scale_npy:
                        np.save(
                            os.path.join(args.output_dir, f"{img_name}_re_input_scale.npy"),
                            input_scale_slice.astype(np.float32),
                        )
                    if args.save_input_scale_png:
                        cv2.imwrite(
                            os.path.join(input_scale_re_dir, f"{img_name}.png"),
                            to_uint8(result_norm),
                        )

            gt_img = None
            if img is not None:
                gt_img = np.squeeze(img[0].numpy() if hasattr(img, "numpy") else img).copy()
                if not per_file_raw:
                    gt_volume_slices.append(np.clip(gt_img, 0.0, 1.0).astype(np.float32))

            output_futures.append(
                output_executor.submit(
                    save_slice_outputs,
                    re_dir,
                    comp_dir,
                    img_name,
                    z_idx,
                    cl_img,
                    result_img,
                    gt_img,
                    args.metrics_align,
                )
            )

    metrics_list = [row for future in output_futures if (row := future.result()) is not None]

    if volume_slices:
        volume = np.stack(volume_slices, axis=-1)
        volume_min = float(np.min(volume))
        volume_max = float(np.max(volume))
        if args.normalize_output_volume:
            volume, _, _ = normalize_volume(volume)
        else:
            volume = np.clip(volume, 0.0, 1.0).astype(np.float32)
        if args.save_re_npy:
            np.save(os.path.join(args.output_dir, f"{img_name}_re.npy"), volume)
        if args.save_input_scale_npy or args.save_input_scale_png:
            input_scale_volume = (volume * (input_max - input_min) + input_min).astype(np.float32)
            if args.save_input_scale_npy:
                np.save(
                    os.path.join(args.output_dir, f"{img_name}_re_input_scale.npy"),
                    input_scale_volume,
                )
            if args.save_input_scale_png:
                for output_z in range(input_scale_volume.shape[2]):
                    scaled = np.clip(
                        (input_scale_volume[:, :, output_z] - input_min) / (input_max - input_min),
                        0.0,
                        1.0,
                    )
                    cv2.imwrite(
                        os.path.join(input_scale_re_dir, f"{img_name}_z{output_z:03d}.png"),
                        to_uint8(scaled),
                    )
        print(
            f"Output volume before saving: min={volume_min:.6g}, "
            f"max={volume_max:.6g}, normalized={args.normalize_output_volume}"
        )

        if args.save_global_png:
            for z_idx in range(volume.shape[2]):
                global_slice = (np.clip(volume[:, :, z_idx], 0.0, 1.0) * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(global_re_dir, f"{img_name}_z{z_idx:03d}.png"),
                    global_slice,
                )
        if args.save_xz_png:
            save_xz_pngs(volume, os.path.join(args.output_dir, "xz"), img_name)
            if len(gt_volume_slices) == volume.shape[2]:
                save_xz_pngs(
                    np.stack(gt_volume_slices, axis=-1),
                    os.path.join(args.output_dir, "xz_gt"),
                    f"{img_name}_gt",
                )

    if metrics_list:
        mean_row = [
            "MEAN",
            float(np.mean([row[1] for row in metrics_list])),
            float(np.mean([row[2] for row in metrics_list])),
            float(np.mean([row[3] for row in metrics_list])),
        ]
        with open(os.path.join(args.output_dir, "image_metrics.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ImageName", "PSNR", "SSIM", "MSE"])
            writer.writerows(metrics_list)
            writer.writerow(mean_row)
        print(
            f"Mean metrics: PSNR={mean_row[1]:.4f}, SSIM={mean_row[2]:.4f}, "
            f"MSE(x1000)={mean_row[3]:.4f}"
        )


def create_argparser():
    defaults = dict(
        gpu_id=1,
        # 测试其他数据需要为空
        # input_raw_dir="",
        # stub数据
        # input_raw_dir="/home/lqg/code_8T/24/lt/data_make/17_360view/slice",
        # 蓝牙数据20
        # input_raw_dir="/home/lqg/code_8T/24/lt/data_make/20_19_47/1024",
        # 蓝牙数据23
        input_raw_dir="/home/lqg/code_8T/24/lt/data_make/23_23_43/1024",
        raw_height=1024,
        raw_width=1024,
        raw_dtype="float32",
        raw_pattern="*.raw",
        raw_order="C",
        raw_volume_name="real_fdk",
        independent_raws=False,

        # # 同源phantom路径
        # input_npy="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/evulate_data/pcb_phantom_npy/cl_fdk_npy/test_phantom_0001_cl_fdk.npy",
        # label_npy="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/evulate_data/pcb_phantom_npy/test_phantom_0001.npy",
        # mpcb路径
        input_npy="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/evulate_data/mpcb_phantom_npy/cl_fdk_npy/phantom_0001_cl_fdk.npy",
        label_npy="/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/evulate_data/mpcb_phantom_npy/phantom_0001.npy",
        data_dir1="",
        data_dir2="",
        batch_size=1,
        sampler="p_sample",  # ddim or p_sample
        ddim_steps=25,
        p_sample_steps=50,
        warm_start_strength=0.8,
        save_global_png=False,
        save_xz_png=False,
        save_re_npy=True,
        normalize_output_volume=False,
        save_input_scale_npy=False,
        save_input_scale_png=False,
        metrics_align="none",

        # [CT] label 模型训练路径
        # model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/checkpoints/first_test/ema_npy_0.9999_250000.pt",
        # [Phantom+guss平滑] label 模型训练路径
        model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/CL_DIFF/checkpoints/phantom_label_guss_lowcontrast_edge5.0_shareall/ema_npy_0.9999_150000.pt",

        output_dir="/home/lqg/code_8T/24/lt/CL_DIFF_v1/result/lowcontrast_edge5.0_shareall/blue23_150000_p50-warm0.8",
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
        timestep_respacing="",  # selected automatically from sampler and step count
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
        normalization_mode="volume",
        percentile_low=1.0,
        percentile_high=99.0,
    )
    config_path = local_config_path("sample")
    defaults = load_local_config(defaults, config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=config_path,
        help="Optional YAML config file (default: sample.ymal).",
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
