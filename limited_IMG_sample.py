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
from torch.utils.data import DataLoader

from guided_diffusion.image_datasets import (
    CLVolumeSliceDataset,
    _pair_npy_files,
    normalize_image,
    validate_normalization_mode,
    volume_normalization_range,
)
from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict, CL_IMG_create_model_and_diffusion
from local_config import apply_local_overrides


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
                "Independent slice mode: each z is tested alone by repeating "
                f"the current slice {self.num_input_slices} times (no 2.5D neighbors). "
                "Per-slice min-max is used; volume/percentile windows are ignored."
            )
        elif self.normalization_mode in ("volume", "percentile"):
            self.input_range = volume_normalization_range(
                self.input_volume,
                self.crop_x,
                self.crop_y,
                self.normalization_mode,
                self.percentile_low,
                self.percentile_high,
            )
            print(
                f"CL normalization ({self.normalization_mode}): "
                f"range=({self.input_range[0]:.6g}, {self.input_range[1]:.6g})"
            )
            if self.label_volume is not None:
                self.label_range = volume_normalization_range(
                    self.label_volume,
                    self.crop_x,
                    self.crop_y,
                    self.normalization_mode,
                    self.percentile_low,
                    self.percentile_high,
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
            normalized = normalize_image(cropped, None)
            cond_stack = np.stack([normalized] * self.num_input_slices, axis=0)
            cond_stack = th.from_numpy(cond_stack[None, ...].astype(np.float32))
            if self.label_volume is None:
                label_slice = None
            else:
                label_slice = np.asarray(
                    self.label_volume[x0:x1, y0:y1, z], dtype=np.float32
                )
                label_slice = normalize_image(label_slice, None)[None, :, :].astype(
                    np.float32
                )
            return label_slice, cond_stack, f"{self.stem}_z{z:03d}"

        z_count = self.input_volume.shape[2]
        half = self.num_input_slices // 2
        z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
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
        self.slice_ranges = [None] * len(self.raw_files)

        crop_h = crop_x[1] - crop_x[0]
        crop_w = crop_y[1] - crop_y[0]
        if crop_h != image_size or crop_w != image_size:
            raise ValueError(f"Crop size ({crop_h}, {crop_w}) does not match image_size={image_size}.")

        self.input_range = None
        if self.independent_raws:
            if requested_mode == "percentile":
                print(
                    "Warning: independent_raws forces per-file min-max; "
                    "percentile stretch is ignored. Set independent_raws: false "
                    "to use p1-p99 on the whole RAW stack."
                )
            print(
                f"Independent RAW mode: {len(self.raw_files)} files from {input_raw_dir}. "
                f"2.5D input repeats each slice {self.num_input_slices} times; "
                "normalization is per file."
            )
        elif self.normalization_mode in ("volume", "percentile"):
            self.input_range = self._compute_volume_range()
            print(
                f"CL normalization ({self.normalization_mode}): "
                f"range=({self.input_range[0]:.6g}, {self.input_range[1]:.6g})"
            )

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
            chunks = [
                self._read_raw_slice(z)[x0:x1, y0:y1].ravel()
                for z in range(len(self))
            ]
            values = np.concatenate(chunks)
            low = float(np.percentile(values, self.percentile_low))
            high = float(np.percentile(values, self.percentile_high))
            if high <= low:
                return float(np.min(values)), float(np.max(values))
            return low, high
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
            normalized = normalize_image(cropped, None)
            cond_stack = np.stack([normalized] * self.num_input_slices, axis=0)
            cond_stack = th.from_numpy(cond_stack[None, ...].astype(np.float32))
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


def normalize_volume(volume):
    """Normalize one complete H x W x Z volume with a shared value range."""
    min_value = float(np.min(volume))
    max_value = float(np.max(volume))
    if max_value == min_value:
        return np.zeros_like(volume, dtype=np.float32), min_value, max_value
    normalized = (volume - min_value) / (max_value - min_value)
    return normalized.astype(np.float32), min_value, max_value


def _valid_value_range(value_range):
    if value_range is None:
        return False
    lo, hi = value_range
    return np.isfinite(lo) and np.isfinite(hi) and hi > lo


def denormalize_from_range(img, value_range):
    lo, hi = value_range
    return img.astype(np.float32) * (hi - lo) + lo


def normalize_to_range(img, value_range):
    lo, hi = value_range
    if hi == lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def remap_pred_to_label_range(pred, input_range, label_range):
    """Map model output from CL-normalized [0,1] into the GT label window [0,1]."""
    pred_raw = denormalize_from_range(pred, input_range)
    return normalize_to_range(pred_raw, label_range)


def prepare_metric_images(pred, gt, input_range, label_range, unified_label_range):
    gt_metric = np.clip(gt, 0.0, 1.0).astype(np.float32)
    if (
        not unified_label_range
        or not _valid_value_range(input_range)
        or not _valid_value_range(label_range)
    ):
        return np.clip(pred, 0.0, 1.0).astype(np.float32), gt_metric
    pred_metric = remap_pred_to_label_range(pred, input_range, label_range)
    return pred_metric, gt_metric


def resolve_metric_ranges(data_source, data_kind, sample_idx, z_idx):
    if data_source is None:
        return None, None

    if data_kind == "volume":
        ds = data_source
        if (
            ds.input_range is not None
            and ds.label_range is not None
            and _valid_value_range(ds.input_range)
            and _valid_value_range(ds.label_range)
        ):
            return ds.input_range, ds.label_range
        if ds.independent_slices and ds.label_volume is not None and z_idx is not None:
            x0, x1 = ds.crop_x
            y0, y1 = ds.crop_y
            cl_slice = ds.input_volume[x0:x1, y0:y1, z_idx]
            gt_slice = ds.label_volume[x0:x1, y0:y1, z_idx]
            return (
                (float(np.min(cl_slice)), float(np.max(cl_slice))),
                (float(np.min(gt_slice)), float(np.max(gt_slice))),
            )
        return None, None

    if data_kind == "batch":
        ds = data_source
        pair_idx, z = ds.indices[sample_idx]
        label_path = ds.label_paths[pair_idx]
        cond_path = ds.cond_paths[pair_idx]
        if ds.normalization_mode in ("volume", "percentile"):
            input_range = ds._normalization_ranges.get(cond_path)
            label_range = ds._normalization_ranges.get(label_path)
            if _valid_value_range(input_range) and _valid_value_range(label_range):
                return input_range, label_range
        x0, x1 = ds.crop_x
        y0, y1 = ds.crop_y
        cond_volume = ds._load_volume(cond_path)
        label_volume = ds._load_volume(label_path)
        cl_slice = cond_volume[x0:x1, y0:y1, z]
        gt_slice = label_volume[x0:x1, y0:y1, z]
        return (
            (float(np.min(cl_slice)), float(np.max(cl_slice))),
            (float(np.min(gt_slice)), float(np.max(gt_slice))),
        )

    return None, None


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
    re_dir,
    comp_dir,
    img_name,
    z_idx,
    cl_img,
    result_img,
    gt_img=None,
    input_range=None,
    label_range=None,
    unified_label_range=True,
):
    """Save one slice and calculate its optional paired-image metrics."""
    slice_name = img_name if z_idx is None else f"{img_name}_z{z_idx:03d}"
    re_path = os.path.join(re_dir, f"{slice_name}.png")
    cv2.imwrite(re_path, to_uint8(result_img))

    metrics = None
    metrics_row = None
    result_for_compare = result_img
    if gt_img is not None:
        result_for_compare, gt_img_metric = prepare_metric_images(
            result_img,
            gt_img,
            input_range=input_range,
            label_range=label_range,
            unified_label_range=unified_label_range,
        )
        p, s, m = indicate(result_for_compare[None, ...], gt_img_metric[None, ...])
        metrics = (float(p), float(s), float(m) * 1000)
        metrics_row = [slice_name, metrics[0], metrics[1], metrics[2]]
        gt_img = gt_img_metric

    comp_path = os.path.join(comp_dir, f"{slice_name}_comparison.png")
    save_comparison(comp_path, cl_img, result_for_compare, gt_img=gt_img, metrics=metrics)
    return metrics_row


def main():
    args = create_argparser().parse_args()
    if args.independent_raws and not args.input_raw_dir and not args.input_npy:
        print(
            "independent_raws is ignored because both input_raw_dir and "
            "input_npy are empty; data_dir batch mode still uses 2.5D neighbors."
        )
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
                "condition_channels", "use_afr", "afr_kernel_size", "use_ild",
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
        data_source = None
        data_kind = "raw"
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
        data_source = data
        data_kind = "volume"
    else:
        if not args.data_dir1 or not args.data_dir2:
            raise ValueError("data_dir1 and data_dir2 are required when input_npy is empty.")
        pairs = _pair_npy_files(args.data_dir1, args.data_dir2)
        data_source = CLVolumeSliceDataset(
            label_paths=[label for label, _ in pairs],
            cond_paths=[cond for _, cond in pairs],
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=(args.crop_x_start, args.crop_x_end),
            crop_y=(args.crop_y_start, args.crop_y_end),
            use_mmap=args.use_mmap,
            normalization_mode=args.normalization_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            augment_condition=False,
        )
        print("Dataset size:", len(data_source))
        data = DataLoader(
            data_source,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        data_kind = "batch"

    if args.metrics_unified_label_range:
        print(
            "Metrics/comparison use unified GT label range: "
            "pred is denormalized from CL window and renormalized to label window."
        )

    input_value_range = getattr(data, "input_range", None)
    input_min = input_max = None
    if (args.save_input_scale_npy or args.save_input_scale_png) and not args.independent_raws:
        if args.normalization_mode not in ("volume", "percentile") or input_value_range is None:
            raise ValueError(
                "Input-scale outputs require normalization_mode='volume' or "
                "'percentile' with a single RAW or NPY input volume."
            )
        input_min, input_max = input_value_range
        if (
            not np.isfinite(input_min)
            or not np.isfinite(input_max)
            or input_max <= input_min
        ):
            raise ValueError(
                f"Invalid input volume range: ({input_min}, {input_max})."
            )

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
                img_name = sample_name
                z_idx = None
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
                    if (
                        slice_min is None
                        or slice_max is None
                        or not np.isfinite(slice_min)
                        or not np.isfinite(slice_max)
                        or slice_max <= slice_min
                    ):
                        raise ValueError(
                            f"Invalid per-file input range for {img_name}: "
                            f"({slice_min}, {slice_max})."
                        )
                    input_scale_slice = (
                        result_norm * (slice_max - slice_min) + slice_min
                    ).astype(np.float32)
                    if args.save_input_scale_npy:
                        np.save(
                            os.path.join(args.output_dir, f"{img_name}_re_input_scale.npy"),
                            input_scale_slice,
                        )
                    if args.save_input_scale_png:
                        png_slice = (result_norm * 255).astype(np.uint8)
                        cv2.imwrite(
                            os.path.join(input_scale_re_dir, f"{img_name}.png"),
                            png_slice,
                        )

            gt_img = None
            if img is not None:
                gt_img = np.squeeze(img[0].numpy() if hasattr(img, "numpy") else img).copy()

            metric_input_range, metric_label_range = resolve_metric_ranges(
                data_source,
                data_kind,
                sample_idx,
                z_idx,
            )

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
                    metric_input_range,
                    metric_label_range,
                    args.metrics_unified_label_range,
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

        input_scale_volume = None
        if args.save_input_scale_npy or args.save_input_scale_png:
            input_scale_volume = (
                volume * (input_max - input_min) + input_min
            ).astype(np.float32)

        if args.save_input_scale_npy:
            input_scale_path = os.path.join(
                args.output_dir, f"{img_name}_re_input_scale.npy"
            )
            np.save(input_scale_path, input_scale_volume)
            print(
                f"Input-scale output saved: min={float(np.min(input_scale_volume)):.6g}, "
                f"max={float(np.max(input_scale_volume)):.6g}, "
                f"input_range=({input_min:.6g}, {input_max:.6g})"
            )

        if args.save_input_scale_png:
            for z_idx in range(input_scale_volume.shape[2]):
                input_scale_slice = input_scale_volume[:, :, z_idx]
                png_slice = (
                    np.clip(
                        (input_scale_slice - input_min) / (input_max - input_min),
                        0.0,
                        1.0,
                    )
                    * 255
                ).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        input_scale_re_dir, f"{img_name}_z{z_idx:03d}.png"
                    ),
                    png_slice,
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

    if metrics_list:
        with open(os.path.join(args.output_dir, "image_metrics.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ImageName", "PSNR", "SSIM", "MSE"])
            writer.writerows(metrics_list)


def create_argparser():
    defaults = dict(
        gpu_id=0,
        # 测试其他数据需要为空
        # input_raw_dir="",
        # stub数据
        # input_raw_dir="/home/lqg/code_8T/24/lt/data_make/17_360view/slice",
        # 蓝牙数据20
        # input_raw_dir="/home/lqg/code_8T/24/lt/data_make/20_19_47/1024",
        # 蓝牙数据23
        # input_raw_dir="/home/lqg/code_8T/24/lt/data_make/23_23_43/1024",
        # pcb14数据
        input_raw_dir="/home/lqg/code_8T/24/lt/data_make/pcb14/10",
        raw_height=1024,
        raw_width=1024,
        raw_dtype="float32",
        raw_pattern="*.raw",
        raw_order="C",
        raw_volume_name="real_fdk",
        # True: 文件夹内每个 raw 单独测试，2.5D 重复同一张，不做邻层拼接
        independent_raws=True,

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
        warm_start_strength=0.25,
        save_global_png=False,
        save_re_npy=False,
        normalize_output_volume=False,
        # Also save the normalized reconstruction mapped to the input volume range.
        save_input_scale_npy=False,
        # Save PNG slices rendered from the input-scale reconstruction volume.
        save_input_scale_png=False,
        # Remap pred from CL normalization window to GT label window before metrics/display.
        metrics_unified_label_range=True,

        # [CT] label 模型训练路径
        # model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/checkpoints/first_test/ema_npy_0.9999_250000.pt",
        # [Phantom+guss平滑] label 模型训练路径
        # model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/CL_DIFF_attention_24_12/checkpoints/phantom_label_guss_lowcontrast_edge5.0_shareall/ema_npy_0.9999_150000.pt",
        # [ct_degraded] label 模型训练路径
        model_path="/home/lqg/code_8T/24/lt/CL_DIFF_v1/CL_DIFF_attention_24_12/checkpoints/ct_degraded_lowcontrast_edge5.0_shareall/ema_npy_0.9999_150000.pt",
        
        
        output_dir="/home/lqg/code_8T/24/lt/CL_DIFF_v1/result/ct_degraded_attention_edge5.0_shareall/pcb14-10_150000_p50-warm0.5",
        max_samples=0,
        slover_data="no",
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
        use_ild=False,
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
        normalization_mode="volume",  # volume / slice / percentile
        percentile_low=1.0,
        percentile_high=99.0,
    )
    apply_local_overrides(defaults, __file__)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
