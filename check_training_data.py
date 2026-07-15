import argparse
import os

import imageio.v2 as imageio
import numpy as np

from guided_diffusion.image_datasets import CLVolumeSliceDataset, _pair_npy_files
from guided_diffusion.script_util import add_dict_to_argparser


def summarize_array(name, arr):
    print(
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={float(np.min(arr)):.6g}, max={float(np.max(arr)):.6g}, "
        f"mean={float(np.mean(arr)):.6g}"
    )


def inspect_sample(dataset, idx):
    label, cond, sample_name = dataset[idx]
    pair_idx, z = dataset.indices[idx]
    z_count = dataset._load_volume(dataset.label_paths[pair_idx]).shape[2]
    half = dataset.num_input_slices // 2
    z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

    print(f"\nSample index {idx}: {sample_name}")
    print(f"volume={os.path.basename(dataset.label_paths[pair_idx])}, z={z}, condition_z={z_indices}")
    summarize_array("label", label)
    summarize_array("condition", cond)
    return label, cond, sample_name


def save_png(path, arr):
    arr = np.squeeze(arr)
    arr = np.clip(arr, 0.0, 1.0)
    imageio.imwrite(path, (arr * 255).astype(np.uint8))


def save_preview(dataset, idx, output_dir):
    label, cond, sample_name = inspect_sample(dataset, idx)
    os.makedirs(output_dir, exist_ok=True)

    save_png(os.path.join(output_dir, f"{sample_name}_label_ct.png"), label)
    for channel_idx in range(cond.shape[0]):
        save_png(
            os.path.join(output_dir, f"{sample_name}_cond_cl_ch{channel_idx}.png"),
            cond[channel_idx],
        )
    print(f"Saved preview images to: {output_dir}")


def main():
    args = create_argparser().parse_args()
    pairs = _pair_npy_files(args.data_dir1, args.data_dir2)
    print(f"Paired volumes: {len(pairs)}")
    for i, (label_path, cond_path) in enumerate(pairs[:args.preview_pairs]):
        label = np.load(label_path, mmap_mode="r")
        cond = np.load(cond_path, mmap_mode="r")
        print(f"\nPair {i}")
        print(f"CT: {label_path}")
        print(f"CL: {cond_path}")
        print(f"CT shape={label.shape}, dtype={label.dtype}")
        print(f"CL shape={cond.shape}, dtype={cond.dtype}")
        if label.shape != cond.shape:
            raise ValueError(f"Shape mismatch: {label_path} vs {cond_path}")

    dataset = CLVolumeSliceDataset(
        label_paths=[label for label, _ in pairs],
        cond_paths=[cond for _, cond in pairs],
        image_size=args.image_size,
        num_input_slices=args.condition_channels,
        crop_x=(args.crop_x_start, args.crop_x_end),
        crop_y=(args.crop_y_start, args.crop_y_end),
        use_mmap=True,
    )
    print(f"\nTotal slice samples: {len(dataset)}")
    print(
        f"Crop: x={args.crop_x_start}:{args.crop_x_end}, "
        f"y={args.crop_y_start}:{args.crop_y_end}, size={args.image_size}x{args.image_size}"
    )

    first_z_count = np.load(pairs[0][0], mmap_mode="r").shape[2]
    preview_indices = [0]
    if first_z_count > 1:
        preview_indices.append(first_z_count - 1)
    if len(dataset) > first_z_count:
        preview_indices.append(first_z_count)

    for idx in preview_indices:
        if args.save_preview:
            save_preview(dataset, idx, args.preview_output_dir)
        else:
            inspect_sample(dataset, idx)


def create_argparser():
    defaults = dict(
        data_dir1="/home/lqg/code_8T/24/lt/data_make/CL-data_make/ct_label_npy",
        data_dir2="/home/lqg/code_8T/24/lt/data_make/CL-data_make/cl_label_npy",
        image_size=768,
        condition_channels=3,
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        preview_pairs=3,
        save_preview=True,
        preview_output_dir="debug_data_preview",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
