"""Export slices from 3-D NPY volumes as grayscale PNG files.

By default, all slices from one volume share the same display range. This
preserves slice-to-slice grayscale differences and is therefore suitable for
checking training data.
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np


# ===== Edit these settings before running the script =====
# INPUT_PATH can be one .npy file or a directory containing .npy files.
INPUT_PATH = r"/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/ct_label_npy/phantom_0001.npy"
OUTPUT_DIR = r"/home/lqg/code_8T/24/lt/data_make/CL-data_make/output/train_data/ct_label_npy/case/phantom_0001"


# The project volume layout is (x, y, z), so z-slice export uses axis 2.
AXIS = 2

# "volume" preserves inter-slice grayscale differences.
# "slice" stretches every slice independently for structure inspection.
# "fixed" uses VALUE_MIN and VALUE_MAX below.
NORMALIZATION = "volume"
VALUE_MIN = None
VALUE_MAX = None


def collect_npy_files(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Input file is not an NPY file: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".npy"),
            key=lambda item: item.name.lower(),
        )
        if not files:
            raise FileNotFoundError(f"No .npy files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def validate_volume(volume, path):
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D NPY volume, but {path} has shape {volume.shape}."
        )
    nonfinite_count = int(volume.size - np.count_nonzero(np.isfinite(volume)))
    if nonfinite_count:
        raise ValueError(
            f"Volume contains {nonfinite_count} NaN or Inf value(s): {path}"
        )


def value_range(volume, normalization_mode, value_min, value_max):
    if normalization_mode == "fixed":
        if value_min is None or value_max is None:
            raise ValueError(
                "--value-min and --value-max are required when "
                "--normalization fixed is used."
            )
        if value_max <= value_min:
            raise ValueError("--value-max must be greater than --value-min.")
        return float(value_min), float(value_max)
    if normalization_mode == "volume":
        return float(np.min(volume)), float(np.max(volume))
    return None


def normalize_slice(slice_data, display_range):
    if display_range is None:
        min_value = float(np.min(slice_data))
        max_value = float(np.max(slice_data))
    else:
        min_value, max_value = display_range

    if max_value == min_value:
        return np.zeros(slice_data.shape, dtype=np.float32)
    normalized = (slice_data.astype(np.float32) - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0)


def export_volume(path, output_root, axis, normalization_mode, value_min, value_max):
    volume = np.load(path, mmap_mode="r")
    validate_volume(volume, path)
    axis = axis % volume.ndim
    display_range = value_range(volume, normalization_mode, value_min, value_max)

    volume_output_dir = output_root / path.stem
    volume_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = volume_output_dir / "slice_statistics.csv"

    print(
        f"Exporting {path}: shape={volume.shape}, dtype={volume.dtype}, "
        f"axis={axis}, slices={volume.shape[axis]}"
    )
    if display_range is not None:
        print(
            f"  Shared display range: [{display_range[0]:.6g}, "
            f"{display_range[1]:.6g}]"
        )

    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            ["slice_index", "png_file", "min", "max", "mean", "std"]
        )

        for slice_index in range(volume.shape[axis]):
            slice_data = np.asarray(
                np.take(volume, slice_index, axis=axis), dtype=np.float32
            )
            normalized = normalize_slice(slice_data, display_range)
            png_data = np.round(normalized * 255.0).astype(np.uint8)
            png_name = f"{path.stem}_axis{axis}_{slice_index:04d}.png"
            png_path = volume_output_dir / png_name
            if not cv2.imwrite(os.fspath(png_path), png_data):
                raise OSError(f"Failed to write PNG file: {png_path}")

            writer.writerow(
                [
                    slice_index,
                    png_name,
                    f"{float(np.min(slice_data)):.9g}",
                    f"{float(np.max(slice_data)):.9g}",
                    f"{float(np.mean(slice_data, dtype=np.float64)):.9g}",
                    f"{float(np.std(slice_data, dtype=np.float64)):.9g}",
                ]
            )

    print(f"  PNG directory: {volume_output_dir}")
    print(f"  Statistics: {report_path}")
    return volume.shape[axis]


def main():
    if AXIS not in (0, 1, 2):
        raise ValueError("AXIS must be 0, 1, or 2.")
    if NORMALIZATION not in ("volume", "slice", "fixed"):
        raise ValueError(
            'NORMALIZATION must be "volume", "slice", or "fixed".'
        )

    input_files = collect_npy_files(INPUT_PATH)
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    for input_file in input_files:
        total_slices += export_volume(
            input_file,
            output_root,
            AXIS,
            NORMALIZATION,
            VALUE_MIN,
            VALUE_MAX,
        )

    print(
        f"Done: exported {total_slices} slice(s) from "
        f"{len(input_files)} volume(s) to {output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
