"""
Validate 2940x2940 RAW slices and resize them to the v1 input format.

The script validates every input file before writing any resized output.
Output files remain one XY slice per RAW file, so they can be consumed by
limited_IMG_sample.py with raw_height=1024 and raw_width=1024.

Machine-specific paths can go in prepare_raw_slices.local.yaml (gitignored).
"""

import csv
import glob
import os
import re

import cv2
import numpy as np

from local_config import apply_local_overrides


_INTERP_MAP = {
    "INTER_AREA":    cv2.INTER_AREA,
    "INTER_LINEAR":  cv2.INTER_LINEAR,
    "INTER_CUBIC":   cv2.INTER_CUBIC,
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}

DEFAULTS = dict(
    # ==== 输入 ====
    input_dir="/home/lqg/code_8T/24/sl/裸板/pcb14/10/10",
    raw_pattern="*.raw",
    raw_dtype="float32",
    raw_height=2940,
    raw_width=2940,
    # 内存布局：C = 行优先（C 语言）, F = 列优先（Fortran）
    raw_order="C",

    # ==== 输出 ====
    output_dir="/home/lqg/code_8T/24/lt/data_make/pcb14/10",
    output_height=1024,
    output_width=1024,
    output_dtype="float32",
    # 插值方法: INTER_AREA / INTER_LINEAR / INTER_CUBIC / INTER_NEAREST / INTER_LANCZOS4
    resize_interpolation="INTER_AREA",

    # ==== 行为控制 ====
    # false: 跳过已有且大小正确的输出；true: 强制覆盖
    overwrite=False,
    # 在 resize 前只验证前 N 个文件；0 = 验证全部
    validate_limit=10,
    # 每处理多少个文件打印一次进度
    progress_interval=10,
)

INPUT_DIR = DEFAULTS["input_dir"]
OUTPUT_DIR = DEFAULTS["output_dir"]
RAW_PATTERN = DEFAULTS["raw_pattern"]
RAW_DTYPE = np.dtype(DEFAULTS["raw_dtype"])
RAW_HEIGHT = DEFAULTS["raw_height"]
RAW_WIDTH = DEFAULTS["raw_width"]
RAW_ORDER = DEFAULTS["raw_order"]
OUTPUT_HEIGHT = DEFAULTS["output_height"]
OUTPUT_WIDTH = DEFAULTS["output_width"]
OUTPUT_DTYPE = np.dtype(DEFAULTS["output_dtype"])
RESIZE_INTERPOLATION = _INTERP_MAP[DEFAULTS["resize_interpolation"]]
OVERWRITE = DEFAULTS["overwrite"]
VALIDATE_LIMIT = DEFAULTS["validate_limit"]
PROGRESS_INTERVAL = DEFAULTS["progress_interval"]


def apply_runtime_config(config):
    global INPUT_DIR, OUTPUT_DIR, RAW_PATTERN, RAW_DTYPE
    global RAW_HEIGHT, RAW_WIDTH, RAW_ORDER
    global OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_DTYPE, RESIZE_INTERPOLATION
    global OVERWRITE, VALIDATE_LIMIT, PROGRESS_INTERVAL

    INPUT_DIR = config["input_dir"]
    OUTPUT_DIR = config["output_dir"]
    RAW_PATTERN = config["raw_pattern"]
    RAW_DTYPE = np.dtype(config["raw_dtype"])
    RAW_HEIGHT = int(config["raw_height"])
    RAW_WIDTH = int(config["raw_width"])
    RAW_ORDER = config["raw_order"]
    OUTPUT_HEIGHT = int(config["output_height"])
    OUTPUT_WIDTH = int(config["output_width"])
    OUTPUT_DTYPE = np.dtype(config["output_dtype"])
    interp_key = config["resize_interpolation"]
    if interp_key not in _INTERP_MAP:
        raise ValueError(
            f"Unknown resize_interpolation {interp_key!r}. "
            f"Choose from: {list(_INTERP_MAP)}"
        )
    RESIZE_INTERPOLATION = _INTERP_MAP[interp_key]
    OVERWRITE = bool(config["overwrite"])
    VALIDATE_LIMIT = int(config["validate_limit"])
    PROGRESS_INTERVAL = int(config["progress_interval"])


def load_config():
    config = apply_local_overrides(DEFAULTS.copy(), __file__)
    apply_runtime_config(config)
    return config


def natural_sort_key(path):
    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def expected_bytes(height, width, dtype):
    return height * width * dtype.itemsize


def validate_files(paths):
    expected_values = RAW_HEIGHT * RAW_WIDTH
    expected_file_bytes = expected_bytes(RAW_HEIGHT, RAW_WIDTH, RAW_DTYPE)
    records = []
    errors = []

    print(f"Found {len(paths)} RAW files.")
    print(f"Expected shape: ({RAW_HEIGHT}, {RAW_WIDTH}), dtype: {RAW_DTYPE}, order: {RAW_ORDER}")
    print(f"Expected bytes per file: {expected_file_bytes}")

    for index, path in enumerate(paths):
        file_bytes = os.path.getsize(path)
        record = {
            "index": index,
            "input_file": path,
            "input_bytes": file_bytes,
            "values": 0,
            "finite": False,
            "min": "",
            "max": "",
            "status": "OK",
            "error": "",
        }

        if file_bytes != expected_file_bytes:
            record["status"] = "ERROR"
            record["error"] = (
                f"file size {file_bytes} != expected {expected_file_bytes}"
            )
            errors.append(record)
            records.append(record)
            continue

        try:
            values = np.fromfile(path, dtype=RAW_DTYPE)
            record["values"] = int(values.size)
            if values.size != expected_values:
                raise ValueError(
                    f"value count {values.size} != expected {expected_values}"
                )
            if not np.isfinite(values).all():
                raise ValueError("contains NaN or Inf")

            record["finite"] = True
            record["min"] = float(values.min())
            record["max"] = float(values.max())
        except Exception as exc:
            record["status"] = "ERROR"
            record["error"] = str(exc)
            errors.append(record)

        records.append(record)

        if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(paths):
            print(f"Validated {index + 1}/{len(paths)} files.")

    return records, errors


def write_validation_report(records):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "raw_validation_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "input_file",
                "input_bytes",
                "values",
                "finite",
                "min",
                "max",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return report_path


def resize_files(paths):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_paths = []
    skipped_paths = []
    expected_output_bytes = expected_bytes(
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        OUTPUT_DTYPE,
    )

    for index, input_path in enumerate(paths):
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
        if os.path.exists(output_path) and not OVERWRITE:
            output_bytes = os.path.getsize(output_path)
            if output_bytes != expected_output_bytes:
                raise FileExistsError(
                    f"Existing output has an unexpected size: {output_path} "
                    f"({output_bytes} bytes, expected {expected_output_bytes}). "
                    "Remove the incomplete file or set overwrite=true in local config."
                )
            skipped_paths.append(output_path)
            if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(paths):
                print(
                    f"Processed {index + 1}/{len(paths)} files "
                    f"(resized={len(output_paths)}, skipped={len(skipped_paths)})."
                )
            continue

        source = np.fromfile(input_path, dtype=RAW_DTYPE)
        source = source.reshape((RAW_HEIGHT, RAW_WIDTH), order=RAW_ORDER)
        resized = cv2.resize(
            source,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=RESIZE_INTERPOLATION,
        )

        if resized.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH):
            raise RuntimeError(
                f"Unexpected resized shape {resized.shape} for {input_path}"
            )
        if not np.isfinite(resized).all():
            raise RuntimeError(f"Resized output contains NaN or Inf: {input_path}")

        resized.astype(OUTPUT_DTYPE, copy=False).tofile(output_path)
        output_paths.append(output_path)

        if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(paths):
            print(
                f"Processed {index + 1}/{len(paths)} files "
                f"(resized={len(output_paths)}, skipped={len(skipped_paths)})."
            )

    return output_paths, skipped_paths


def main():
    load_config()
    input_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, RAW_PATTERN)),
        key=natural_sort_key,
    )
    if not input_paths:
        raise FileNotFoundError(
            f"No files found in {INPUT_DIR!r} matching {RAW_PATTERN!r}."
        )

    validation_paths = (
        input_paths[:VALIDATE_LIMIT] if VALIDATE_LIMIT > 0 else input_paths
    )
    print(
        f"Validating {len(validation_paths)} of {len(input_paths)} files "
        "before resizing."
    )
    records, errors = validate_files(validation_paths)
    report_path = write_validation_report(records)
    print(f"Validation report: {report_path}")

    if errors:
        print(f"Validation failed: {len(errors)} file(s) have errors.")
        for record in errors[:10]:
            print(f"  {record['input_file']}: {record['error']}")
        raise RuntimeError("Fix RAW validation errors before resizing.")

    output_paths, skipped_paths = resize_files(input_paths)
    print(f"Resize complete: {len(output_paths)} new file(s)")
    print(f"Skipped existing: {len(skipped_paths)} file(s)")
    print(f"Total ready: {len(output_paths) + len(skipped_paths)} file(s)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output shape: ({OUTPUT_HEIGHT}, {OUTPUT_WIDTH}), dtype: {OUTPUT_DTYPE}")


if __name__ == "__main__":
    main()
