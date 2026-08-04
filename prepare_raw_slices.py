"""
Validate 2940x2940 RAW slices and resize them to the v1 input format.

The script validates every input file before writing any resized output.
Output files remain one XY slice per RAW file, so they can be consumed by
limited_IMG_sample.py with raw_height=1024 and raw_width=1024.
"""

import csv
import glob
import os
import re

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Change INPUT_DIR if the real RAW directory is different on the server.
INPUT_DIR = "/home/lqg/code_8T/24/lt/data_make/20_19_47/slice"
OUTPUT_DIR = "/home/lqg/code_8T/24/lt/data_make/20_19_47/1024"

RAW_PATTERN = "*.raw"
RAW_DTYPE = np.dtype("float32")
RAW_HEIGHT = 2940
RAW_WIDTH = 2940
RAW_ORDER = "C"

OUTPUT_HEIGHT = 1024
OUTPUT_WIDTH = 1024
OUTPUT_DTYPE = np.dtype("float32")

# Prevent accidental overwriting of an earlier conversion.
OVERWRITE = False

# Validate only the first N naturally sorted files before resizing all files.
# Set to 0 to validate the complete input directory.
VALIDATE_LIMIT = 10


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
    print(f"Expected shape: ({RAW_HEIGHT}, {RAW_WIDTH})")
    print(f"Expected dtype: {RAW_DTYPE}")
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

        if (index + 1) % 10 == 0 or index + 1 == len(paths):
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

    for index, input_path in enumerate(paths):
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
        if os.path.exists(output_path) and not OVERWRITE:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Set OVERWRITE=True only when replacement is intended."
            )

        source = np.fromfile(input_path, dtype=RAW_DTYPE)
        source = source.reshape((RAW_HEIGHT, RAW_WIDTH), order=RAW_ORDER)
        resized = cv2.resize(
            source,
            (OUTPUT_WIDTH, OUTPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        if resized.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH):
            raise RuntimeError(
                f"Unexpected resized shape {resized.shape} for {input_path}"
            )
        if not np.isfinite(resized).all():
            raise RuntimeError(f"Resized output contains NaN or Inf: {input_path}")

        resized.astype(OUTPUT_DTYPE, copy=False).tofile(output_path)
        output_paths.append(output_path)

        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"Resized {index + 1}/{len(paths)} files.")

    return output_paths


def main():
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

    output_paths = resize_files(input_paths)
    print(f"Resize complete: {len(output_paths)} files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output shape: ({OUTPUT_HEIGHT}, {OUTPUT_WIDTH})")
    print(f"Output dtype: {OUTPUT_DTYPE}")


if __name__ == "__main__":
    main()
