import glob
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


def augment_low_contrast_condition(
    cond_stack,
    probability=0.5,
    contrast_min=0.3,
    contrast_max=0.8,
    noise_std=0.02,
    blur_probability=0.25,
):
    """Make CL conditions look like low-contrast real measurements.

    The same contrast factor is applied to all z-neighbor channels so the
    2.5D geometry remains consistent. The phantom target is never modified.
    """
    if probability <= 0 or np.random.random() >= probability:
        return cond_stack

    stack = cond_stack.astype(np.float32, copy=True)
    contrast = np.random.uniform(contrast_min, contrast_max)
    mean_value = np.mean(stack, dtype=np.float32)
    stack = mean_value + contrast * (stack - mean_value)

    if blur_probability > 0 and np.random.random() < blur_probability:
        # A small local average simulates the loss of weak edge contrast.
        padded = np.pad(stack, ((0, 0), (1, 1), (1, 1)), mode="edge")
        stack = (
            padded[:, :-2, :-2]
            + padded[:, 1:-1, :-2]
            + padded[:, 2:, :-2]
            + padded[:, :-2, 1:-1]
            + padded[:, 1:-1, 1:-1]
            + padded[:, 2:, 1:-1]
            + padded[:, :-2, 2:]
            + padded[:, 1:-1, 2:]
            + padded[:, 2:, 2:]
        ) / 9.0

    if noise_std > 0:
        stack += np.random.normal(0.0, noise_std, size=stack.shape).astype(np.float32)

    return np.clip(stack, 0.0, 1.0)


def _list_npy_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.npy")))


def _pair_npy_files(label_dir, cond_dir):
    label_files = _list_npy_files(label_dir)
    cond_files = _list_npy_files(cond_dir)
    if not label_files:
        raise FileNotFoundError(f"No CT label npy files found in: {label_dir}")
    if not cond_files:
        raise FileNotFoundError(f"No CL input npy files found in: {cond_dir}")

    cond_by_name = {os.path.basename(path): path for path in cond_files}
    pairs = []
    missing = []
    for label_path in label_files:
        name = os.path.basename(label_path)
        cond_path = cond_by_name.get(name)
        if cond_path is None:
            missing.append(name)
        else:
            pairs.append((label_path, cond_path))

    if missing:
        raise FileNotFoundError(
            "Missing CL npy files for CT labels: {}".format(", ".join(missing[:10]))
        )
    return pairs


def normalize_image(img, value_range=None):
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array")
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D array")

    if value_range is None:
        min_val = float(np.min(img))
        max_val = float(np.max(img))
    else:
        min_val, max_val = value_range
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - min_val) / (max_val - min_val), 0.0, 1.0)


def volume_value_range(volume, crop_x=None, crop_y=None):
    """Return one shared value range for every z-slice in a volume."""
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array")
    x0, x1 = crop_x or (0, volume.shape[0])
    y0, y1 = crop_y or (0, volume.shape[1])
    cropped = volume[x0:x1, y0:y1, :]
    return float(np.min(cropped)), float(np.max(cropped))


class CLVolumeSliceDataset(Dataset):
    """
    Build 2.5D slice samples from paired CT/CL reconstruction volumes.
    The npy volume layout is expected to be (x, y, z).
    """

    def __init__(
        self,
        label_paths,
        cond_paths,
        image_size,
        num_input_slices=3,
        crop_x=(127, 895),
        crop_y=(127, 895),
        use_mmap=True,
        normalization_mode="volume",
        augment_condition=False,
        condition_aug_probability=0.5,
        condition_contrast_min=0.3,
        condition_contrast_max=0.8,
        condition_noise_std=0.02,
        condition_blur_probability=0.25,
    ):
        if num_input_slices % 2 != 1:
            raise ValueError("num_input_slices must be odd, e.g. 3 for [z-1,z,z+1].")

        self.label_paths = label_paths
        self.cond_paths = cond_paths
        self.image_size = image_size
        self.num_input_slices = num_input_slices
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.use_mmap = use_mmap
        if normalization_mode not in ("slice", "volume"):
            raise ValueError("normalization_mode must be 'slice' or 'volume'.")
        self.normalization_mode = normalization_mode
        self.augment_condition = augment_condition
        self.condition_aug_probability = condition_aug_probability
        self.condition_contrast_min = condition_contrast_min
        self.condition_contrast_max = condition_contrast_max
        self.condition_noise_std = condition_noise_std
        self.condition_blur_probability = condition_blur_probability
        self._volume_cache = {}
        self._normalization_ranges = {}

        crop_h = self.crop_x[1] - self.crop_x[0]
        crop_w = self.crop_y[1] - self.crop_y[0]
        if crop_h != self.image_size or crop_w != self.image_size:
            raise ValueError(
                f"Crop size ({crop_h}, {crop_w}) does not match image_size={self.image_size}."
            )

        self.indices = []
        for pair_idx, label_path in enumerate(self.label_paths):
            label_volume = self._load_volume(label_path)
            cond_volume = self._load_volume(self.cond_paths[pair_idx])
            if label_volume.ndim != 3:
                raise ValueError(f"Expected 3D volume layout (x, y, z): {label_path}")
            if label_volume.shape != cond_volume.shape:
                raise ValueError(
                    f"CT/CL volume shape mismatch: {label_path} {label_volume.shape} vs "
                    f"{self.cond_paths[pair_idx]} {cond_volume.shape}"
                )
            if self.normalization_mode == "volume":
                self._normalization_ranges[label_path] = volume_value_range(
                    label_volume, self.crop_x, self.crop_y
                )
                self._normalization_ranges[self.cond_paths[pair_idx]] = volume_value_range(
                    cond_volume, self.crop_x, self.crop_y
                )
            for z in range(label_volume.shape[2]):
                self.indices.append((pair_idx, z))

    def _load_volume(self, path):
        if path not in self._volume_cache:
            mmap_mode = "r" if self.use_mmap else None
            self._volume_cache[path] = np.load(path, mmap_mode=mmap_mode)
        return self._volume_cache[path]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pair_idx, z = self.indices[idx]
        label_path = self.label_paths[pair_idx]
        cond_path = self.cond_paths[pair_idx]

        label_volume = self._load_volume(label_path)
        cond_volume = self._load_volume(cond_path)
        if label_volume.shape != cond_volume.shape:
            raise ValueError(
                f"CT/CL volume shape mismatch: {label_path} {label_volume.shape} vs "
                f"{cond_path} {cond_volume.shape}"
            )

        z_count = label_volume.shape[2]
        half = self.num_input_slices // 2
        z_indices = [min(max(z + offset, 0), z_count - 1) for offset in range(-half, half + 1)]

        x0, x1 = self.crop_x
        y0, y1 = self.crop_y
        label_slice = np.asarray(label_volume[x0:x1, y0:y1, z], dtype=np.float32)
        cond_slices = [
            np.asarray(cond_volume[x0:x1, y0:y1, zi], dtype=np.float32)
            for zi in z_indices
        ]

        label_range = self._normalization_ranges.get(label_path)
        cond_range = self._normalization_ranges.get(cond_path)
        label_slice = normalize_image(label_slice, label_range)[None, :, :].astype(np.float32)
        cond_stack = np.stack(
            [normalize_image(slice_, cond_range) for slice_ in cond_slices], axis=0
        )
        cond_stack = cond_stack.astype(np.float32)
        if self.augment_condition:
            cond_stack = augment_low_contrast_condition(
                cond_stack,
                probability=self.condition_aug_probability,
                contrast_min=self.condition_contrast_min,
                contrast_max=self.condition_contrast_max,
                noise_std=self.condition_noise_std,
                blur_probability=self.condition_blur_probability,
            )

        stem = os.path.splitext(os.path.basename(cond_path))[0]
        sample_name = f"{stem}_z{z:03d}"
        return label_slice, cond_stack, sample_name


def load_CL_IMG_data(
        *,
        data_dir1,
        data_dir2,
        batch_size,
        image_size,
        mode,
        num_input_slices=3,
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
        normalization_mode="volume",
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
        augment_condition=None,
        condition_aug_probability=0.5,
        condition_contrast_min=0.3,
        condition_contrast_max=0.8,
        condition_noise_std=0.02,
        condition_blur_probability=0.25,
):
    if not data_dir1:
        raise ValueError("data_dir1 is required")
    if not data_dir2:
        raise ValueError("data_dir2 is required")

    pairs = _pair_npy_files(data_dir1, data_dir2)
    dataset = CLVolumeSliceDataset(
        label_paths=[label for label, _ in pairs],
        cond_paths=[cond for _, cond in pairs],
        image_size=image_size,
        num_input_slices=num_input_slices,
        crop_x=(crop_x_start, crop_x_end),
        crop_y=(crop_y_start, crop_y_end),
        use_mmap=use_mmap,
        normalization_mode=normalization_mode,
        augment_condition=(mode == "train") if augment_condition is None else augment_condition,
        condition_aug_probability=condition_aug_probability,
        condition_contrast_min=condition_contrast_min,
        condition_contrast_max=condition_contrast_max,
        condition_noise_std=condition_noise_std,
        condition_blur_probability=condition_blur_probability,
    )

    print("Dataset size:", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    if mode == "train":
        while True:
            yield from loader
    elif mode == "test":
        yield from loader
    else:
        raise ValueError(f"Unsupported mode: {mode}")
