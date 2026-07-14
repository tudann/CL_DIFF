import glob
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset


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


def normalize_image(img):
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array")
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D array")

    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


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
        self._volume_cache = {}

        crop_h = self.crop_x[1] - self.crop_x[0]
        crop_w = self.crop_y[1] - self.crop_y[0]
        if crop_h != self.image_size or crop_w != self.image_size:
            raise ValueError(
                f"Crop size ({crop_h}, {crop_w}) does not match image_size={self.image_size}."
            )

        self.indices = []
        for pair_idx, label_path in enumerate(self.label_paths):
            volume = self._load_volume(label_path)
            if volume.ndim != 3:
                raise ValueError(f"Expected 3D volume layout (x, y, z): {label_path}")
            for z in range(volume.shape[2]):
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

        label_slice = normalize_image(label_slice)[None, :, :].astype(np.float32)
        cond_stack = np.stack([normalize_image(slice_) for slice_ in cond_slices], axis=0)
        cond_stack = cond_stack.astype(np.float32)

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
    )

    print("Dataset size:", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
    )

    if mode == "train":
        while True:
            yield from loader
    elif mode == "test":
        yield from loader
    else:
        raise ValueError(f"Unsupported mode: {mode}")
