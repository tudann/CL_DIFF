"""Preview sharp vs blurry edges on real CL slices.

This is an inspection tool only. It does not change training or sampling.
The hypothesis is: in-focus structure has thin (1-2 px) edges, while
interlayer ghosts and missing-cone halos are wider.

Run from this directory:

    python preview_sharp_edges.py

Copy preview_sharp_edges.local.yaml.example to preview_sharp_edges.local.yaml
and point it at a real volume. Start with z_indices: mid, then inspect the
green/red overlay before considering this as a training condition.
"""
import argparse
import os

import cv2
import numpy as np

from guided_diffusion.script_util import add_dict_to_argparser
from limited_IMG_sample import SingleCLRawSliceDataset, SingleCLVolumeDataset
from local_config import apply_local_overrides


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img.astype(np.float32)
    ksize = int(np.ceil(sigma * 6)) | 1
    return cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma)


def sobel_grad(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag, gx, gy


def non_max_suppression(mag, gx, gy):
    """Keep ridge pixels, Canny-style, 4 orientations."""
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    quant = np.round(angle / 45.0) % 4
    padded = np.pad(mag, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    n_ew_a, n_ew_b = padded[1:-1, 2:], padded[1:-1, :-2]
    n_nesw_a, n_nesw_b = padded[:-2, 2:], padded[2:, :-2]
    n_ns_a, n_ns_b = padded[:-2, 1:-1], padded[2:, 1:-1]
    n_nwse_a, n_nwse_b = padded[:-2, :-2], padded[2:, 2:]
    keep = (
        ((quant == 0) & (center >= n_ew_a) & (center >= n_ew_b))
        | ((quant == 1) & (center >= n_nesw_a) & (center >= n_nesw_b))
        | ((quant == 2) & (center >= n_ns_a) & (center >= n_ns_b))
        | ((quant == 3) & (center >= n_nwse_a) & (center >= n_nwse_b))
    )
    out = np.zeros_like(mag, dtype=np.float32)
    out[keep] = mag[keep]
    return out


def measure_half_width(mag, gx, gy, mask, max_r=8):
    """FWHM-like width along the gradient normal, in pixels."""
    ys, xs = np.nonzero(mask)
    width_map = np.zeros_like(mag, dtype=np.float32)
    if ys.size == 0:
        return width_map
    mag_safe = mag + 1e-8
    ny = gy[ys, xs] / mag_safe[ys, xs]
    nx = gx[ys, xs] / mag_safe[ys, xs]
    half = 0.5 * mag[ys, xs]
    height, width = mag.shape
    left = np.full(ys.shape, max_r, dtype=np.float32)
    right = np.full(ys.shape, max_r, dtype=np.float32)
    found_l = np.zeros(ys.shape, dtype=bool)
    found_r = np.zeros(ys.shape, dtype=bool)
    for radius in range(1, max_r + 1):
        for found, acc, sign in ((found_l, left, -1.0), (found_r, right, 1.0)):
            yy = np.clip(np.rint(ys + sign * radius * ny).astype(np.int32), 0, height - 1)
            xx = np.clip(np.rint(xs + sign * radius * nx).astype(np.int32), 0, width - 1)
            hit = (~found) & (mag[yy, xx] < half)
            acc[hit] = radius
            found |= hit
    width_map[ys, xs] = left + right
    return width_map


def heat_bgr(arr):
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    vmax = float(np.percentile(arr[finite], 99.5)) if np.any(arr[finite] > 0) else 1.0
    vmax = max(vmax, 1e-8)
    scaled = np.clip(arr / vmax, 0.0, 1.0)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def overlay_classes(gray, sharp, blurry, uncertain=None):
    canvas = cv2.cvtColor(to_uint8(gray), cv2.COLOR_GRAY2BGR)
    if uncertain is not None:
        canvas[uncertain] = (0.35 * canvas[uncertain] + 0.65 * np.array([0, 255, 255])).astype(
            np.uint8
        )
    canvas[blurry] = (0.25 * canvas[blurry] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    canvas[sharp] = (0.25 * canvas[sharp] + 0.75 * np.array([0, 220, 0])).astype(np.uint8)
    return canvas


def overlay_layer_split(gray, unique_sharp, leak_blurry, other):
    canvas = cv2.cvtColor(to_uint8(gray), cv2.COLOR_GRAY2BGR)
    canvas[other] = (0.45 * canvas[other] + 0.55 * np.array([180, 180, 180])).astype(np.uint8)
    canvas[leak_blurry] = (0.25 * canvas[leak_blurry] + 0.75 * np.array([255, 0, 255])).astype(
        np.uint8
    )
    canvas[unique_sharp] = (0.25 * canvas[unique_sharp] + 0.75 * np.array([255, 255, 0])).astype(
        np.uint8
    )
    return canvas


def draw_inset(canvas, gray, mag, box_size=192, zoom=2):
    """Mark the strongest-gradient window and paste a zoomed crop."""
    height, width = mag.shape
    box_size = min(box_size, height, width)
    heat = cv2.blur(mag.astype(np.float32), (box_size, box_size))
    margin = box_size // 2
    inner = heat[margin : height - margin, margin : width - margin]
    if inner.size == 0:
        return canvas
    iy, ix = np.unravel_index(int(np.argmax(inner)), inner.shape)
    y0 = iy
    x0 = ix
    y1 = y0 + box_size
    x1 = x0 + box_size
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    crop = to_uint8(gray[y0:y1, x0:x1])
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    zoomed = cv2.resize(
        crop_bgr, (box_size * zoom, box_size * zoom), interpolation=cv2.INTER_NEAREST
    )
    zh, zw = zoomed.shape[:2]
    pad = 8
    canvas[pad : pad + zh, width - pad - zw : width - pad] = zoomed
    cv2.rectangle(
        canvas,
        (width - pad - zw, pad),
        (width - pad - 1, pad + zh - 1),
        (0, 255, 255),
        2,
    )
    return canvas


def draw_profiles(img, mag, gx, gy, sharp, blurry, length=18, canvas_w=768, canvas_h=240):
    """1D intensity cuts through the strongest sharp and blurry edge pixels."""
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    mag_safe = mag + 1e-8
    specs = [
        ("sharp", sharp, (0, 180, 0)),
        ("blurry", blurry, (0, 0, 220)),
    ]
    half_h = canvas_h // 2
    for idx, (title, mask, color) in enumerate(specs):
        y0 = idx * half_h
        cv2.putText(
            canvas,
            f"{title} edge profile (intensity vs pixels along normal)",
            (16, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if not np.any(mask):
            cv2.putText(
                canvas,
                "no pixels in this class",
                (16, y0 + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            continue
        score = np.where(mask, mag, -1.0)
        y, x = np.unravel_index(int(np.argmax(score)), mag.shape)
        ny = float(gy[y, x] / mag_safe[y, x])
        nx = float(gx[y, x] / mag_safe[y, x])
        ts = np.arange(-length, length + 1, dtype=np.float32)
        vals = []
        h, w = img.shape
        for t in ts:
            yy = int(np.clip(round(y + t * ny), 0, h - 1))
            xx = int(np.clip(round(x + t * nx), 0, w - 1))
            vals.append(float(img[yy, xx]))
        vals = np.asarray(vals, dtype=np.float32)
        plot_x0, plot_x1 = 40, canvas_w - 20
        plot_y0, plot_y1 = y0 + 36, y0 + half_h - 12
        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (230, 230, 230), -1)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        xs = np.linspace(plot_x0, plot_x1, vals.size).astype(np.int32)
        ys = (plot_y1 - (vals - vmin) / (vmax - vmin) * (plot_y1 - plot_y0 - 4)).astype(
            np.int32
        )
        for i in range(len(xs) - 1):
            cv2.line(canvas, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 2, cv2.LINE_AA)
        mid = (plot_x0 + plot_x1) // 2
        cv2.line(canvas, (mid, plot_y0), (mid, plot_y1), (160, 160, 160), 1)
        cv2.putText(
            canvas,
            f"pixel=({y},{x})",
            (plot_x0 + 8, plot_y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return canvas


def labeled_panel(image_bgr, title, footer=""):
    top_h = 42
    bot_h = 28 if footer else 0
    h, w = image_bgr.shape[:2]
    canvas = np.full((top_h + h + bot_h, w, 3), 255, dtype=np.uint8)
    canvas[top_h : top_h + h] = image_bgr
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    if footer:
        cv2.putText(
            canvas,
            footer,
            (12, top_h + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return canvas


def stack_grid(panels, cols):
    rows = []
    for start in range(0, len(panels), cols):
        chunk = panels[start : start + cols]
        h = max(p.shape[0] for p in chunk)
        padded = []
        for panel in chunk:
            if panel.shape[0] < h:
                pad = np.full((h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
                panel = np.concatenate([panel, pad], axis=0)
            padded.append(panel)
        while len(padded) < cols:
            padded.append(np.full_like(padded[0], 255))
        rows.append(np.concatenate(padded, axis=1))
    return np.concatenate(rows, axis=0)


def parse_z_indices(spec, count, max_slices):
    spec = (spec or "").strip().lower()
    if spec in ("", "auto"):
        n = min(max(int(max_slices), 1), count)
        if n == 1:
            return [count // 2]
        return [int(round(i * (count - 1) / (n - 1))) for i in range(n)]
    if spec == "mid":
        return [count // 2]
    indices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        z = int(token)
        if z < 0 or z >= count:
            raise ValueError(f"z={z} is outside [0, {count - 1}]")
        indices.append(z)
    if not indices:
        raise ValueError("z_indices did not contain any slice indices.")
    return indices


def extract_edges(curr, prev, nxt, args, neighbors_real):
    blurred = gaussian_blur(curr, args.pre_sigma)
    mag, gx, gy = sobel_grad(blurred)
    mag_fine, _, _ = sobel_grad(gaussian_blur(curr, args.fine_sigma))
    mag_coarse, _, _ = sobel_grad(gaussian_blur(curr, args.coarse_sigma))
    sharpness = mag_fine / (mag_coarse + 1e-6)

    ridges = non_max_suppression(mag, gx, gy)
    positive = ridges[ridges > 0]
    if positive.size == 0:
        empty = np.zeros_like(curr, dtype=bool)
        return {
            "mag": mag,
            "gx": gx,
            "gy": gy,
            "sharpness": sharpness,
            "width": np.zeros_like(curr, dtype=np.float32),
            "ridges": ridges,
            "sharp": empty,
            "blurry": empty,
            "uncertain": empty,
            "unique_sharp": empty,
            "leak_blurry": empty,
            "other": empty,
            "n_ridge": 0,
        }

    thresh = float(np.percentile(positive, args.edge_percentile))
    ridge_mask = ridges >= thresh
    width = measure_half_width(mag, gx, gy, ridge_mask, max_r=args.max_width_radius)
    sharp = ridge_mask & (width <= args.sharp_max_width)
    blurry = ridge_mask & (width >= args.blur_min_width)
    uncertain = ridge_mask & (~sharp) & (~blurry)

    unique_sharp = sharp
    leak_blurry = blurry
    other = ridge_mask & (~unique_sharp) & (~leak_blurry)
    if neighbors_real:
        mag_prev, _, _ = sobel_grad(gaussian_blur(prev, args.pre_sigma))
        mag_nxt, _, _ = sobel_grad(gaussian_blur(nxt, args.pre_sigma))
        neigh = 0.5 * (mag_prev + mag_nxt)
        leak = np.abs(curr - 0.5 * (prev + nxt))
        leak_mag, _, _ = sobel_grad(gaussian_blur(leak, args.pre_sigma))
        unique_sharp = sharp & (mag > args.unique_ratio * neigh)
        leak_blurry = blurry & (leak_mag > args.leak_mag_ratio * (np.mean(leak_mag) + 1e-6))
        other = ridge_mask & (~unique_sharp) & (~leak_blurry)

    return {
        "mag": mag,
        "gx": gx,
        "gy": gy,
        "sharpness": sharpness,
        "width": width,
        "ridges": ridges,
        "sharp": sharp,
        "blurry": blurry,
        "uncertain": uncertain,
        "unique_sharp": unique_sharp,
        "leak_blurry": leak_blurry,
        "other": other,
        "n_ridge": int(ridge_mask.sum()),
    }


def save_slice_preview(output_dir, name, curr, prev, nxt, label, result, neighbors_real):
    overlay = overlay_classes(curr, result["sharp"], result["blurry"], result["uncertain"])
    overlay = draw_inset(overlay, curr, result["mag"])
    if neighbors_real:
        split = overlay_layer_split(
            curr, result["unique_sharp"], result["leak_blurry"], result["other"]
        )
        split_footer = "cyan=in-focus candidate  magenta=ghost candidate  gray=other"
    else:
        split = cv2.cvtColor(to_uint8(curr), cv2.COLOR_GRAY2BGR)
        cv2.putText(
            split,
            "2.5D split skipped (independent / identical neighbors)",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
        split_footer = "enable neighboring z (independent_raws: false)"

    if label is not None:
        label_mag, _, _ = sobel_grad(label)
        gt_panel = heat_bgr(label_mag)
        gt_title = "CT label gradient (reference)"
        gt_footer = "green CL edges should follow these if the prior is right"
    else:
        gt_panel = heat_bgr(np.where(result["n_ridge"] > 0, result["width"], 0.0))
        gt_title = "edge width map (px)"
        gt_footer = "thin=sharp  wide=blurry"

    n_sharp = int(result["sharp"].sum())
    n_blur = int(result["blurry"].sum())
    n_unc = int(result["uncertain"].sum())
    panels = [
        labeled_panel(cv2.cvtColor(to_uint8(curr), cv2.COLOR_GRAY2BGR), "CL (as model sees it)"),
        labeled_panel(heat_bgr(result["mag"]), "gradient magnitude"),
        labeled_panel(
            heat_bgr(result["sharpness"]),
            "fine/coarse sharpness",
            "high = survives extra blur",
        ),
        labeled_panel(
            overlay,
            "sharp vs blurry overlay",
            f"green={n_sharp}  red={n_blur}  yellow={n_unc}",
        ),
        labeled_panel(split, "2.5D: unique vs leakage", split_footer),
        labeled_panel(gt_panel, gt_title, gt_footer),
    ]
    grid = stack_grid(panels, cols=3)
    panel_path = os.path.join(output_dir, f"{name}_edges.png")
    profile_path = os.path.join(output_dir, f"{name}_profiles.png")
    mask_path = os.path.join(output_dir, f"{name}_masks.png")
    cv2.imwrite(panel_path, grid)
    cv2.imwrite(
        profile_path,
        draw_profiles(
            curr, result["mag"], result["gx"], result["gy"], result["sharp"], result["blurry"]
        ),
    )
    mask_panel = np.concatenate(
        [
            labeled_panel(cv2.cvtColor((result["sharp"].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR), "sharp mask"),
            labeled_panel(cv2.cvtColor((result["blurry"].astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR), "blurry mask"),
        ],
        axis=1,
    )
    cv2.imwrite(mask_path, mask_panel)
    print(
        f"{name}: ridges={result['n_ridge']} sharp={n_sharp} "
        f"blurry={n_blur} uncertain={n_unc} -> {panel_path}"
    )


def make_self_test_image(size=384):
    img = np.full((size, size), 0.28, dtype=np.float32)
    img[80:300, 70:88] = 0.92
    ghost = np.zeros_like(img)
    ghost[80:300, 240:270] = 0.85
    ghost = gaussian_blur(ghost, 3.2)
    img = np.clip(img + 0.85 * ghost, 0.0, 1.0)
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0.0, 0.012, img.shape).astype(np.float32), 0.0, 1.0)
    return img


def run_self_test(args, output_dir):
    img = make_self_test_image()
    result = extract_edges(img, img, img, args, neighbors_real=False)
    save_slice_preview(output_dir, "self_test", img, img, img, None, result, False)
    sharp_left = int(result["sharp"][80:300, 60:100].sum())
    blur_right = int(result["blurry"][80:300, 220:300].sum())
    print(
        f"self-test check: sharp pixels on left bar={sharp_left}, "
        f"blurry pixels on right ghost={blur_right}"
    )
    print("If the left bar is green and the right smear is red, the extractor is working.")


def build_dataset(args):
    crop_x = (args.crop_x_start, args.crop_x_end)
    crop_y = (args.crop_y_start, args.crop_y_end)
    if args.input_raw_dir:
        return SingleCLRawSliceDataset(
            input_raw_dir=args.input_raw_dir,
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=crop_x,
            crop_y=crop_y,
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
    if args.input_npy:
        return SingleCLVolumeDataset(
            input_npy=args.input_npy,
            label_npy=args.label_npy,
            image_size=args.image_size,
            num_input_slices=args.condition_channels,
            crop_x=crop_x,
            crop_y=crop_y,
            use_mmap=args.use_mmap,
            normalization_mode=args.normalization_mode,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            independent_slices=args.independent_raws,
        )
    raise ValueError("Set input_npy or input_raw_dir, or pass --self_test true.")


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.self_test:
        run_self_test(args, args.output_dir)
        if not args.input_npy and not args.input_raw_dir:
            return

    dataset = build_dataset(args)
    z_list = parse_z_indices(args.z_indices, len(dataset), args.max_slices)
    print(f"Previewing {len(z_list)} slice(s): {z_list}")
    if args.independent_raws:
        print(
            "independent_raws=true: neighbors are repeated, so the 2.5D panel "
            "cannot separate in-focus structure from leakage. Set it false to "
            "use [z-1, z, z+1]."
        )

    for z in z_list:
        label, cond, name = dataset[z]
        cond_np = cond[0].detach().cpu().numpy()
        center = cond_np.shape[0] // 2
        prev = cond_np[max(center - 1, 0)]
        curr = cond_np[center]
        nxt = cond_np[min(center + 1, cond_np.shape[0] - 1)]
        neighbors_real = (not args.independent_raws) and (
            float(np.mean(np.abs(prev - curr))) > 1e-6
            or float(np.mean(np.abs(nxt - curr))) > 1e-6
        )
        label_np = None if label is None else np.asarray(label[0], dtype=np.float32)
        result = extract_edges(curr, prev, nxt, args, neighbors_real)
        save_slice_preview(
            args.output_dir, name, curr, prev, nxt, label_np, result, neighbors_real
        )


def create_argparser():
    defaults = dict(
        self_test=False,
        input_raw_dir="",
        raw_height=1024,
        raw_width=1024,
        raw_dtype="float32",
        raw_pattern="*.raw",
        raw_order="C",
        raw_volume_name="real_fdk",
        independent_raws=False,
        input_npy="",
        label_npy="",
        output_dir="debug_sharp_edges",
        image_size=768,
        condition_channels=3,
        crop_x_start=127,
        crop_x_end=895,
        crop_y_start=127,
        crop_y_end=895,
        use_mmap=True,
        normalization_mode="percentile",
        percentile_low=1.0,
        percentile_high=99.0,
        z_indices="mid",
        max_slices=5,
        pre_sigma=0.6,
        fine_sigma=0.8,
        coarse_sigma=2.5,
        edge_percentile=88.0,
        sharp_max_width=3.0,
        blur_min_width=5.0,
        max_width_radius=8,
        unique_ratio=1.35,
        leak_mag_ratio=1.5,
    )
    apply_local_overrides(defaults, __file__)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
