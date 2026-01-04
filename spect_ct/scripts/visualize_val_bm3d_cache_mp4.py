#!/usr/bin/env python3
"""
Visualize SPECT229 validation BM3D caches (and their corresponding thinned projections)
for 4 validation patients in ONE MP4.

Layout (per frame = one view):
  - rows: patients (val indices 0..3)
  - cols: for each dose: [thin, bm3d] => 10 columns total
  - each cell: image for that patient/dose/view

Display convention:
  - For low-dose (xk), we multiply the displayed image by k so that all doses
    share the same dynamic range as 20s (matches the training/validation MP4 style).
  - Optional log1p for readability.

Cache convention (readable, produced by rename_val_bm3d_cache_readable.py):
  datasets/SPECT229/<patient>/bm3d_cache/seed{seed}/{label}_bm3d.npy
  datasets/SPECT229/<patient>/bm3d_cache/seed{seed}/{label}_thin.npy

Example:
  python spect_ct/scripts/visualize_val_bm3d_cache_mp4.py --seed 123 --out spect_ct/results/val_bm3d_cache_seed123.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm


def _seed_dir(cache_root: Path, seed: int) -> Path:
    return cache_root / f"seed{int(seed)}"


def _load_u16_proj(path: Path, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint16)
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(f"Invalid projection size: {path} expected {expected}, got {arr.size}")
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


def _to_u8_gray(x: np.ndarray, vmax: float, log1p: bool) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), 0.0, float(vmax))
    if log1p:
        x = np.log1p(x) / np.log1p(float(vmax))
    else:
        x = x / float(vmax)
    return (x * 255.0).round().clip(0, 255).astype(np.uint8)


def _make_frame(
    rows: List[Tuple[str, List[Tuple[str, float, np.ndarray, np.ndarray]]]],
    vi: int,
    vmax20_by_patient: List[float],
    log1p: bool,
) -> np.ndarray:
    """
    rows:
      [
        (patient_name, [(label, k, thin_vol(60,H,W), bm3d_vol(60,H,W)), ... 5 doses]),
        ...
      ]
    """
    from PIL import Image, ImageDraw, ImageFont

    H = int(rows[0][1][0][2].shape[1])
    W = int(rows[0][1][0][2].shape[2])
    n_rows = len(rows)
    n_doses = len(rows[0][1])
    n_cols = n_doses * 2  # thin + bm3d per dose

    header_h = 18
    pad = 2
    canvas = Image.new("RGB", (n_cols * W, header_h + n_rows * (H + pad)), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()

    # Column labels: each dose has two columns
    col_labels = [lab for (lab, _, _, _) in rows[0][1]]
    for d, lab in enumerate(col_labels):
        draw.text(((2 * d) * W + 4, 2), f"{lab}|thin", fill=(255, 255, 255), font=font)
        draw.text(((2 * d + 1) * W + 4, 2), f"{lab}|bm3d", fill=(255, 255, 255), font=font)
    draw.text((n_cols * W - 140, 2), f"view={vi:02d}", fill=(200, 200, 200), font=font)

    # Rows
    for i, (patient, doses) in enumerate(rows):
        vmax20 = float(vmax20_by_patient[i])
        for d, (lab, k, thin_vol, bm3d_vol) in enumerate(doses):
            y0 = header_h + i * (H + pad)

            # thin
            img_t = thin_vol[vi].astype(np.float32, copy=False) * float(k)
            u8_t = _to_u8_gray(img_t, vmax=vmax20, log1p=log1p)
            im_t = Image.fromarray(np.repeat(u8_t[:, :, None], 3, axis=2), mode="RGB")
            x0 = (2 * d) * W
            canvas.paste(im_t, (x0, y0))

            # bm3d
            img_b = bm3d_vol[vi].astype(np.float32, copy=False) * float(k)
            u8_b = _to_u8_gray(img_b, vmax=vmax20, log1p=log1p)
            im_b = Image.fromarray(np.repeat(u8_b[:, :, None], 3, axis=2), mode="RGB")
            x1 = (2 * d + 1) * W
            canvas.paste(im_b, (x1, y0))

            if d == 0:
                draw.text((x0 + 4, y0 + 2), patient, fill=(255, 255, 255), font=font)

    return np.asarray(canvas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="datasets/SPECT229")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--factors", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", action="store_true", default=False)
    parser.add_argument("--out", type=str, default="spect_ct/results/val_bm3d_cache_overview.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    log1p = bool(args.log1p) and (not bool(args.no_log1p))
    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        raise FileNotFoundError(dataroot)

    patients = sorted([p for p in dataroot.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not patients:
        raise RuntimeError(f"No patient dirs under {dataroot}")

    labels = [("20s", 1.0)] + [(f"x{k}", float(k)) for k in args.factors]

    rows = []
    vmax20_by_patient = []

    for idx in args.val_indices:
        if idx >= len(patients):
            raise IndexError(f"val index {idx} out of range (patients={len(patients)})")
        pdir = patients[idx]
        pname = pdir.name
        cache_root = pdir / "bm3d_cache"
        cache_dir = _seed_dir(cache_root, int(args.seed))
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Missing cache dir: {cache_dir}. Run precompute_validation_bm3d.py --seed {args.seed} --overwrite"
            )

        # Determine vmax20 from original 20s projection
        proj_files = list(pdir.glob("*_Proj4Filter.dat"))
        if not proj_files:
            raise FileNotFoundError(f"No *_Proj4Filter.dat under {pdir}")
        proj20 = _load_u16_proj(proj_files[0])
        vmax20_by_patient.append(float(np.max(proj20)))

        doses = []
        for lab, k in labels:
            thin_path = cache_dir / f"{lab}_thin.npy"
            bm3d_path = cache_dir / f"{lab}_bm3d.npy"
            if not thin_path.exists():
                raise FileNotFoundError(f"Missing thin cache: {thin_path} (run rename/precompute first)")
            if not bm3d_path.exists():
                raise FileNotFoundError(f"Missing BM3D cache: {bm3d_path} (run rename/precompute first)")
            thin_vol = np.load(thin_path).astype(np.float32, copy=False)
            bm3d_vol = np.load(bm3d_path).astype(np.float32, copy=False)
            if thin_vol.shape[0] != 60 or bm3d_vol.shape[0] != 60:
                raise ValueError(f"Unexpected shape thin={thin_vol.shape} bm3d={bm3d_vol.shape} for {pname}/{lab}")
            doses.append((lab, k, thin_vol, bm3d_vol))
        rows.append((pname, doses))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    with imageio.get_writer(
        str(out_path),
        format="FFMPEG",
        fps=int(args.fps),
        codec="libx264",
        macro_block_size=1,
        # Let imageio decide pix_fmt to avoid duplicate -pix_fmt warnings.
        ffmpeg_params=["-crf", str(int(args.crf))],
    ) as w:
        for vi in tqdm(range(60), desc="write-mp4", dynamic_ncols=True):
            fr = _make_frame(rows, vi=vi, vmax20_by_patient=vmax20_by_patient, log1p=log1p)
            w.append_data(fr)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


