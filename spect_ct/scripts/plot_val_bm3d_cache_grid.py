#!/usr/bin/env python3
"""
Plot a static grid for validation cached low-dose thins and BM3D results.

Grid:
  - rows: patients
  - cols: for each dose: thin | bm3d   => 10 columns (20s/x2/x3/x4/x5 × 2)

Display:
  - low-dose images are multiplied by k (dose factor) for display so all doses share 20s scale
  - per-patient vmax20 is from original 20s projection file max
  - optional log1p

Example:
  python spect_ct/scripts/plot_val_bm3d_cache_grid.py --seed 123 --view 0 --out spect_ct/results/val_cache_view00.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_u16_proj(path: Path, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint16)
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(f"Invalid projection size: {path} expected {expected}, got {arr.size}")
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


def _norm_for_display(x: np.ndarray, vmax: float, log1p: bool) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), 0.0, float(vmax))
    if log1p:
        x = np.log1p(x) / np.log1p(float(vmax))
    else:
        x = x / float(vmax)
    return x


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, default="datasets/SPECT229")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--patients", type=str, nargs="+", default=["AnYufeng", "BaTunasong", "BaYasu", "BaiYukun"])
    p.add_argument("--factors", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--log1p", action="store_true", default=True)
    p.add_argument("--no-log1p", action="store_true", default=False)
    p.add_argument("--out", type=str, default="spect_ct/results/val_cache_grid.png")
    args = p.parse_args()

    log1p = bool(args.log1p) and (not bool(args.no_log1p))
    v = int(args.view)
    if v < 0 or v >= 60:
        raise ValueError("--view must be in [0,59]")

    root = Path(args.dataroot)
    labels: List[Tuple[str, float]] = [("20s", 1.0)] + [(f"x{k}", float(k)) for k in args.factors]

    # Load arrays
    rows = []
    vmax20s = []
    for pname in args.patients:
        pdir = root / pname
        cache_dir = pdir / "bm3d_cache" / f"seed{int(args.seed)}"
        if not cache_dir.exists():
            raise FileNotFoundError(cache_dir)
        proj_files = list(pdir.glob("*_Proj4Filter.dat"))
        if not proj_files:
            raise FileNotFoundError(f"No *_Proj4Filter.dat under {pdir}")
        proj20 = _load_u16_proj(proj_files[0])
        vmax20s.append(float(np.max(proj20)))

        cols = []
        for lab, k in labels:
            thin = np.load(cache_dir / f"{lab}_thin.npy").astype(np.float32, copy=False)
            bm3d = np.load(cache_dir / f"{lab}_bm3d.npy").astype(np.float32, copy=False)
            cols.append((lab, float(k), thin, bm3d))
        rows.append((pname, cols))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_doses = len(labels)
    n_cols = n_doses * 2

    fig_w = 2.0 * n_cols
    fig_h = 2.0 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=160)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for i, (pname, cols) in enumerate(rows):
        vmax = vmax20s[i]
        for d, (lab, k, thin, bm3d) in enumerate(cols):
            # thin
            ax0 = axes[i, 2 * d]
            img0 = thin[v] * k
            ax0.imshow(_norm_for_display(img0, vmax=vmax, log1p=log1p), cmap="gray", vmin=0, vmax=1)
            ax0.set_xticks([]); ax0.set_yticks([])
            if i == 0:
                ax0.set_title(f"{lab}\nthin", fontsize=8)
            if d == 0:
                ax0.set_ylabel(pname, fontsize=9)

            # bm3d
            ax1 = axes[i, 2 * d + 1]
            img1 = bm3d[v] * k
            ax1.imshow(_norm_for_display(img1, vmax=vmax, log1p=log1p), cmap="gray", vmin=0, vmax=1)
            ax1.set_xticks([]); ax1.set_yticks([])
            if i == 0:
                ax1.set_title(f"{lab}\nbm3d", fontsize=8)

    fig.suptitle(f"Val cache grid | seed={int(args.seed)} | view={v:02d} | log1p={log1p}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


