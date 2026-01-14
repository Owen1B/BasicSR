#!/usr/bin/env python
"""
Plot per-row total counts curve for a projection volume (V,H,W).

row_sum[r] = sum_{v,w} proj[v,r,w]

This helps identify rows with abnormal drops (e.g. padding borders or stripe artifacts).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_proj_u16(path: str, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    expected = int(views) * int(h) * int(w)
    if arr.size != expected:
        raise ValueError(f"Invalid projection size: {path}. Expected {expected} uint16 values, got {arr.size}.")
    return arr.reshape(int(views), int(h), int(w)).astype(np.float64, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Path to *_Proj4Filter.dat")
    ap.add_argument("--out", required=True, help="Output png path")
    ap.add_argument("--title", default="", help="Plot title")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    ap.add_argument("--smooth", type=int, default=1, help="Simple moving average window for row_sum (>=1)")
    args = ap.parse_args()

    proj_path = str(args.proj)
    vol = load_proj_u16(proj_path, views=60, h=128, w=128)  # (60,128,128) float64

    row_sum = vol.sum(axis=(0, 2))  # (H,)
    row_sum = row_sum.astype(np.float64, copy=False)

    # Moving average for readability
    win = int(max(1, args.smooth))
    if win > 1:
        k = np.ones((win,), dtype=np.float64) / float(win)
        row_sum_s = np.convolve(row_sum, k, mode="same")
    else:
        row_sum_s = row_sum

    # Relative drop: compare to local neighborhood
    # d[r] = row_sum[r] - row_sum[r-1]
    d1 = np.diff(row_sum_s, prepend=row_sum_s[0])

    # Identify candidate "big drops" (most negative diffs), excluding zeros
    # We also compute a robust threshold based on percentile.
    neg = d1.copy()
    thr = float(np.percentile(neg, 1.0))  # very negative tail
    cand = np.where(d1 <= thr)[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(row_sum.shape[0], dtype=np.int64)

        fig = plt.figure(figsize=(10, 5), dpi=160)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        ax1.plot(x, row_sum_s, lw=1.5, color="black")
        ax1.set_ylabel("row_sum (counts)")
        if args.logy:
            ax1.set_yscale("log")
        if args.title:
            ax1.set_title(str(args.title))

        # Mark zero rows
        zero_rows = np.where(row_sum == 0)[0]
        if zero_rows.size > 0:
            ax1.scatter(zero_rows, np.maximum(row_sum_s[zero_rows], 1.0), s=12, color="red", label="zero row")
            ax1.legend(loc="best", fontsize=8)

        # Diff plot
        ax2.plot(x, d1, lw=1.0, color="#1f77b4")
        ax2.axhline(0.0, lw=0.8, color="gray", alpha=0.6)
        ax2.set_xlabel("row index (0=top)")
        ax2.set_ylabel("Δ row_sum")

        # Mark big drops
        if cand.size > 0:
            ax2.scatter(cand, d1[cand], s=14, color="orange", label="large drops (p1)")
            ax2.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        raise RuntimeError(f"Failed to plot: {e}")

    # Print a short summary for quick inspection
    print("proj:", proj_path)
    print("out:", str(out_path))
    print("zero_rows:", np.where(row_sum == 0)[0].tolist())
    # Top-10 most negative drops
    idx = np.argsort(d1)[:10]
    print("top_negative_drops (row, d1):", [(int(i), float(d1[i])) for i in idx])


if __name__ == "__main__":
    main()














