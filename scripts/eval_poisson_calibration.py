#!/usr/bin/env python3
"""
评测降噪模型的“泊松拟合/校准”情况（投影域）

思想（与现有 analyze_poisson_calibration.py 一致）：
给定观测 y（原始投影计数）与模型输出 λ̂（降噪后的“均值估计”，count domain），
若 y~Poisson(λ)，则在 λ̂ 分桶条件下：
  Var(y - λ̂ | λ̂≈v) ≈ v

输出：
- poisson_calibration_plot.png
- poisson_calibration_bins.csv
- poisson_calibration_summary.txt

支持：
- 指定 ckpt/config
- 指定病人列表（patients），或者用 glob 扫描
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

from prime.pipeline.inference import denoise_views, load_basicsr_net
from prime.pipeline.io import load_projection_i16

@dataclass
class BinStats:
    n: np.ndarray
    sum_lambda: np.ndarray
    sum_r: np.ndarray
    sum_r2: np.ndarray

    @classmethod
    def zeros(cls, num_bins: int) -> "BinStats":
        z = np.zeros(num_bins, dtype=np.float64)
        return cls(n=z.copy(), sum_lambda=z.copy(), sum_r=z.copy(), sum_r2=z.copy())

    def update(self, bin_idx: np.ndarray, lam: np.ndarray, r: np.ndarray, num_bins: int) -> None:
        bin_idx = bin_idx.astype(np.int64, copy=False)
        self.n += np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
        self.sum_lambda += np.bincount(bin_idx, weights=lam, minlength=num_bins).astype(np.float64)
        self.sum_r += np.bincount(bin_idx, weights=r, minlength=num_bins).astype(np.float64)
        self.sum_r2 += np.bincount(bin_idx, weights=r * r, minlength=num_bins).astype(np.float64)

    def finalize(self, min_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        valid = n >= float(min_count)
        mean_lambda = np.zeros_like(n, dtype=np.float64)
        mean_r = np.zeros_like(n, dtype=np.float64)
        var_r = np.zeros_like(n, dtype=np.float64)
        mean_lambda[valid] = self.sum_lambda[valid] / n[valid]
        mean_r[valid] = self.sum_r[valid] / n[valid]
        var_r[valid] = self.sum_r2[valid] / n[valid] - mean_r[valid] ** 2
        return valid, n, mean_lambda, mean_r, var_r


def _resolve_proj_paths(spect229_dir: Path, patients: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patients:
        proj = spect229_dir / p / f"{p}_Proj4Filter.dat"
        if proj.exists():
            out.append(proj)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="评测泊松拟合：Var(residual) vs mean(denoised)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    parser.add_argument("--patients", nargs="*", default=None, help="可选：指定病人列表（不传则用 --glob）")
    parser.add_argument("--glob", type=str, default="datasets/SPECT229/*/*_Proj4Filter.dat", help="投影文件 glob")
    parser.add_argument("--num-files", type=int, default=20, help="不指定 patients 时，截取前 N 个文件")
    parser.add_argument("--max-value", type=float, default=150.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--max-bin", type=float, default=150.0)
    parser.add_argument("--min-count", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20020113)
    parser.add_argument("--results-root", type=str, default="outputs", help="统一结果根目录")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名（将输出到 results/<exp-name>/...）")
    parser.add_argument("--out-dir", type=str, default=None, help="可选：手动指定输出目录（覆盖默认 results-root/exp-name）")
    args = parser.parse_args()

    # 提示：请在 recipe 项目根目录运行（确保能 import prime）
    try:
        _ = sys.path  # noqa: F841
    except Exception:
        pass

    # 延迟导入 BasicSR：保证 `--help` 不依赖 cv2 等可选依赖
    try:
        from basicsr.archs import build_network  # type: ignore  # noqa: F401
        from basicsr.utils.options import yaml_load  # type: ignore  # noqa: F401
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "") or ""
        if missing == "cv2" or "cv2" in str(e):
            raise ModuleNotFoundError(
                "缺少依赖 cv2（opencv-python）。请先在当前 Python 环境安装后重试：\n"
                "  python -m pip install opencv-python\n"
            ) from e
        raise

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    loaded = load_basicsr_net(Path(args.config), Path(args.checkpoint), device=device)
    net = loaded.net
    used_key = loaded.used_key

    # paths
    if args.patients:
        proj_paths = _resolve_proj_paths(Path(args.spect229_dir), list(args.patients))
    else:
        proj_paths = sorted(Path(".").glob(str(args.glob)))[: max(1, int(args.num_files))]
    if len(proj_paths) == 0:
        raise FileNotFoundError("没有找到任何投影文件")

    bin_edges = np.arange(0.0, float(args.max_bin) + float(args.bin_width), float(args.bin_width), dtype=np.float32)
    num_bins = int(len(bin_edges))
    stats = BinStats.zeros(num_bins=num_bins)

    total_pixels = 0
    num_neg = 0

    for p in proj_paths:
        proj_f = load_projection_i16(p).astype(np.float32, copy=False)
        num_neg += int(np.sum(proj_f < 0))
        proj_f = np.clip(proj_f, 0.0, None)
        den = denoise_views(net, proj_f, max_value=float(args.max_value), device=device)
        for i in range(proj_f.shape[0]):
            y = proj_f[i].reshape(-1).astype(np.float64, copy=False)
            lam = den[i].reshape(-1).astype(np.float64, copy=False)
            r = y - lam
            idx = np.digitize(lam, bin_edges, right=False) - 1
            idx = np.clip(idx, 0, num_bins - 1)
            stats.update(idx, lam, r, num_bins=num_bins)
            total_pixels += lam.size

    valid, n, mean_lambda, mean_r, var_r = stats.finalize(min_count=int(args.min_count))
    ratio = np.zeros_like(mean_lambda)
    ratio[valid] = var_r[valid] / np.maximum(mean_lambda[valid], 1e-8)

    # weighted fit: var ≈ a*mean + b
    x = mean_lambda[valid]
    y = var_r[valid]
    w = n[valid]
    if x.size >= 2:
        A = np.vstack([x, np.ones_like(x)]).T
        W = np.diag(w / np.maximum(w.max(), 1.0))
        coef = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
        a, b = float(coef[0]), float(coef[1])
    else:
        a, b = float("nan"), float("nan")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.results_root) / str(args.exp_name) / "poisson_fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    centers = np.zeros(num_bins, dtype=np.float64)
    if num_bins >= 2:
        centers[:-1] = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    centers[-1] = float(args.max_bin) + float(args.bin_width) / 2.0

    csv_path = out_dir / "poisson_calibration_bins.csv"
    header = "bin_center,n,mean_lambda,var_residual,ratio_var_over_mean,valid"
    rows = np.stack([centers, n, mean_lambda, var_r, ratio, valid.astype(np.float64)], axis=1)
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

    fig = plt.figure(figsize=(7.2, 5.4), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(
        mean_lambda[valid],
        var_r[valid],
        s=np.clip((n[valid] / np.maximum(n[valid].max(), 1.0)) * 80.0, 8.0, 80.0),
        alpha=0.65,
        label=f"bins (n≥{int(args.min_count)})",
    )
    max_plot = float(args.max_bin)
    ax.plot([0, max_plot], [0, max_plot], "--", color="black", linewidth=1.0, label="Poisson ideal: var=mean")
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(0, max_plot, 200)
        ax.plot(xs, a * xs + b, color="#d62728", linewidth=1.5, label=f"fit: y={a:.3f}x+{b:.3f}")
    ax.set_xlim(0, max_plot)
    ax.set_ylim(0, max_plot)
    ax.set_xlabel("mean(denoised) in bin (count)")
    ax.set_ylabel("var(y - denoised) in bin")
    ax.set_title(f"Poisson calibration\nckpt={Path(args.checkpoint).name} ({used_key}), files={len(proj_paths)}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "poisson_calibration_plot.png")
    plt.close(fig)

    # 额外输出：更“分析向”的多子图（兼容历史 residual_variance_analysis.png 的用途）
    fig = plt.figure(figsize=(12.8, 6.4), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

    # (1) var vs mean（含理想线与拟合线）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        mean_lambda[valid],
        var_r[valid],
        s=np.clip((n[valid] / np.maximum(n[valid].max(), 1.0)) * 60.0, 6.0, 60.0),
        alpha=0.65,
    )
    ax1.plot([0, max_plot], [0, max_plot], "--", color="black", linewidth=1.0, label="ideal: var=mean")
    if np.isfinite(a) and np.isfinite(b):
        xs = np.linspace(0, max_plot, 200)
        ax1.plot(xs, a * xs + b, color="#d62728", linewidth=1.5, label=f"fit: y={a:.3f}x+{b:.3f}")
    ax1.set_xlim(0, max_plot)
    ax1.set_ylim(0, max_plot)
    ax1.set_xlabel("mean(denoised) (count)")
    ax1.set_ylabel("var(y - denoised)")
    ax1.set_title("Residual variance vs mean(denoised)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    # (2) ratio = var/mean（理想=1）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(
        mean_lambda[valid],
        ratio[valid],
        s=np.clip((n[valid] / np.maximum(n[valid].max(), 1.0)) * 60.0, 6.0, 60.0),
        alpha=0.65,
        color="#1f77b4",
    )
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="ideal: 1.0")
    ax2.set_xlim(0, max_plot)
    ax2.set_ylim(0.0, max(2.0, float(np.nanpercentile(ratio[valid], 99.0)) if np.any(valid) else 2.0))
    ax2.set_xlabel("mean(denoised) (count)")
    ax2.set_ylabel("var/mean")
    ax2.set_title("Ratio: var(residual) / mean(denoised)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    # (3) bias = mean residual
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        mean_lambda[valid],
        mean_r[valid],
        s=np.clip((n[valid] / np.maximum(n[valid].max(), 1.0)) * 60.0, 6.0, 60.0),
        alpha=0.65,
        color="#2ca02c",
    )
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax3.set_xlim(0, max_plot)
    # 对称显示
    m = float(np.nanpercentile(np.abs(mean_r[valid]), 99.0)) if np.any(valid) else 1.0
    ax3.set_ylim(-max(1.0, m), max(1.0, m))
    ax3.set_xlabel("mean(denoised) (count)")
    ax3.set_ylabel("mean(y - denoised)")
    ax3.set_title("Residual mean (bias) vs mean(denoised)")
    ax3.grid(True, alpha=0.25)

    # (4) histogram of ratio
    ax4 = fig.add_subplot(gs[1, 1])
    rr = ratio[valid]
    rr = rr[np.isfinite(rr)]
    if rr.size > 0:
        ax4.hist(rr, bins=40, color="#9467bd", alpha=0.8, edgecolor="black")
        ax4.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax4.axvline(float(np.mean(rr)), color="orange", linewidth=1.5, label=f"mean={np.mean(rr):.3f}")
        ax4.legend(loc="upper right", fontsize=8)
    ax4.set_xlabel("var/mean")
    ax4.set_ylabel("bins")
    ax4.set_title("Distribution of var/mean across bins")
    ax4.grid(True, alpha=0.25)

    fig.suptitle(f"Residual variance analysis\nckpt={Path(args.checkpoint).name} ({used_key}), files={len(proj_paths)}", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_variance_analysis.png")
    plt.close(fig)

    with open(out_dir / "poisson_calibration_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"loaded_key: {used_key}\n")
        f.write(f"files: {len(proj_paths)}\n")
        f.write(f"total_pixels: {total_pixels}\n")
        f.write(f"neg_pixels_in_raw_proj (clipped to 0): {num_neg}\n")
        f.write(f"bin_width: {args.bin_width}\n")
        f.write(f"max_bin: {args.max_bin}\n")
        f.write(f"min_count_per_bin: {args.min_count}\n")
        f.write(f"fit_a: {a}\n")
        f.write(f"fit_b: {b}\n")


if __name__ == "__main__":
    main()
