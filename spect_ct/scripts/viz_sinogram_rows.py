#!/usr/bin/env python3
"""
把投影数据 (60,128,128) 按“每一行 y”切成 128 个 sinogram（每个 60x128），并可视化其中一个。

定义（以数组形状 (views, H, W) 为准）：
- 第 y 行的 sinogram: proj[:, y, :] -> shape=(60, 128)

输出：
- 默认保存一张示例图：sinogram_row<y>.png
- 可选保存全部行：sinograms_all.npy (shape=(128,60,128))
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_proj_i16(path: Path, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    x = np.fromfile(str(path), dtype=np.int16)
    expected = int(views * h * w)
    if x.size != expected:
        raise ValueError(f"投影大小不匹配: {path}，期望={expected} 实际={x.size}")
    return x.reshape(views, h, w).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 (60,128,128) 投影切成按行的 sinogram 并可视化")
    parser.add_argument("--proj", type=str, default=None, help="投影 .dat 路径（int16，60x128x128）。不传则用 --patient 从 SPECT229 查找")
    parser.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    parser.add_argument("--patient", type=str, default=None, help="病人名（用于拼 SPECT229 路径）")
    parser.add_argument("--row", type=int, default=64, help="可视化第几行 y（0~127，默认 64；当 --num-rows>1 时作为兜底）")
    parser.add_argument("--num-rows", type=int, default=1, help="一次可视化多少行正弦图（默认1；设为10即可）")
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="可选：指定行索引列表，逗号分隔，例如 0,16,32,48,64,80,96,112,120,127（会覆盖 --num-rows/--row）",
    )
    parser.add_argument("--log1p", action="store_true", help="对显示使用 log1p 增强对比度（不改变原数据）")
    parser.add_argument("--save-individual", action="store_true", help="当 --num-rows>1 时，额外保存每一行的单张 sinogram PNG")
    parser.add_argument("--save-all", action="store_true", help="保存所有 128 行 sinogram 为一个 .npy（shape=128x60x128）")
    parser.add_argument("--results-root", type=str, default="spect_ct/results", help="统一结果根目录")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名（输出到 results/<exp-name>/sinograms/）")
    args = parser.parse_args()

    if args.proj:
        proj_path = Path(args.proj)
        patient = proj_path.parent.name
    else:
        if not args.patient:
            raise ValueError("必须提供 --proj 或 --patient")
        patient = str(args.patient)
        proj_path = Path(args.spect229_dir) / patient / f"{patient}_Proj4Filter.dat"

    if not proj_path.exists():
        raise FileNotFoundError(f"找不到投影文件: {proj_path}")

    proj = load_proj_i16(proj_path)  # (60,128,128)
    v, h, w = proj.shape
    if (v, h, w) != (60, 128, 128):
        raise ValueError(f"投影 shape 异常: {proj.shape}（期望 60x128x128）")

    y = int(args.row)
    if not (0 <= y < h):
        raise ValueError(f"--row 超范围: {y}（应在 0~{h-1}）")

    # 128 个 sinogram： (H, V, W) = (128,60,128)
    sinograms = np.transpose(proj, (1, 0, 2)).astype(np.float32, copy=False)

    out_dir = Path(args.results_root) / str(args.exp_name) / "sinograms" / patient
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.save_all):
        np.save(out_dir / "sinograms_all.npy", sinograms)

    # Resolve which rows to visualize
    if args.rows:
        rows = [int(x.strip()) for x in str(args.rows).split(",") if x.strip() != ""]
        if len(rows) == 0:
            raise ValueError("--rows 解析为空")
    else:
        k = int(args.num_rows)
        if k <= 1:
            rows = [y]
        else:
            rows = np.linspace(0, h - 1, k).round().astype(int).tolist()

    # clamp & unique (preserve order)
    rows2 = []
    seen = set()
    for r in rows:
        rr = int(np.clip(int(r), 0, h - 1))
        if rr not in seen:
            seen.add(rr)
            rows2.append(rr)
    rows = rows2

    def _disp(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(x, 0.0, None)) if args.log1p else x

    if len(rows) == 1:
        yy = rows[0]
        s = sinograms[yy]  # (60,128)
        disp = _disp(s)

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
        im = ax.imshow(disp, aspect="auto", cmap="gray")
        ax.set_title(f"{patient} | sinogram row y={yy} | shape={s.shape} | log1p={bool(args.log1p)}", fontweight="bold")
        ax.set_xlabel("detector column (x)")
        ax.set_ylabel("view index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_png = out_dir / f"sinogram_row{yy:03d}.png"
        fig.savefig(out_png)
        plt.close(fig)

        print(f"✅ proj: {proj_path}")
        print(f"✅ saved: {out_png}")
    else:
        n = len(rows)
        cols = 5 if n >= 5 else n
        rows_n = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 2.6 * rows_n), dpi=180)
        if rows_n == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows_n == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.expand_dims(axes, 1)

        stack = np.stack([_disp(sinograms[yy]) for yy in rows], axis=0)
        vmin = float(np.min(stack))
        vmax = float(np.percentile(stack, 99.5))

        for i, yy in enumerate(rows):
            r = i // cols
            c = i % cols
            ax = axes[r, c]
            ax.imshow(_disp(sinograms[yy]), aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"y={yy}", fontsize=10, fontweight="bold")
            ax.set_xlabel("x", fontsize=9)
            ax.set_ylabel("view", fontsize=9)

            if bool(args.save_individual):
                s = sinograms[yy]
                fig1, ax1 = plt.subplots(figsize=(10, 4.5), dpi=180)
                ax1.imshow(_disp(s), aspect="auto", cmap="gray")
                ax1.set_title(f"{patient} | sinogram row y={yy} | shape=(60,128) | log1p={bool(args.log1p)}", fontweight="bold")
                ax1.set_xlabel("detector column (x)")
                ax1.set_ylabel("view index")
                fig1.tight_layout()
                fig1.savefig(out_dir / f"sinogram_row{yy:03d}.png")
                plt.close(fig1)

        for j in range(n, rows_n * cols):
            r = j // cols
            c = j % cols
            axes[r, c].axis("off")

        fig.suptitle(f"{patient} | {n} sinogram rows | each=(60,128) | log1p={bool(args.log1p)}", fontweight="bold")
        fig.tight_layout()
        out_png = out_dir / f"sinogram_{n}rows.png"
        fig.savefig(out_png)
        plt.close(fig)

        print(f"✅ proj: {proj_path}")
        print(f"✅ saved: {out_png}")
    if args.save_all:
        print(f"✅ saved: {out_dir/'sinograms_all.npy'}  shape={sinograms.shape}")


if __name__ == "__main__":
    main()


