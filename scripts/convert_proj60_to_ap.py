#!/usr/bin/env python3
"""
数据处理：把投影 (views,H,W) 转换成“前/后位”(A/P) 两通道数据

默认规则（最常见、最明确）：
- anterior = view[0]
- posterior = view[(anterior + views//2) % views]（与 anterior 对径）

也支持：
- 指定 anterior/posterior 的 view index
- 指定 window（对中心 view 做邻域平均，提高稳定性）

输出：
- 默认保存为 int16 的 .dat（shape=(2,H,W) 展平写入）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prime.pipeline.io import load_projection, save_dat_f32, save_dat_i16

def _mean_window(v: np.ndarray, center: int, window: int) -> np.ndarray:
    """window=0 表示只取 center；window=1 表示 [center-1,center,center+1]"""
    center = int(center) % int(v.shape[0])
    if window <= 0:
        return v[center]
    idx = [(center + k) % v.shape[0] for k in range(-window, window + 1)]
    return np.mean(v[idx], axis=0)


def _parse_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3:
        raise ValueError(f"--input-shape 必须是 views,h,w（例如 60,128,128），实际: {spec}")
    if min(shp) <= 0:
        raise ValueError(f"--input-shape 每个维度都必须 >0，实际: {spec}")
    return int(shp[0]), int(shp[1]), int(shp[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="把投影转为前/后位两通道（支持可配置 shape/dtype）")
    parser.add_argument("--input", type=str, required=True, help="输入 .dat（shape 由 --input-shape 指定）")
    parser.add_argument("--input-shape", type=str, default="60,128,128", help="输入投影 shape=views,h,w")
    parser.add_argument(
        "--input-dtype",
        type=str,
        default="int16",
        choices=["int16", "uint16", "float32"],
        help="输入投影 dtype",
    )
    parser.add_argument("--results-root", type=str, default="outputs", help="统一结果根目录")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名（将输出到 results/<exp-name>/...）")
    parser.add_argument("--output", type=str, default=None, help="可选：手动指定输出 .dat（覆盖默认 results-root/exp-name）")
    parser.add_argument("--anterior-idx", type=int, default=0, help="前位视角索引（默认 0）")
    parser.add_argument("--posterior-idx", type=int, default=None, help="后位视角索引（默认按半周自动推断）")
    parser.add_argument("--window", type=int, default=0, help="邻域平均半径（0=不平均）")
    parser.add_argument("--clip-nonneg", action="store_true", help="输出前把负值 clip 到 0")
    parser.add_argument("--dtype", type=str, default="int16", choices=["int16", "float32"], help="输出数据类型")
    args = parser.parse_args()

    inp = Path(args.input)
    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.results_root) / str(args.exp_name) / "converted_ap" / f"{inp.stem}_AP.dat"
    input_shape = _parse_shape(str(args.input_shape))
    v = load_projection(inp, shape=input_shape, dtype=str(args.input_dtype))

    n_views = int(v.shape[0])
    anterior_idx = int(args.anterior_idx) % max(1, n_views)
    if args.posterior_idx is not None:
        posterior_idx = int(args.posterior_idx) % max(1, n_views)
    else:
        posterior_idx = int((anterior_idx + n_views // 2) % max(1, n_views))
    a = _mean_window(v, anterior_idx, int(args.window))
    p = _mean_window(v, posterior_idx, int(args.window))
    ap = np.stack([a, p], axis=0)  # (2,H,W)

    if args.clip_nonneg:
        ap = np.clip(ap, 0.0, None)

    if args.dtype == "int16":
        save_dat_i16(out, ap, round_values=True, clip_nonneg=bool(args.clip_nonneg))
    else:
        save_dat_f32(out, ap, clip_nonneg=bool(args.clip_nonneg))

    print(f"✅ 已保存: {out}")
    print(f"   shape: {ap.shape}, dtype(out): {args.dtype}")
    print(f"   input_shape={args.input_shape}, input_dtype={args.input_dtype}")
    print(f"   anterior_idx={anterior_idx}, posterior_idx={posterior_idx}, window={args.window}")


if __name__ == "__main__":
    main()
