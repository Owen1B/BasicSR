#!/usr/bin/env python3
"""
数据处理：把 (60,128,128) 的投影转换成“前/后位”(A/P) 两通道数据

默认规则（最常见、最明确）：
- anterior = view[0]
- posterior = view[30]（与 view0 对径）

也支持：
- 指定 anterior/posterior 的 view index
- 指定 window（对中心 view 做邻域平均，提高稳定性）

输出：
- 默认保存为 int16 的 .dat（shape=(2,128,128) 展平写入）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from spect_ct.pipeline.io import load_projection_i16

def _mean_window(v: np.ndarray, center: int, window: int) -> np.ndarray:
    """window=0 表示只取 center；window=1 表示 [center-1,center,center+1]"""
    if window <= 0:
        return v[center]
    idx = [(center + k) % v.shape[0] for k in range(-window, window + 1)]
    return np.mean(v[idx], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 60-view 投影转为前/后位两通道")
    parser.add_argument("--input", type=str, required=True, help="输入 .dat（int16，60x128x128）")
    parser.add_argument("--results-root", type=str, default="spect_ct/results", help="统一结果根目录")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名（将输出到 results/<exp-name>/...）")
    parser.add_argument("--output", type=str, default=None, help="可选：手动指定输出 .dat（覆盖默认 results-root/exp-name）")
    parser.add_argument("--anterior-idx", type=int, default=0)
    parser.add_argument("--posterior-idx", type=int, default=30)
    parser.add_argument("--window", type=int, default=0, help="邻域平均半径（0=不平均）")
    parser.add_argument("--clip-nonneg", action="store_true", help="输出前把负值 clip 到 0")
    parser.add_argument("--dtype", type=str, default="int16", choices=["int16", "float32"], help="输出数据类型")
    args = parser.parse_args()

    inp = Path(args.input)
    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.results_root) / str(args.exp_name) / "converted_ap" / f"{inp.stem}_AP.dat"
    v = load_projection_i16(inp)

    a = _mean_window(v, int(args.anterior_idx), int(args.window))
    p = _mean_window(v, int(args.posterior_idx), int(args.window))
    ap = np.stack([a, p], axis=0)  # (2,128,128)

    if args.clip_nonneg:
        ap = np.clip(ap, 0.0, None)

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dtype == "int16":
        np.round(ap).astype(np.int16).tofile(str(out))
    else:
        ap.astype(np.float32).tofile(str(out))

    print(f"✅ 已保存: {out}")
    print(f"   shape: {ap.shape}, dtype(out): {args.dtype}")
    print(f"   anterior_idx={args.anterior_idx}, posterior_idx={args.posterior_idx}, window={args.window}")


if __name__ == "__main__":
    main()



