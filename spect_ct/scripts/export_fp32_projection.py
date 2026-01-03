#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Allow running as a script from repo root without setting PYTHONPATH
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spect_ct.pipeline.inference import denoise_views, load_basicsr_net  # noqa: E402
from spect_ct.pipeline.io import load_projection_i16  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Export fp32 denoised projection (no round/clip save).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--input-proj", type=str, required=True, help="path to original_projection.dat (int16)")
    p.add_argument("--out", type=str, required=True, help="output .npy path")
    p.add_argument("--max-value", type=float, default=150.0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    config = Path(args.config)
    ckpt = Path(args.ckpt)
    inp = Path(args.input_proj)
    out = Path(args.out)

    proj = load_projection_i16(inp)  # (60,128,128) float32
    loaded = load_basicsr_net(config, ckpt, device=str(args.device), require_ema=True)
    den = denoise_views(loaded.net, proj, max_value=float(args.max_value), device=str(args.device)).astype(np.float32, copy=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), den)
    print(f"[OK] wrote fp32 denoised projection: {out}  shape={den.shape} dtype={den.dtype}")


if __name__ == "__main__":
    main()


