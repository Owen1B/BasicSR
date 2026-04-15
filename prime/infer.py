from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prime import register_all
from prime.pipeline.inference import denoise_views, load_basicsr_net
from prime.pipeline.io import load_projection, save_dat_f32, save_dat_i16


def _parse_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3 or min(shp) <= 0:
        raise ValueError(f"Invalid --input-shape: {spec}. Expected views,h,w (e.g. 60,128,128).")
    return int(shp[0]), int(shp[1]), int(shp[2])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SPECT inference entrypoint (Real-ESRGAN-style standalone script)")
    p.add_argument("--opt", required=True, type=str, help="YAML config path (for network_g definition)")
    p.add_argument("--model", required=True, type=str, help="checkpoint path (.pth)")
    p.add_argument("--input", required=True, type=str, help="input projection dat path")
    p.add_argument("--input-shape", type=str, default="60,128,128", help="input shape: views,h,w")
    p.add_argument(
        "--input-dtype",
        type=str,
        default="int16",
        choices=["int16", "uint16", "float32"],
        help="input dat dtype",
    )
    p.add_argument("--output", required=True, type=str, help="output dat path")
    p.add_argument(
        "--output-dtype",
        type=str,
        default="float32",
        choices=["float32", "int16"],
        help="output dat dtype",
    )
    p.add_argument("--max-value", type=float, default=150.0, help="count-domain normalization max")
    p.add_argument("--device", type=str, default="cuda", help="torch device, e.g. cuda or cpu")
    p.add_argument("--allow-non-ema", action="store_true", help="allow loading params/raw if params_ema missing")
    p.add_argument("--clip-nonneg", action="store_true", help="clip output to non-negative before save")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    register_all()

    opt_path = Path(args.opt)
    model_path = Path(args.model)
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not opt_path.exists():
        raise FileNotFoundError(f"Config not found: {opt_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    in_shape = _parse_shape(args.input_shape)
    proj = load_projection(in_path, shape=in_shape, dtype=str(args.input_dtype))

    loaded = load_basicsr_net(
        config_path=opt_path,
        checkpoint_path=model_path,
        device=str(args.device),
        require_ema=not bool(args.allow_non_ema),
    )

    out = denoise_views(
        loaded.net,
        proj_count=proj,
        max_value=float(args.max_value),
        device=str(args.device),
    )

    if bool(args.clip_nonneg):
        out = np.clip(out, 0.0, None)

    if str(args.output_dtype) == "int16":
        save_dat_i16(out_path, out, round_values=True, clip_nonneg=bool(args.clip_nonneg))
    else:
        save_dat_f32(out_path, out, clip_nonneg=bool(args.clip_nonneg))

    print(f"Saved: {out_path}")
    print(
        f"input={in_path} shape={tuple(proj.shape)} dtype={args.input_dtype}; "
        f"output_dtype={args.output_dtype}; ckpt_key={loaded.used_key}"
    )


if __name__ == "__main__":
    main()
