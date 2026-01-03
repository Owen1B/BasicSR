#!/usr/bin/env python3
"""
可视化：衰减图(PostAtten) 与 重建体(recon) 的“同步旋转 MIP”对齐检查 GIF（单病人）。

输出：1x2 布局（左=recon MIP，右=atten MIP），两者使用相同角度旋转。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as a script from repo root without setting PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spect_ct.pipeline.io import load_atten_f32, load_recon_f32
from spect_ct.pipeline.viz import compute_mip, create_subplot_with_colorbar, rotate_volume_around_x


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 recon vs atten 同步旋转对齐 GIF（1x2）")
    parser.add_argument("--patient", type=str, required=True)
    parser.add_argument("--recon", type=str, required=True, help="重建 .dat（float32，128x128x128）")
    parser.add_argument("--atten", type=str, required=True, help="衰减 .dat（float32，通常 128x128x128）")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--log1p", action="store_true", help="recon 显示用 log1p")
    parser.add_argument("--results-root", type=str, default="spect_ct/results")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--out-gif", type=str, default=None, help="可选：手动指定输出 gif（覆盖默认）")
    args = parser.parse_args()

    patient = str(args.patient)
    recon_path = Path(args.recon)
    atten_path = Path(args.atten)
    if not recon_path.exists():
        raise FileNotFoundError(f"recon not found: {recon_path}")
    if not atten_path.exists():
        raise FileNotFoundError(f"atten not found: {atten_path}")

    recon = load_recon_f32(recon_path, shape=(128, 128, 128))
    atten = load_atten_f32(atten_path)
    if recon.shape != atten.shape:
        raise ValueError(f"shape mismatch: recon={recon.shape}, atten={atten.shape}")

    vmax_recon = float(np.percentile(recon[recon > 0], 99.95)) if np.any(recon > 0) else float(recon.max())
    vmax_atten = float(np.percentile(atten[atten > 0], 99.9)) if np.any(atten > 0) else float(atten.max())

    if args.out_gif:
        out_gif = Path(args.out_gif)
    else:
        out_gif = Path(args.results_root) / str(args.exp_name) / "recon_atten_alignment" / patient / f"{patient}_recon_atten_align.gif"
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    frames_out: list[Image.Image] = []
    angle_start = 90.0
    angle_step = 360.0 / float(int(args.frames))
    for i in range(int(args.frames)):
        angle = angle_start + i * angle_step
        r_rot = rotate_volume_around_x(recon, angle)
        a_rot = rotate_volume_around_x(atten, angle)
        r_mip = np.flipud(compute_mip(r_rot, axis=2))
        a_mip = np.flipud(compute_mip(a_rot, axis=2))

        img_r = create_subplot_with_colorbar(
            r_mip, 0.0, vmax_recon, cmap_name="hot", title=f"Recon MIP ({angle:.1f}°)", use_log1p=bool(args.log1p)
        )
        img_a = create_subplot_with_colorbar(a_mip, 0.0, vmax_atten, cmap_name="gray", title=f"Atten MIP ({angle:.1f}°)", use_log1p=False)

        w, h = img_r.width, img_r.height
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(img_r, (0, 0))
        canvas.paste(img_a, (w, 0))
        frames_out.append(canvas)

    frames_out[0].save(
        out_gif,
        save_all=True,
        append_images=frames_out[1:],
        duration=int(1000.0 / max(float(args.fps), 1e-6)),
        loop=0,
        optimize=False,
    )
    print(f"✅ saved: {out_gif}")


if __name__ == "__main__":
    main()


