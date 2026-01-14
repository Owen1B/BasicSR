#!/usr/bin/env python3
from __future__ import annotations

"""
Render a validation-style MP4 (same style as BasicSR training MP4) for ONE model:

Rows (5): 20s + thinned x2/x3/x4/x5 (longer time on top)
Cols (3): Original | Denoised(ema) | Poisson(ema)

This is a faithful standalone adaptation of:
  `basicsr/models/spect3d_model.py::_generate_validation_mp4_thin_factors`

Key behaviors kept identical:
  - Fixed thins loaded from dataset cache: datasets/SPECT229/<patient>/bm3d_cache/seed{seed}/xk_thin.npy (or md5 fallback)
  - Display gray columns in 20s-equivalent domain: multiply each row by k and use the SAME vmax20 (from 20s original).
  - Poisson(ema) sampled ONCE per row (stable across frames) with deterministic RNG based on seed and k.
  - Overlay counts in W and % diff vs theory (C20/k) for columns 2/3; Original column shows only "{sec}s(xk)/W".
  - DejaVuSans font, header style, and mp4 writer args (macro_block_size=1, yuv420p, -crf).
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np


def _md5_16(s: str) -> str:
    import hashlib
    return hashlib.md5(str(s).encode()).hexdigest()[:16]


def _counts_int(x: np.ndarray) -> np.ndarray:
    y = np.rint(x.astype(np.float64, copy=False))
    y = np.clip(y, 0.0, None)
    return y.astype(np.int64, copy=False)


def _norm_to_u8(x: np.ndarray, *, vmax: float, use_log1p: bool) -> np.ndarray:
    vmax = float(vmax)
    if vmax < 1e-6:
        vmax = 1.0
    y = np.clip(x.astype(np.float32, copy=False), 0.0, vmax)
    if bool(use_log1p):
        y = np.log1p(y) / np.log1p(vmax)
    else:
        y = y / vmax
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def _fmt_counts_w_pct(actual: float, theory: float) -> str:
    theory = float(theory)
    actual = float(actual)
    w = actual / 1.0e4
    if w >= 100:
        cstr = f"C={w:.0f}w"
    elif w >= 10:
        cstr = f"C={w:.1f}w"
    else:
        cstr = f"C={w:.2f}w"
    if theory <= 0:
        return cstr
    pct = (actual - theory) / theory * 100.0
    pstr = f"{pct:+.0f}%" if abs(pct) >= 10 else f"{pct:+.1f}%"
    return f"{cstr} ({pstr})"


def _fmt_w_int(actual: float) -> str:
    w = float(actual) / 1.0e4
    return f"{int(round(w))}W"


def _load_font(size: int = 12):
    from PIL import ImageFont  # type: ignore
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _pick_ckpt(exp_dir: Path, *, prefer_iter: Optional[int] = None) -> Path:
    models = Path(exp_dir) / "models"
    if prefer_iter is not None:
        p = models / f"net_g_{int(prefer_iter)}.pth"
        if p.is_file():
            return p
    best = models / "net_g_best.pth"
    if best.is_file():
        return best
    latest = models / "net_g_latest.pth"
    if latest.is_file():
        return latest
    # max iter
    best_it = -1
    best_p = None
    for p in models.glob("net_g_*.pth"):
        m = re.search(r"net_g_(\\d+)\\.pth", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_it:
            best_it = it
            best_p = p
    if best_p is None:
        raise FileNotFoundError(f"No checkpoint found under: {models}")
    return best_p


def _load_cached_thin(*, patient_dir: Path, lq_path: str, label: str, seed: int, require_cached: bool) -> np.ndarray:
    cand_root = Path(patient_dir) / "bm3d_cache"
    ds_cache_dir = None
    if cand_root.exists():
        seed_dir = cand_root / f"seed{int(seed)}"
        ds_cache_dir = seed_dir if seed_dir.exists() else cand_root
    xk = None
    if ds_cache_dir is not None:
        thin_path = ds_cache_dir / f"{label}_thin.npy"
        if thin_path.exists():
            xk = np.load(thin_path).astype(np.float32, copy=False)
        else:
            key = _md5_16(f"{str(lq_path)}__{label}")
            thin_path2 = ds_cache_dir / f"{key}_thin.npy"
            if thin_path2.exists():
                xk = np.load(thin_path2).astype(np.float32, copy=False)
    if xk is None:
        if require_cached:
            raise FileNotFoundError(
                f"Missing cached low-dose split: patient={patient_dir.name} label={label} "
                f"(expected under {cand_root}/seed{seed})."
            )
        raise RuntimeError("On-the-fly thinning is disabled; please precompute caches.")
    return xk


def _infer_denoised(*, config: Path, ckpt: Path, proj: np.ndarray, max_value: float, device: str) -> np.ndarray:
    from spect_ct.pipeline.inference import load_basicsr_net, denoise_views
    loaded = load_basicsr_net(config, ckpt, device=device, require_ema=True)
    out = denoise_views(loaded.net, proj.astype(np.float32, copy=False), max_value=float(max_value), device=device)
    return np.clip(out.astype(np.float32, copy=False), 0.0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, required=True, help="patient folder name under datasets/SPECT229")
    ap.add_argument("--out", type=str, required=True, help="output mp4 path")
    ap.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thin-factors", type=str, default="2,3,4,5")
    ap.add_argument("--require-cached", action="store_true")
    ap.set_defaults(require_cached=True)
    ap.add_argument("--max-value", type=float, default=150.0)
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--no-log1p", dest="log1p", action="store_false")
    ap.set_defaults(log1p=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--exp", type=str, required=True, help="experiment dir (e.g. experiments/n2n_spect229_60view3d_unet_projonly_full_multidose)")
    ap.add_argument("--iter", type=int, default=None, help="optional checkpoint iter (e.g. 13000)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--label", type=str, default="Denoised(ema)")
    args = ap.parse_args()

    from PIL import Image, ImageDraw  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    spect229_dir = Path(args.spect229_dir)
    patient_dir = spect229_dir / str(args.patient)
    if not patient_dir.exists():
        raise FileNotFoundError(f"patient_dir not found: {patient_dir}")

    lq_path = str((patient_dir / f"{args.patient}_Proj4Filter.dat").resolve())
    if not Path(lq_path).exists():
        raise FileNotFoundError(f"lq projection not found: {lq_path}")

    from spect_ct.pipeline.io import load_projection_i16
    proj_u16 = load_projection_i16(Path(lq_path)).astype(np.float32, copy=False)
    y20i = _counts_int(proj_u16)
    c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
    if c20 <= 0:
        c20 = 1.0
    vmax20 = float(np.max(y20i.astype(np.float32, copy=False)))
    if vmax20 < 1e-6:
        vmax20 = 1.0

    factors = []
    for part in str(args.thin_factors).split(","):
        part = part.strip()
        if part:
            factors.append(float(part))
    factors = [k for k in factors if float(k) > 1.0]

    base_rows: list[tuple[str, float, np.ndarray]] = [("20s", 1.0, y20i.astype(np.float32, copy=False))]
    for k in factors:
        lab = f"x{k:g}"
        xk = _load_cached_thin(patient_dir=patient_dir, lq_path=lq_path, label=lab, seed=int(args.seed), require_cached=bool(args.require_cached))
        base_rows.append((lab, float(k), xk.astype(np.float32, copy=False)))

    def _sec(label: str, k: float) -> float:
        return 20.0 if label == "20s" else 20.0 / max(float(k), 1e-12)

    base_rows.sort(key=lambda t: _sec(t[0], t[1]), reverse=True)

    from spect_ct.pipeline.experiment import find_config_for_experiment
    exp_dir = Path(args.exp)
    cfg = Path(args.config) if args.config else find_config_for_experiment(exp_dir)
    ckpt = Path(args.ckpt) if args.ckpt else _pick_ckpt(exp_dir, prefer_iter=args.iter)

    mv = float(args.max_value)
    use_log1p = bool(args.log1p)
    seed = int(args.seed)

    # Per-row inference + poisson (sample once per row)
    rows = []
    for label, k, orig in base_rows:
        den_ema = _infer_denoised(config=cfg, ckpt=ckpt, proj=orig, max_value=mv, device=str(args.device))
        rng = np.random.default_rng(seed + int(round(float(k) * 1000.0)))
        poi = rng.poisson(lam=den_ema).astype(np.float32)
        rows.append(
            dict(
                label=label,
                k=float(k),
                sec=_sec(label, k),
                orig=orig,
                ema=den_ema,
                poi=poi,
                theory=(c20 / max(float(k), 1e-12)),
            )
        )

    for r in rows:
        r["c_orig"] = float(np.sum(_counts_int(r["orig"]).astype(np.float64, copy=False)))
        r["c_ema"] = float(np.sum(r["ema"].astype(np.float64, copy=False)))
        r["c_poi"] = float(np.sum(_counts_int(r["poi"]).astype(np.float64, copy=False)))

    W, H = 128, 128
    header_h = 22
    font = _load_font(12)
    col_labels = ["Original", str(args.label), "Poisson(ema)"]

    def _g2rgb(u8: np.ndarray) -> Image.Image:
        return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")

    turns = int(args.turns)
    if turns < 1:
        turns = 1

    frames = []
    for fi in range(60 * turns):
        vi = int(fi % 60)
        row_imgs = []
        for r in rows:
            k = float(r["k"])
            a = _norm_to_u8(r["orig"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            b = _norm_to_u8(r["ema"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            c = _norm_to_u8(r["poi"][vi] * k, vmax=vmax20, use_log1p=use_log1p)

            row_canvas = Image.new("RGB", (W * 3, H), color=(0, 0, 0))
            row_canvas.paste(_g2rgb(a), (0 * W, 0))
            row_canvas.paste(_g2rgb(b), (1 * W, 0))
            row_canvas.paste(_g2rgb(c), (2 * W, 0))

            draw = ImageDraw.Draw(row_canvas)
            if str(r["label"]).startswith("x"):
                row_label = f"{r['sec']:.0f}s(x{k:g})"
            else:
                row_label = f"{r['sec']:.0f}s"
            draw.text((4, 4), f"{row_label} / {_fmt_w_int(float(r['c_orig']))}", fill=255, font=font)
            t = float(r["theory"])
            draw.text((1 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema"]), t), fill=255, font=font)
            draw.text((2 * W + 4, 4), _fmt_counts_w_pct(float(r["c_poi"]), t), fill=255, font=font)
            row_imgs.append(row_canvas)

        full = Image.new("RGB", (W * 3, header_h + H * len(row_imgs)), color=(0, 0, 0))
        draw = ImageDraw.Draw(full)
        for j, lab in enumerate(col_labels):
            draw.text((j * W + 4, 3), lab, fill=255, font=font)
        # Requested: do NOT display view index.
        for i, im in enumerate(row_imgs):
            full.paste(im, (0, header_h + i * H))
        frames.append(full)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(out),
        format="FFMPEG",
        fps=float(args.fps) if float(args.fps) > 0 else 10.0,
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(int(args.crf))],
    ) as w:
        for fr in frames:
            w.append_data(np.asarray(fr.convert("RGB")))

    print(f"[OK] wrote mp4: {out}")
    print(f"[OK] ckpt: {ckpt}  config: {cfg}")


if __name__ == "__main__":
    main()


