#!/usr/bin/env python3
from __future__ import annotations

"""
Render a training-style MP4 for REAL low-dose acquisitions (not synthetic thinning).

Rows: 20s baseline + low-dose acquisitions discovered under a lowdose root (e.g. 6s-2/, 4s-3/, 2s-7/).
Cols (3): Original | Denoised(ema) | Poisson(ema)

Key behaviors aligned with existing training mp4 style:
  - Display gray columns in 20s-equivalent domain: multiply each row by k and use SAME vmax20 (from 20s baseline).
  - Estimate k from total counts: k ≈ C20 / C_low (per file).
  - Poisson(ema) sampled ONCE per row (stable across frames) with deterministic RNG based on seed and row index.
  - Overlay counts in W and % diff vs theory (C20/k) for denoised/poisson; Original shows only "{tag} / {sec}s / W".
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


V, H, W = 60, 128, 128
N = V * H * W


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


def _load_u16_proj(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    if arr.size != N:
        raise ValueError(f"unexpected proj size: {path} elems={arr.size} expected={N}")
    return arr.reshape(V, H, W).astype(np.float32, copy=False)


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
    best_it = -1
    best_p = None
    for p in models.glob("net_g_*.pth"):
        m = re.search(r"net_g_(\d+)\.pth", p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_it:
            best_it = it
            best_p = p
    if best_p is None:
        raise FileNotFoundError(f"No checkpoint found under: {models}")
    return best_p


def _infer_denoised(*, config: Path, ckpt: Path, proj: np.ndarray, max_value: float, device: str) -> np.ndarray:
    from spect_ct.pipeline.inference import load_basicsr_net, denoise_views
    loaded = load_basicsr_net(config, ckpt, device=device, require_ema=True)
    out = denoise_views(loaded.net, proj.astype(np.float32, copy=False), max_value=float(max_value), device=device)
    return np.clip(out.astype(np.float32, copy=False), 0.0, None)


def _sec_from_tag(tag: str) -> Optional[float]:
    # Tags like "2s-7/ProjectionImage3" or "6s-2/ProjectionImage1"
    m = re.search(r"(^|/)(\d+(?:\.\d+)?)s", str(tag))
    if not m:
        return None
    try:
        return float(m.group(2))
    except Exception:
        return None


@dataclass(frozen=True)
class Row:
    tag: str
    sec: float
    k: float
    orig: np.ndarray
    ema: np.ndarray
    poi: np.ndarray
    c_orig: float
    c_ema: float
    c_poi: float
    theory: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, required=True, help="patient name (used to find 20s baseline under datasets/SPECT229 and lowdose root)")
    ap.add_argument("--lowdose-root", type=str, default=None, help="root dir, default: datasets/SPECT229_lowdose/<patient>")
    ap.add_argument("--baseline-proj", type=str, default=None, help="20s baseline proj, default: datasets/SPECT229/<patient>/<patient>_Proj4Filter.dat")

    ap.add_argument("--exp", type=str, required=True, help="experiment dir for model (e.g. experiments/n2n_spect229_60view3d_unet_projonly_full_multidose)")
    ap.add_argument("--iter", type=int, default=None, help="optional checkpoint iter")
    ap.add_argument("--config", type=str, default=None, help="optional explicit YAML config path")
    ap.add_argument("--ckpt", type=str, default=None, help="optional explicit checkpoint path")
    ap.add_argument("--out", type=str, required=True, help="output mp4 path")

    ap.add_argument("--max-value", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--no-log1p", dest="log1p", action="store_false")
    ap.set_defaults(log1p=True)

    ap.add_argument("--include", type=str, default="2s-7,4s-3,6s-2", help="comma list of subdirs under lowdose root to include")
    ap.add_argument("--pattern", type=str, default="ProjectionImage*.dat", help="glob pattern within each include dir")
    args = ap.parse_args()

    from PIL import Image, ImageDraw  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    patient = str(args.patient)
    low_root = Path(args.lowdose_root) if args.lowdose_root else (Path("datasets/SPECT229_lowdose") / patient)
    base_proj = Path(args.baseline_proj) if args.baseline_proj else (Path("datasets/SPECT229") / patient / f"{patient}_Proj4Filter.dat")
    if not base_proj.is_file():
        raise FileNotFoundError(f"baseline 20s proj not found: {base_proj}")
    if not low_root.exists():
        raise FileNotFoundError(f"lowdose_root not found: {low_root}")

    proj20 = _load_u16_proj(base_proj)
    y20i = _counts_int(proj20)
    c20 = float(y20i.astype(np.float64).sum())
    if c20 <= 0:
        c20 = 1.0
    vmax20 = float(y20i.astype(np.float32).max())
    if vmax20 < 1e-6:
        vmax20 = 1.0

    include_dirs = [s.strip() for s in str(args.include).split(",") if s.strip() != ""]
    files: list[tuple[str, Path]] = []
    for d in include_dirs:
        dd = low_root / d
        for p in sorted(dd.glob(str(args.pattern))):
            tag = f"{d}/{p.stem}"
            files.append((tag, p))

    if len(files) == 0:
        raise FileNotFoundError(f"No lowdose projections found under {low_root} (include={include_dirs}, pattern={args.pattern})")

    # Resolve model config/ckpt
    from spect_ct.pipeline.experiment import find_config_for_experiment

    exp_dir = Path(args.exp)
    cfg = Path(args.config) if args.config else find_config_for_experiment(exp_dir)
    ckpt = Path(args.ckpt) if args.ckpt else _pick_ckpt(exp_dir, prefer_iter=args.iter)

    mv = float(args.max_value)
    seed = int(args.seed)
    use_log1p = bool(args.log1p)

    # Build rows: 20s baseline + each lowdose file.
    base_rows: list[tuple[str, float, float, np.ndarray]] = []
    base_rows.append(("20s", 20.0, 1.0, proj20))
    for tag, p in files:
        proj = _load_u16_proj(p)
        c = float(_counts_int(proj).astype(np.float64).sum())
        k = float(c20 / max(c, 1e-12))
        sec = _sec_from_tag(tag)
        if sec is None:
            sec = 20.0 / max(k, 1e-12)
        base_rows.append((tag, float(sec), float(k), proj))

    # Sort: longer seconds on top (like training mp4)
    base_rows.sort(key=lambda t: float(t[1]), reverse=True)

    rows: list[Row] = []
    for ri, (tag, sec, k, orig) in enumerate(base_rows):
        den_ema = _infer_denoised(config=cfg, ckpt=ckpt, proj=orig, max_value=mv, device=str(args.device))
        rng = np.random.default_rng(seed + int(ri) * 10007)
        poi = rng.poisson(lam=den_ema).astype(np.float32)
        c_orig = float(_counts_int(orig).astype(np.float64).sum())
        c_ema = float(den_ema.astype(np.float64).sum())
        c_poi = float(_counts_int(poi).astype(np.float64).sum())
        theory = float(c20 / max(float(k), 1e-12))
        rows.append(
            Row(
                tag=str(tag),
                sec=float(sec),
                k=float(k),
                orig=orig.astype(np.float32, copy=False),
                ema=den_ema.astype(np.float32, copy=False),
                poi=poi.astype(np.float32, copy=False),
                c_orig=c_orig,
                c_ema=c_ema,
                c_poi=c_poi,
                theory=theory,
            )
        )

    # Render frames
    font = _load_font(12)
    header_h = 22
    col_labels = ["Original", "Denoised(ema)", "Poisson(ema)"]
    w0, h0 = 128, 128

    def _g2rgb(u8: np.ndarray) -> Image.Image:
        return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")

    frames = []
    for vi in range(60):
        row_imgs = []
        for r in rows:
            kk = float(r.k)
            # 20s-equivalent display
            a = _norm_to_u8(r.orig[vi] * kk, vmax=vmax20, use_log1p=use_log1p)
            b = _norm_to_u8(r.ema[vi] * kk, vmax=vmax20, use_log1p=use_log1p)
            c = _norm_to_u8(r.poi[vi] * kk, vmax=vmax20, use_log1p=use_log1p)
            row_canvas = Image.new("RGB", (w0 * 3, h0), color=(0, 0, 0))
            row_canvas.paste(_g2rgb(a), (0 * w0, 0))
            row_canvas.paste(_g2rgb(b), (1 * w0, 0))
            row_canvas.paste(_g2rgb(c), (2 * w0, 0))
            draw = ImageDraw.Draw(row_canvas)

            # Left label: include sec + k + orig W
            if r.tag == "20s":
                left = f"20s / {_fmt_w_int(r.c_orig)}"
            else:
                left = f"{r.tag} / {r.sec:.0f}s(k≈{r.k:.2f}) / {_fmt_w_int(r.c_orig)}"
            draw.text((4, 4), left, fill=255, font=font)

            # Den/poi: counts vs theory
            t = float(r.theory)
            draw.text((1 * w0 + 4, 4), _fmt_counts_w_pct(float(r.c_ema), t), fill=255, font=font)
            draw.text((2 * w0 + 4, 4), _fmt_counts_w_pct(float(r.c_poi), t), fill=255, font=font)
            row_imgs.append(row_canvas)

        full = Image.new("RGB", (w0 * 3, header_h + h0 * len(row_imgs)), color=(0, 0, 0))
        draw = ImageDraw.Draw(full)
        for j, lab in enumerate(col_labels):
            draw.text((j * w0 + 4, 3), lab, fill=255, font=font)
        draw.text((w0 * 3 - 120, 3), f"view={vi:02d}", fill=255, font=font)
        for i, im in enumerate(row_imgs):
            full.paste(im, (0, header_h + i * h0))
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
    print(f"[OK] baseline: {base_proj}  C20={c20/1e4:.2f}W")
    print(f"[OK] ckpt: {ckpt}  config: {cfg}")


if __name__ == "__main__":
    main()







