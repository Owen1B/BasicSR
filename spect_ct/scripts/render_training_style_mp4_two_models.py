#!/usr/bin/env python3
from __future__ import annotations

"""
Render a validation-style MP4 (same style as BasicSR training MP4) but comparing TWO models.

This script is intentionally a faithful copy/adaptation of the core rendering logic from:
  `basicsr/models/spect3d_model.py::_generate_validation_mp4_thin_factors`

Target layout requested:
  Rows (5): 20s + thinned x2/x3/x4/x5 (longer time on top)
  Cols (8): Original | BM3D | ModelB(ema) | ModelA(ema) | Original diff | BM3D diff | ModelB diff | ModelA diff

Notes:
  - Gray columns are displayed in 20s-equivalent domain: multiply each row by k and use SAME vmax20.
  - Diff is computed in count domain: diff_cnt = ema_lowdose*k - ema20.
  - Diff colormap uses the SAME scale as Anscombe residual (dv_res) to match training MP4 behavior.
  - Fixed thins/BM3D are loaded from dataset cache:
      datasets/SPECT229/<patient>/bm3d_cache/seed{seed}/
      - Prefer readable: {x2,x3,x4,x5}_thin.npy and {20s,x2,x3,x4,x5}_bm3d.npy
      - Fallback: md5(lq_path+"__"+label)[:16]_{thin|bm3d}.npy
"""

import argparse
import os
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


def _anscombe_forward(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(x + 3.0 / 8.0, 0.0))


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


def _signed_to_bwr(x: np.ndarray, *, vmax: float) -> np.ndarray:
    vmax = float(vmax)
    if vmax < 1e-6:
        vmax = 1.0
    t = np.clip(x.astype(np.float32, copy=False), -vmax, vmax) / vmax
    u = (t + 1.0) * 0.5
    try:
        import matplotlib
        matplotlib.use("Agg")
        try:
            cmap = matplotlib.colormaps.get_cmap("bwr")
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap("bwr")
        return (cmap(u)[..., :3] * 255.0).round().astype(np.uint8)
    except Exception:
        rgb = np.zeros((*u.shape, 3), dtype=np.float32)
        lo = u <= 0.5
        hi = ~lo
        a = (u[lo] / 0.5)[:, None]
        rgb[lo] = (1 - a) * np.array([0, 0, 255], dtype=np.float32) + a * np.array([255, 255, 255], dtype=np.float32)
        b = ((u[hi] - 0.5) / 0.5)[:, None]
        rgb[hi] = (1 - b) * np.array([255, 255, 255], dtype=np.float32) + b * np.array([255, 0, 0], dtype=np.float32)
        return rgb.round().clip(0, 255).astype(np.uint8)


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
    """Format total counts in integer W (1e4). Example: 140W."""
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
    # Prefer seed-scoped readable cache if present
    cand_root = Path(patient_dir) / "bm3d_cache"
    ds_cache_dir = None
    if cand_root.exists():
        seed_dir = cand_root / f"seed{int(seed)}"
        ds_cache_dir = seed_dir if seed_dir.exists() else cand_root
    xk = None
    if ds_cache_dir is not None:
        # readable first
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
        # optional fallback: deterministic on-the-fly thinning (not identical to training when caches exist)
        rng = np.random.default_rng(int(seed) + int(round(float(label[1:]) * 1000.0)))
        # This branch requires y20i passed in by caller; avoid accidental use.
        raise RuntimeError("On-the-fly thinning is disabled in this script; please precompute caches.")
    return xk


def _load_cached_bm3d(*, patient_dir: Path, lq_path: str, label: str, seed: int, require_cached: bool) -> np.ndarray:
    cand_root = Path(patient_dir) / "bm3d_cache"
    ds_cache_dir = None
    if cand_root.exists():
        seed_dir = cand_root / f"seed{int(seed)}"
        ds_cache_dir = seed_dir if seed_dir.exists() else cand_root
    arr = None
    if ds_cache_dir is not None:
        readable = ds_cache_dir / f"{label}_bm3d.npy"
        if readable.exists():
            arr = np.load(readable).astype(np.float32, copy=False)
        else:
            key = _md5_16(f"{str(lq_path)}__{label}")
            p2 = ds_cache_dir / f"{key}_bm3d.npy"
            if p2.exists():
                arr = np.load(p2).astype(np.float32, copy=False)
    if arr is None:
        if require_cached:
            raise FileNotFoundError(
                f"Missing cached BM3D: patient={patient_dir.name} label={label} "
                f"(expected under {cand_root}/seed{seed})."
            )
        raise RuntimeError("BM3D on-the-fly is disabled in this script; please precompute caches.")
    return np.clip(arr.astype(np.float32, copy=False), 0.0, None)


def _infer_denoised(*, config: Path, ckpt: Path, proj: np.ndarray, max_value: float, device: str) -> np.ndarray:
    # Use the same loader as toolbox: strict EMA weights.
    from spect_ct.pipeline.inference import load_basicsr_net, denoise_views
    loaded = load_basicsr_net(config, ckpt, device=device, require_ema=True)
    out = denoise_views(loaded.net, proj.astype(np.float32, copy=False), max_value=float(max_value), device=device)
    return np.clip(out.astype(np.float32, copy=False), 0.0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, required=True, help="patient folder name under datasets/SPECT229")
    ap.add_argument("--val-sample-name", type=str, default=None, help="suffix name like BaiYukun_Proj4Filter_003 (used only for output filename)")
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
    ap.add_argument("--turns", type=int, default=2, help="number of full 60-view cycles to render (e.g. 2 => 120 frames)")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--exp-a", type=str, required=True, help="experiment dir for model A (e.g. experiments/n2n_spect229_60view3d_unet_projonly_full_multidose)")
    ap.add_argument("--iter-a", type=int, default=None, help="optional checkpoint iter for model A (e.g. 13000)")
    ap.add_argument("--config-a", type=str, default=None)
    ap.add_argument("--ckpt-a", type=str, default=None)
    ap.add_argument("--label-a", type=str, default="60view3d(ema)")

    ap.add_argument("--exp-b", type=str, required=True, help="experiment dir for model B (e.g. experiments/n2n_spect229_singleview_patch64)")
    ap.add_argument("--iter-b", type=int, default=None)
    ap.add_argument("--config-b", type=str, default=None)
    ap.add_argument("--ckpt-b", type=str, default=None)
    ap.add_argument("--label-b", type=str, default="patch64(ema)")

    args = ap.parse_args()

    from PIL import Image, ImageDraw  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    spect229_dir = Path(args.spect229_dir)
    patient_dir = spect229_dir / str(args.patient)
    if not patient_dir.exists():
        raise FileNotFoundError(f"patient_dir not found: {patient_dir}")

    # lq_path in training validation is full path to *_Proj4Filter.dat
    lq_path = str((patient_dir / f"{args.patient}_Proj4Filter.dat").resolve())
    if not Path(lq_path).exists():
        raise FileNotFoundError(f"lq projection not found: {lq_path}")

    # Load 20s projection (int16 dataset), convert to count-int for visualization math.
    from spect_ct.pipeline.io import load_projection_i16
    proj_u16 = load_projection_i16(Path(lq_path)).astype(np.float32, copy=False)
    y20i = _counts_int(proj_u16)
    c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
    if c20 <= 0:
        c20 = 1.0
    vmax20 = float(np.max(y20i.astype(np.float32, copy=False)))
    if vmax20 < 1e-6:
        vmax20 = 1.0

    # Rows: 20s + x2/x3/x4/x5 from cached thins (same as training MP4).
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

    # Resolve model configs/ckpts
    from spect_ct.pipeline.experiment import find_config_for_experiment

    exp_a = Path(args.exp_a)
    exp_b = Path(args.exp_b)
    cfg_a = Path(args.config_a) if args.config_a else find_config_for_experiment(exp_a)
    cfg_b = Path(args.config_b) if args.config_b else find_config_for_experiment(exp_b)
    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else _pick_ckpt(exp_a, prefer_iter=args.iter_a)
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else _pick_ckpt(exp_b, prefer_iter=args.iter_b)

    mv = float(args.max_value)
    use_log1p = bool(args.log1p)

    # Compute per-row: BM3D + modelA EMA + modelB EMA + residual(ans) for dv scale + diff
    rows = []
    for label, k, orig in base_rows:
        den_bm3d = _load_cached_bm3d(patient_dir=patient_dir, lq_path=lq_path, label=label, seed=int(args.seed), require_cached=bool(args.require_cached))
        ema_a = _infer_denoised(config=cfg_a, ckpt=ckpt_a, proj=orig, max_value=mv, device=str(args.device))
        ema_b = _infer_denoised(config=cfg_b, ckpt=ckpt_b, proj=orig, max_value=mv, device=str(args.device))
        res_a = (_anscombe_forward(orig) - _anscombe_forward(ema_a)).astype(np.float32, copy=False)
        res_b = (_anscombe_forward(orig) - _anscombe_forward(ema_b)).astype(np.float32, copy=False)
        rows.append(
            dict(
                label=label,
                k=float(k),
                sec=_sec(label, k),
                orig=orig.astype(np.float32, copy=False),
                bm3d=den_bm3d.astype(np.float32, copy=False),
                ema_a=ema_a,
                ema_b=ema_b,
                res_a=res_a,
                res_b=res_b,
                theory=(c20 / max(float(k), 1e-12)),
            )
        )

    # Find 20s references (orig + BM3D + EMA outputs for both models)
    orig20 = None
    bm3d20 = None
    ema20_a = None
    ema20_b = None
    for r in rows:
        if r["label"] == "20s":
            orig20 = r["orig"]
            bm3d20 = r["bm3d"]
            ema20_a = r["ema_a"]
            ema20_b = r["ema_b"]
            break
    if ema20_a is None:
        ema20_a = rows[0]["ema_a"]
    if ema20_b is None:
        ema20_b = rows[0]["ema_b"]
    if bm3d20 is None:
        bm3d20 = rows[0]["bm3d"]
    if orig20 is None:
        orig20 = rows[0]["orig"]

    diffs_a = []
    diffs_b = []
    for r in rows:
        k = float(r["k"])
        r["diff_orig"] = (r["orig"] * k - orig20).astype(np.float32, copy=False)
        r["diff_bm3d"] = (r["bm3d"] * k - bm3d20).astype(np.float32, copy=False)
        r["diff_a"] = (r["ema_a"] * k - ema20_a).astype(np.float32, copy=False)
        r["diff_b"] = (r["ema_b"] * k - ema20_b).astype(np.float32, copy=False)
        r["c_orig"] = float(np.sum(_counts_int(r["orig"]).astype(np.float64, copy=False)))
        r["c_bm3d"] = float(np.sum(_counts_int(r["bm3d"]).astype(np.float64, copy=False)))
        r["c_ema_a"] = float(np.sum(r["ema_a"].astype(np.float64, copy=False)))
        r["c_ema_b"] = float(np.sum(r["ema_b"].astype(np.float64, copy=False)))
        diffs_a.append(np.abs(r["res_a"]).reshape(-1))
        diffs_b.append(np.abs(r["res_b"]).reshape(-1))

    flat_a = np.concatenate(diffs_a, axis=0) if diffs_a else np.array([1.0], dtype=np.float32)
    flat_b = np.concatenate(diffs_b, axis=0) if diffs_b else np.array([1.0], dtype=np.float32)
    dv_a = float(np.percentile(flat_a, 99.5))
    dv_b = float(np.percentile(flat_b, 99.5))
    if dv_a < 1e-6:
        dv_a = 1.0
    if dv_b < 1e-6:
        dv_b = 1.0
    # Use a shared vmax for all diff columns (keeps diffs directly comparable).
    dv = float(max(dv_a, dv_b))

    # ---- render frames (same style as training MP4) ----
    W, H = 128, 128
    header_h = 22
    font = _load_font(12)
    # Requested layout: put all gray columns first, then all diff columns.
    # Keep model blocks swapped (B first, then A).
    col_labels = [
        "Original",
        "BM3D",
        str(args.label_b),
        str(args.label_a),
        "Original diff",
        "BM3D diff",
        f"{str(args.label_b)} diff",
        f"{str(args.label_a)} diff",
    ]

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
            b = _norm_to_u8(r["bm3d"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            # Gray columns: model blocks swapped (B first, then A)
            c = _norm_to_u8(r["ema_b"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            d = _norm_to_u8(r["ema_a"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            # Diff columns (all share the same dv)
            e = _signed_to_bwr(r["diff_orig"][vi], vmax=dv)
            f = _signed_to_bwr(r["diff_bm3d"][vi], vmax=dv)
            g = _signed_to_bwr(r["diff_b"][vi], vmax=dv)
            h = _signed_to_bwr(r["diff_a"][vi], vmax=dv)

            row_canvas = Image.new("RGB", (W * 8, H), color=(0, 0, 0))
            row_canvas.paste(_g2rgb(a), (0 * W, 0))
            row_canvas.paste(_g2rgb(b), (1 * W, 0))
            row_canvas.paste(_g2rgb(c), (2 * W, 0))
            row_canvas.paste(_g2rgb(d), (3 * W, 0))
            row_canvas.paste(Image.fromarray(e, mode="RGB"), (4 * W, 0))
            row_canvas.paste(Image.fromarray(f, mode="RGB"), (5 * W, 0))
            row_canvas.paste(Image.fromarray(g, mode="RGB"), (6 * W, 0))
            row_canvas.paste(Image.fromarray(h, mode="RGB"), (7 * W, 0))

            draw = ImageDraw.Draw(row_canvas)
            if str(r["label"]).startswith("x"):
                row_label = f"{r['sec']:.0f}s(x{k:g})"
            else:
                row_label = f"{r['sec']:.0f}s"
            draw.text((4, 4), f"{row_label} / {_fmt_w_int(float(r['c_orig']))}", fill=255, font=font)
            t = float(r["theory"])
            # Counts on gray columns only (same spirit as training MP4). Skip all diff columns.
            draw.text((1 * W + 4, 4), _fmt_counts_w_pct(float(r["c_bm3d"]), t), fill=255, font=font)
            draw.text((2 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema_b"]), t), fill=255, font=font)
            draw.text((3 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema_a"]), t), fill=255, font=font)
            row_imgs.append(row_canvas)

        full = Image.new("RGB", (W * 8, header_h + H * len(row_imgs)), color=(0, 0, 0))
        draw = ImageDraw.Draw(full)
        for j, lab in enumerate(col_labels):
            draw.text((j * W + 4, 3), lab, fill=255, font=font)
        # Keep training-style header: show a single shared vmax for diffs.
        # Requested: do NOT display view index.
        draw.text((W * 8 - 170, 3), f"ans_vmax={dv:.3g}", fill=255, font=font)
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
    print(f"[OK] modelA ckpt: {ckpt_a}  config: {cfg_a}")
    print(f"[OK] modelB ckpt: {ckpt_b}  config: {cfg_b}")


if __name__ == "__main__":
    main()


