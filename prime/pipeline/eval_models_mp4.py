from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from prime.pipeline.io import load_projection, load_projection_f32
from prime.pipeline.media import write_mp4_frames
from prime.pipeline.val_cache import (
    anscombe_forward as _anscombe_forward,
    bm3d_anscombe_denoise as _bm3d_anscombe_denoise,
    counts_int as _counts_int,
    load_cached_bm3d as _load_cached_bm3d,
    load_cached_thin as _load_cached_thin,
    pick_ckpt as _pick_ckpt,
    poisson_thin_binomial as _poisson_thin_binomial,
    seed_cache_dir as _seed_cache_dir,
)


@dataclass(frozen=True)
class ModelPredictor:
    label: str
    predict_fn: Callable[[np.ndarray], np.ndarray]
    config_path: Path | None = None
    checkpoint_path: Path | None = None


def _norm_to_u8(x: np.ndarray, *, vmax: float, use_log1p: bool) -> np.ndarray:
    vmax = max(float(vmax), 1e-6)
    y = np.clip(x.astype(np.float32, copy=False), 0.0, vmax)
    if bool(use_log1p):
        y = np.log1p(y) / np.log1p(vmax)
    else:
        y = y / vmax
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def _signed_to_bwr(x: np.ndarray, *, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-6)
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


def _vmax_from_ref(x: np.ndarray, *, percentile: float) -> float:
    p = float(percentile)
    if p >= 100.0:
        v = float(np.max(x.astype(np.float32, copy=False)))
    else:
        v = float(np.percentile(x.astype(np.float32, copy=False), p))
    return max(v, 1.0)


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
    return f"{int(round(float(actual) / 1.0e4))}W"


def _load_font(size: int = 12):
    from PIL import ImageFont  # type: ignore

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _sec_from_label(label: str, k: float) -> float:
    return 20.0 if label == "20s" else 20.0 / max(float(k), 1e-12)


def _parse_thin_factors(spec: str) -> list[float]:
    out: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return [k for k in out if float(k) > 1.0]


def _parse_proj_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3 or min(shp) <= 0:
        raise ValueError(f"--proj-shape must be like views,h,w and >0, got: {spec}")
    return int(shp[0]), int(shp[1]), int(shp[2])


def _parse_proj_dtype(spec: str) -> np.dtype:
    key = str(spec).strip().lower()
    m = {"int16": np.int16, "uint16": np.uint16, "float32": np.float32}
    if key not in m:
        raise ValueError(f"--proj-dtype must be one of {sorted(m.keys())}, got: {spec}")
    return np.dtype(m[key])


def _build_rows(
    *,
    proj_20s: np.ndarray,
    patient_dir: Path | None,
    lq_path: str | None,
    predictors: list[ModelPredictor],
    thin_factors: list[float],
    seed: int,
    require_cached: bool,
    allow_on_the_fly_cache: bool,
    bm3d_sigma: float,
    bm3d_workers: int,
    include_bm3d: bool,
) -> tuple[list[dict], float]:
    y20i = _counts_int(proj_20s)
    c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
    if c20 <= 0:
        c20 = 1.0

    base_rows: list[tuple[str, float, np.ndarray]] = [("20s", 1.0, y20i.astype(np.float32, copy=False))]
    for k in thin_factors:
        lab = f"x{k:g}"
        xk: np.ndarray | None = None
        if patient_dir is not None and lq_path is not None:
            try:
                xk = _load_cached_thin(
                    patient_dir=patient_dir,
                    lq_path=lq_path,
                    label=lab,
                    seed=int(seed),
                    require_cached=bool(require_cached),
                )
            except FileNotFoundError:
                xk = None

        if xk is None:
            if not bool(allow_on_the_fly_cache):
                raise FileNotFoundError(
                    f"Missing cached thin split for {lab}. "
                    f"Set allow_on_the_fly_cache=True or precompute cache."
                )
            xk = _poisson_thin_binomial(y20i, float(k), int(seed))
            if patient_dir is not None:
                np.save(_seed_cache_dir(patient_dir, int(seed)) / f"{lab}_thin.npy", xk.astype(np.float32, copy=False))

        base_rows.append((lab, float(k), xk.astype(np.float32, copy=False)))

    base_rows.sort(key=lambda t: _sec_from_label(t[0], t[1]), reverse=True)

    rows: list[dict] = []
    for label, k, orig in base_rows:
        pred_map: dict[str, np.ndarray] = {}
        if bool(include_bm3d):
            bm3d = None
            if patient_dir is not None and lq_path is not None:
                try:
                    bm3d = _load_cached_bm3d(
                        patient_dir=patient_dir,
                        lq_path=lq_path,
                        label=label,
                        seed=int(seed),
                        require_cached=bool(require_cached),
                    )
                except FileNotFoundError:
                    bm3d = None

            if bm3d is None:
                if not bool(allow_on_the_fly_cache):
                    raise FileNotFoundError(
                        f"Missing cached BM3D for {label}. "
                        f"Set allow_on_the_fly_cache=True or precompute cache."
                    )
                bm3d = _bm3d_anscombe_denoise(
                    orig,
                    sigma_psd=float(bm3d_sigma),
                    num_workers=int(bm3d_workers),
                )
                if patient_dir is not None:
                    np.save(_seed_cache_dir(patient_dir, int(seed)) / f"{label}_bm3d.npy", bm3d.astype(np.float32, copy=False))

            pred_map["BM3D"] = np.clip(bm3d.astype(np.float32, copy=False), 0.0, None)
        for spec in predictors:
            den = spec.predict_fn(orig.astype(np.float32, copy=False))
            pred_map[str(spec.label)] = np.clip(den.astype(np.float32, copy=False), 0.0, None)

        rows.append(
            dict(
                label=label,
                k=float(k),
                sec=_sec_from_label(label, float(k)),
                orig=orig.astype(np.float32, copy=False),
                preds=pred_map,
                theory=(c20 / max(float(k), 1e-12)),
            )
        )

    return rows, float(c20)


def _prepare_rows(rows: list[dict], *, model_labels: list[str], diff_percentile: float) -> float:
    row20 = next((r for r in rows if str(r["label"]) == "20s"), rows[0])
    orig20 = row20["orig"]
    pred20 = {m: row20["preds"][m] for m in model_labels}

    all_abs_res: list[np.ndarray] = []
    for r in rows:
        k = float(r["k"])
        r["c_orig"] = float(np.sum(_counts_int(r["orig"]).astype(np.float64, copy=False)))
        r["diff_orig"] = (r["orig"] * k - orig20).astype(np.float32, copy=False)
        r["diffs"] = {}
        r["counts"] = {}
        for m in model_labels:
            pm = r["preds"][m]
            r["diffs"][m] = (pm * k - pred20[m]).astype(np.float32, copy=False)
            r["counts"][m] = float(np.sum(pm.astype(np.float64, copy=False)))
            all_abs_res.append(np.abs(_anscombe_forward(r["orig"]) - _anscombe_forward(pm)).reshape(-1))

    flat = np.concatenate(all_abs_res, axis=0) if all_abs_res else np.array([1.0], dtype=np.float32)
    dv = float(np.percentile(flat, float(diff_percentile)))
    return max(dv, 1.0)


def _build_frames(
    *,
    rows: list[dict],
    model_labels: list[str],
    turns: int,
    vmax20: float,
    dv: float,
    use_log1p: bool,
    show_global_text: bool,
    show_factor_col: bool,
    factor_col_width: int,
):
    from PIL import Image, ImageDraw  # type: ignore

    if not rows:
        return []
    n_views = int(rows[0]["orig"].shape[0])
    H = int(rows[0]["orig"].shape[1])
    W = int(rows[0]["orig"].shape[2])
    header_h = 22
    font = _load_font(12)

    col_labels: list[str] = ["Original", "Original diff"]
    for m in model_labels:
        col_labels.extend([str(m), f"{m} diff"])
    n_cols = len(col_labels)
    fcw = max(48, int(factor_col_width))

    def _g2rgb(u8: np.ndarray):
        return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")

    turns = int(max(1, int(turns)))
    frames = []
    for fi in range(n_views * turns):
        vi = int(fi % n_views)
        row_imgs = []
        for r in rows:
            k = float(r["k"])
            row_w = W * n_cols + (fcw if bool(show_factor_col) else 0)
            row_canvas = Image.new("RGB", (row_w, H), color=(0, 0, 0))
            draw = ImageDraw.Draw(row_canvas)
            x_off = fcw if bool(show_factor_col) else 0

            row_label = f"{r['sec']:.0f}s(x{k:g})" if str(r["label"]).startswith("x") else f"{r['sec']:.0f}s"
            if bool(show_factor_col):
                # Dedicated left factor column: makes dose multiplier visually explicit.
                draw.rectangle([(0, 0), (fcw - 1, H - 1)], fill=(18, 18, 18))
                k_txt = "x1" if abs(float(k) - 1.0) < 1e-6 else f"x{k:g}"
                s_txt = f"{float(r['sec']):.1f}s"
                draw.text((6, 6), k_txt, fill=255, font=font)
                draw.text((6, 24), s_txt, fill=200, font=font)
                draw.text((6, 42), _fmt_w_int(float(r["c_orig"])), fill=200, font=font)
            else:
                draw.text((4, 4), f"{row_label} / {_fmt_w_int(float(r['c_orig']))}", fill=255, font=font)
            orig_img = _norm_to_u8(r["orig"][vi] * k, vmax=vmax20, use_log1p=bool(use_log1p))
            orig_diff = _signed_to_bwr(r["diff_orig"][vi], vmax=float(dv))
            row_canvas.paste(_g2rgb(orig_img), (x_off + 0, 0))
            row_canvas.paste(Image.fromarray(orig_diff, mode="RGB"), (x_off + W, 0))
            for mi, m in enumerate(model_labels):
                pred = r["preds"][m]
                img = _norm_to_u8(pred[vi] * k, vmax=vmax20, use_log1p=bool(use_log1p))
                diff = _signed_to_bwr(r["diffs"][m][vi], vmax=float(dv))
                x0 = x_off + (mi + 1) * 2 * W
                row_canvas.paste(_g2rgb(img), (x0, 0))
                row_canvas.paste(Image.fromarray(diff, mode="RGB"), (x0 + W, 0))
                draw.text((x0 + 4, 4), _fmt_counts_w_pct(float(r["counts"][m]), float(r["theory"])), fill=255, font=font)
            row_imgs.append(row_canvas)

        full_w = W * n_cols + (fcw if bool(show_factor_col) else 0)
        full = Image.new("RGB", (full_w, header_h + H * len(row_imgs)), color=(0, 0, 0))
        draw = ImageDraw.Draw(full)
        if bool(show_factor_col):
            draw.text((6, 3), "Dose", fill=255, font=font)
        for j, lab in enumerate(col_labels):
            draw.text(((fcw if bool(show_factor_col) else 0) + j * W + 4, 3), lab, fill=255, font=font)
        if bool(show_global_text):
            draw.text((full_w - 170, 3), f"ans_vmax={dv:.3g}", fill=255, font=font)
        for i, im in enumerate(row_imgs):
            full.paste(im, (0, header_h + i * H))
        frames.append(full)

    return frames


def generate_eval_mp4_from_projection(
    *,
    proj_20s: np.ndarray,
    predictors: list[ModelPredictor],
    out_path: Path | str,
    thin_factors: list[float] | None = None,
    seed: int = 123,
    patient_dir: Path | None = None,
    lq_path: str | None = None,
    require_cached: bool = True,
    allow_on_the_fly_cache: bool = False,
    bm3d_sigma: float = 1.0,
    bm3d_workers: int = 0,
    include_bm3d: bool = False,
    use_log1p: bool = True,
    show_global_text: bool = True,
    fps: float = 10.0,
    crf: int = 18,
    turns: int = 2,
    diff_percentile: float = 99.5,
    vmax_percentile: float = 100.0,
    show_factor_col: bool = True,
    factor_col_width: int = 64,
) -> Path:
    thin_list = list(thin_factors) if thin_factors is not None else [2.0, 3.0, 4.0, 5.0]
    rows, _ = _build_rows(
        proj_20s=np.asarray(proj_20s, dtype=np.float32),
        patient_dir=patient_dir,
        lq_path=lq_path,
        predictors=predictors,
        thin_factors=thin_list,
        seed=int(seed),
        require_cached=bool(require_cached),
        allow_on_the_fly_cache=bool(allow_on_the_fly_cache),
        bm3d_sigma=float(bm3d_sigma),
        bm3d_workers=int(bm3d_workers),
        include_bm3d=bool(include_bm3d),
    )
    y20i = _counts_int(rows[0]["orig"])
    vmax20 = _vmax_from_ref(y20i, percentile=float(vmax_percentile))
    model_labels = []
    if bool(include_bm3d):
        model_labels.append("BM3D")
    model_labels.extend([str(p.label) for p in predictors])
    dv = _prepare_rows(rows, model_labels=model_labels, diff_percentile=float(diff_percentile))
    frames = _build_frames(
        rows=rows,
        model_labels=model_labels,
        turns=int(turns),
        vmax20=float(vmax20),
        dv=float(dv),
        use_log1p=bool(use_log1p),
        show_global_text=bool(show_global_text),
        show_factor_col=bool(show_factor_col),
        factor_col_width=int(factor_col_width),
    )
    out = Path(out_path)
    write_mp4_frames(
        out,
        frames,
        fps=float(fps),
        crf=int(crf),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    return out


def _parse_model_spec(spec: str) -> tuple[str, Path, int | None]:
    if "=" not in str(spec):
        raise ValueError(f"--model-exp format must be label=exp_dir[@iter], got: {spec}")
    label, rhs = str(spec).split("=", 1)
    label = label.strip()
    rhs = rhs.strip()
    if not label or not rhs:
        raise ValueError(f"--model-exp format must be label=exp_dir[@iter], got: {spec}")
    iter_id = None
    exp_str = rhs
    m = re.match(r"^(.*)@(\d+)$", rhs)
    if m:
        exp_str = m.group(1)
        iter_id = int(m.group(2))
    return label, Path(exp_str), iter_id


def _parse_model_ckpt_spec(spec: str) -> tuple[str, Path, Path]:
    """Parse --model-ckpt as: label=ckpt_path|config_path"""
    if "=" not in str(spec):
        raise ValueError(f"--model-ckpt format must be label=ckpt_path|config_path, got: {spec}")
    label, rhs = str(spec).split("=", 1)
    label = label.strip()
    rhs = rhs.strip()
    if not label or not rhs or "|" not in rhs:
        raise ValueError(f"--model-ckpt format must be label=ckpt_path|config_path, got: {spec}")
    ckpt_str, cfg_str = rhs.rsplit("|", 1)
    ckpt = Path(ckpt_str.strip())
    cfg = Path(cfg_str.strip())
    if not ckpt:
        raise ValueError(f"empty checkpoint path in --model-ckpt: {spec}")
    if not cfg:
        raise ValueError(f"empty config path in --model-ckpt: {spec}")
    return label, ckpt, cfg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output projection mp4 path.")
    ap.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    ap.add_argument("--patient-dir", type=str, default=None)
    ap.add_argument("--input-proj", type=str, default=None)
    ap.add_argument("--proj-shape", type=str, default="60,128,128")
    ap.add_argument("--proj-dtype", type=str, default="int16", choices=["int16", "uint16", "float32"])
    ap.add_argument("--model-exp", type=str, action="append", default=[], help="Repeatable: label=exp_dir[@iter]")
    ap.add_argument(
        "--model-ckpt",
        type=str,
        action="append",
        default=[],
        help="Repeatable: label=ckpt_path|config_path (exact checkpoint, no auto-pick).",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thin-factors", type=str, default="2,3,4,5")
    ap.add_argument("--require-cached", dest="require_cached", action="store_true")
    ap.add_argument("--no-require-cached", dest="require_cached", action="store_false")
    ap.add_argument("--allow-on-the-fly-cache", action="store_true")
    ap.set_defaults(require_cached=True, allow_on_the_fly_cache=False)
    ap.add_argument("--bm3d-sigma", type=float, default=1.0)
    ap.add_argument("--bm3d-workers", type=int, default=0)
    ap.add_argument("--with-bm3d", action="store_true", help="Include BM3D as one additional model.")
    ap.add_argument("--max-value", type=float, default=150.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--no-log1p", dest="log1p", action="store_false")
    ap.set_defaults(log1p=True)
    ap.add_argument("--show-global-text", action="store_true")
    ap.add_argument("--no-global-text", dest="show_global_text", action="store_false")
    ap.set_defaults(show_global_text=True)
    ap.add_argument("--show-factor-col", action="store_true", help="Show dedicated left column for dose factor.")
    ap.add_argument("--no-factor-col", dest="show_factor_col", action="store_false")
    ap.set_defaults(show_factor_col=True)
    ap.add_argument("--factor-col-width", type=int, default=64, help="Left factor column width in pixels.")
    ap.add_argument("--diff-percentile", type=float, default=99.5)
    ap.add_argument("--vmax-percentile", type=float, default=100.0)
    return ap


def main() -> None:
    from prime.pipeline.experiment import find_config_for_experiment
    from prime.pipeline.inference import denoise_views, load_basicsr_net

    args = build_arg_parser().parse_args()
    if not args.model_exp and not args.model_ckpt:
        raise ValueError("At least one model must be provided via --model-exp or --model-ckpt.")
    proj_shape = _parse_proj_shape(str(args.proj_shape))
    proj_dtype = _parse_proj_dtype(str(args.proj_dtype))
    thin_factors = _parse_thin_factors(str(args.thin_factors))

    spect229_dir = Path(str(args.spect229_dir))
    legacy_patient_dir = spect229_dir / str(args.patient)
    if args.patient_dir:
        patient_dir = Path(str(args.patient_dir))
    elif legacy_patient_dir.exists():
        patient_dir = legacy_patient_dir
    elif args.input_proj:
        patient_dir = Path(str(args.input_proj)).resolve().parent
    else:
        patient_dir = legacy_patient_dir

    if args.input_proj:
        lq_path = str(Path(str(args.input_proj)).resolve())
    else:
        lq_path = str((patient_dir / f"{args.patient}_Proj4Filter.dat").resolve())
    if not Path(lq_path).exists():
        raise FileNotFoundError(f"lq projection not found: {lq_path}")

    proj_20s = load_projection(Path(lq_path), shape=proj_shape, dtype=proj_dtype).astype(np.float32, copy=False)

    predictors: list[ModelPredictor] = []
    for spec in args.model_exp or []:
        label, exp_dir, prefer_iter = _parse_model_spec(str(spec))
        cfg = find_config_for_experiment(exp_dir)
        ckpt = _pick_ckpt(exp_dir, prefer_iter=prefer_iter)
        net = load_basicsr_net(cfg, ckpt, device=str(args.device), require_ema=True).net
        mv = float(args.max_value)

        def _predict(x: np.ndarray, *, _net=net, _mv=mv, _device=str(args.device)) -> np.ndarray:
            y = denoise_views(_net, x.astype(np.float32, copy=False), max_value=float(_mv), device=str(_device))
            return np.clip(y.astype(np.float32, copy=False), 0.0, None)

        predictors.append(
            ModelPredictor(
                label=str(label),
                predict_fn=_predict,
                config_path=Path(cfg),
                checkpoint_path=Path(ckpt),
            )
        )

    for spec in args.model_ckpt or []:
        label, ckpt, cfg = _parse_model_ckpt_spec(str(spec))
        net = load_basicsr_net(cfg, ckpt, device=str(args.device), require_ema=True).net
        mv = float(args.max_value)

        def _predict_ckpt(x: np.ndarray, *, _net=net, _mv=mv, _device=str(args.device)) -> np.ndarray:
            y = denoise_views(_net, x.astype(np.float32, copy=False), max_value=float(_mv), device=str(_device))
            return np.clip(y.astype(np.float32, copy=False), 0.0, None)

        predictors.append(
            ModelPredictor(
                label=str(label),
                predict_fn=_predict_ckpt,
                config_path=Path(cfg),
                checkpoint_path=Path(ckpt),
            )
        )

    out = generate_eval_mp4_from_projection(
        proj_20s=proj_20s,
        predictors=predictors,
        out_path=Path(str(args.out)),
        thin_factors=thin_factors,
        seed=int(args.seed),
        patient_dir=Path(patient_dir) if patient_dir is not None else None,
        lq_path=lq_path,
        require_cached=bool(args.require_cached),
        allow_on_the_fly_cache=bool(args.allow_on_the_fly_cache),
        bm3d_sigma=float(args.bm3d_sigma),
        bm3d_workers=int(args.bm3d_workers),
        include_bm3d=bool(args.with_bm3d),
        use_log1p=bool(args.log1p),
        show_global_text=bool(args.show_global_text),
        fps=float(args.fps),
        crf=int(args.crf),
        turns=int(args.turns),
        diff_percentile=float(args.diff_percentile),
        vmax_percentile=float(args.vmax_percentile),
        show_factor_col=bool(args.show_factor_col),
        factor_col_width=int(args.factor_col_width),
    )

    print(f"[OK] wrote mp4: {out}")
    if bool(args.with_bm3d):
        print("[OK] columns are pairs: [Original, Original diff] + [BM3D, BM3D diff] + [model, model diff]...")
    else:
        print("[OK] columns are pairs: [Original, Original diff] + [model, model diff]...")
    for p in predictors:
        print(f"[OK] model: {p.label}  ckpt: {p.checkpoint_path}  config: {p.config_path}")


if __name__ == "__main__":
    main()
