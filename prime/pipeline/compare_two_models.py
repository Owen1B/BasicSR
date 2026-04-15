"""
Render a validation-style MP4 (same style as BasicSR training MP4) but comparing TWO models.

This module is intentionally a faithful copy/adaptation of the core rendering logic from:
  `prime/models/_legacy/spect3d_model.py::_generate_validation_mp4_thin_factors`

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

Also supports a second-stage reconstruction MP4:
  - Uses OSEM (`prime/pipeline/osem.py::run_osem_reconstruction`) to reconstruct volumes
    from the first-stage exported projection .dat files.
  - Renders a rotating MIP visualization (hot colormap) with the same row ordering (20s, 10s, 6.7s, 5s, 4s).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
from prime.pipeline.io import load_projection_f32, load_recon_f32, save_projection_f32
from prime.pipeline.media import write_mp4_frames


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


def _norm_to_u8_cmap(x: np.ndarray, *, vmax: float, cmap_name: str = "hot") -> np.ndarray:
    from prime.pipeline.viz import scalar_to_u8_cmap

    return scalar_to_u8_cmap(
        np.clip(x.astype(np.float32, copy=False), 0.0, None),
        vmin=0.0,
        vmax=max(float(vmax), 1e-6),
        cmap_name=str(cmap_name),
    )


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


def _vmax_from_ref(x: np.ndarray, *, percentile: float) -> float:
    """Compute a robust vmax from a reference array.

    - If percentile >= 100: returns max(x)
    - Else: returns percentile(x)
    """
    p = float(percentile)
    if p >= 100.0:
        v = float(np.max(x.astype(np.float32, copy=False)))
    else:
        v = float(np.percentile(x.astype(np.float32, copy=False), p))
    if v < 1e-6:
        v = 1.0
    return v


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


def _fmt_counts_w_pct_base(actual: float, base: float) -> str:
    """Format counts in w + pct vs base (base<=0 => no pct)."""
    base = float(base)
    actual = float(actual)
    w = actual / 1.0e4
    if w >= 100:
        cstr = f"C={w:.0f}w"
    elif w >= 10:
        cstr = f"C={w:.1f}w"
    else:
        cstr = f"C={w:.2f}w"
    if base <= 0:
        return cstr
    pct = (actual - base) / base * 100.0
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


def _infer_denoised(*, net, proj: np.ndarray, max_value: float, device: str) -> np.ndarray:
    from prime.pipeline.inference import denoise_views

    out = denoise_views(net, proj.astype(np.float32, copy=False), max_value=float(max_value), device=device)
    return np.clip(out.astype(np.float32, copy=False), 0.0, None)


def _parse_thin_factors_csv(spec: str) -> list[float]:
    out: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return [k for k in out if float(k) > 1.0]


def _parse_recon_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3:
        raise ValueError(f"--recon-shape must be like 128,128,128, got: {spec}")
    return int(shp[0]), int(shp[1]), int(shp[2])


def _parse_proj_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3:
        raise ValueError(f"--proj-shape must be like views,h,w (e.g. 60,128,128), got: {spec}")
    if min(shp) <= 0:
        raise ValueError(f"--proj-shape values must be >0, got: {spec}")
    return int(shp[0]), int(shp[1]), int(shp[2])


def _parse_proj_dtype(spec: str) -> np.dtype:
    key = str(spec).strip().lower()
    m = {"int16": np.int16, "uint16": np.uint16, "float32": np.float32}
    if key not in m:
        raise ValueError(f"--proj-dtype must be one of {sorted(m.keys())}, got: {spec}")
    return np.dtype(m[key])


def _resolve_osem_resources(
    *,
    patient_dir: Path,
    osem_dir: str,
    par_file: str | None,
    atten_file: str | None,
    orbit_file: str | None,
) -> tuple[Path, Path, Path, Path]:
    par_path = Path(par_file) if par_file else None
    if par_path is None:
        cands = list(Path(patient_dir).glob("*ParamFile*.par")) + list(Path(patient_dir).glob("*.par"))
        cands = [p for p in cands if p.is_file()]
        if not cands:
            raise FileNotFoundError(f"No par file found under: {patient_dir}")
        par_path = cands[0]

    atten_path = Path(atten_file) if atten_file else None
    if atten_path is None:
        cands = list(Path(patient_dir).glob("*PostAtten*.dat")) + list(Path(patient_dir).glob("*Atten*.dat"))
        cands = [p for p in cands if p.is_file()]
        if not cands:
            raise FileNotFoundError(f"No atten file found under: {patient_dir}")
        atten_path = cands[0]

    osem_path = Path(str(osem_dir))
    orbit_path = Path(orbit_file) if orbit_file else (osem_path / "orbit.orb")
    if not orbit_path.exists():
        raise FileNotFoundError(f"orbit file not found: {orbit_path}")
    return Path(par_path), Path(atten_path), osem_path, Path(orbit_path)


def _run_or_load_osem(
    *,
    proj_file: Path,
    patient_name: str,
    par_file: Path,
    orbit_file: Path,
    atten_file: Path,
    output_name: str,
    iterations: int,
    osem_dir: Path,
    final_output_dir: Path,
    output_filename: str,
    timeout_sec: int,
) -> Path:
    from prime.pipeline.osem import ensure_osem_reconstruction

    return Path(
        ensure_osem_reconstruction(
            proj_file=Path(proj_file),
            patient_name=str(patient_name),
            par_file=Path(par_file),
            orbit_file=Path(orbit_file),
            atten_file=Path(atten_file),
            output_name=str(output_name),
            iterations=int(iterations),
            prj_data_type=1,
            osem_dir=Path(osem_dir),
            final_output_dir=Path(final_output_dir),
            output_filename=str(output_filename),
            timeout_sec=int(timeout_sec),
            overwrite=False,
        )
    )


def _export_projection_rows_dat(rows: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        lab = str(r["label"])
        save_projection_f32(out_dir / f"{lab}_orig_f32.dat", r["orig"])
        save_projection_f32(out_dir / f"{lab}_bm3d_f32.dat", r["bm3d"])
        save_projection_f32(out_dir / f"{lab}_ema_b_f32.dat", r["ema_b"])
        save_projection_f32(out_dir / f"{lab}_ema_a_f32.dat", r["ema_a"])


def _sec_from_label(label: str, k: float) -> float:
    return 20.0 if label == "20s" else 20.0 / max(float(k), 1e-12)


def _build_rows_from_exported_dat(
    *,
    dat_dir: Path,
    thin_factors: list[float],
    proj_shape: tuple[int, int, int],
    vmax_percentile: float,
    two_model: bool,
) -> tuple[list[dict], float]:
    if not dat_dir.exists():
        raise FileNotFoundError(f"--load-proj-dat-dir not found: {dat_dir}")

    def _load_dat(name: str) -> np.ndarray:
        p = dat_dir / name
        if not p.exists():
            raise FileNotFoundError(f"missing dat: {p}")
        return load_projection_f32(
            p,
            views=int(proj_shape[0]),
            h=int(proj_shape[1]),
            w=int(proj_shape[2]),
        )

    def _load_dat_optional(name: str) -> np.ndarray | None:
        p = dat_dir / name
        if not p.exists():
            return None
        return load_projection_f32(
            p,
            views=int(proj_shape[0]),
            h=int(proj_shape[1]),
            w=int(proj_shape[2]),
        )

    rows: list[dict] = []
    # 20s
    o20 = _load_dat("20s_orig_f32.dat")
    b20 = _load_dat("20s_bm3d_f32.dat")
    ea20 = _load_dat("20s_ema_a_f32.dat")
    eb20 = _load_dat_optional("20s_ema_b_f32.dat") if bool(two_model) else None
    if eb20 is None:
        eb20 = ea20
    rows.append(
        dict(
            label="20s",
            k=1.0,
            sec=_sec_from_label("20s", 1.0),
            orig=o20,
            bm3d=b20,
            ema_a=ea20,
            ema_b=eb20,
            res_a=(_anscombe_forward(o20) - _anscombe_forward(ea20)).astype(np.float32, copy=False),
            res_b=(_anscombe_forward(o20) - _anscombe_forward(eb20)).astype(np.float32, copy=False),
        )
    )

    for k in thin_factors:
        lab = f"x{k:g}"
        o = _load_dat(f"{lab}_orig_f32.dat")
        b = _load_dat(f"{lab}_bm3d_f32.dat")
        ea = _load_dat(f"{lab}_ema_a_f32.dat")
        eb = _load_dat_optional(f"{lab}_ema_b_f32.dat") if bool(two_model) else None
        if eb is None:
            eb = ea
        rows.append(
            dict(
                label=lab,
                k=float(k),
                sec=_sec_from_label(lab, float(k)),
                orig=o,
                bm3d=b,
                ema_a=ea,
                ema_b=eb,
                res_a=(_anscombe_forward(o) - _anscombe_forward(ea)).astype(np.float32, copy=False),
                res_b=(_anscombe_forward(o) - _anscombe_forward(eb)).astype(np.float32, copy=False),
            )
        )

    rows.sort(key=lambda r: float(r["sec"]), reverse=True)
    y20i = _counts_int(rows[0]["orig"])
    c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
    if c20 <= 0:
        c20 = 1.0
    for r in rows:
        r["theory"] = c20 / max(float(r["k"]), 1e-12)
    vmax20 = _vmax_from_ref(y20i, percentile=float(vmax_percentile))
    return rows, vmax20


def _build_rows_from_model_inference(
    *,
    args,
    patient_dir: Path,
    lq_path: str,
    thin_factors: list[float],
    proj_shape: tuple[int, int, int],
    proj_dtype: np.dtype,
) -> tuple[list[dict], float, Path, Path, Path | None, Path | None]:
    from prime.pipeline.io import load_projection
    from prime.pipeline.experiment import find_config_for_experiment
    from prime.pipeline.inference import load_basicsr_net

    proj_u16 = load_projection(Path(lq_path), shape=proj_shape, dtype=proj_dtype).astype(np.float32, copy=False)
    y20i = _counts_int(proj_u16)
    c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
    if c20 <= 0:
        c20 = 1.0
    vmax20 = _vmax_from_ref(y20i, percentile=float(args.vmax_percentile))

    base_rows: list[tuple[str, float, np.ndarray]] = [("20s", 1.0, y20i.astype(np.float32, copy=False))]
    for k in thin_factors:
        lab = f"x{k:g}"
        try:
            xk = _load_cached_thin(
                patient_dir=patient_dir,
                lq_path=lq_path,
                label=lab,
                seed=int(args.seed),
                require_cached=bool(args.require_cached),
            )
        except FileNotFoundError:
            if not bool(args.allow_on_the_fly_cache):
                raise
            xk = _poisson_thin_binomial(y20i, float(k), int(args.seed))
            np.save(_seed_cache_dir(patient_dir, int(args.seed)) / f"{lab}_thin.npy", xk.astype(np.float32, copy=False))
        base_rows.append((lab, float(k), xk.astype(np.float32, copy=False)))

    base_rows.sort(key=lambda t: _sec_from_label(t[0], t[1]), reverse=True)

    two_model = bool(
        args.two_model or args.exp_b or args.ckpt_b or args.config_b or (args.iter_b is not None)
    )

    exp_a = Path(args.exp_a)
    exp_b = Path(args.exp_b) if args.exp_b else None
    cfg_a = Path(args.config_a) if args.config_a else find_config_for_experiment(exp_a)
    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else _pick_ckpt(exp_a, prefer_iter=args.iter_a)
    net_a = load_basicsr_net(cfg_a, ckpt_a, device=str(args.device), require_ema=True).net
    cfg_b: Path | None = None
    ckpt_b: Path | None = None
    net_b = None
    if bool(two_model):
        if args.config_b:
            cfg_b = Path(args.config_b)
        elif exp_b is not None:
            cfg_b = find_config_for_experiment(exp_b)
        else:
            raise ValueError("Two-model mode requires --exp-b or --config-b.")

        if args.ckpt_b:
            ckpt_b = Path(args.ckpt_b)
        elif exp_b is not None:
            ckpt_b = _pick_ckpt(exp_b, prefer_iter=args.iter_b)
        else:
            raise ValueError("Two-model mode requires --exp-b or --ckpt-b.")

        net_b = load_basicsr_net(cfg_b, ckpt_b, device=str(args.device), require_ema=True).net

    mv = float(args.max_value)
    rows: list[dict] = []
    for label, k, orig in base_rows:
        try:
            den_bm3d = _load_cached_bm3d(
                patient_dir=patient_dir,
                lq_path=lq_path,
                label=label,
                seed=int(args.seed),
                require_cached=bool(args.require_cached),
            )
        except FileNotFoundError:
            if not bool(args.allow_on_the_fly_cache):
                raise
            den_bm3d = _bm3d_anscombe_denoise(
                orig,
                sigma_psd=float(args.bm3d_sigma),
                num_workers=int(args.bm3d_workers),
            )
            np.save(_seed_cache_dir(patient_dir, int(args.seed)) / f"{label}_bm3d.npy", den_bm3d.astype(np.float32, copy=False))
        ema_a = _infer_denoised(net=net_a, proj=orig, max_value=mv, device=str(args.device))
        ema_b = (
            _infer_denoised(net=net_b, proj=orig, max_value=mv, device=str(args.device))
            if net_b is not None
            else ema_a
        )
        rows.append(
            dict(
                label=label,
                k=float(k),
                sec=_sec_from_label(label, k),
                orig=orig.astype(np.float32, copy=False),
                bm3d=den_bm3d.astype(np.float32, copy=False),
                ema_a=ema_a,
                ema_b=ema_b,
                res_a=(_anscombe_forward(orig) - _anscombe_forward(ema_a)).astype(np.float32, copy=False),
                res_b=(_anscombe_forward(orig) - _anscombe_forward(ema_b)).astype(np.float32, copy=False),
                theory=(c20 / max(float(k), 1e-12)),
            )
        )
    return rows, vmax20, cfg_a, ckpt_a, cfg_b, ckpt_b


def _prepare_rows_diff_and_counts(rows: list[dict], *, diff_percentile: float) -> float:
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
    dv_a = float(np.percentile(flat_a, float(diff_percentile)))
    dv_b = float(np.percentile(flat_b, float(diff_percentile)))
    if dv_a < 1e-6:
        dv_a = 1.0
    if dv_b < 1e-6:
        dv_b = 1.0
    return float(max(dv_a, dv_b))


def _build_projection_frames(
    *,
    rows: list[dict],
    label_a: str,
    label_b: str,
    two_model: bool,
    turns: int,
    vmax20: float,
    dv: float,
    use_log1p: bool,
    show_global_text: bool,
):
    from PIL import Image, ImageDraw  # type: ignore

    if not rows:
        return []
    n_views = int(rows[0]["orig"].shape[0])
    H = int(rows[0]["orig"].shape[1])
    W = int(rows[0]["orig"].shape[2])
    header_h = 22
    font = _load_font(12)
    if bool(two_model):
        col_labels = [
            "Original",
            "BM3D",
            str(label_b),
            str(label_a),
            "Original",
            "BM3D",
            str(label_b),
            str(label_a),
        ]
    else:
        col_labels = [
            "Original",
            "BM3D",
            str(label_a),
            "Original",
            "BM3D",
            str(label_a),
        ]
    n_cols = len(col_labels)

    def _g2rgb(u8: np.ndarray):
        return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")

    turns = int(max(1, int(turns)))
    frames = []
    for fi in range(n_views * turns):
        vi = int(fi % n_views)
        row_imgs = []
        for r in rows:
            k = float(r["k"])
            a = _norm_to_u8(r["orig"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            b = _norm_to_u8(r["bm3d"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            c = _norm_to_u8(r["ema_b"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            d = _norm_to_u8(r["ema_a"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
            e = _signed_to_bwr(r["diff_orig"][vi], vmax=dv)
            f = _signed_to_bwr(r["diff_bm3d"][vi], vmax=dv)
            g = _signed_to_bwr(r["diff_b"][vi], vmax=dv)
            h = _signed_to_bwr(r["diff_a"][vi], vmax=dv)

            row_canvas = Image.new("RGB", (W * n_cols, H), color=(0, 0, 0))
            row_canvas.paste(_g2rgb(a), (0 * W, 0))
            row_canvas.paste(_g2rgb(b), (1 * W, 0))
            if bool(two_model):
                row_canvas.paste(_g2rgb(c), (2 * W, 0))
                row_canvas.paste(_g2rgb(d), (3 * W, 0))
                row_canvas.paste(Image.fromarray(e, mode="RGB"), (4 * W, 0))
                row_canvas.paste(Image.fromarray(f, mode="RGB"), (5 * W, 0))
                row_canvas.paste(Image.fromarray(g, mode="RGB"), (6 * W, 0))
                row_canvas.paste(Image.fromarray(h, mode="RGB"), (7 * W, 0))
            else:
                row_canvas.paste(_g2rgb(d), (2 * W, 0))
                row_canvas.paste(Image.fromarray(e, mode="RGB"), (3 * W, 0))
                row_canvas.paste(Image.fromarray(f, mode="RGB"), (4 * W, 0))
                row_canvas.paste(Image.fromarray(h, mode="RGB"), (5 * W, 0))

            draw = ImageDraw.Draw(row_canvas)
            row_label = f"{r['sec']:.0f}s(x{k:g})" if str(r["label"]).startswith("x") else f"{r['sec']:.0f}s"
            draw.text((4, 4), f"{row_label} / {_fmt_w_int(float(r['c_orig']))}", fill=255, font=font)
            t = float(r["theory"])
            draw.text((1 * W + 4, 4), _fmt_counts_w_pct(float(r["c_bm3d"]), t), fill=255, font=font)
            if bool(two_model):
                draw.text((2 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema_b"]), t), fill=255, font=font)
                draw.text((3 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema_a"]), t), fill=255, font=font)
            else:
                draw.text((2 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema_a"]), t), fill=255, font=font)
            row_imgs.append(row_canvas)

        full = Image.new("RGB", (W * n_cols, header_h + H * len(row_imgs)), color=(0, 0, 0))
        draw = ImageDraw.Draw(full)
        for j, lab in enumerate(col_labels):
            draw.text((j * W + 4, 3), lab, fill=255, font=font)
        if bool(show_global_text):
            draw.text((W * n_cols - 170, 3), f"ans_vmax={dv:.3g}", fill=255, font=font)
        for i, im in enumerate(row_imgs):
            full.paste(im, (0, header_h + i * H))
        frames.append(full)
    return frames


def _projection_dat_dir_from_args(args: argparse.Namespace) -> Path | None:
    if args.load_proj_dat_dir:
        return Path(str(args.load_proj_dat_dir))
    if args.save_proj_dat_dir:
        return Path(str(args.save_proj_dat_dir))
    return None


def _parse_osem_iters_csv(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, required=True, help="patient/sample identifier (used in output filenames)")
    ap.add_argument("--val-sample-name", type=str, default=None, help="suffix name like BaiYukun_Proj4Filter_003 (used only for output filename)")
    ap.add_argument("--out", type=str, required=False, help="output mp4 path (projection MP4). Required unless you only run recon-only outputs.")
    ap.add_argument("--spect229-dir", type=str, default="datasets/SPECT229", help="legacy dataset root for SPECT229-style layout")
    ap.add_argument("--patient-dir", type=str, default=None, help="optional explicit patient directory (cache/par/atten auto-discovery root)")
    ap.add_argument("--input-proj", type=str, default=None, help="optional explicit projection .dat path (overrides legacy <patient>_Proj4Filter.dat lookup)")
    ap.add_argument("--proj-shape", type=str, default="60,128,128", help="projection shape as views,h,w")
    ap.add_argument("--proj-dtype", type=str, default="int16", choices=["int16", "uint16", "float32"], help="input projection dtype for --input-proj / legacy projection file")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thin-factors", type=str, default="2,3,4,5")
    ap.add_argument("--require-cached", dest="require_cached", action="store_true")
    ap.add_argument("--no-require-cached", dest="require_cached", action="store_false")
    ap.add_argument("--allow-on-the-fly-cache", action="store_true", help="If cache missing, compute thinning/BM3D on the fly and save into datasets cache")
    ap.add_argument("--bm3d-sigma", type=float, default=1.0, help="BM3D sigma in Anscombe domain (training uses ~1.0)")
    ap.add_argument("--bm3d-workers", type=int, default=0, help="BM3D parallel workers across views (0/1 disables multiprocessing)")
    ap.set_defaults(require_cached=True, allow_on_the_fly_cache=False)
    ap.add_argument("--max-value", type=float, default=150.0)
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--no-log1p", dest="log1p", action="store_false")
    ap.set_defaults(log1p=True)
    ap.add_argument("--show-global-text", action="store_true", help="Show global header text like ans_vmax/recon_vmax.")
    ap.add_argument("--no-global-text", dest="show_global_text", action="store_false", help="Hide global header text like ans_vmax/recon_vmax.")
    ap.set_defaults(show_global_text=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--turns", type=int, default=2, help="number of full view cycles to render")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--two-model",
        action="store_true",
        help="Enable explicit two-model comparison. By default, only model A is rendered.",
    )

    ap.add_argument("--exp-a", type=str, required=False, help="experiment dir for model A (e.g. experiments/converged_tomo_3d_nview)")
    ap.add_argument("--iter-a", type=int, default=None, help="optional checkpoint iter for model A (e.g. 13000)")
    ap.add_argument("--config-a", type=str, default=None)
    ap.add_argument("--ckpt-a", type=str, default=None)
    ap.add_argument("--label-a", type=str, default="60view3d(ema)")

    ap.add_argument("--exp-b", type=str, required=False, help="experiment dir for model B (e.g. experiments/converged_tomo_2d_nview)")
    ap.add_argument("--iter-b", type=int, default=None)
    ap.add_argument("--config-b", type=str, default=None)
    ap.add_argument("--ckpt-b", type=str, default=None)
    ap.add_argument("--label-b", type=str, default="patch64(ema)")

    ap.add_argument(
        "--load-proj-dat-dir",
        type=str,
        default=None,
        help="If set, skip inference/BM3D and load projections from this directory (expects {20s,x2,x3,x4,x5}_{orig,bm3d,ema_a}_f32.dat, and optional ema_b).",
    )
    ap.add_argument(
        "--save-proj-dat-dir",
        type=str,
        default=None,
        help="If set, write per-dose projection .dat files for orig/bm3d/modelA/(optional modelB) into this directory (float32, shape=proj-shape).",
    )
    ap.add_argument(
        "--diff-percentile",
        type=float,
        default=99.5,
        help="Percentile (of |Anscombe residual|) used to set shared diff vmax. Training uses 99.5; try 99.9 for less saturation.",
    )
    ap.add_argument(
        "--vmax-percentile",
        type=float,
        default=100.0,
        help="Percentile used to set display vmax for projection/reconstruction tiles (100=max). Use 99.5 for percentile vmax; applies to all MP4s.",
    )

    # ===== Stage2: reconstruction MP4 (from exported projections) =====
    ap.add_argument("--recon-mp4-out", type=str, default=None, help="If set, also reconstruct volumes from projection dats and render a reconstruction MP4.")
    ap.add_argument("--recon-dir", type=str, default=None, help="Cache dir for recon outputs (default: <out_dir>/reconstructions).")
    ap.add_argument("--osem-dir", type=str, default="osemreocnexe", help="Directory containing osemrecon.exe and orbit.orb")
    ap.add_argument("--par-file", type=str, default=None, help="Optional explicit ParamFile.par path (default: datasets/SPECT229/<patient>/*ParamFile*.par)")
    ap.add_argument("--atten-file", type=str, default=None, help="Optional explicit PostAtten.dat path (default: datasets/SPECT229/<patient>/*PostAtten*.dat)")
    ap.add_argument("--orbit-file", type=str, default=None, help="Optional explicit orbit file (default: <osem-dir>/orbit.orb)")
    ap.add_argument("--osem-iterations", type=int, default=10)
    ap.add_argument("--osem-timeout-sec", type=int, default=600)
    ap.add_argument("--recon-shape", type=str, default="128,128,128", help="Reconstruction volume shape, e.g. 128,128,128")
    ap.add_argument("--recon-colormap", type=str, default="hot")
    ap.add_argument("--recon-fps", type=float, default=None, help="FPS for recon MP4 (default: same as --fps)")
    ap.add_argument("--recon-crf", type=int, default=None, help="CRF for recon MP4 (default: same as --crf)")
    ap.add_argument("--recon-turns", type=int, default=None, help="Turns for recon MP4 (default: same as --turns)")

    # ===== Extra MP4 #1: 3 projections + 3 reconstructions (poisson recon auto-run) =====
    ap.add_argument(
        "--proj3recon-mp4-out",
        type=str,
        default=None,
        help="If set, render a 5-row x 6-col MP4: [orig proj, ema(60view3d) proj, poisson(ema) proj, orig recon, ema recon, poisson recon].",
    )
    ap.add_argument("--poisson-seed", type=int, default=None, help="Seed for Poisson sampling (default: seed+2026).")
    ap.add_argument("--poisson-tag", type=str, default="poi", help="Tag name for poisson projection/recon cache files.")

    # ===== Extra MP4 #2: OSEM iteration sweep on 20s (orig vs ema) =====
    ap.add_argument(
        "--osem-sweep-mp4-out",
        type=str,
        default=None,
        help="If set, render 2-row x N-col MP4 sweeping OSEM iterations on 20s: row1=orig, row2=ema. Uses a shared global_vmax across all tiles.",
    )
    ap.add_argument("--osem-sweep-iters", type=str, default="5,10,15,20,25,30", help="Comma-separated iterations list for sweep MP4.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    from PIL import Image, ImageDraw  # type: ignore

    if not args.out and not args.recon_mp4_out and not args.proj3recon_mp4_out and not args.osem_sweep_mp4_out:
        raise SystemExit("No outputs specified. Provide --out and/or --recon-mp4-out / --proj3recon-mp4-out / --osem-sweep-mp4-out.")

    two_model = bool(args.two_model or args.exp_b or args.ckpt_b or args.config_b or (args.iter_b is not None))
    if not args.load_proj_dat_dir and not args.exp_a:
        raise SystemExit("--exp-a is required unless --load-proj-dat-dir is provided.")
    if bool(two_model) and (not args.load_proj_dat_dir):
        has_b_from_exp = bool(args.exp_b)
        has_b_explicit = bool(args.config_b and args.ckpt_b)
        if not (has_b_from_exp or has_b_explicit):
            raise SystemExit("Two-model mode requires --exp-b or (--config-b and --ckpt-b).")

    proj_shape = _parse_proj_shape(str(args.proj_shape))
    proj_dtype = _parse_proj_dtype(str(args.proj_dtype))

    spect229_dir = Path(args.spect229_dir)
    legacy_patient_dir = spect229_dir / str(args.patient)
    if args.patient_dir:
        patient_dir = Path(str(args.patient_dir))
    elif legacy_patient_dir.exists():
        patient_dir = legacy_patient_dir
    elif args.input_proj:
        patient_dir = Path(str(args.input_proj)).resolve().parent
    else:
        patient_dir = legacy_patient_dir

    # Either:
    # - standard path: load 20s from dataset + thin cache / compute, then infer models
    # - fast path: load everything from previously exported projection .dat files (no inference)
    if args.input_proj:
        lq_path = str(Path(str(args.input_proj)).resolve())
    else:
        lq_path = str((patient_dir / f"{args.patient}_Proj4Filter.dat").resolve())
    thin_factors = _parse_thin_factors_csv(str(args.thin_factors))
    model_meta: tuple[Path, Path, Path, Path] | None = None
    if args.load_proj_dat_dir:
        rows, vmax20 = _build_rows_from_exported_dat(
            dat_dir=Path(str(args.load_proj_dat_dir)),
            thin_factors=thin_factors,
            proj_shape=proj_shape,
            vmax_percentile=float(args.vmax_percentile),
            two_model=bool(two_model),
        )
    else:
        if not Path(lq_path).exists():
            raise FileNotFoundError(f"lq projection not found: {lq_path}")
        rows, vmax20, cfg_a, ckpt_a, cfg_b, ckpt_b = _build_rows_from_model_inference(
            args=args,
            patient_dir=patient_dir,
            lq_path=lq_path,
            thin_factors=thin_factors,
            proj_shape=proj_shape,
            proj_dtype=proj_dtype,
        )
        model_meta = (cfg_a, ckpt_a, cfg_b, ckpt_b)

    use_log1p = bool(args.log1p)
    num_views = int(rows[0]["orig"].shape[0]) if rows else int(proj_shape[0])

    # Optional: export projections for reconstruction-domain experiments
    if args.save_proj_dat_dir:
        out_dir = Path(str(args.save_proj_dat_dir))
        _export_projection_rows_dat(rows, out_dir)
        print(f"[OK] wrote projection .dat files to: {out_dir}")
    dv = _prepare_rows_diff_and_counts(rows, diff_percentile=float(args.diff_percentile))
    frames = _build_projection_frames(
        rows=rows,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        two_model=bool(two_model),
        turns=int(args.turns),
        vmax20=float(vmax20),
        dv=float(dv),
        use_log1p=bool(use_log1p),
        show_global_text=bool(args.show_global_text),
    )

    if args.out:
        out = Path(args.out)
        write_mp4_frames(
            out,
            frames,
            fps=float(args.fps),
            crf=int(args.crf),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

        print(f"[OK] wrote mp4: {out}")
        if model_meta is not None:
            cfg_a, ckpt_a, cfg_b, ckpt_b = model_meta
            print(f"[OK] modelA ckpt: {ckpt_a}  config: {cfg_a}")
            if cfg_b is not None and ckpt_b is not None:
                print(f"[OK] modelB ckpt: {ckpt_b}  config: {cfg_b}")
        else:
            print(f"[OK] loaded projections from: {args.load_proj_dat_dir}")

    # ===== Stage2: reconstruction MP4 =====
    if args.recon_mp4_out:
        from prime.pipeline.viz import compute_mip, rotate_volume_around_x

        par_file, atten_file, osem_dir, orbit_file = _resolve_osem_resources(
            patient_dir=patient_dir,
            osem_dir=str(args.osem_dir),
            par_file=args.par_file,
            atten_file=args.atten_file,
            orbit_file=args.orbit_file,
        )

        # Recon cache dir
        recon_dir = Path(args.recon_dir) if args.recon_dir else (Path(args.recon_mp4_out).parent / "reconstructions")
        recon_dir.mkdir(parents=True, exist_ok=True)

        dz, dy, dx = _parse_recon_shape(str(args.recon_shape))

        # Helper to load recon volume
        def _load_recon_dat(p: Path) -> np.ndarray:
            return load_recon_f32(p, shape=(dz, dy, dx))

        # Determine projection dat dir for running OSEM
        proj_dat_dir = _projection_dat_dir_from_args(args)
        if proj_dat_dir is None:
            # If no dat dir was specified, write projections to recon_dir/projections for OSEM input.
            proj_dat_dir = recon_dir / "projections"
            _export_projection_rows_dat(rows, proj_dat_dir)

        # Run / load reconstructions
        methods = [
            ("orig", "orig_f32", "Original"),
            ("bm3d", "bm3d_f32", "BM3D"),
            ("ema_b", "ema_b_f32", str(args.label_b)),
            ("ema_a", "ema_a_f32", str(args.label_a)),
        ]

        recon_rows = []
        for r in rows:
            lab = str(r["label"])
            k = float(r["k"])
            rec = {"label": lab, "k": k, "sec": float(r["sec"])}
            for key, tag, _ in methods:
                proj_path = proj_dat_dir / f"{lab}_{tag}.dat"
                if not proj_path.exists():
                    raise FileNotFoundError(f"Missing projection dat for recon: {proj_path}")
                out_name = f"{lab}_{key}"
                out_fn = f"{args.patient}_{lab}_{key}_OSEMIter{int(args.osem_iterations)}.dat"
                out_path = _run_or_load_osem(
                    proj_file=proj_path,
                    patient_name=str(args.patient),
                    par_file=Path(par_file),
                    orbit_file=Path(orbit_file),
                    atten_file=Path(atten_file),
                    output_name=out_name,
                    iterations=int(args.osem_iterations),
                    osem_dir=Path(osem_dir),
                    final_output_dir=recon_dir,
                    output_filename=out_fn,
                    timeout_sec=int(args.osem_timeout_sec),
                )
                rec[key] = _load_recon_dat(out_path)
            recon_rows.append(rec)

        # Sort like rows
        recon_rows.sort(key=lambda rr: float(rr["sec"]), reverse=True)

        # Determine recon vmax based on 20s volumes (max/percentile across methods)
        row20 = next((rr for rr in recon_rows if rr["label"] == "20s"), recon_rows[0])
        vmax_recon = _vmax_from_ref(
            np.concatenate(
                [
                    np.clip(row20["orig"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                    np.clip(row20["bm3d"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                    np.clip(row20["ema_b"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                    np.clip(row20["ema_a"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                ],
                axis=0,
            ),
            percentile=float(args.vmax_percentile),
        )

        # Render recon MP4: 5 rows x 4 cols, rotating MIP
        sample_mip = np.flipud(compute_mip(rotate_volume_around_x(row20["orig"], 90.0), axis=2))
        H = int(sample_mip.shape[0])
        W = int(sample_mip.shape[1])
        header_h = 22
        font = _load_font(12)
        col_labels = ["Original Recon", "BM3D Recon", f"{str(args.label_b)} Recon", f"{str(args.label_a)} Recon"]

        turns2 = int(args.recon_turns) if args.recon_turns is not None else int(args.turns)
        if turns2 < 1:
            turns2 = 1
        fps2 = float(args.recon_fps) if args.recon_fps is not None else float(args.fps)
        crf2 = int(args.recon_crf) if args.recon_crf is not None else int(args.crf)

        frames2 = []
        for fi in range(num_views * turns2):
            vi = int(fi % num_views)
            recon_angle = 90.0 + float(vi) * (360.0 / float(num_views))
            row_imgs = []
            for rr in recon_rows:
                k = float(rr["k"])
                # rotate -> mip -> flipud, then scale by k to 20s-equivalent domain
                mip_o = np.flipud(compute_mip(rotate_volume_around_x(rr["orig"], recon_angle), axis=2)) * k
                mip_b = np.flipud(compute_mip(rotate_volume_around_x(rr["bm3d"], recon_angle), axis=2)) * k
                mip_pb = np.flipud(compute_mip(rotate_volume_around_x(rr["ema_b"], recon_angle), axis=2)) * k
                mip_pa = np.flipud(compute_mip(rotate_volume_around_x(rr["ema_a"], recon_angle), axis=2)) * k

                uo = _norm_to_u8_cmap(mip_o, vmax=vmax_recon, cmap_name=str(args.recon_colormap))
                ub = _norm_to_u8_cmap(mip_b, vmax=vmax_recon, cmap_name=str(args.recon_colormap))
                upb = _norm_to_u8_cmap(mip_pb, vmax=vmax_recon, cmap_name=str(args.recon_colormap))
                upa = _norm_to_u8_cmap(mip_pa, vmax=vmax_recon, cmap_name=str(args.recon_colormap))

                row_canvas = Image.new("RGB", (W * 4, H), color=(0, 0, 0))
                row_canvas.paste(Image.fromarray(uo, mode="RGB"), (0 * W, 0))
                row_canvas.paste(Image.fromarray(ub, mode="RGB"), (1 * W, 0))
                row_canvas.paste(Image.fromarray(upb, mode="RGB"), (2 * W, 0))
                row_canvas.paste(Image.fromarray(upa, mode="RGB"), (3 * W, 0))

                # counts overlay (recon total activity)
                c0 = float(np.sum(np.clip(rr["orig"].astype(np.float64, copy=False), 0.0, None)))
                cb = float(np.sum(np.clip(rr["bm3d"].astype(np.float64, copy=False), 0.0, None)))
                cB = float(np.sum(np.clip(rr["ema_b"].astype(np.float64, copy=False), 0.0, None)))
                cA = float(np.sum(np.clip(rr["ema_a"].astype(np.float64, copy=False), 0.0, None)))

                draw = ImageDraw.Draw(row_canvas)
                # row label: show seconds only (20s/10s/6.7s/5s/4s)
                if rr["label"] == "20s":
                    row_label = "20s"
                else:
                    row_label = f"{(20.0 / float(rr['k'])):.1f}s"
                draw.text((4, 4), f"{row_label} / {_fmt_w_int(c0)}", fill=255, font=font)
                draw.text((1 * W + 4, 4), _fmt_counts_w_pct_base(cb, c0), fill=255, font=font)
                draw.text((2 * W + 4, 4), _fmt_counts_w_pct_base(cB, c0), fill=255, font=font)
                draw.text((3 * W + 4, 4), _fmt_counts_w_pct_base(cA, c0), fill=255, font=font)

                row_imgs.append(row_canvas)

            full2 = Image.new("RGB", (W * 4, header_h + H * len(row_imgs)), color=(0, 0, 0))
            draw = ImageDraw.Draw(full2)
            for j, lab in enumerate(col_labels):
                draw.text((j * W + 4, 3), lab, fill=255, font=font)
            if bool(args.show_global_text):
                draw.text((W * 4 - 220, 3), f"recon_vmax={vmax_recon:.3g}", fill=255, font=font)
            for i, im in enumerate(row_imgs):
                full2.paste(im, (0, header_h + i * H))
            frames2.append(full2)

        out2 = Path(str(args.recon_mp4_out))
        write_mp4_frames(
            out2,
            frames2,
            fps=float(fps2),
            crf=int(crf2),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

        print(f"[OK] wrote recon mp4: {out2}")

    # ===== Extra MP4 #1: 3 projections + 3 reconstructions (poisson recon auto-run) =====
    if args.proj3recon_mp4_out:
        from prime.pipeline.viz import compute_mip, rotate_volume_around_x

        proj_dat_dir = _projection_dat_dir_from_args(args)
        if proj_dat_dir is None:
            raise RuntimeError("--proj3recon-mp4-out requires --load-proj-dat-dir (or --save-proj-dat-dir).")
        proj_dat_dir = Path(proj_dat_dir)

        par_file, atten_file, osem_dir, orbit_file = _resolve_osem_resources(
            patient_dir=patient_dir,
            osem_dir=str(args.osem_dir),
            par_file=args.par_file,
            atten_file=args.atten_file,
            orbit_file=args.orbit_file,
        )

        recon_dir = Path(args.recon_dir) if args.recon_dir else (Path(args.proj3recon_mp4_out).parent / "reconstructions")
        recon_dir.mkdir(parents=True, exist_ok=True)

        dz, dy, dx = _parse_recon_shape(str(args.recon_shape))

        def _load_recon_dat(p: Path) -> np.ndarray:
            return load_recon_f32(p, shape=(dz, dy, dx))

        # Poisson projections (create if missing)
        poi_seed = int(args.poisson_seed) if args.poisson_seed is not None else int(args.seed) + 2026
        rng = np.random.default_rng(poi_seed)
        poi_tag = str(args.poisson_tag)

        def _load_proj_dat(p: Path) -> np.ndarray:
            return load_projection_f32(
                p,
                views=int(proj_shape[0]),
                h=int(proj_shape[1]),
                w=int(proj_shape[2]),
            )

        for r in rows:
            lab = str(r["label"])
            poi_p = proj_dat_dir / f"{lab}_{poi_tag}_f32.dat"
            if not poi_p.exists():
                # Poisson sample from EMA (use ema_a as 60view3d ema)
                ema_p = proj_dat_dir / f"{lab}_ema_a_f32.dat"
                ema = _load_proj_dat(ema_p)
                poi = rng.poisson(lam=np.clip(ema, 0.0, None)).astype(np.float32)
                save_projection_f32(poi_p, poi)

        # Load recons into table
        recon_rows = []
        for r in rows:
            lab = str(r["label"])
            k = float(r["k"])
            orig_p = proj_dat_dir / f"{lab}_orig_f32.dat"
            ema_p = proj_dat_dir / f"{lab}_ema_a_f32.dat"
            poi_p = proj_dat_dir / f"{lab}_{poi_tag}_f32.dat"
            if not (orig_p.exists() and ema_p.exists() and poi_p.exists()):
                raise FileNotFoundError(f"Missing projection dats for {lab} under {proj_dat_dir}")

            o_path = _run_or_load_osem(
                proj_file=orig_p,
                patient_name=str(args.patient),
                par_file=Path(par_file),
                orbit_file=Path(orbit_file),
                atten_file=Path(atten_file),
                output_name=f"{lab}_orig",
                iterations=int(args.osem_iterations),
                osem_dir=Path(osem_dir),
                final_output_dir=recon_dir,
                output_filename=f"{args.patient}_{lab}_orig_OSEMIter{int(args.osem_iterations)}.dat",
                timeout_sec=int(args.osem_timeout_sec),
            )
            e_path = _run_or_load_osem(
                proj_file=ema_p,
                patient_name=str(args.patient),
                par_file=Path(par_file),
                orbit_file=Path(orbit_file),
                atten_file=Path(atten_file),
                output_name=f"{lab}_ema",
                iterations=int(args.osem_iterations),
                osem_dir=Path(osem_dir),
                final_output_dir=recon_dir,
                output_filename=f"{args.patient}_{lab}_ema_OSEMIter{int(args.osem_iterations)}.dat",
                timeout_sec=int(args.osem_timeout_sec),
            )
            p_path = _run_or_load_osem(
                proj_file=poi_p,
                patient_name=str(args.patient),
                par_file=Path(par_file),
                orbit_file=Path(orbit_file),
                atten_file=Path(atten_file),
                output_name=f"{lab}_{poi_tag}",
                iterations=int(args.osem_iterations),
                osem_dir=Path(osem_dir),
                final_output_dir=recon_dir,
                output_filename=f"{args.patient}_{lab}_{poi_tag}_OSEMIter{int(args.osem_iterations)}.dat",
                timeout_sec=int(args.osem_timeout_sec),
            )

            recon_rows.append(
                dict(
                    label=lab,
                    k=k,
                    sec=float(r["sec"]),
                    orig_proj=_load_proj_dat(orig_p),
                    ema_proj=_load_proj_dat(ema_p),
                    poi_proj=_load_proj_dat(poi_p),
                    orig_recon=_load_recon_dat(o_path),
                    ema_recon=_load_recon_dat(e_path),
                    poi_recon=_load_recon_dat(p_path),
                )
            )

        recon_rows.sort(key=lambda rr: float(rr["sec"]), reverse=True)

        # Projection vmax20 based on 20s integer counts (same as training; optionally percentile)
        row20p = next((rr for rr in recon_rows if rr["label"] == "20s"), recon_rows[0])
        vmax20 = _vmax_from_ref(_counts_int(row20p["orig_proj"]), percentile=float(args.vmax_percentile))

        # Recon vmax based on 20s volumes (max/percentile across 3 methods)
        row20r = next((rr for rr in recon_rows if rr["label"] == "20s"), recon_rows[0])
        vmax_recon = _vmax_from_ref(
            np.concatenate(
                [
                    np.clip(row20r["orig_recon"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                    np.clip(row20r["ema_recon"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                    np.clip(row20r["poi_recon"].astype(np.float32, copy=False), 0.0, None).reshape(-1),
                ],
                axis=0,
            ),
            percentile=float(args.vmax_percentile),
        )

        # Render MP4 (5 rows x 6 cols), auto-adapt tile size to projection/recon dimensions
        proj_h = int(recon_rows[0]["orig_proj"][0].shape[0])
        proj_w = int(recon_rows[0]["orig_proj"][0].shape[1])
        sample_mip = np.flipud(compute_mip(rotate_volume_around_x(row20r["orig_recon"], 90.0), axis=2))
        recon_h = int(sample_mip.shape[0])
        recon_w = int(sample_mip.shape[1])
        H = max(proj_h, recon_h)
        W = max(proj_w, recon_w)
        header_h = 22
        font = _load_font(12)
        col_labels = ["Orig Proj", "EMA Proj", "Poisson Proj", "Orig Recon", "EMA Recon", "Poisson Recon"]

        turns3 = int(args.turns)
        if turns3 < 1:
            turns3 = 1
        out3 = Path(str(args.proj3recon_mp4_out))
        out3.parent.mkdir(parents=True, exist_ok=True)

        frames3 = []
        for fi in range(num_views * turns3):
            vi = int(fi % num_views)
            recon_angle = 90.0 + float(vi) * (360.0 / float(num_views))
            row_imgs = []
            for rr in recon_rows:
                k = float(rr["k"])
                # projections in 20s-equivalent domain
                a = _norm_to_u8(rr["orig_proj"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                b = _norm_to_u8(rr["ema_proj"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                c = _norm_to_u8(rr["poi_proj"][vi] * k, vmax=vmax20, use_log1p=use_log1p)

                # recon MIPs in 20s-equivalent domain
                mip_o = np.flipud(compute_mip(rotate_volume_around_x(rr["orig_recon"], recon_angle), axis=2)) * k
                mip_e = np.flipud(compute_mip(rotate_volume_around_x(rr["ema_recon"], recon_angle), axis=2)) * k
                mip_p = np.flipud(compute_mip(rotate_volume_around_x(rr["poi_recon"], recon_angle), axis=2)) * k

                ro = _norm_to_u8_cmap(mip_o, vmax=vmax_recon, cmap_name=str(args.recon_colormap))
                re = _norm_to_u8_cmap(mip_e, vmax=vmax_recon, cmap_name=str(args.recon_colormap))
                rp = _norm_to_u8_cmap(mip_p, vmax=vmax_recon, cmap_name=str(args.recon_colormap))

                row_canvas = Image.new("RGB", (W * 6, H), color=(0, 0, 0))
                tiles = [
                    np.repeat(a[:, :, None], 3, axis=2),
                    np.repeat(b[:, :, None], 3, axis=2),
                    np.repeat(c[:, :, None], 3, axis=2),
                    ro,
                    re,
                    rp,
                ]
                for j, tile in enumerate(tiles):
                    th, tw = int(tile.shape[0]), int(tile.shape[1])
                    ox = j * W + max(0, (W - tw) // 2)
                    oy = max(0, (H - th) // 2)
                    row_canvas.paste(Image.fromarray(tile.astype(np.uint8, copy=False), mode="RGB"), (ox, oy))

                # counts overlays:
                # - projection totals vs original projection per-row
                # - recon totals (activity) vs original recon per-row
                c0 = float(np.sum(_counts_int(rr["orig_proj"]).astype(np.float64, copy=False)))
                c1 = float(np.sum(rr["ema_proj"].astype(np.float64, copy=False)))
                c2 = float(np.sum(_counts_int(rr["poi_proj"]).astype(np.float64, copy=False)))
                a0 = float(np.sum(np.clip(rr["orig_recon"].astype(np.float64, copy=False), 0.0, None)))
                a1 = float(np.sum(np.clip(rr["ema_recon"].astype(np.float64, copy=False), 0.0, None)))
                a2 = float(np.sum(np.clip(rr["poi_recon"].astype(np.float64, copy=False), 0.0, None)))
                draw = ImageDraw.Draw(row_canvas)
                # seconds label
                if rr["label"] == "20s":
                    row_label = "20s"
                else:
                    row_label = f"{(20.0 / float(rr['k'])):.1f}s"
                draw.text((4, 4), f"{row_label} / {_fmt_w_int(c0)}", fill=255, font=font)
                draw.text((1 * W + 4, 4), _fmt_counts_w_pct_base(c1, c0), fill=255, font=font)
                draw.text((2 * W + 4, 4), _fmt_counts_w_pct_base(c2, c0), fill=255, font=font)
                # recon activity overlays (same format, but base = orig recon activity)
                draw.text((3 * W + 4, 4), f"{_fmt_w_int(a0)}", fill=255, font=font)
                draw.text((4 * W + 4, 4), _fmt_counts_w_pct_base(a1, a0), fill=255, font=font)
                draw.text((5 * W + 4, 4), _fmt_counts_w_pct_base(a2, a0), fill=255, font=font)

                row_imgs.append(row_canvas)

            full3 = Image.new("RGB", (W * 6, header_h + H * len(row_imgs)), color=(0, 0, 0))
            draw = ImageDraw.Draw(full3)
            for j, lab in enumerate(col_labels):
                draw.text((j * W + 4, 3), lab, fill=255, font=font)
            if bool(args.show_global_text):
                # Right-align global info to avoid overlapping with the rightmost column headers.
                info = f"proj_vmax={vmax20:.3g}  recon_vmax={vmax_recon:.3g}"
                try:
                    x0, y0, x1, y1 = draw.textbbox((0, 0), info, font=font)
                    tw = int(x1 - x0)
                except Exception:
                    # Fallback: rough estimate (works fine for DejaVuSans)
                    tw = int(len(info) * 6)
                draw.text((W * 6 - tw - 4, 3), info, fill=255, font=font)
            for i, im in enumerate(row_imgs):
                full3.paste(im, (0, header_h + i * H))
            frames3.append(full3)

        write_mp4_frames(
            out3,
            frames3,
            fps=float(args.fps),
            crf=int(args.crf),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

        print(f"[OK] wrote proj3recon mp4: {out3}")

    # ===== Extra MP4 #2: OSEM iteration sweep on 20s (orig vs ema) =====
    if args.osem_sweep_mp4_out:
        from prime.pipeline.viz import compute_mip, rotate_volume_around_x

        proj_dat_dir = _projection_dat_dir_from_args(args)
        if proj_dat_dir is None:
            raise RuntimeError("--osem-sweep-mp4-out requires --load-proj-dat-dir (or --save-proj-dat-dir).")
        proj_dat_dir = Path(proj_dat_dir)

        par_file, atten_file, osem_dir, orbit_file = _resolve_osem_resources(
            patient_dir=patient_dir,
            osem_dir=str(args.osem_dir),
            par_file=args.par_file,
            atten_file=args.atten_file,
            orbit_file=args.orbit_file,
        )

        recon_dir = Path(args.recon_dir) if args.recon_dir else (Path(args.osem_sweep_mp4_out).parent / "reconstructions_sweep")
        recon_dir.mkdir(parents=True, exist_ok=True)

        dz, dy, dx = _parse_recon_shape(str(args.recon_shape))

        def _load_recon_dat(p: Path) -> np.ndarray:
            return load_recon_f32(p, shape=(dz, dy, dx))

        # Parse iter list
        iters = _parse_osem_iters_csv(str(args.osem_sweep_iters))
        if not iters:
            raise ValueError("--osem-sweep-iters is empty")

        # Load 20s projections
        orig_p = proj_dat_dir / "20s_orig_f32.dat"
        ema_p = proj_dat_dir / "20s_ema_a_f32.dat"
        if not (orig_p.exists() and ema_p.exists()):
            raise FileNotFoundError(f"Missing 20s projections under {proj_dat_dir}")

        # Run recons for both rows across iterations
        vols_orig = []
        vols_ema = []
        for it in iters:
            fn_o = f"{args.patient}_20s_orig_iter{int(it)}.dat"
            fn_e = f"{args.patient}_20s_ema_iter{int(it)}.dat"
            p_o = _run_or_load_osem(
                proj_file=orig_p,
                patient_name=str(args.patient),
                par_file=Path(par_file),
                orbit_file=Path(orbit_file),
                atten_file=Path(atten_file),
                output_name=f"20s_orig_iter{int(it)}",
                iterations=int(it),
                osem_dir=Path(osem_dir),
                final_output_dir=recon_dir,
                output_filename=fn_o,
                timeout_sec=int(args.osem_timeout_sec),
            )
            p_e = _run_or_load_osem(
                proj_file=ema_p,
                patient_name=str(args.patient),
                par_file=Path(par_file),
                orbit_file=Path(orbit_file),
                atten_file=Path(atten_file),
                output_name=f"20s_ema_iter{int(it)}",
                iterations=int(it),
                osem_dir=Path(osem_dir),
                final_output_dir=recon_dir,
                output_filename=fn_e,
                timeout_sec=int(args.osem_timeout_sec),
            )
            vols_orig.append(_load_recon_dat(p_o))
            vols_ema.append(_load_recon_dat(p_e))

        # Global vmax across all 12 vols (max/percentile)
        vmax_sweep = _vmax_from_ref(
            np.concatenate(
                [
                    np.clip(v.astype(np.float32, copy=False), 0.0, None).reshape(-1)
                    for v in (vols_orig + vols_ema)
                ],
                axis=0,
            ),
            percentile=float(args.vmax_percentile),
        )

        # Render MP4: 2 rows x len(iters) cols
        sample_mip = np.flipud(compute_mip(rotate_volume_around_x(vols_orig[0], 90.0), axis=2))
        H = int(sample_mip.shape[0])
        W = int(sample_mip.shape[1])
        header_h = 22
        font = _load_font(12)
        col_labels = [f"Iter{it}" for it in iters]
        turns2 = int(args.recon_turns) if args.recon_turns is not None else int(args.turns)
        if turns2 < 1:
            turns2 = 1
        fps2 = float(args.recon_fps) if args.recon_fps is not None else float(args.fps)
        crf2 = int(args.recon_crf) if args.recon_crf is not None else int(args.crf)

        outp = Path(str(args.osem_sweep_mp4_out))
        outp.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for fi in range(num_views * turns2):
            vi = int(fi % num_views)
            recon_angle = 90.0 + float(vi) * (360.0 / float(num_views))
            rows_img = []
            for row_name, vols in [("Orig(20s)", vols_orig), (f"EMA({str(args.label_a)})", vols_ema)]:
                row_canvas = Image.new("RGB", (W * len(iters), H), color=(0, 0, 0))
                for j, vol in enumerate(vols):
                    mip = np.flipud(compute_mip(rotate_volume_around_x(vol, recon_angle), axis=2))
                    rgb = _norm_to_u8_cmap(mip, vmax=vmax_sweep, cmap_name=str(args.recon_colormap))
                    row_canvas.paste(Image.fromarray(rgb, mode="RGB"), (j * W, 0))
                # row label on first tile
                draw = ImageDraw.Draw(row_canvas)
                draw.text((4, 4), row_name, fill=255, font=font)
                rows_img.append(row_canvas)

            full = Image.new("RGB", (W * len(iters), header_h + H * 2), color=(0, 0, 0))
            draw = ImageDraw.Draw(full)
            for j, lab in enumerate(col_labels):
                draw.text((j * W + 4, 3), lab, fill=255, font=font)
            if bool(args.show_global_text):
                draw.text((W * len(iters) - 220, 3), f"recon_vmax={vmax_sweep:.3g}", fill=255, font=font)
            full.paste(rows_img[0], (0, header_h))
            full.paste(rows_img[1], (0, header_h + H))
            frames.append(full)

        write_mp4_frames(
            outp,
            frames,
            fps=float(fps2),
            crf=int(crf2),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

        print(f"[OK] wrote osem sweep mp4: {outp}")


if __name__ == "__main__":
    main()
