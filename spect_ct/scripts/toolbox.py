#!/usr/bin/env python3
from __future__ import annotations

"""
SPECT229 Toolbox (interactive + CLI)

Goals:
- Select experiment dir -> default latest ckpt -> select patient indices -> run denoise or denoise+recon.
- Batch analyze multiple experiments: denoise (fast) for many, recon (slow) for selected few, plus metrics.

Design:
- If run with no args: interactive menu (questionary if available, else input()).
- If args provided: argparse subcommands (list/run/batch).
"""

import argparse
import fnmatch
import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Allow running as a script from repo root without setting PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spect_ct.pipeline.experiment import (
    find_config_for_experiment,
    list_experiments,
    parse_patient_indices,
    pick_latest_ckpt,
    resolve_patients,
)
from spect_ct.pipeline.io import load_projection_i16, load_recon_f32, save_projection_i16_round_clip
from spect_ct.pipeline.metrics import PoissonBinStats, lpips_views_mean, psnr_ssim_recon
from spect_ct.pipeline.original_recon_cache import ensure_original_recon, resolve_patient_paths


def _maybe_questionary():
    try:
        import questionary  # type: ignore
        return questionary
    except Exception:
        return None


def _load_toolbox_defaults() -> dict:
    """Load interactive defaults from `spect_ct/scripts/toolbox_defaults.json` if present."""
    p = Path(__file__).with_name("toolbox_defaults.json")
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _apply_torch_backend_settings(*, disable_cudnn: bool) -> None:
    """Apply torch backend flags early to avoid CuDNN-related instability.

    Note: We intentionally import torch lazily so `list/--help` still works in minimal envs.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return
    if bool(disable_cudnn):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    else:
        # Default BasicSR behavior: allow cudnn autotune for speed.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def _read_yaml_max_value(config_path: Path, default: float = 150.0) -> float:
    try:
        import yaml
    except Exception:
        return float(default)
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        mv = (((cfg or {}).get("datasets", {}) or {}).get("train", {}) or {}).get("max_value", None)
        if mv is None:
            mv = (((cfg or {}).get("datasets", {}) or {}).get("val", {}) or {}).get("max_value", None)
        if mv is None:
            mv = (cfg or {}).get("max_value", None)
        return float(mv) if mv is not None else float(default)
    except Exception:
        return float(default)


def _default_orbit_file() -> Path:
    return Path(__file__).resolve().parents[1] / "osemreocnexe" / "orbit.orb"


def _default_results_root() -> Path:
    return Path("spect_ct") / "results"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _input_yes_no(prompt: str, *, default: bool = True) -> bool:
    """Yes/No prompt for input() fallback.

    - Press Enter to accept default (so default=True means Enter => Yes).
    - Accept common falsy tokens for No: n/no/false/0.
    """
    suffix = " [Y/n, 回车=Y]: " if default else " [y/N, 回车=N]: "
    s = input(prompt + suffix).strip().lower()
    if s == "":
        return bool(default)
    if s in {"n", "no", "false", "0"}:
        return False
    if s in {"y", "yes", "true", "1"}:
        return True
    # Any other input: be forgiving and fall back to default.
    return bool(default)


def _save_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if len(rows) == 0:
        return
    _ensure_dir(path.parent)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_poisson_calibration(out_png: Path, stats: dict[str, np.ndarray], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    valid = stats["valid"].astype(bool)
    mean_lam = stats["mean_lambda"][valid]
    var_r = stats["var_residual"][valid]
    ratio = stats["ratio_var_over_mean"][valid]
    if mean_lam.size == 0:
        return
    _ensure_dir(out_png.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    ax0, ax1 = axes
    ax0.plot(mean_lam, var_r, ".", markersize=2)
    ax0.plot(mean_lam, mean_lam, "--", linewidth=1)
    ax0.set_xlabel("mean(lambda_hat)")
    ax0.set_ylabel("var(y - lambda_hat)")
    ax0.set_title("Var(residual) vs mean(lambda)")
    ax0.grid(True, alpha=0.25)

    ax1.plot(mean_lam, ratio, ".", markersize=2)
    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xlabel("mean(lambda_hat)")
    ax1.set_ylabel("var/mean")
    ax1.set_title("ratio var/mean")
    ax1.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


@dataclass(frozen=True)
class RunConfig:
    exp_dir: Path
    config: Path
    ckpt: Path
    ckpt_id: str
    spect229_dir: Path
    results_root: Path
    exp_name: str
    max_value: float
    iterations: int
    views_per_subset: Optional[int]
    device: str
    use_log1p: bool
    recon_limit: Optional[int]
    compute_metrics: bool
    poisson_bin_width: float
    poisson_max_bin: float
    poisson_min_count: int
    overwrite: bool
    rerun_denoise: bool
    rerun_recon: bool
    mp4: bool
    mp4_only: bool
    gif: bool
    # Synthetic low-dose inference via independent thinning from 20s (default disabled).
    synthetic_thin_factors: Optional[str]
    synthetic_thin_seed: int
    synthetic_include_20s: bool


def _ckpt_id_from_path(ckpt: Path) -> str:
    """Stable identifier for bucketing outputs by checkpoint."""
    ckpt = Path(ckpt)
    m = re.search(r"net_g[_-](\d+)\.pth", ckpt.name)
    if m:
        return f"iter{int(m.group(1))}"
    if ckpt.name == "net_g_latest.pth" and ckpt.parent.is_dir():
        best = 0
        for p in ckpt.parent.glob("net_g_*.pth"):
            mm = re.search(r"net_g_(\d+)\.pth", p.name)
            if mm:
                best = max(best, int(mm.group(1)))
        return f"iter{best}" if best > 0 else "latest"
    return ckpt.stem


def _pick_ckpt_best_or_latest(models_dir: Path) -> Path:
    """Pick ckpt with priority: net_g_best.pth -> net_g_latest.pth -> max iter net_g_<iter>.pth."""
    models_dir = Path(models_dir)
    best = models_dir / "net_g_best.pth"
    if best.is_file():
        return best
    return pick_latest_ckpt(models_dir)


def _list_ckpts(models_dir: Path) -> list[Path]:
    """List available checkpoints under models_dir (latest first, then iters desc)."""
    models_dir = Path(models_dir)
    out: list[Path] = []
    best = models_dir / "net_g_best.pth"
    if best.is_file():
        out.append(best)
    latest = models_dir / "net_g_latest.pth"
    if latest.is_file():
        out.append(latest)
    iters: list[tuple[int, Path]] = []
    for p in models_dir.glob("net_g_*.pth"):
        m = re.search(r"net_g_(\d+)\.pth", p.name)
        if m:
            iters.append((int(m.group(1)), p))
    iters.sort(key=lambda t: t[0], reverse=True)
    out.extend([p for _, p in iters])
    # de-dup
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _parse_factors(s: Optional[str]) -> list[float]:
    s = "" if s is None else str(s).strip()
    if s == "":
        return []
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(float(part))
    return out


def _counts_int(x: np.ndarray) -> np.ndarray:
    """Round+clip to int64 counts."""
    y = np.round(x.astype(np.float64, copy=False))
    y = np.clip(y, 0.0, None)
    return y.astype(np.int64, copy=False)


def _poisson_thin_binomial(y: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Independent thinning: x ~ Binomial(y, p) elementwise."""
    p = float(p)
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    return rng.binomial(y, p).astype(np.int64, copy=False)


def run_experiment(
    *,
    rc: RunConfig,
    patients: list[str],
    stage: str,
) -> None:
    stage = str(stage).lower().strip()
    orbit_file = _default_orbit_file()

    # Bucket results by ckpt id: different ckpts should not overwrite each other.
    exp_root = rc.results_root / rc.exp_name
    # New layout:
    #   results/<exp_name>/<ckpt_id>/<patient>/{projections,reconstructions,gifs/...}
    out_dir = exp_root / rc.ckpt_id
    _ensure_dir(out_dir)
    print(f"[INFO] exp={rc.exp_name} stage={stage} patients={len(patients)} out_dir={out_dir}")
    print(f"[INFO] config={rc.config}")
    print(f"[INFO] ckpt={rc.ckpt}")
    print(f"[INFO] ckpt_id={rc.ckpt_id}")
    if not Path(rc.ckpt).is_file():
        raise FileNotFoundError(f"checkpoint not found: {rc.ckpt}")
    if rc.compute_metrics:
        print(f"[INFO] metrics: summary_dir={out_dir / '_summary'}")

    # Matplotlib is needed for GIF generation utilities imported by the pipeline.
    # We keep this import here (not at module import time) so list/--help works in minimal envs.
    try:
        from spect_ct.pipeline.threeproj3recon import process_patient_3proj3recon  # noqa: F401
    except ModuleNotFoundError as e:
        if "matplotlib" in str(e):
            raise ModuleNotFoundError(
                "缺少依赖 matplotlib（用于 GIF/可视化）。请在你的训练环境里安装：\n"
                "  python -m pip install matplotlib\n"
                "然后再运行 toolbox 的 run/batch。"
            ) from e
        raise

    # Metrics accumulators
    lpips_rows: list[dict] = []
    recon_rows: list[dict] = []
    # Poisson calibration accum (y=original proj, lam=denoised proj)
    bin_edges = np.arange(0.0, float(rc.poisson_max_bin) + float(rc.poisson_bin_width), float(rc.poisson_bin_width), dtype=np.float32)
    pois_stats = PoissonBinStats.create(bin_edges=bin_edges)

    # Decide which patients to reconstruct (slow)
    recon_patients = patients
    if rc.recon_limit is not None:
        recon_patients = patients[: max(0, int(rc.recon_limit))]
    recon_patients_set = set(recon_patients)

    thin_factors = _parse_factors(getattr(rc, "synthetic_thin_factors", None))
    use_synth_thin = len(thin_factors) > 0
    if use_synth_thin:
        print(
            "[INFO] synthetic-thin enabled: "
            + json.dumps(
                {
                    "factors": thin_factors,
                    "seed": int(getattr(rc, "synthetic_thin_seed", 123)),
                    "include_20s": bool(getattr(rc, "synthetic_include_20s", True)),
                },
                ensure_ascii=False,
            )
        )

    for pi, patient in enumerate(patients):
        print(f"[INFO] [{pi+1:03d}/{len(patients):03d}] patient={patient}")
        ppaths = resolve_patient_paths(rc.spect229_dir, patient)
        input_proj = ppaths.proj
        par_file = ppaths.par
        atten_file = ppaths.atten

        # Decide per-patient stage
        do_recon = patient in recon_patients_set

        # If stage includes recon, ensure original recon cache exists under dataset dir.
        orig_recon_path = None
        if do_recon and stage in ["recon", "all", "recon_only", "full"]:
            try:
                orig_recon_path = ensure_original_recon(
                    spect229_dir=rc.spect229_dir,
                    patient=patient,
                    iterations=rc.iterations,
                    views_per_subset=rc.views_per_subset,
                    orbit_file=orbit_file,
                    overwrite=False,
                )
            except RuntimeError as e:
                # Common on Linux containers (osemrecon.exe not runnable).
                print(f"[WARN][{patient}] original recon skipped: {e}")
                do_recon = False
                orig_recon_path = None

        def _run_one(*, run_patient: str, in_proj: Path, orig_recon_override: Optional[Path]) -> Optional[object]:
            # Existing outputs (for skip-by-default behavior)
            patient_out = out_dir / run_patient
            den_file = patient_out / "projections" / "denoised_projection_f32.dat"
            den_p_file = patient_out / "projections" / "denoised_poisson_projection_f32.dat"
            recon_den = patient_out / "reconstructions" / f"{run_patient}_denoised_OSEMReconed_Iter{int(rc.iterations)}.dat"
            recon_poi = patient_out / "reconstructions" / f"{run_patient}_denoised_poisson_OSEMReconed_Iter{int(rc.iterations)}.dat"
            have_denoise = den_file.exists() and den_p_file.exists()
            have_recon = recon_den.exists() and recon_poi.exists()

            # Desired outputs
            gif_path = patient_out / "gifs" / f"{run_patient}_4proj_4recon_2x4.gif"
            mp4_path = patient_out / "gifs" / f"{run_patient}_4proj_4recon_2x4.mp4"
            want_mp4_any = bool(getattr(rc, "mp4", False)) or bool(getattr(rc, "mp4_only", False))
            want_mp4 = want_mp4_any and (bool(rc.overwrite) or (not mp4_path.exists()))
            # If mp4-only: never generate gif. If not mp4-only: generate gif only when missing/overwrite.
            want_gif = bool(getattr(rc, "gif", True)) and (not bool(getattr(rc, "mp4_only", False))) and (bool(rc.overwrite) or (not gif_path.exists()))

            # Map stage to pipeline stages (per-run)
            if stage in ["denoise", "infer"]:
                pipe_stage = "denoise"
                # default: if projections already exist, skip denoise and just re-draw GIF
                skip_denoise = (not bool(rc.rerun_denoise)) and bool(have_denoise)
                skip_recon = True
            elif stage in ["recon", "recon_only"]:
                pipe_stage = "all"
                skip_denoise = (not bool(rc.rerun_denoise)) and bool(have_denoise)
                skip_recon = (not bool(rc.rerun_recon)) and bool(have_recon)
                if not do_recon:
                    return None
            elif stage in ["all", "full"]:
                pipe_stage = "all"
                skip_denoise = (not bool(rc.rerun_denoise)) and bool(have_denoise)
                if not do_recon:
                    skip_recon = True
                else:
                    skip_recon = ((not bool(rc.rerun_recon)) and bool(have_recon))
            else:
                raise ValueError(f"unsupported stage: {stage}")

            try:
                return process_patient_3proj3recon(
                    patient=run_patient,
                    checkpoint=rc.ckpt,
                    config=rc.config,
                    input_proj=in_proj,
                    par_file=par_file,
                    orbit_file=orbit_file,
                    atten_file=atten_file,
                    output_dir=out_dir,
                    max_value=float(rc.max_value),
                    iterations=int(rc.iterations),
                    views_per_subset=rc.views_per_subset,
                    # Let pipeline skip if counts+requested visuals already exist (unless overwrite).
                    skip_existing=(not bool(rc.overwrite)),
                    skip_denoise=bool(skip_denoise),
                    skip_recon=bool(skip_recon),
                    use_log1p=bool(rc.use_log1p),
                    device=str(rc.device),
                    stage=str(pipe_stage),
                    original_recon_path=orig_recon_override,
                    overwrite=bool(rc.overwrite),
                    mp4=want_mp4,
                    gif=want_gif,
                )
            except RuntimeError as e:
                print(f"[WARN][{run_patient}] pipeline failed (maybe OSEM not runnable): {e}")
                return None

        outputs_list: list[object] = []
        if use_synth_thin:
            # Optional baseline 20s
            if bool(getattr(rc, "synthetic_include_20s", True)):
                out0 = _run_one(run_patient=patient, in_proj=input_proj, orig_recon_override=orig_recon_path)
                if out0 is not None:
                    outputs_list.append(out0)

            # Generate x2/x3/x4/x5 from the 20s projection
            y20 = load_projection_i16(input_proj).astype(np.float32, copy=False)
            y20i = _counts_int(y20)
            base_seed = int(getattr(rc, "synthetic_thin_seed", 123)) + int(zlib.adler32(patient.encode("utf-8")))
            for k in thin_factors:
                kk = float(k)
                if kk <= 1.0:
                    continue
                rng_k = np.random.default_rng(base_seed + int(round(kk * 1000.0)))
                xk = _poisson_thin_binomial(y20i, p=1.0 / kk, rng=rng_k).astype(np.float32, copy=False)
                tag = f"x{kk:g}"
                in_dir = out_dir / f"{patient}__{tag}" / "_inputs"
                in_dir.mkdir(parents=True, exist_ok=True)
                in_file = in_dir / f"{patient}_thin_{tag}_from20s.dat"
                if bool(rc.overwrite) or (not in_file.exists()):
                    save_projection_i16_round_clip(in_file, xk)
                # For low-dose recons, do NOT reuse cached 20s original recon.
                outk = _run_one(run_patient=f"{patient}__{tag}", in_proj=in_file, orig_recon_override=None)
                if outk is not None:
                    outputs_list.append(outk)
        else:
            out0 = _run_one(run_patient=patient, in_proj=input_proj, orig_recon_override=orig_recon_path)
            if out0 is None:
                continue
            outputs_list.append(out0)

        # Projection-domain metrics (fast): fp32-only (no int16 artifacts).
        if rc.compute_metrics:
            try:
                from spect_ct.pipeline.io import load_projection_f32
                # PSNR/SSIM (range-aware): prefer BasicSR metrics (cv2-based) in KAIR env; fallback otherwise.
                try:
                    from basicsr.metrics import calculate_psnr_range, calculate_ssim_range  # type: ignore
                    _use_basicsr = True
                except Exception:
                    _use_basicsr = False
                    from spect_ct.pipeline.simple_metrics import psnr_range, ssim_range

                def _psnr_ssim_views_mean(pred: np.ndarray, ref: np.ndarray, data_range: float) -> tuple[float, float]:
                    psnrs = []
                    ssims = []
                    dr = float(data_range)
                    for i in range(int(ref.shape[0])):
                        r2 = np.clip(ref[i].astype(np.float32, copy=False), 0.0, dr)
                        p2 = np.clip(pred[i].astype(np.float32, copy=False), 0.0, dr)
                        if _use_basicsr:
                            psnrs.append(float(calculate_psnr_range(p2[:, :, None], r2[:, :, None], crop_border=0, data_range=float(dr), input_order="HWC")))
                            ssims.append(float(calculate_ssim_range(p2[:, :, None], r2[:, :, None], crop_border=0, data_range=float(dr), input_order="HWC")))
                        else:
                            psnrs.append(float(psnr_range(p2, r2, data_range=dr)))
                            ssims.append(float(ssim_range(p2, r2, data_range=dr)))
                    return float(np.mean(psnrs)) if psnrs else float("nan"), float(np.mean(ssims)) if ssims else float("nan")

                mv = float(rc.max_value)
                for out_obj in outputs_list:
                    outputs = out_obj  # type: ignore
                    proj_dir = Path(outputs.proj_dir)
                    # fp32-only layout (backward compat: fallback to legacy int16 names if present)
                    orig_f32 = proj_dir / "original_projection_f32.dat"
                    orig_i16 = proj_dir / "original_projection.dat"
                    if orig_f32.exists():
                        orig_proj = load_projection_f32(orig_f32).astype(np.float32, copy=False)
                    else:
                        orig_proj = load_projection_i16(orig_i16).astype(np.float32, copy=False)

                    den_fp32 = load_projection_f32(proj_dir / "denoised_projection_f32.dat").astype(np.float32, copy=False)

                    den_p_f32 = proj_dir / "denoised_poisson_projection_f32.dat"
                    den_p_i16 = proj_dir / "denoised_poisson_projection.dat"
                    if den_p_f32.exists():
                        den_p = load_projection_f32(den_p_f32).astype(np.float32, copy=False)
                    else:
                        den_p = load_projection_i16(den_p_i16).astype(np.float32, copy=False)

                    # LPIPS
                    lp_d_fp32 = lpips_views_mean(pred=den_fp32, ref=orig_proj, max_value=float(rc.max_value), nets=("alex", "vgg"))
                    lp_p = lpips_views_mean(pred=den_p, ref=orig_proj, max_value=float(rc.max_value), nets=("alex", "vgg"))

                    psnr_fp32, ssim_fp32 = _psnr_ssim_views_mean(den_fp32, orig_proj, mv)
                    psnr_p, ssim_p = _psnr_ssim_views_mean(den_p, orig_proj, mv)
                    psnr_qgap, ssim_qgap = float("nan"), float("nan")

                    patient_name = Path(outputs.proj_dir).parents[0].name
                    lpips_rows.append(
                        {
                            "patient": patient_name,
                            "lpips_denoised_fp32_alex": lp_d_fp32.get("alex", float("nan")),
                            "lpips_denoised_fp32_vgg": lp_d_fp32.get("vgg", float("nan")),
                            "psnr_denoised_fp32": psnr_fp32,
                            "ssim_denoised_fp32": ssim_fp32,
                            "lpips_poisson_alex": lp_p.get("alex", float("nan")),
                            "lpips_poisson_vgg": lp_p.get("vgg", float("nan")),
                            "psnr_poisson": psnr_p,
                            "ssim_poisson": ssim_p,
                            "psnr_u16_vs_fp32": psnr_qgap,
                            "ssim_u16_vs_fp32": ssim_qgap,
                        }
                    )

                    # Poisson calibration accumulation: residual = y - lam_hat
                    y = np.clip(orig_proj, 0.0, None)
                    lam = np.clip(den_fp32, 0.0, None)
                    for vi in range(int(y.shape[0])):
                        pois_stats.update(lam[vi], y[vi])
            except Exception:
                pass

        # Recon-domain metrics (slow, only when recon was done)
        if rc.compute_metrics and do_recon and orig_recon_path is not None and not skip_recon:
            try:
                r_ref = load_recon_f32(orig_recon_path, shape=(128, 128, 128))
                # Only compute recon metrics for the baseline 20s run (patient name exactly matches).
                base_out = None
                for out_obj in outputs_list:
                    o = out_obj  # type: ignore
                    if Path(o.recon_dir).parents[0].name == patient:
                        base_out = o
                        break
                if base_out is None:
                    raise FileNotFoundError("baseline outputs not found for recon metrics")
                r_den = load_recon_f32(Path(base_out.recon_dir) / f"{patient}_denoised_OSEMReconed_Iter{rc.iterations}.dat", shape=(128, 128, 128))
                r_poi = load_recon_f32(Path(base_out.recon_dir) / f"{patient}_denoised_poisson_OSEMReconed_Iter{rc.iterations}.dat", shape=(128, 128, 128))

                m_den = psnr_ssim_recon(pred=r_den, ref=r_ref, crop_border=0, data_range=None)
                m_poi = psnr_ssim_recon(pred=r_poi, ref=r_ref, crop_border=0, data_range=None)
                recon_rows.append(
                    {
                        "patient": patient,
                        "psnr_denoised": m_den.psnr,
                        "ssim_denoised": m_den.ssim,
                        "psnr_poisson": m_poi.psnr,
                        "ssim_poisson": m_poi.ssim,
                        "data_range_ref_mip": m_den.data_range,
                        "orig_recon_path": str(orig_recon_path),
                    }
                )
            except Exception:
                pass

    # Save summaries
    if rc.compute_metrics:
        # New layout: results/<exp>/<ckpt_id>/_summary/*
        summary_dir = out_dir / "_summary"
        _ensure_dir(summary_dir)
        _save_csv(summary_dir / "lpips_proj.csv", lpips_rows)
        _save_csv(summary_dir / "recon_psnr_ssim.csv", recon_rows)

        # Poisson calibration summary
        stats = pois_stats.finalize(min_count=int(rc.poisson_min_count))
        # Save as npz + png
        np.savez_compressed(summary_dir / "poisson_calibration_bins.npz", **stats)
        _plot_poisson_calibration(summary_dir / "poisson_calibration.png", stats, title=f"{rc.exp_name} poisson calibration")
        print(f"[INFO] wrote: {summary_dir / 'lpips_proj.csv'}")
        print(f"[INFO] wrote: {summary_dir / 'poisson_calibration.png'}")
        print(f"[INFO] wrote: {summary_dir / 'poisson_calibration_bins.npz'}")


def _cmd_list(args: argparse.Namespace) -> None:
    exps = list_experiments(Path(args.experiments_root))
    for i, p in enumerate(exps):
        print(f"[{i:03d}] {p.name}  ({p})")


def _cmd_run(args: argparse.Namespace) -> None:
    _apply_torch_backend_settings(disable_cudnn=bool(getattr(args, "disable_cudnn", False)))
    exp_dir = Path(args.exp)
    config = Path(args.config) if args.config else find_config_for_experiment(exp_dir)
    ckpt = Path(args.ckpt) if args.ckpt else _pick_ckpt_best_or_latest(exp_dir / "models")
    ckpt_id = _ckpt_id_from_path(ckpt)
    spect229_dir = Path(args.spect229_dir)
    results_root = Path(args.results_root)
    max_value = float(args.max_value) if args.max_value is not None else _read_yaml_max_value(config, default=150.0)

    all_patients = resolve_patients(spect229_dir)
    sel_patients = parse_patient_indices(all_patients, indices=args.patients, index_range=args.patients_range)

    rc = RunConfig(
        exp_dir=exp_dir,
        config=config,
        ckpt=ckpt,
        ckpt_id=str(ckpt_id),
        spect229_dir=spect229_dir,
        results_root=results_root,
        exp_name=str(args.exp_name) if args.exp_name else exp_dir.name,
        max_value=max_value,
        iterations=int(args.iterations),
        views_per_subset=int(args.views_per_subset) if args.views_per_subset is not None else None,
        device=str(args.device),
        use_log1p=bool(args.log1p),
        recon_limit=int(args.recon_limit) if args.recon_limit is not None else None,
        compute_metrics=bool(args.metrics),
        poisson_bin_width=float(args.poisson_bin_width),
        poisson_max_bin=float(args.poisson_max_bin),
        poisson_min_count=int(args.poisson_min_count),
        overwrite=bool(args.overwrite),
        rerun_denoise=not bool(getattr(args, "no_rerun_denoise", False)),
        rerun_recon=not bool(getattr(args, "no_rerun_recon", False)),
        mp4=bool(getattr(args, "mp4", False)) or bool(getattr(args, "mp4_only", False)),
        mp4_only=bool(getattr(args, "mp4_only", False)),
        gif=bool(getattr(args, "gif", True)),
        synthetic_thin_factors=str(getattr(args, "synthetic_thin_factors", "") or ""),
        synthetic_thin_seed=int(getattr(args, "synthetic_thin_seed", 123)),
        synthetic_include_20s=bool(getattr(args, "synthetic_include_20s", True)),
    )

    run_experiment(rc=rc, patients=sel_patients, stage=str(args.stage))


def _cmd_batch(args: argparse.Namespace) -> None:
    _apply_torch_backend_settings(disable_cudnn=bool(getattr(args, "disable_cudnn", False)))
    root = Path(args.experiments_root)
    exps = list_experiments(root)
    if args.match:
        exps = [p for p in exps if fnmatch.fnmatch(p.name, str(args.match))]
    if len(exps) == 0:
        print("[WARN] no experiments matched.")
        return

    spect229_dir = Path(args.spect229_dir)
    all_patients = resolve_patients(spect229_dir)
    sel_patients = parse_patient_indices(all_patients, indices=args.patients, index_range=args.patients_range)

    for exp_dir in exps:
        config = Path(args.config) if args.config else find_config_for_experiment(exp_dir)
        ckpt = Path(args.ckpt) if args.ckpt else _pick_ckpt_best_or_latest(exp_dir / "models")
        ckpt_id = _ckpt_id_from_path(ckpt)
        max_value = float(args.max_value) if args.max_value is not None else _read_yaml_max_value(config, default=150.0)
        rc = RunConfig(
            exp_dir=exp_dir,
            config=config,
            ckpt=ckpt,
            ckpt_id=str(ckpt_id),
            spect229_dir=spect229_dir,
            results_root=Path(args.results_root),
            exp_name=str(args.exp_name_prefix) + exp_dir.name if args.exp_name_prefix else exp_dir.name,
            max_value=max_value,
            iterations=int(args.iterations),
            views_per_subset=int(args.views_per_subset) if args.views_per_subset is not None else None,
            device=str(args.device),
            use_log1p=bool(args.log1p),
            recon_limit=int(args.recon_limit) if args.recon_limit is not None else None,
            compute_metrics=bool(args.metrics),
            poisson_bin_width=float(args.poisson_bin_width),
            poisson_max_bin=float(args.poisson_max_bin),
            poisson_min_count=int(args.poisson_min_count),
            overwrite=bool(args.overwrite),
            rerun_denoise=not bool(getattr(args, "no_rerun_denoise", False)),
            rerun_recon=not bool(getattr(args, "no_rerun_recon", False)),
            mp4=bool(getattr(args, "mp4", False)) or bool(getattr(args, "mp4_only", False)),
            mp4_only=bool(getattr(args, "mp4_only", False)),
            gif=bool(getattr(args, "gif", True)),
            synthetic_thin_factors=str(getattr(args, "synthetic_thin_factors", "") or ""),
            synthetic_thin_seed=int(getattr(args, "synthetic_thin_seed", 123)),
            synthetic_include_20s=bool(getattr(args, "synthetic_include_20s", True)),
        )
        run_experiment(rc=rc, patients=sel_patients, stage=str(args.stage))


def _interactive() -> None:
    q = _maybe_questionary()
    experiments_root = Path("experiments")
    spect229_dir = Path("datasets/SPECT229")
    results_root = _default_results_root()
    dft = _load_toolbox_defaults()

    exps = list_experiments(experiments_root)
    if len(exps) == 0:
        print(f"[ERR] no experiments found under: {experiments_root}")
        return

    if q is not None:
        exp_choice = q.select("选择实验目录", choices=[{"name": p.name, "value": str(p)} for p in exps]).ask()
        if not exp_choice:
            return
        exp_dir = Path(exp_choice)
        ckpts = _list_ckpts(exp_dir / "models")
        if len(ckpts) == 0:
            print(f"[ERR] no ckpt found under: {exp_dir / 'models'}")
            return
        ckpt_choice = q.select(
            "选择 ckpt（默认 latest/最大iter 在最前）",
            choices=[{"name": p.name, "value": str(p)} for p in ckpts],
        ).ask()
        if not ckpt_choice:
            return
        ckpt = Path(ckpt_choice)
        stage = q.select("选择 stage", choices=["denoise", "recon", "recon_only", "all"]).ask() or "denoise"
        patients_range = q.text("选择病人序号范围（例如 0:20，留空=全部）", default="").ask()
        recon_limit_s = q.text("重建数量上限（慢；留空=全部/或stage=denoise时忽略）", default="").ask()
        # 不再提示：从 toolbox_defaults.json 读取（可在文件里改）
        log1p = bool(dft.get("log1p", True))
        metrics = bool(dft.get("metrics", True))
        disable_cudnn = bool(dft.get("disable_cudnn", True))
        overwrite = bool(dft.get("overwrite", True))
        rerun_denoise = bool(dft.get("rerun_denoise", False))
        rerun_recon = bool(dft.get("rerun_recon", False))
    else:
        print("[WARN] questionary 未安装，已降级到 input() 交互。")
        for i, p in enumerate(exps):
            print(f"[{i}] {p.name}")
        exp_dir = exps[int(input("选择实验编号: ").strip())]
        ckpts = _list_ckpts(exp_dir / "models")
        for i, p in enumerate(ckpts):
            print(f"[{i}] {p.name}")
        ckpt = ckpts[int(input("选择 ckpt 编号 [0]: ").strip() or "0")]
        stage = input("stage (denoise/recon/all) [denoise]: ").strip() or "denoise"
        patients_range = input("病人范围 0:20 (空=全部): ").strip()
        recon_limit_s = input("重建上限 (空=全部): ").strip()
        # 不再提示：从 toolbox_defaults.json 读取（可在文件里改）
        log1p = bool(dft.get("log1p", True))
        metrics = bool(dft.get("metrics", True))
        disable_cudnn = bool(dft.get("disable_cudnn", True))
        overwrite = bool(dft.get("overwrite", True))
        rerun_denoise = bool(dft.get("rerun_denoise", False))
        rerun_recon = bool(dft.get("rerun_recon", False))

    print(
        "[INFO] interactive defaults: "
        + json.dumps(
            {
                "log1p": bool(log1p),
                "metrics": bool(metrics),
                "disable_cudnn": bool(disable_cudnn),
                "overwrite": bool(overwrite),
                "rerun_denoise": bool(rerun_denoise),
                "rerun_recon": bool(rerun_recon),
            },
            ensure_ascii=False,
        )
    )

    _apply_torch_backend_settings(disable_cudnn=bool(disable_cudnn))
    config = find_config_for_experiment(exp_dir)
    ckpt_id = _ckpt_id_from_path(ckpt)
    max_value = _read_yaml_max_value(config, default=150.0)
    all_patients = resolve_patients(spect229_dir)
    sel_patients = parse_patient_indices(all_patients, index_range=(patients_range or None))
    recon_limit = int(recon_limit_s) if recon_limit_s else None

    rc = RunConfig(
        exp_dir=exp_dir,
        config=config,
        ckpt=ckpt,
        ckpt_id=str(ckpt_id),
        spect229_dir=spect229_dir,
        results_root=results_root,
        exp_name=exp_dir.name,
        max_value=max_value,
        iterations=int(dft.get("iterations", 10)),
        views_per_subset=int(dft["views_per_subset"]) if dft.get("views_per_subset", None) is not None else None,
        device=str(dft.get("device", "cuda")),
        use_log1p=bool(log1p),
        recon_limit=recon_limit,
        compute_metrics=bool(metrics),
        poisson_bin_width=float(dft.get("poisson_bin_width", 1.0)),
        poisson_max_bin=float(max_value),
        poisson_min_count=int(dft.get("poisson_min_count", 20000)),
        overwrite=bool(overwrite),
        rerun_denoise=bool(rerun_denoise),
        rerun_recon=bool(rerun_recon),
        mp4=False,
        mp4_only=False,
        synthetic_thin_factors=str(dft.get("synthetic_thin_factors", "") or ""),
        synthetic_thin_seed=int(dft.get("synthetic_thin_seed", 123)),
        synthetic_include_20s=bool(dft.get("synthetic_include_20s", True)),
    )
    run_experiment(rc=rc, patients=sel_patients, stage=str(stage))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SPECT229 toolbox (interactive + CLI)")
    sub = p.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="list experiments under experiments_root")
    p_list.add_argument("--experiments-root", type=str, default="experiments")
    p_list.set_defaults(func=_cmd_list)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
        sp.add_argument("--results-root", type=str, default=str(_default_results_root()))
        sp.add_argument("--patients", type=str, default=None, help="comma indices, e.g. 0,1,2")
        sp.add_argument("--patients-range", type=str, default=None, help="slice range, e.g. 0:20")
        sp.add_argument("--iterations", type=int, default=10)
        sp.add_argument("--views-per-subset", type=int, default=None, help="override Number_of_Views_Per_Subset in .par")
        sp.add_argument("--recon-limit", type=int, default=None, help="only recon first K patients (slow)")
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--max-value", type=float, default=None, help="projection max_value for inference/LPIPS normalization")
        sp.add_argument("--log1p", action="store_true")
        sp.add_argument("--no-log1p", dest="log1p", action="store_false")
        sp.set_defaults(log1p=True)
        sp.add_argument("--metrics", action="store_true", help="compute metrics & summaries")
        sp.add_argument("--poisson-bin-width", type=float, default=1.0)
        sp.add_argument("--poisson-max-bin", type=float, default=150.0)
        sp.add_argument("--poisson-min-count", type=int, default=20000)
        # NOTE: recon_only = strictly do reconstruction only (reuse existing projections under results).
        sp.add_argument("--stage", type=str, default="denoise", choices=["denoise", "recon", "recon_only", "all"])
        sp.add_argument(
            "--overwrite",
            action="store_true",
            help="overwrite outputs (reuse *_000 dir for GIFs, overwrite same-name files)",
        )
        sp.add_argument(
            "--no-rerun-denoise",
            action="store_true",
            help="do NOT rerun denoise if projections/*.dat already exist under results (skip denoise by default)",
        )
        sp.add_argument(
            "--no-rerun-recon",
            action="store_true",
            help="do NOT rerun recon if reconstructions/*.dat already exist under results (skip recon by default)",
        )
        sp.add_argument(
            "--disable-cudnn",
            action="store_true",
            help="disable torch.backends.cudnn for stability (may be slower)",
        )
        sp.add_argument("--mp4", action="store_true", help="also generate mp4 (truecolor) for each patient")
        sp.add_argument("--mp4-only", dest="mp4_only", action="store_true", help="only generate mp4 (skip gif)")
        sp.set_defaults(mp4=False, mp4_only=False)
        sp.add_argument("--gif", action="store_true", help="generate gif visualization (default: enabled)")
        sp.add_argument("--no-gif", dest="gif", action="store_false", help="disable gif generation (denoise only)")
        sp.set_defaults(gif=True)

        # Synthetic low-dose inference via 20s thinning (default OFF).
        sp.add_argument(
            "--synthetic-thin-factors",
            type=str,
            default="",
            help="(default OFF) independent thinning from 20s by factors, e.g. '2,3,4,5' (means p=1/k).",
        )
        sp.add_argument("--synthetic-thin-seed", type=int, default=123, help="seed for thinning RNG (stable across runs)")
        sp.add_argument(
            "--synthetic-include-20s",
            action="store_true",
            help="also run baseline 20s alongside synthetic xk rows (default: enabled)",
        )
        sp.add_argument(
            "--synthetic-no-20s",
            dest="synthetic_include_20s",
            action="store_false",
            help="when using --synthetic-thin-factors, do NOT run the baseline 20s row",
        )
        sp.set_defaults(synthetic_include_20s=True)

    p_run = sub.add_parser("run", help="run one experiment")
    p_run.add_argument("--exp", type=str, required=True, help="experiment dir under experiments/")
    p_run.add_argument("--config", type=str, default=None, help="override config yml; default auto-detect from exp dir")
    p_run.add_argument("--ckpt", type=str, default=None, help="override checkpoint; default latest under exp/models/")
    p_run.add_argument("--exp-name", type=str, default=None, help="output name under spect_ct/results/")
    add_common(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_batch = sub.add_parser("batch", help="batch run multiple experiments")
    p_batch.add_argument("--experiments-root", type=str, default="experiments")
    p_batch.add_argument("--match", type=str, default=None, help="fnmatch pattern for experiment names, e.g. n2n_spect229_*")
    p_batch.add_argument("--config", type=str, default=None, help="optional override config for all")
    p_batch.add_argument("--ckpt", type=str, default=None, help="optional override ckpt for all")
    p_batch.add_argument("--exp-name-prefix", type=str, default=None)
    add_common(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    # ---------------------------------------------------------------------
    # Recon-only from existing results (no ckpt/config needed)
    # ---------------------------------------------------------------------
    def _cmd_recon_results(args: argparse.Namespace) -> None:
        """Run OSEM reconstructions for an existing results tree produced by denoise stage.

        Expected layout (fp32-only):
          <results_dir>/<patient>/projections/{original_projection_f32,denoised_projection_f32,denoised_poisson_projection_f32}.dat
          <results_dir>/<patient>/reconstructions/  (will be filled)
        """
        from spect_ct.pipeline.osem import run_osem_reconstruction
        from spect_ct.pipeline.original_recon_cache import resolve_patient_paths

        results_dir = Path(args.results_dir)
        spect229_dir = Path(args.spect229_dir)
        orbit_file = Path(args.orbit_file) if args.orbit_file else (Path(__file__).resolve().parents[1] / "osemreocnexe" / "orbit.orb")
        if not orbit_file.exists():
            raise FileNotFoundError(f"orbit file not found: {orbit_file}")

        if not results_dir.exists():
            raise FileNotFoundError(f"results_dir not found: {results_dir}")

        # Patient selection: either explicit list, or all subdirs.
        all_patients = sorted([p.name for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
        sel_patients = parse_patient_indices(all_patients, indices=getattr(args, "patients", None), index_range=getattr(args, "patients_range", None))
        if len(sel_patients) == 0:
            print("[WARN] no patients selected.")
            return

        it = int(args.iterations)
        vps = int(args.views_per_subset) if args.views_per_subset is not None else None
        rerun = bool(args.rerun_recon)
        synth_f32 = bool(getattr(args, "synth_f32", False))

        for i, patient in enumerate(sel_patients):
            print(f"[INFO] [{i+1:03d}/{len(sel_patients):03d}] recon patient={patient}")
            # Resolve dataset-side par/atten
            ppaths = resolve_patient_paths(spect229_dir, patient)

            proj_dir = results_dir / patient / "projections"
            recon_dir = results_dir / patient / "reconstructions"
            recon_dir.mkdir(parents=True, exist_ok=True)

            # Recon jobs (fp32-only, PrjDataType=1).
            recon_jobs: list[tuple[Path, str, int]] = [
                (proj_dir / "original_projection_f32.dat", "original", 1),
                (proj_dir / "denoised_projection_f32.dat", "denoised_fp32", 1),
                (proj_dir / "denoised_poisson_projection_f32.dat", "denoised_poisson", 1),
            ]
            missing = [str(p) for p, _, _ in recon_jobs if not Path(p).exists()]
            if missing:
                print(f"[WARN][{patient}] missing projections, skip: {missing}")
                continue

            for proj_file, recon_name, prj_dtype in recon_jobs:
                out = recon_dir / f"{patient}_{recon_name}_OSEMReconed_Iter{it}.dat"
                if out.exists() and not rerun:
                    continue
                try:
                    run_osem_reconstruction(
                        proj_file=Path(proj_file),
                        patient_name=patient,
                        par_file=ppaths.par,
                        orbit_file=orbit_file,
                        atten_file=ppaths.atten,
                        output_name=recon_name,
                        iterations=it,
                        views_per_subset=vps,
                        prj_data_type=int(prj_dtype),
                        final_output_dir=recon_dir,
                        timeout_sec=int(args.timeout_sec),
                    )
                except Exception as e:
                    print(f"[WARN][{patient}] recon failed ({recon_name}): {e}")
                    # keep going for other patients
                    continue

    p_recon = sub.add_parser("recon-results", help="run OSEM reconstructions for an existing results tree (no ckpt/config)")
    p_recon.add_argument("--results-dir", type=str, required=True, help="path like spect_ct/results/<exp>/<ckpt_id>")
    p_recon.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    p_recon.add_argument("--orbit-file", type=str, default=None, help="override orbit.orb (default: spect_ct/osemreocnexe/orbit.orb)")
    p_recon.add_argument("--patients", type=str, default=None, help="comma indices based on sorted patient dirs, e.g. 0,1,2")
    p_recon.add_argument("--patients-range", type=str, default=None, help="slice range, e.g. 0:20")
    p_recon.add_argument("--iterations", type=int, default=10)
    p_recon.add_argument("--views-per-subset", type=int, default=None, help="override Number_of_Views_Per_Subset in .par")
    p_recon.add_argument("--timeout-sec", type=int, default=600)
    p_recon.add_argument("--rerun-recon", action="store_true", help="overwrite existing reconstructions/*.dat if present")
    p_recon.add_argument(
        "--synth-f32",
        action="store_true",
        help="(legacy) kept for backward compat; fp32-only layout no longer synthesizes from int16 outputs",
    )
    p_recon.set_defaults(func=_cmd_recon_results)

    # ---------------------------------------------------------------------
    # GIF+counts from existing results only (skip denoise + skip recon)
    # ---------------------------------------------------------------------
    def _cmd_gif_results(args: argparse.Namespace) -> None:
        from spect_ct.pipeline.threeproj3recon import generate_gif_counts_from_existing

        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"results_dir not found: {results_dir}")

        all_patients = sorted([p.name for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_")])
        sel_patients = parse_patient_indices(all_patients, indices=getattr(args, "patients", None), index_range=getattr(args, "patients_range", None))
        if len(sel_patients) == 0:
            print("[WARN] no patients selected.")
            return

        it = int(args.iterations)
        mv = float(args.max_value)
        use_log1p = bool(getattr(args, "log1p", True))
        skip_existing = bool(getattr(args, "skip_existing", True)) and (not bool(getattr(args, "force", False)))
        want_mp4 = bool(getattr(args, "mp4", False))
        mp4_only = bool(getattr(args, "mp4_only", False))
        want_gif = (not mp4_only)

        for i, patient in enumerate(sel_patients):
            print(f"[INFO] [{i+1:03d}/{len(sel_patients):03d}] gif_counts patient={patient}")
            pdir = results_dir / patient
            try:
                generate_gif_counts_from_existing(
                    patient=patient,
                    patient_dir=pdir,
                    iterations=it,
                    max_value=mv,
                    use_log1p=use_log1p,
                    skip_existing=skip_existing,
                    mp4=want_mp4,
                    gif=want_gif,
                )
            except Exception as e:
                print(f"[WARN][{patient}] gif_counts skipped: {e}")
                continue

    p_gif = sub.add_parser("gif-results", help="generate 2x3 GIF + counts.txt from existing projections+recons (no denoise/recon)")
    p_gif.add_argument("--results-dir", type=str, required=True, help="path like spect_ct/results/<exp>/<ckpt_id>")
    p_gif.add_argument("--patients", type=str, default=None, help="comma indices based on sorted patient dirs, e.g. 0,1,2")
    p_gif.add_argument("--patients-range", type=str, default=None, help="slice range, e.g. 0:20")
    p_gif.add_argument("--iterations", type=int, default=10)
    p_gif.add_argument("--max-value", type=float, default=150.0)
    p_gif.add_argument("--log1p", action="store_true")
    p_gif.add_argument("--no-log1p", dest="log1p", action="store_false")
    p_gif.set_defaults(log1p=True)
    p_gif.add_argument("--skip-existing", action="store_true", help="skip if target gif + counts.txt already exist")
    p_gif.add_argument("--force", action="store_true", help="force re-generate gif even if it exists")
    p_gif.add_argument("--mp4", action="store_true", help="also generate mp4 (truecolor) alongside outputs")
    p_gif.add_argument("--mp4-only", dest="mp4_only", action="store_true", help="only generate mp4 (skip gif)")
    p_gif.set_defaults(skip_existing=True, force=False)
    p_gif.set_defaults(mp4=False, mp4_only=False)
    p_gif.set_defaults(func=_cmd_gif_results)

    return p


def main() -> None:
    parser = build_parser()
    import sys
    if len(sys.argv) == 1:
        _interactive()
        return
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()


