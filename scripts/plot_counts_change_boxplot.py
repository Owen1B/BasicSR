#!/usr/bin/env python3
from __future__ import annotations

"""
Plot boxplots of total-count percentage change vs ORIGINAL for different denoisers.

This is meant to accompany the validation-style MP4 comparison.

For one patient:
  - Build rows: 20s + x2/x3/x4/x5 (from cached thins if available; optional on-the-fly + cache)
  - Get BM3D (cached; optional on-the-fly + cache) using:
        Anscombe forward -> BM3D(sigma=1.0) -> unbiased inverse
  - Get ModelA(ema), ModelB(ema) via prime inference loader (require_ema=True)
  - For each dose (row), compute per-view total counts, then percent change:
        pct = (method_view_counts - orig_view_counts) / orig_view_counts * 100
  - Save a grouped boxplot PNG: x-axis=dose (20s,10s,6.7s,5s,4s), each group contains BM3D/ModelB/ModelA.
"""

import argparse
from pathlib import Path

import numpy as np

from prime.pipeline.val_cache import (
    bm3d_anscombe_denoise as _bm3d_anscombe_denoise,
    counts_int as _counts_int,
    load_cached_bm3d as _load_cached_bm3d,
    load_cached_thin as _load_cached_thin,
    pick_ckpt as _pick_ckpt,
    poisson_thin_binomial as _poisson_thin_binomial,
    seed_cache_dir as _seed_cache_dir,
)


def _pct_change(method: np.ndarray, base: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    denom = np.maximum(base, eps)
    return (method - base) / denom * 100.0


def _parse_proj_shape(spec: str) -> tuple[int, int, int]:
    shp = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if len(shp) != 3:
        raise ValueError(f"--proj-shape must be views,h,w (e.g. 60,128,128), got: {spec}")
    if min(shp) <= 0:
        raise ValueError(f"--proj-shape values must be >0, got: {spec}")
    return int(shp[0]), int(shp[1]), int(shp[2])


def _parse_proj_dtype(spec: str) -> np.dtype:
    key = str(spec).strip().lower()
    mapping = {"int16": np.int16, "uint16": np.uint16, "float32": np.float32}
    if key not in mapping:
        raise ValueError(f"--proj-dtype must be one of {sorted(mapping.keys())}, got: {spec}")
    return np.dtype(mapping[key])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=str, default=None, help="sample/patient id for title and legacy path lookup")
    ap.add_argument("--spect229-dir", type=str, default="datasets/SPECT229")
    ap.add_argument(
        "--patient-dir",
        type=str,
        default=None,
        help="optional explicit patient directory (cache root + legacy file lookup root)",
    )
    ap.add_argument("--input-proj", type=str, default=None, help="optional explicit projection .dat path")
    ap.add_argument("--proj-shape", type=str, default="60,128,128", help="projection shape as views,h,w")
    ap.add_argument(
        "--proj-dtype",
        type=str,
        default="int16",
        choices=["int16", "uint16", "float32"],
        help="input projection dtype",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thin-factors", type=str, default="2,3,4,5")
    ap.add_argument("--require-cached", dest="require_cached", action="store_true")
    ap.add_argument("--no-require-cached", dest="require_cached", action="store_false")
    ap.add_argument("--allow-on-the-fly-cache", action="store_true")
    ap.set_defaults(require_cached=True, allow_on_the_fly_cache=False)

    ap.add_argument("--bm3d-sigma", type=float, default=1.0)
    ap.add_argument("--bm3d-workers", type=int, default=0)

    ap.add_argument("--max-value", type=float, default=150.0)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--exp-a", type=str, required=True)
    ap.add_argument("--iter-a", type=int, default=None)
    ap.add_argument("--config-a", type=str, default=None)
    ap.add_argument("--ckpt-a", type=str, default=None)
    ap.add_argument("--label-a", type=str, default="ModelA(ema)")

    ap.add_argument("--exp-b", type=str, required=True)
    ap.add_argument("--iter-b", type=int, default=None)
    ap.add_argument("--config-b", type=str, default=None)
    ap.add_argument("--ckpt-b", type=str, default=None)
    ap.add_argument("--label-b", type=str, default="ModelB(ema)")

    ap.add_argument("--out", type=str, required=True, help="output .png path")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.patient:
        patient = str(args.patient)
    elif args.input_proj:
        patient = Path(str(args.input_proj)).stem
    elif args.patient_dir:
        patient = Path(str(args.patient_dir)).name
    else:
        raise ValueError("请提供 --patient，或通过 --input-proj/--patient-dir 推断样本名")

    spect229_dir = Path(args.spect229_dir)
    legacy_patient_dir = spect229_dir / patient
    if args.patient_dir:
        patient_dir = Path(str(args.patient_dir))
    elif legacy_patient_dir.exists():
        patient_dir = legacy_patient_dir
    elif args.input_proj:
        patient_dir = Path(str(args.input_proj)).resolve().parent
    else:
        patient_dir = legacy_patient_dir
    if not patient_dir.exists():
        raise FileNotFoundError(f"patient_dir not found: {patient_dir}")

    if args.input_proj:
        lq_path = Path(str(args.input_proj)).resolve()
    else:
        lq_path = (patient_dir / f"{patient}_Proj4Filter.dat").resolve()
    if not lq_path.exists():
        raise FileNotFoundError(f"lq projection not found: {lq_path}")

    from prime.pipeline.io import load_projection
    proj_shape = _parse_proj_shape(str(args.proj_shape))
    proj_dtype = _parse_proj_dtype(str(args.proj_dtype))
    proj_u16 = load_projection(lq_path, shape=proj_shape, dtype=proj_dtype).astype(np.float32, copy=False)
    y20i = _counts_int(proj_u16)

    # rows
    factors = []
    for part in str(args.thin_factors).split(","):
        part = part.strip()
        if part:
            factors.append(float(part))
    factors = [k for k in factors if float(k) > 1.0]

    base_rows: list[tuple[str, float, np.ndarray]] = [("20s", 1.0, y20i.astype(np.float32, copy=False))]
    for k in factors:
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
            if bool(args.require_cached) and not bool(args.allow_on_the_fly_cache):
                raise
            xk = _poisson_thin_binomial(y20i, float(k), int(args.seed))
            if bool(args.allow_on_the_fly_cache):
                np.save(
                    _seed_cache_dir(patient_dir, int(args.seed)) / f"{lab}_thin.npy",
                    xk.astype(np.float32, copy=False),
                )

        base_rows.append((lab, float(k), xk.astype(np.float32, copy=False)))

    def _sec(label: str, k: float) -> float:
        return 20.0 if label == "20s" else 20.0 / max(float(k), 1e-12)

    base_rows.sort(key=lambda t: _sec(t[0], t[1]), reverse=True)

    # model resolve
    from prime.pipeline.experiment import find_config_for_experiment
    exp_a = Path(args.exp_a)
    exp_b = Path(args.exp_b)
    cfg_a = Path(args.config_a) if args.config_a else find_config_for_experiment(exp_a)
    cfg_b = Path(args.config_b) if args.config_b else find_config_for_experiment(exp_b)
    ckpt_a = Path(args.ckpt_a) if args.ckpt_a else _pick_ckpt(exp_a, prefer_iter=args.iter_a)
    ckpt_b = Path(args.ckpt_b) if args.ckpt_b else _pick_ckpt(exp_b, prefer_iter=args.iter_b)

    from prime.pipeline.inference import load_basicsr_net, denoise_views
    net_a = load_basicsr_net(cfg_a, ckpt_a, device=str(args.device), require_ema=True).net
    net_b = load_basicsr_net(cfg_b, ckpt_b, device=str(args.device), require_ema=True).net

    mv = float(args.max_value)

    # dose-wise collections: label -> pct array (len=60)
    pct_by_label_bm3d: dict[str, np.ndarray] = {}
    pct_by_label_a: dict[str, np.ndarray] = {}
    pct_by_label_b: dict[str, np.ndarray] = {}

    for label, k, orig in base_rows:
        # BM3D
        try:
            bm = _load_cached_bm3d(
                patient_dir=patient_dir,
                lq_path=lq_path,
                label=label,
                seed=int(args.seed),
                require_cached=bool(args.require_cached),
            )
        except FileNotFoundError:
            if bool(args.require_cached) and not bool(args.allow_on_the_fly_cache):
                raise
            bm = _bm3d_anscombe_denoise(orig, sigma_psd=float(args.bm3d_sigma), num_workers=int(args.bm3d_workers))
            if bool(args.allow_on_the_fly_cache):
                np.save(
                    _seed_cache_dir(patient_dir, int(args.seed)) / f"{label}_bm3d.npy",
                    bm.astype(np.float32, copy=False),
                )

        # models (EMA) in count domain
        ema_a = denoise_views(net_a, orig.astype(np.float32, copy=False), max_value=mv, device=str(args.device))
        ema_b = denoise_views(net_b, orig.astype(np.float32, copy=False), max_value=mv, device=str(args.device))
        ema_a = np.clip(ema_a.astype(np.float32, copy=False), 0.0, None)
        ema_b = np.clip(ema_b.astype(np.float32, copy=False), 0.0, None)

        # per-view totals
        base_v = np.sum(_counts_int(orig).astype(np.float64, copy=False), axis=(1, 2))
        bm_v = np.sum(_counts_int(bm).astype(np.float64, copy=False), axis=(1, 2))
        a_v = np.sum(ema_a.astype(np.float64, copy=False), axis=(1, 2))
        b_v = np.sum(ema_b.astype(np.float64, copy=False), axis=(1, 2))

        pct_by_label_bm3d[str(label)] = _pct_change(bm_v, base_v)
        pct_by_label_a[str(label)] = _pct_change(a_v, base_v)
        pct_by_label_b[str(label)] = _pct_change(b_v, base_v)

    # Build grouped boxplot: 3 boxes per dose (BM3D, B, A)
    dose_order = [lab for (lab, _, _) in base_rows]
    dose_names = []
    for lab, k, _ in base_rows:
        if str(lab) == "20s":
            dose_names.append("20s")
        else:
            sec = 20.0 / float(k)
            dose_names.append(f"{sec:.1f}s")

    group_size = 3
    positions = []
    data = []
    tick_pos = []
    for gi, lab in enumerate(dose_order):
        base_x = gi * (group_size + 1)
        positions.extend([base_x + 0, base_x + 1, base_x + 2])
        data.extend([
            pct_by_label_bm3d[str(lab)],
            pct_by_label_b[str(lab)],
            pct_by_label_a[str(lab)],
        ])
        tick_pos.append(base_x + 1)

    fig = plt.figure(figsize=(12, 5), dpi=200)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(
        data,
        positions=positions,
        showfliers=True,
        whis=1.5,
    )
    ax.set_xticks(tick_pos, dose_names)
    ax.tick_params(axis="x", rotation=0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("% change vs Original (per-view total counts)")
    ax.set_title(f"{patient}: count change vs original (BM3D / {args.label_b} / {args.label_a})")
    ax.grid(True, axis="y", alpha=0.25)

    # Legend (simple, color by method)
    colors = ["#4C78A8", "#F58518", "#54A24B"]  # bm3d, B, A
    for i, box in enumerate(bp["boxes"]):
        box.set_color(colors[i % 3])
        box.set_linewidth(1.3)
    for i, med in enumerate(bp["medians"]):
        med.set_color(colors[i % 3])
        med.set_linewidth(1.8)
    ax.plot([], [], color=colors[0], label="BM3D")
    ax.plot([], [], color=colors[1], label=str(args.label_b))
    ax.plot([], [], color=colors[2], label=str(args.label_a))
    ax.legend(loc="upper right", frameon=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote: {out}")
    print(f"[OK] N samples per dose per method: {int(y20i.shape[0])} (doses={len(base_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
