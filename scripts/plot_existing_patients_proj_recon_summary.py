#!/usr/bin/env python3
from __future__ import annotations

"""
Summarize existing patients' projection+reconstruction totals across dose levels and models.

This script reads already-generated artifacts under:
  outputs/compare_two_models/<patient>/
    projections/{label}_{method}_f32.dat
    reconstructions/<patient>_{label}_{method}_OSEMIter{iters}.dat

It produces:
  - A 2-row figure:
      row1: projection total counts % change vs ORIGINAL (per-dose, across patients)
      row2: reconstruction total activity % change vs ORIGINAL (per-dose, across patients)
  - A CSV with one row per (patient, dose, method, modality).

Notes:
  - Projection "orig/bm3d/poi" totals are computed like training: round+clip then sum.
  - Projection "ema_*" totals are computed by direct float sum (no rounding).
  - Reconstruction totals are clip then sum (float).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from prime.pipeline.io import load_projection_f32, load_recon_f32
from prime.pipeline.val_cache import counts_int as _counts_int


def _counts_int_sum(x: np.ndarray) -> float:
    return float(np.sum(_counts_int(x).astype(np.float64, copy=False)))


def _counts_float_sum(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64, copy=False)))


def _recon_sum(x: np.ndarray) -> float:
    y = np.clip(x.astype(np.float64, copy=False), 0.0, None)
    return float(np.sum(y))


def _pct(method: float, base: float, eps: float = 1e-12) -> float:
    b = float(base)
    if b <= eps:
        return 0.0
    return (float(method) - b) / b * 100.0


def _dose_name(label: str, k: float) -> str:
    if label == "20s":
        return "20s"
    sec = 20.0 / float(max(k, 1e-12))
    return f"{sec:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/compare_two_models", help="Root results dir")
    ap.add_argument("--osem-iters", type=int, default=10, help="Which OSEM iter to look for in filenames")
    ap.add_argument("--out-png", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--methods", type=str, default="bm3d,ema_a,ema_b,poi", help="Comma list of methods to include (orig is baseline)")
    ap.add_argument("--doses", type=str, default="20s,x2,x3,x4,x5", help="Comma list of dose labels to include")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    doses = [d.strip() for d in str(args.doses).split(",") if d.strip()]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    patients = sorted([p for p in root.iterdir() if p.is_dir() and (p / "projections").exists()])
    if not patients:
        raise RuntimeError(f"No patients with projections/ found under: {root}")

    rows_csv: list[dict[str, object]] = []

    # Collect per (dose, method) across patients for two modalities
    proj_pct: dict[tuple[str, str], list[float]] = {}
    recon_pct: dict[tuple[str, str], list[float]] = {}

    for pdir in patients:
        pname = pdir.name
        proj_dir = pdir / "projections"
        rec_dir = pdir / "reconstructions"

        def load_proj(label: str, method: str) -> np.ndarray:
            f = proj_dir / f"{label}_{method}_f32.dat"
            if not f.exists():
                raise FileNotFoundError(f)
            return load_projection_f32(f)

        def load_recon(label: str, method: str) -> np.ndarray:
            # accept both ema and ema_a/ema_b naming in cache
            method_cands = [str(method)]
            if str(method) == "ema":
                method_cands.extend(["ema_a", "ema_b"])
            elif str(method) in {"ema_a", "ema_b"}:
                method_cands.append("ema")
            method_cands = list(dict.fromkeys(method_cands))
            cands = [rec_dir / f"{pname}_{label}_{m}_OSEMIter{int(args.osem_iters)}.dat" for m in method_cands]
            f = next((x for x in cands if x.exists()), None)
            if f is None:
                raise FileNotFoundError(f"Missing recon for {pname} {label} {method} under {rec_dir}")
            return load_recon_f32(f, shape=(128, 128, 128))

        # base totals per dose from original
        base_proj_tot: dict[str, float] = {}
        base_rec_tot: dict[str, float] = {}

        for d in doses:
            orig_proj = load_proj(d, "orig")
            base_proj_tot[d] = _counts_int_sum(orig_proj)
            # recon may be missing for some patients/doses; skip if missing
            try:
                orig_rec = load_recon(d, "orig")
                base_rec_tot[d] = _recon_sum(orig_rec)
            except FileNotFoundError:
                base_rec_tot[d] = float("nan")

        for d in doses:
            # infer k from label
            k = 1.0 if d == "20s" else float(d[1:])  # x2 -> 2
            dose_str = _dose_name(d, k)
            bproj = base_proj_tot[d]
            brec = base_rec_tot[d]

            for m in methods:
                # projection totals
                try:
                    x = load_proj(d, m)
                except FileNotFoundError:
                    continue
                if m.startswith("ema"):
                    tot = _counts_float_sum(x)
                else:
                    tot = _counts_int_sum(x)
                pctv = _pct(tot, bproj)
                proj_pct.setdefault((dose_str, m), []).append(pctv)
                rows_csv.append(
                    dict(patient=pname, modality="proj", dose=dose_str, label=d, k=k, method=m, total=tot, base=bproj, pct=pctv)
                )

                # reconstruction totals (if available)
                if not np.isfinite(brec):
                    continue
                try:
                    rvol = load_recon(d, m)
                except FileNotFoundError:
                    continue
                rtot = _recon_sum(rvol)
                rpct = _pct(rtot, brec)
                recon_pct.setdefault((dose_str, m), []).append(rpct)
                rows_csv.append(
                    dict(patient=pname, modality="recon", dose=dose_str, label=d, k=k, method=m, total=rtot, base=brec, pct=rpct)
                )

    # Write CSV
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.out_png).with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "modality", "dose", "label", "k", "method", "total", "base", "pct"])
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    # Plot: grouped boxplots
    dose_order = []
    for d in doses:
        k = 1.0 if d == "20s" else float(d[1:])
        dose_order.append(_dose_name(d, k))

    method_order = methods
    colors = {
        "bm3d": "#4C78A8",
        "ema_b": "#F58518",
        "ema_a": "#54A24B",
        "poi": "#B279A2",
    }

    def plot_ax(ax, data_dict, title: str):
        group = len(method_order)
        xs = []
        ys = []
        for i, dose in enumerate(dose_order):
            base_x = i * (group + 1)
            for j, m in enumerate(method_order):
                xs.append(base_x + j)
                ys.append(data_dict.get((dose, m), []))
        bp = ax.boxplot(ys, positions=xs, widths=0.75, showfliers=True, whis=1.5)
        for idx, box in enumerate(bp["boxes"]):
            m = method_order[idx % group]
            box.set_color(colors.get(m, "black"))
            box.set_linewidth(1.2)
        for idx, med in enumerate(bp["medians"]):
            m = method_order[idx % group]
            med.set_color(colors.get(m, "black"))
            med.set_linewidth(1.6)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        tick_pos = [i * (group + 1) + (group - 1) / 2.0 for i in range(len(dose_order))]
        ax.set_xticks(tick_pos, dose_order)
        ax.set_title(title)
        ax.set_ylabel("% change vs original")
        ax.grid(True, axis="y", alpha=0.25)

    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plot_ax(ax1, proj_pct, f"Projection total counts change (across patients, n={len(patients)})")
    plot_ax(ax2, recon_pct, f"Reconstruction total activity change (across patients, n={len(patients)})")

    # legend
    handles = []
    labels = []
    for m in method_order:
        handles.append(plt.Line2D([0], [0], color=colors.get(m, "black"), lw=3))
        labels.append(m)
    ax1.legend(handles, labels, loc="upper right", frameon=True, title="method")

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] patients: {len(patients)}")
    print(f"[OK] wrote png: {out_png}")
    print(f"[OK] wrote csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








