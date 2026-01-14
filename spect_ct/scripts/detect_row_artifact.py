#!/usr/bin/env python
"""
Detect a horizontal "dark line" artifact across views in SPECT projection volumes.

We quantify, per view, the row-wise mean profile and measure a local dip score:
  dip(r) = mean(neighbors around r) - mean(row r)

Then we aggregate across views to see if a consistent row index shows a strong dip
(e.g. near the top ~5% of height).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def load_proj_u16(path: str, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    expected = int(views) * int(h) * int(w)
    if arr.size != expected:
        raise ValueError(f"Invalid projection size: {path}. Expected {expected} uint16 values, got {arr.size}.")
    return arr.reshape(int(views), int(h), int(w)).astype(np.float32, copy=False)


def build_net_from_yaml(yml_path: str) -> Tuple[torch.nn.Module, float, Dict]:
    import yaml
    from basicsr.archs import build_network

    with open(yml_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    net_opt = opt.get("network_g", None)
    if not isinstance(net_opt, dict):
        raise ValueError(f"Missing network_g in yml: {yml_path}")
    net = build_network(net_opt)
    # use train dataset max_value as default
    mv = None
    try:
        mv = float((((opt.get("datasets") or {}).get("train") or {}).get("max_value")))
    except Exception:
        mv = None
    if mv is None or mv < 1e-6:
        mv = 150.0
    return net, float(mv), opt


@torch.no_grad()
def infer_denoised_count(
    proj_count: np.ndarray,
    *,
    net: torch.nn.Module,
    max_value: float,
    device: str,
) -> np.ndarray:
    # proj_count: (V,H,W) float32 in count domain
    mv = float(max(max_value, 1e-6))
    x = (proj_count / mv).astype(np.float32, copy=False)
    xt = torch.from_numpy(x[None, None, ...]).to(device=device, dtype=torch.float32)  # (1,1,V,H,W)
    net = net.to(device)
    net.eval()
    yt = net(xt)
    yt = torch.clamp(yt, min=0.0)
    y = yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
    return y


def row_dip_score(
    vol: np.ndarray,
    *,
    win: int = 2,
) -> np.ndarray:
    """Return per-view per-row dip score: (V,H)"""
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected (V,H,W), got {vol.shape}")
    v, h, w = vol.shape
    # row mean per view
    rm = vol.mean(axis=2)  # (V,H)
    # neighbor mean using a window excluding center row
    dip = np.zeros((v, h), dtype=np.float32)
    for r in range(h):
        lo = max(0, r - win)
        hi = min(h, r + win + 1)
        idx = [i for i in range(lo, hi) if i != r]
        if len(idx) == 0:
            dip[:, r] = 0.0
        else:
            nbr = rm[:, idx].mean(axis=1)
            dip[:, r] = (nbr - rm[:, r])
    # normalize by local magnitude to make it comparable
    denom = np.maximum(np.abs(rm), 1.0)
    return dip / denom


def summarize_dip(dip_vh: np.ndarray, *, top_frac: float = 0.2) -> Dict:
    dip_vh = np.asarray(dip_vh, dtype=np.float32)
    v, h = dip_vh.shape
    rmax = int(max(1, round(float(top_frac) * h)))
    rmax = min(rmax, h)
    region = dip_vh[:, :rmax]  # (V, rmax)
    # per-view best row in top region
    best_r = region.argmax(axis=1)  # (V,)
    best_val = region.max(axis=1)  # (V,)

    # robust consensus row
    r_med = int(np.median(best_r))
    # how many views agree within +/-1 row
    agree = float(np.mean(np.abs(best_r - r_med) <= 1))

    return {
        "views": int(v),
        "height": int(h),
        "top_rows": int(rmax),
        "median_row": int(r_med),
        "agree_frac_pm1": float(agree),
        "best_val_mean": float(best_val.mean()),
        "best_val_median": float(np.median(best_val)),
        "best_row_hist": np.bincount(best_r, minlength=rmax).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Path to *_Proj4Filter.dat (uint16, VxHxW).")
    ap.add_argument("--yml", required=True, help="YAML config containing network_g (to build net).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pth) to load (use params_ema if present).")
    ap.add_argument("--out_dir", required=True, help="Output directory for report/plots.")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--param_key", default="params_ema", help="params_ema|params")
    ap.add_argument("--view_stride", type=int, default=1, help="Subsample views before analysis (e.g. 2 -> 30 views).")
    ap.add_argument("--top_frac", type=float, default=0.2, help="Only search top fraction of rows for artifact.")
    ap.add_argument("--win", type=int, default=2, help="Neighbor window for dip score (rows).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proj = load_proj_u16(args.proj, views=60, h=128, w=128)
    if args.view_stride and args.view_stride > 1:
        proj = proj[:: int(args.view_stride)]

    net, mv, _ = build_net_from_yaml(args.yml)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and args.param_key in ckpt:
        sd = ckpt[args.param_key]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        sd = ckpt["params"]
    else:
        sd = ckpt
    # strip module.
    sd2 = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        sd2[k] = v
    net.load_state_dict(sd2, strict=True)

    den = infer_denoised_count(proj, net=net, max_value=mv, device=args.device)

    dip_orig = row_dip_score(proj, win=int(args.win))
    dip_den = row_dip_score(den, win=int(args.win))

    s0 = summarize_dip(dip_orig, top_frac=float(args.top_frac))
    s1 = summarize_dip(dip_den, top_frac=float(args.top_frac))

    # Save quick numeric report
    import json

    report = {
        "proj": str(args.proj),
        "ckpt": str(args.ckpt),
        "yml": str(args.yml),
        "device": str(args.device),
        "param_key": str(args.param_key),
        "view_stride": int(args.view_stride),
        "max_value": float(mv),
        "dip_top_frac": float(args.top_frac),
        "dip_win": int(args.win),
        "orig": s0,
        "denoised": s1,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save a diagnostic plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_heat(ax, dip, title: str):
            ax.imshow(dip, aspect="auto", cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("row")
            ax.set_ylabel("view")

        fig = plt.figure(figsize=(12, 6), dpi=140)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        plot_heat(ax1, dip_orig, f"orig dip (view_stride={int(args.view_stride)})")
        plot_heat(ax2, dip_den, "denoised dip")
        fig.tight_layout()
        fig.savefig(out_dir / "dip_heat.png")
        plt.close(fig)

        # Also plot median-row profiles around the suspicious area
        r0 = int(s0["median_row"])
        r1 = int(s1["median_row"])
        h = int(dip_orig.shape[1])
        r_lo = max(0, min(r0, r1) - 8)
        r_hi = min(h, max(r0, r1) + 9)
        x = np.arange(r_lo, r_hi)
        m0 = dip_orig[:, r_lo:r_hi].mean(axis=0)
        m1 = dip_den[:, r_lo:r_hi].mean(axis=0)
        fig = plt.figure(figsize=(10, 3), dpi=140)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, m0, label="orig mean dip")
        ax.plot(x, m1, label="den mean dip")
        ax.axvline(r0, color="k", lw=1, ls="--", alpha=0.5)
        ax.axvline(r1, color="k", lw=1, ls=":", alpha=0.5)
        ax.set_title("Mean dip profile around detected row (higher = darker line)")
        ax.set_xlabel("row")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dip_profile.png")
        plt.close(fig)
    except Exception:
        pass

    print("Wrote:", str(out_dir / "report.json"))
    print("orig:", s0)
    print("denoised:", s1)


if __name__ == "__main__":
    main()














