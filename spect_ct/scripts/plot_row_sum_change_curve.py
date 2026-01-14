#!/usr/bin/env python
"""
Plot per-row total-count change before/after denoising.

Given a projection volume shaped (V,H,W):
  row_sum[r] = sum_{v,w} proj[v,r,w]

We compute row_sum for:
  - orig (observed counts)
  - denoised (model output, count domain)
and plot:
  - row_sum_orig and row_sum_den
  - delta = den - orig
  - relative delta = delta / max(orig, 1)
"""

from __future__ import annotations

import argparse
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
    mv = None
    try:
        mv = float((((opt.get("datasets") or {}).get("train") or {}).get("max_value")))
    except Exception:
        mv = None
    if mv is None or mv < 1e-6:
        mv = 150.0
    return net, float(mv), opt


def load_state_dict(ckpt_path: str, param_key: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and param_key in ckpt:
        sd = ckpt[param_key]
    elif isinstance(ckpt, dict) and "params" in ckpt:
        sd = ckpt["params"]
    else:
        sd = ckpt
    sd2 = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        sd2[k] = v
    return sd2


@torch.no_grad()
def infer_denoised_count(
    proj_count: np.ndarray,
    *,
    net: torch.nn.Module,
    max_value: float,
    device: str,
) -> np.ndarray:
    mv = float(max(max_value, 1e-6))
    x = (proj_count / mv).astype(np.float32, copy=False)
    xt = torch.from_numpy(x[None, None, ...]).to(device=device, dtype=torch.float32)  # (1,1,V,H,W)
    net = net.to(device)
    net.eval()
    yt = net(xt)
    yt = torch.clamp(yt, min=0.0)
    y = yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Path to *_Proj4Filter.dat")
    ap.add_argument("--yml", required=True, help="YAML config containing network_g")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pth")
    ap.add_argument("--param_key", default="params_ema", help="params_ema|params")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="", help="Plot title")
    ap.add_argument("--logy", action="store_true", help="Log scale for row_sum plot")
    ap.add_argument("--topk", type=int, default=15, help="Print top-K rows by |delta| (excluding zero rows)")
    args = ap.parse_args()

    proj = load_proj_u16(args.proj, views=60, h=128, w=128)  # (V,H,W)
    net, mv, _ = build_net_from_yaml(args.yml)
    sd = load_state_dict(args.ckpt, str(args.param_key))
    net.load_state_dict(sd, strict=True)
    device = str(args.device).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    den = infer_denoised_count(proj, net=net, max_value=mv, device=device)

    row_sum_o = proj.astype(np.float64, copy=False).sum(axis=(0, 2))  # (H,)
    row_sum_d = den.astype(np.float64, copy=False).sum(axis=(0, 2))
    delta = row_sum_d - row_sum_o
    rel = delta / np.maximum(row_sum_o, 1.0)

    # Exclude fully-zero rows for ranking
    nz = row_sum_o > 0
    idx = np.arange(row_sum_o.shape[0], dtype=np.int64)
    idx_nz = idx[nz]
    if idx_nz.size > 0:
        order = np.argsort(np.abs(delta[idx_nz]))[::-1]
        topk = int(max(1, args.topk))
        sel = idx_nz[order[:topk]]
        print("Top rows by |delta| (exclude zero rows):")
        for r in sel.tolist():
            print(
                f"  row {r:3d}: orig={row_sum_o[r]:.0f} den={row_sum_d[r]:.0f} "
                f"delta={delta[r]:+.0f} rel={rel[r]:+.4f}"
            )
    else:
        print("All rows are zero? Unexpected.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(row_sum_o.shape[0], dtype=np.int64)
    fig = plt.figure(figsize=(10, 7), dpi=160)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    ax1.plot(x, row_sum_o, color="black", lw=1.2, label="orig")
    ax1.plot(x, row_sum_d, color="#d62728", lw=1.0, label="denoised")
    ax1.set_ylabel("row_sum")
    if args.logy:
        ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=8)
    if args.title:
        ax1.set_title(str(args.title))

    ax2.plot(x, delta, color="#1f77b4", lw=1.0)
    ax2.axhline(0.0, lw=0.8, color="gray", alpha=0.6)
    ax2.set_ylabel("delta (den-orig)")

    ax3.plot(x, rel, color="#2ca02c", lw=1.0)
    ax3.axhline(0.0, lw=0.8, color="gray", alpha=0.6)
    ax3.set_ylabel("rel delta")
    ax3.set_xlabel("row (0=top)")

    # mark zero rows
    zero_rows = np.where(row_sum_o == 0)[0]
    for ax in (ax1, ax2, ax3):
        for r in zero_rows.tolist():
            ax.axvline(r, color="gray", lw=0.3, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()














