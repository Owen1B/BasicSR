#!/usr/bin/env python
"""
Visualize SPECT "sinogram-like" slices before/after denoising.

Your data is a projection volume shaped (V, H, W) where V=number of views.
To create a sinogram-like 2D image, we take a fixed row r and plot:
  S(r) = vol[:, r, :]  -> shape (V, W)

This makes it easy to spot a horizontal artifact line (fixed r) across views.
We output:
  - orig sinogram
  - denoised sinogram
  - (den - orig) difference

We also optionally plot a "mean-over-H" sinogram: vol.mean(axis=1) -> (V,W).
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
    # train dataset max_value as default
    mv = None
    try:
        mv = float((((opt.get("datasets") or {}).get("train") or {}).get("max_value")))
    except Exception:
        mv = None
    if mv is None or mv < 1e-6:
        mv = 150.0
    return net, float(mv), opt


def load_state_dict(ckpt_path: str, param_key: str = "params_ema") -> Dict[str, torch.Tensor]:
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


def _norm_vis(x: np.ndarray, *, log1p: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    return np.log1p(x) if log1p else x


def save_fig(
    *,
    out_path: Path,
    orig: np.ndarray,
    den: np.ndarray,
    title: str,
    log1p: bool,
    scale_mode: str,
    p_low: float,
    p_high: float,
    cmap_main: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = _norm_vis(orig, log1p=log1p)
    d = _norm_vis(den, log1p=log1p)
    diff = d - o

    p_low = float(p_low)
    p_high = float(p_high)
    if p_high <= p_low:
        p_low, p_high = 1.0, 99.5

    def _robust_range(x: np.ndarray) -> Tuple[float, float]:
        xf = x.reshape(-1).astype(np.float32, copy=False)
        lo = float(np.percentile(xf, p_low))
        hi = float(np.percentile(xf, p_high))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo + 1e-12:
            hi = float(np.max(xf)) if xf.size else 1.0
        if hi <= lo + 1e-12:
            hi = lo + 1.0
        return lo, hi

    # Scale: shared (fair) vs per_panel (more visible detail)
    if str(scale_mode).lower() == "per_panel":
        vmin_o, vmax_o = _robust_range(o)
        vmin_d, vmax_d = _robust_range(d)
    else:
        vmin_o, vmax_o = _robust_range(np.concatenate([o.reshape(-1), d.reshape(-1)]))
        vmin_d, vmax_d = vmin_o, vmax_o
    # Diff scale
    dv = float(np.percentile(np.abs(diff).reshape(-1), max(50.0, p_high)))
    if dv < 1e-6:
        dv = float(np.max(np.abs(diff)))
    if dv < 1e-6:
        dv = 1.0

    # Keep the displayed x/y ratio faithful to the data shape (V x W).
    # With aspect='equal', the panel width/height will be ~ W/V.
    v, w = int(o.shape[0]), int(o.shape[1])
    ratio = float(w) / float(max(v, 1))
    base_h = 4.0
    panel_w = base_h * ratio
    fig_w = max(10.0, 3.0 * panel_w)  # 3 panels
    fig = plt.figure(figsize=(fig_w, base_h), dpi=160)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(o, aspect="equal", cmap=str(cmap_main), vmin=vmin_o, vmax=vmax_o, interpolation="nearest")
    ax1.set_title("orig")
    ax1.set_xlabel("col (W)")
    ax1.set_ylabel("view (V)")

    ax2.imshow(d, aspect="equal", cmap=str(cmap_main), vmin=vmin_d, vmax=vmax_d, interpolation="nearest")
    ax2.set_title("denoised")
    ax2.set_xlabel("col (W)")
    ax2.set_ylabel("view (V)")

    ax3.imshow(diff, aspect="equal", cmap="bwr", vmin=-dv, vmax=dv, interpolation="nearest")
    ax3.set_title("diff (den-orig)")
    ax3.set_xlabel("col (W)")
    ax3.set_ylabel("view (V)")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Path to *_Proj4Filter.dat (uint16, VxHxW).")
    ap.add_argument("--yml", required=True, help="YAML config containing network_g (to build net).")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (.pth) to load.")
    ap.add_argument("--param_key", default="params_ema", help="params_ema|params")
    ap.add_argument("--device", default="cuda", help="cuda|cpu")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--rows", default="17,6", help="Comma-separated row indices to visualize (sinogram vol[:,row,:]).")
    ap.add_argument("--log1p", action="store_true", help="Use log1p for visualization.")
    ap.add_argument("--scale", default="per_panel", choices=["per_panel", "shared"], help="Color scaling mode.")
    ap.add_argument("--p_low", type=float, default=1.0, help="Lower percentile for robust scaling.")
    ap.add_argument("--p_high", type=float, default=99.5, help="Upper percentile for robust scaling.")
    ap.add_argument("--cmap", default="gray", help="Colormap for orig/den panels (e.g., gray, magma).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    proj = load_proj_u16(args.proj, views=60, h=128, w=128)

    net, mv, _ = build_net_from_yaml(args.yml)
    sd = load_state_dict(args.ckpt, param_key=str(args.param_key))
    net.load_state_dict(sd, strict=True)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    den = infer_denoised_count(proj, net=net, max_value=mv, device=device)

    # Parse rows
    rows = []
    for tok in str(args.rows).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            rows.append(int(tok))
        except Exception:
            pass
    if not rows:
        rows = [17, 6]

    # Save per-row sinograms
    base = Path(args.proj).stem
    for r in rows:
        r0 = int(max(0, min(int(proj.shape[1]) - 1, r)))
        s_orig = proj[:, r0, :]  # (V,W)
        s_den = den[:, r0, :]
        title = f"{base} row={r0}  (sinogram=vol[:,row,:])  log1p={bool(args.log1p)}"
        save_fig(
            out_path=out_dir / f"{base}_row{r0:03d}_sinogram_scale{str(args.scale)}.png",
            orig=s_orig,
            den=s_den,
            title=title,
            log1p=bool(args.log1p),
            scale_mode=str(args.scale),
            p_low=float(args.p_low),
            p_high=float(args.p_high),
            cmap_main=str(args.cmap),
        )

    # Also a mean-over-H sinogram (V,W)
    s_orig_m = proj.mean(axis=1)
    s_den_m = den.mean(axis=1)
    save_fig(
        out_path=out_dir / f"{base}_meanH_sinogram_scale{str(args.scale)}.png",
        orig=s_orig_m,
        den=s_den_m,
        title=f"{base} mean-over-H  (sinogram=mean_y vol)  log1p={bool(args.log1p)}",
        log1p=bool(args.log1p),
        scale_mode=str(args.scale),
        p_low=float(args.p_low),
        p_high=float(args.p_high),
        cmap_main=str(args.cmap),
    )

    print("Wrote images to:", str(out_dir))


if __name__ == "__main__":
    main()


