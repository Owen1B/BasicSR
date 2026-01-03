from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class ReconMetrics:
    psnr: float
    ssim: float
    data_range: float


def _as_hwc_rgb_from_gray(img2d: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale image to HWC RGB (repeat channels) for LPIPS."""
    x = np.asarray(img2d, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D image, got shape={x.shape}")
    return np.repeat(x[:, :, None], 3, axis=2)


def lpips_views_mean(
    *,
    pred: np.ndarray,
    ref: np.ndarray,
    max_value: float = 150.0,
    nets: Iterable[str] = ("alex", "vgg"),
    device: Optional[str] = None,
    view_stride: int = 1,
) -> dict[str, float]:
    """LPIPS averaged over views (projection domain).

    - pred/ref: (V,H,W) in count domain.
    - Normalization to [0,1] uses fixed max_value (not per-patient).
    """
    from basicsr.metrics import calculate_lpips
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mv = float(max_value)
    if mv <= 0:
        mv = 1.0
    nets2 = [str(x).lower() for x in nets]
    stride = max(1, int(view_stride))

    out: dict[str, float] = {}
    for net in nets2:
        vals = []
        for i in range(0, int(pred.shape[0]), stride):
            a = np.clip(ref[i], 0.0, mv) / mv
            b = np.clip(pred[i], 0.0, mv) / mv
            vals.append(calculate_lpips(_as_hwc_rgb_from_gray(b), _as_hwc_rgb_from_gray(a), input_order="HWC", net=net, device=device))
        out[net] = float(np.mean(vals)) if len(vals) else float("nan")
    return out


def lpips_recon_slices_mean(
    *,
    pred: np.ndarray,
    ref: np.ndarray,
    data_range: Optional[float] = None,
    axis: int = 2,
    nets: Iterable[str] = ("alex", "vgg"),
    device: Optional[str] = None,
    slice_stride: int = 1,
) -> dict[str, float]:
    """LPIPS averaged over 2D slices in recon domain.

    This avoids the "only on MIP" shortcut and is less sensitive to single-slice artifacts.

    - pred/ref: 3D volumes (Z,Y,X) float32.
    - axis: which axis to slice over (default 2 -> 128 slices of 128x128 if shape is 128^3).
    - Normalization: clip to [0,data_range] then divide by data_range to [0,1] before LPIPS.
    """
    from basicsr.metrics import calculate_lpips
    import torch

    p = np.asarray(pred, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: pred={p.shape}, ref={r.shape}")
    if p.ndim != 3:
        raise ValueError(f"expected 3D recon volumes, got pred.shape={p.shape}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_range is None:
        data_range = float(np.max(r)) if np.isfinite(r).any() else 1.0
        if data_range <= 0:
            data_range = 1.0
    dr = float(data_range)

    ax = int(axis)
    stride = max(1, int(slice_stride))
    nets2 = [str(x).lower() for x in nets]

    out: dict[str, float] = {}
    n_slices = int(p.shape[ax])
    for net in nets2:
        vals = []
        for si in range(0, n_slices, stride):
            ps = np.take(p, si, axis=ax)
            rs = np.take(r, si, axis=ax)
            # Normalize to [0,1]
            ps = np.clip(ps, 0.0, dr) / dr
            rs = np.clip(rs, 0.0, dr) / dr
            vals.append(
                calculate_lpips(
                    _as_hwc_rgb_from_gray(ps),
                    _as_hwc_rgb_from_gray(rs),
                    input_order="HWC",
                    net=net,
                    device=device,
                )
            )
        out[net] = float(np.mean(vals)) if len(vals) else float("nan")
    return out


def psnr_ssim_recon(
    *,
    pred: np.ndarray,
    ref: np.ndarray,
    crop_border: int = 0,
    data_range: Optional[float] = None,
) -> ReconMetrics:
    """PSNR/SSIM in recon domain, computed on a fixed 2D MIP image (axis=2).

    We keep metrics in linear domain (no log1p), per your choice "viz only".
    """
    from spect_ct.pipeline.viz import compute_mip
    # Prefer BasicSR metrics (cv2-based) in KAIR env; fallback to numpy implementation if unavailable.
    try:
        from basicsr.metrics import calculate_psnr_range, calculate_ssim_range  # type: ignore
        _use_basicsr = True
    except Exception:
        _use_basicsr = False
        from spect_ct.pipeline.simple_metrics import psnr_range, ssim_range

    p = np.asarray(pred, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: pred={p.shape}, ref={r.shape}")
    if p.ndim != 3:
        raise ValueError(f"expected 3D recon volumes, got pred.shape={p.shape}")

    # Use deterministic MIP (no rotation): (Z,Y,X) -> MIP over X => (Z,Y)
    p_mip = compute_mip(p, axis=2).astype(np.float32, copy=False)
    r_mip = compute_mip(r, axis=2).astype(np.float32, copy=False)

    if data_range is None:
        data_range = float(np.max(r_mip)) if np.isfinite(r_mip).any() else 1.0
        if data_range <= 0:
            data_range = 1.0

    # crop_border is kept for API parity (rarely used here).
    cb = int(crop_border)
    if cb > 0:
        p_eval = p_mip[cb:-cb, cb:-cb]
        r_eval = r_mip[cb:-cb, cb:-cb]
    else:
        p_eval = p_mip
        r_eval = r_mip
    if _use_basicsr:
        # Metrics expect HWC; use 1-channel HWC.
        p_hwc = np.clip(p_eval, 0.0, float(data_range))[:, :, None]
        r_hwc = np.clip(r_eval, 0.0, float(data_range))[:, :, None]
        psnr = float(calculate_psnr_range(p_hwc, r_hwc, crop_border=0, data_range=float(data_range), input_order="HWC"))
        ssim = float(calculate_ssim_range(p_hwc, r_hwc, crop_border=0, data_range=float(data_range), input_order="HWC"))
    else:
        psnr = float(psnr_range(np.clip(p_eval, 0.0, float(data_range)), np.clip(r_eval, 0.0, float(data_range)), data_range=float(data_range)))
        ssim = float(ssim_range(np.clip(p_eval, 0.0, float(data_range)), np.clip(r_eval, 0.0, float(data_range)), data_range=float(data_range)))
    return ReconMetrics(psnr=psnr, ssim=ssim, data_range=float(data_range))


@dataclass
class PoissonBinStats:
    bin_edges: np.ndarray
    n: np.ndarray
    sum_lam: np.ndarray
    sum_r: np.ndarray
    sum_r2: np.ndarray

    @classmethod
    def create(cls, *, bin_edges: np.ndarray) -> "PoissonBinStats":
        be = np.asarray(bin_edges, dtype=np.float64)
        m = int(len(be))
        z = np.zeros(m, dtype=np.float64)
        return cls(bin_edges=be, n=z.copy(), sum_lam=z.copy(), sum_r=z.copy(), sum_r2=z.copy())

    def update(self, lam: np.ndarray, y: np.ndarray) -> None:
        lam_f = np.asarray(lam, dtype=np.float64).reshape(-1)
        y_f = np.asarray(y, dtype=np.float64).reshape(-1)
        r = y_f - lam_f
        idx = np.digitize(lam_f, self.bin_edges, right=False) - 1
        idx = np.clip(idx, 0, len(self.bin_edges) - 1).astype(np.int64, copy=False)
        self.n += np.bincount(idx, minlength=len(self.bin_edges)).astype(np.float64, copy=False)
        self.sum_lam += np.bincount(idx, weights=lam_f, minlength=len(self.bin_edges)).astype(np.float64, copy=False)
        self.sum_r += np.bincount(idx, weights=r, minlength=len(self.bin_edges)).astype(np.float64, copy=False)
        self.sum_r2 += np.bincount(idx, weights=r * r, minlength=len(self.bin_edges)).astype(np.float64, copy=False)

    def finalize(self, *, min_count: int = 20000) -> dict[str, np.ndarray]:
        n = self.n
        valid = n >= float(min_count)
        mean_lam = np.zeros_like(n)
        mean_r = np.zeros_like(n)
        var_r = np.zeros_like(n)
        mean_lam[valid] = self.sum_lam[valid] / n[valid]
        mean_r[valid] = self.sum_r[valid] / n[valid]
        var_r[valid] = self.sum_r2[valid] / n[valid] - mean_r[valid] ** 2
        ratio = np.zeros_like(n)
        ratio[valid] = var_r[valid] / np.maximum(mean_lam[valid], 1e-8)
        centers = self.bin_edges.astype(np.float64)
        return {
            "bin_center": centers,
            "n": n,
            "valid": valid.astype(np.int32),
            "mean_lambda": mean_lam,
            "mean_residual": mean_r,
            "var_residual": var_r,
            "ratio_var_over_mean": ratio,
        }


