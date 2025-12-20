from __future__ import annotations

import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_poisson_fit(
    noise: np.ndarray = None,
    gt: np.ndarray = None,
    img: np.ndarray = None,
    img2: np.ndarray = None,
    input_order: str = 'HWC',
    eps: float = 1e-6,
    bin_size: float = 1.0,
    bin_center: str = 'nearest_int',
    min_pixels_per_bin: int = 256,
    min_gt: float = 0.0,
    **kwargs,
) -> float:
    """Poisson conformity of predicted noise using GT as λ, computed by binning pixels.

    For a Poisson observation y ~ Pois(λ):
    - Residual noise (y - λ) has Var(y-λ) = Var(y) = λ, and mean ≈ 0.

    Here we take predicted noise = (LQ - Pred) in *count domain* and use GT as λ proxy.
    We group pixels by their GT intensity (binning with `bin_size`) across the whole image
    (and across both views/channels if present), then for each bin compute:
      - var_bin = Var(noise | GT in bin)
      - mean_gt_bin = Mean(GT | GT in bin)
    and measure relative deviation: |var_bin - mean_gt_bin| / (mean_gt_bin + eps).

    The final score is the pixel-count-weighted mean of the per-bin deviations.
    Lower is better; 0 means perfect Poisson conformity.
    """
    # Support both explicit names and img/img2 aliases
    if noise is None:
        noise = img
    if gt is None:
        gt = img2

    if noise is None or gt is None:
        raise ValueError('Must provide noise/img and gt/img2')

    noise = np.asarray(noise, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if input_order != 'HWC':
        raise ValueError(f'input_order={input_order} not supported')

    # Flatten across all pixels (and channels/views if present)
    if noise.ndim == 3:
        noise_flat = noise.reshape(-1)
        gt_flat = gt.reshape(-1)
    elif noise.ndim == 2:
        noise_flat = noise.reshape(-1)
        gt_flat = gt.reshape(-1)
    else:
        raise ValueError(f'noise ndim={noise.ndim} not supported')

    if bin_size <= 0:
        raise ValueError('bin_size must be > 0')

    # Valid mask: use GT as λ proxy; require it to be non-negative and >= min_gt
    gt_flat = np.clip(gt_flat, 0.0, None)
    valid = gt_flat >= float(min_gt)
    if not np.any(valid):
        # No valid pixels -> worst (cannot evaluate)
        return float('inf')

    gt_v = gt_flat[valid]
    noise_v = noise_flat[valid]

    # Bin by GT intensity
    # - nearest_int: bins centered at integer k -> [k-0.5, k+0.5) when bin_size=1
    # - left_edge:   bins as [k, k+1) when bin_size=1
    bs = float(bin_size)
    if bin_center not in ('nearest_int', 'left_edge'):
        raise ValueError("bin_center must be 'nearest_int' or 'left_edge'")
    if bin_center == 'nearest_int':
        bin_idx = np.floor((gt_v / bs) + 0.5).astype(np.int64)
    else:
        bin_idx = np.floor(gt_v / bs).astype(np.int64)
    if bin_idx.size == 0:
        return float('inf')

    nbins = int(bin_idx.max()) + 1
    counts = np.bincount(bin_idx, minlength=nbins).astype(np.int64)
    sum_gt = np.bincount(bin_idx, weights=gt_v, minlength=nbins).astype(np.float64)
    sum_noise = np.bincount(bin_idx, weights=noise_v, minlength=nbins).astype(np.float64)
    sum_noise2 = np.bincount(bin_idx, weights=noise_v * noise_v, minlength=nbins).astype(np.float64)

    # Per-bin mean/var of noise
    mean_gt = sum_gt / np.maximum(counts, 1)
    mean_noise = sum_noise / np.maximum(counts, 1)
    mean_noise2 = sum_noise2 / np.maximum(counts, 1)
    var_noise = np.maximum(mean_noise2 - mean_noise * mean_noise, 0.0)

    # Keep bins with enough pixels and positive mean_gt
    keep = (counts >= int(min_pixels_per_bin)) & (mean_gt > eps)
    if not np.any(keep):
        return float('inf')

    rel_err = np.abs(var_noise[keep] - mean_gt[keep]) / (mean_gt[keep] + eps)
    weights = counts[keep].astype(np.float64)
    score = float(np.sum(rel_err * weights) / np.sum(weights))
    return score

