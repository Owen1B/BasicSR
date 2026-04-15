from __future__ import annotations

import numpy as np


def psnr_range(pred: np.ndarray, ref: np.ndarray, *, data_range: float) -> float:
    """PSNR for arrays in [0, data_range]. Works for 2D or any shape."""
    dr = float(data_range)
    if dr <= 0:
        dr = 1.0
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    mse = float(np.mean((p - r) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(dr) - 10.0 * np.log10(mse))


def _gaussian_kernel1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    size = int(size)
    if size % 2 == 0:
        size += 1
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1.0
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k) if np.sum(k) > 0 else 1.0
    return k


def _conv1d_reflect(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """1D convolution with reflect padding, output same length as x."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    half = int(len(k) // 2)
    if half <= 0:
        return x.copy()
    xp = np.pad(x, (half, half), mode="reflect")
    y = np.convolve(xp, k, mode="valid")
    return y


def _gaussian_filter2d(img: np.ndarray, *, win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Separable gaussian filter for 2D arrays."""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got shape={x.shape}")
    k = _gaussian_kernel1d(size=int(win_size), sigma=float(sigma))
    # Convolve rows
    tmp = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        tmp[i, :] = _conv1d_reflect(x[i, :], k)
    # Convolve cols
    out = np.empty_like(tmp, dtype=np.float64)
    for j in range(tmp.shape[1]):
        out[:, j] = _conv1d_reflect(tmp[:, j], k)
    return out


def ssim_range(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    data_range: float,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """SSIM for 2D grayscale arrays in [0, data_range], gaussian window.

    Lightweight implementation to avoid cv2/skimage dependency.
    """
    dr = float(data_range)
    if dr <= 0:
        dr = 1.0
    x = np.asarray(ref, dtype=np.float64)
    y = np.asarray(pred, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"expected 2D images, got ref={x.shape} pred={y.shape}")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: ref={x.shape} pred={y.shape}")

    C1 = (float(K1) * dr) ** 2
    C2 = (float(K2) * dr) ** 2

    mu_x = _gaussian_filter2d(x, win_size=win_size, sigma=sigma)
    mu_y = _gaussian_filter2d(y, win_size=win_size, sigma=sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian_filter2d(x * x, win_size=win_size, sigma=sigma) - mu_x2
    sigma_y2 = _gaussian_filter2d(y * y, win_size=win_size, sigma=sigma) - mu_y2
    sigma_xy = _gaussian_filter2d(x * y, win_size=win_size, sigma=sigma) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


