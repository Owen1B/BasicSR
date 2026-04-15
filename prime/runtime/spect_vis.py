from __future__ import annotations

import numpy as np

from basicsr.utils.img_util import imwrite
from prime.runtime.anscombe import anscombe_forward


def _gray_u8(x: np.ndarray, vmax: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vmax = float(vmax)
    if vmax <= 0:
        vmax = float(np.max(x) if np.max(x) > 0 else 1.0)
    y = np.clip(x / vmax, 0.0, 1.0)
    u = (y * 255.0).round().astype(np.uint8)
    return np.stack([u, u, u], axis=2)


def _diverging_u8(x: np.ndarray, vlim: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    vlim = float(vlim)
    if vlim <= 0:
        vlim = float(np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0)

    t = np.clip(x / vlim, -1.0, 1.0)
    a = np.abs(t)

    white = np.array([255, 255, 255], dtype=np.float32)
    red = np.array([0, 0, 255], dtype=np.float32)
    blue = np.array([255, 0, 0], dtype=np.float32)

    out = np.empty((x.shape[0], x.shape[1], 3), dtype=np.float32)
    pos = t >= 0
    neg = ~pos

    out[pos] = (1.0 - a[pos])[..., None] * white + a[pos][..., None] * red
    out[neg] = (1.0 - a[neg])[..., None] * white + a[neg][..., None] * blue
    return np.clip(out, 0, 255).round().astype(np.uint8)


def save_spect_2view_grid(
    save_path: str,
    lq_cnt: np.ndarray,
    pred_cnt: np.ndarray,
    gt_cnt: np.ndarray,
    vmax_cnt: float,
    vlim_res: float,
    vlim_anscombe: float | None = None,
) -> None:
    if lq_cnt.ndim != 3 or lq_cnt.shape[2] != 2:
        raise ValueError(f"lq_cnt must be HWC with C=2, got {lq_cnt.shape}")
    if pred_cnt.shape != lq_cnt.shape or gt_cnt.shape != lq_cnt.shape:
        raise ValueError("pred_cnt and gt_cnt must match lq_cnt shape")

    h, w, _ = lq_cnt.shape
    tiles_rows = []
    for view in range(2):
        lq = lq_cnt[:, :, view]
        pred = pred_cnt[:, :, view]
        gt = gt_cnt[:, :, view]

        neg_noise = 2.0 * pred - lq
        lq_anscombe = anscombe_forward(np.clip(lq, 0.0, None))
        pred_anscombe = anscombe_forward(np.clip(pred, 0.0, None))
        noise_anscombe = lq_anscombe - pred_anscombe
        noise_cnt = lq - pred
        err_cnt = pred - gt

        if vlim_anscombe is None:
            vlim_anscombe = float(np.max(np.abs(noise_anscombe)))
            if vlim_anscombe <= 0:
                vlim_anscombe = 1.0

        t1 = _gray_u8(lq, vmax_cnt)
        t2 = _gray_u8(pred, vmax_cnt)
        t3 = _gray_u8(neg_noise, vmax_cnt)
        t4 = _gray_u8(gt, vmax_cnt)
        t5 = _diverging_u8(noise_anscombe, vlim_anscombe)
        t6 = _diverging_u8(noise_cnt, vlim_res)
        t7 = _diverging_u8(err_cnt, vlim_res)

        row = np.concatenate([t1, t2, t3, t4, t5, t6, t7], axis=1)
        assert row.shape[0] == h and row.shape[1] == w * 7
        tiles_rows.append(row)

    grid = np.concatenate(tiles_rows, axis=0)
    imwrite(grid, save_path)
