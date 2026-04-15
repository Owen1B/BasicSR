from __future__ import annotations

from typing import Tuple

import numpy as np


def _to_nonnegative_int_trials(x_scaled: np.ndarray, round_mode: str) -> np.ndarray:
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    x_scaled = np.clip(x_scaled, 0.0, None)
    if round_mode == "nearest":
        n = np.rint(x_scaled)
    elif round_mode == "floor":
        n = np.floor(x_scaled)
    else:
        raise ValueError(f"Unsupported round_mode: {round_mode}")
    return np.clip(n, 0.0, None).astype(np.int64, copy=False)


def poisson_thinning_complement_split(
    x_scaled: np.ndarray,
    prob: float,
    round_mode: str,
    random_swap: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson thinning with complementary splits: y~Binomial(n,p), z=n-y."""
    n = _to_nonnegative_int_trials(x_scaled, round_mode=round_mode)
    y = np.random.binomial(n=n, p=prob).astype(np.int64, copy=False)
    z = (n - y).astype(np.int64, copy=False)

    y_f = y.astype(np.float32, copy=False)
    z_f = z.astype(np.float32, copy=False)
    if random_swap and (np.random.rand() < 0.5):
        return z_f, y_f
    return y_f, z_f


def poisson_thinning_same_dose_split(
    x_scaled: np.ndarray,
    k_factor: float,
    round_mode: str,
    random_swap: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson thinning with two same-dose splits via multinomial thinning."""
    kk = float(k_factor)
    if kk <= 1.0:
        raise ValueError(f"k_factor must be > 1. Got: {k_factor}")

    n = _to_nonnegative_int_trials(x_scaled, round_mode=round_mode)
    p1 = 1.0 / kk
    y1 = np.random.binomial(n=n, p=p1).astype(np.int64, copy=False)
    rem = (n - y1).astype(np.int64, copy=False)
    p2 = 1.0 / (kk - 1.0)
    y2 = np.random.binomial(n=rem, p=p2).astype(np.int64, copy=False)

    y1_f = y1.astype(np.float32, copy=False)
    y2_f = y2.astype(np.float32, copy=False)
    if random_swap and (np.random.rand() < 0.5):
        return y2_f, y1_f
    return y1_f, y2_f
