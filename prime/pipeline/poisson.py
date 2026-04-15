from __future__ import annotations

import numpy as np


def poisson_sample(lam: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """y ~ Poisson(lam)，lam 会先 clip 到非负。"""
    lam = np.clip(lam, 0.0, None)
    if rng is None:
        return np.random.poisson(lam=lam).astype(np.float32)
    return rng.poisson(lam=lam).astype(np.float32)











