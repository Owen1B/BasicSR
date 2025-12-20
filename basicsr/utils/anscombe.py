"""Anscombe variance-stabilizing transform utilities.

This file is intentionally lightweight (numpy-only) so it can be used in
datasets and validation code paths.

We use the standard Anscombe transform for Poisson data:

    A(x) = 2 * sqrt(x + 3/8)

References:
  - F. J. Anscombe, "The transformation of Poisson, binomial and negative-binomial data", 1948.
  - M. Makitalo, A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", 2011.
"""

from __future__ import annotations

import numpy as np


def anscombe_forward(x: np.ndarray) -> np.ndarray:
    """Anscombe forward transform: A(x) = 2*sqrt(x + 3/8).

    Args:
        x: Non-negative Poisson counts. Values < 0 will be clipped to 0.

    Returns:
        Transformed array, float32.
    """
    x = np.asarray(x)
    x = np.clip(x, 0.0, None)
    return (2.0 * np.sqrt(x + 3.0 / 8.0)).astype(np.float32, copy=False)


def anscombe_inverse_algebraic(y: np.ndarray) -> np.ndarray:
    """Algebraic inverse approximation.

    A^{-1}(y) ≈ (y/2)^2 - 3/8
    """
    y = np.asarray(y).astype(np.float32, copy=False)
    return np.maximum((y * 0.5) ** 2 - 3.0 / 8.0, 0.0).astype(np.float32, copy=False)


def anscombe_inverse_unbiased(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Makitalo-Foi unbiased inverse approximation (commonly used).

    This is a practical, numerically stable approximation used in many
    implementations to reduce bias after inverse VST.
    """
    y = np.asarray(y).astype(np.float32, copy=False)
    y_safe = np.maximum(y, eps)
    inv = (
        (y * 0.5) ** 2
        - 1.0 / 8.0
        + (1.0 / 4.0) * np.sqrt(3.0 / 2.0) * (1.0 / y_safe)
        - (11.0 / 8.0) * (1.0 / (y_safe**2))
        + (5.0 / 8.0) * np.sqrt(3.0 / 2.0) * (1.0 / (y_safe**3))
        - (1.0 / 8.0) * (1.0 / (y_safe**4))
    )
    return np.maximum(inv, 0.0).astype(np.float32, copy=False)



