"""Poisson Negative Log-Likelihood Loss for count data.

This module provides loss functions for Poisson noise modeling, which is
common in low-count imaging applications like SPECT, PET, and X-ray CT.

Key features:
- Supports both linear and Anscombe normalization
- Computes loss in count domain for correct Poisson likelihood
- Precise Anscombe inverse without unnecessary clipping
- GPU-friendly PyTorch implementation
"""
import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss


def anscombe_forward_torch(x: torch.Tensor) -> torch.Tensor:
    """Anscombe forward transform: A(x) = 2*sqrt(x + 3/8).

    Args:
        x: Non-negative tensor (count domain)

    Returns:
        Transformed tensor (Anscombe domain)
    """
    # Use softplus-like clamping to avoid negative values while being differentiable
    x_safe = torch.clamp(x, min=0.0)
    return 2.0 * torch.sqrt(x_safe + 0.375)


def anscombe_inverse_algebraic_torch(y: torch.Tensor) -> torch.Tensor:
    """Algebraic inverse approximation: A^{-1}(y) ≈ (y/2)^2 - 3/8.

    Args:
        y: Tensor in Anscombe domain

    Returns:
        Tensor in count domain (may have small negative values near zero)
    """
    return (y * 0.5) ** 2 - 0.375


def anscombe_inverse_unbiased_torch(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Makitalo-Foi unbiased inverse approximation.

    This provides a more accurate inverse that reduces bias, especially
    for low count values.

    Reference:
        M. Makitalo, A. Foi, "Optimal inversion of the Anscombe transformation
        in low-count Poisson image denoising", 2011.

    Args:
        y: Tensor in Anscombe domain
        eps: Small constant for numerical stability

    Returns:
        Tensor in count domain
    """
    y_safe = torch.clamp(y, min=eps)

    inv = (
        (y * 0.5) ** 2
        - 0.125  # -1/8
        + 0.25 * (1.5 ** 0.5) / y_safe  # (1/4) * sqrt(3/2) / y
        - 1.375 / (y_safe ** 2)  # -11/8 / y^2
        + 0.625 * (1.5 ** 0.5) / (y_safe ** 3)  # (5/8) * sqrt(3/2) / y^3
        - 0.125 / (y_safe ** 4)  # -1/8 / y^4
    )

    return inv


def poisson_nll_loss_count(pred_count, target_count, weight=None, eps=1e-8, reduction='mean', full=True):
    """Poisson Negative Log-Likelihood loss in count domain.

    Uses PyTorch's built-in poisson_nll_loss for numerical stability.

    Full NLL = lambda - k*log(lambda) + log(k!)

    When full=True (default), adds Stirling approximation of log(k!) term:
        log(k!) ≈ k*log(k) - k + 0.5*log(2*pi*k)
    This makes the loss approximately 0 when pred ≈ target.

    Args:
        pred_count: Predicted count (lambda), shape (N, C, H, W)
        target_count: Observed count (k), shape (N, C, H, W)
        weight: Optional per-element weights, shape (N, C, H, W)
        eps: Small constant for numerical stability
        reduction: 'none' | 'mean' | 'sum'
        full: If True, add Stirling approximation of log(k!) term

    Returns:
        Poisson NLL loss
    """
    eps = float(eps)

    # Clamp predictions to be positive (Poisson rate must be > 0)
    pred_safe = torch.clamp(pred_count, min=eps)

    # Use log(lambda) as input to PyTorch's poisson_nll_loss
    log_pred = torch.log(pred_safe)

    # Clamp target to be non-negative
    target_safe = torch.clamp(target_count, min=0.0)

    # Compute Poisson NLL using PyTorch's implementation
    # log_input=True means we pass log(lambda), not lambda
    # full=True adds Stirling approximation of log(k!) for better numerical range
    loss = torch.nn.functional.poisson_nll_loss(
        log_pred, target_safe,
        log_input=True,
        full=full,
        reduction='none'
    )

    # Apply weight if provided
    if weight is not None:
        loss = loss * weight

    # Reduction
    if reduction == 'mean':
        if weight is not None:
            return loss.sum() / (weight.sum() + eps)
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


@LOSS_REGISTRY.register()
class PoissonNLLLoss(nn.Module):
    """Poisson Negative Log-Likelihood loss for count data.

    This loss converts normalized model outputs back to count domain and
    computes the Poisson NLL. It supports both linear and Anscombe normalization.

    For Anscombe normalization:
    - Data is normalized as: y_norm = A(x) / A(max_value)
    - Inverse: x = A^{-1}(y_norm * A(max_value))
    - Loss is computed in count domain for correct Poisson likelihood

    IMPORTANT: When using Anscombe normalization, the inverse transform is
    applied WITHOUT hard clipping to preserve gradient flow and avoid
    introducing bias. Small negative values may occur but are handled
    gracefully in the loss computation.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Reduction method. Supported: 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): Small constant for log stability. Default: 1e-8.
        max_value (float): Maximum count value used for normalization. Default: 100.0.
        norm_type (str): Normalization type. Supported: 'linear' | 'anscombe'. Default: 'linear'.
        anscombe_inverse (str): Anscombe inverse method. Supported: 'algebraic' | 'unbiased'. Default: 'unbiased'.
        use_soft_clamp (bool): Use soft clamping instead of hard clamp. Default: True.

    Example config:
        pixel_opt:
            type: PoissonNLLLoss
            loss_weight: 1.0
            max_value: 150.0
            norm_type: linear  # or 'anscombe'
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
        eps: float = 1e-8,
        max_value: float = 100.0,
        norm_type: str = 'linear',
        anscombe_inverse: str = 'unbiased',
        use_soft_clamp: bool = True,
    ):
        super(PoissonNLLLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')
        if norm_type not in ['linear', 'anscombe']:
            raise ValueError(f'Unsupported norm_type: {norm_type}')
        if anscombe_inverse not in ['algebraic', 'unbiased']:
            raise ValueError(f'Unsupported anscombe_inverse: {anscombe_inverse}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = float(eps)
        self.max_value = float(max_value)
        self.norm_type = norm_type
        self.anscombe_inverse = anscombe_inverse
        self.use_soft_clamp = use_soft_clamp

        # Pre-compute Anscombe of max_value for efficiency
        if self.norm_type == 'anscombe':
            self.register_buffer(
                'anscombe_max',
                torch.tensor(2.0 * (self.max_value + 0.375) ** 0.5, dtype=torch.float32)
            )

    def inverse_from_norm_tensor(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to count domain.

        For linear normalization:
            x = y_norm * max_value

        For Anscombe normalization:
            y_anscombe = y_norm * A(max_value)
            x = A^{-1}(y_anscombe)

        Args:
            y_norm: Normalized tensor in [0, 1] range, shape (N, C, H, W)

        Returns:
            Count domain tensor, shape (N, C, H, W)
        """
        if self.norm_type == 'linear':
            # Linear: simple scaling, minimal clipping
            if self.use_soft_clamp:
                # Soft clamp: allow small negative but scale up positive
                return y_norm * self.max_value
            else:
                return torch.clamp(y_norm, min=0.0) * self.max_value

        # Anscombe normalization
        # Scale to Anscombe domain
        y_anscombe = y_norm * self.anscombe_max

        # Apply inverse transform
        if self.anscombe_inverse == 'unbiased':
            x_count = anscombe_inverse_unbiased_torch(y_anscombe, self.eps)
        else:
            x_count = anscombe_inverse_algebraic_torch(y_anscombe)

        # Note: We intentionally do NOT hard-clip here to preserve precision
        # The loss function handles negative values gracefully
        return x_count

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        vmax: float = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute Poisson NLL loss.

        Args:
            pred: Predicted tensor in normalized domain, shape (N, C, H, W).
                  Will be converted to count domain (treated as lambda).
            target: Target tensor in normalized domain, shape (N, C, H, W).
                    Will be converted to count domain (treated as observed counts).
            weight: Optional element-wise weights, shape (N, C, H, W).
            vmax: Optional override for max_value (per-batch).

        Returns:
            Poisson NLL loss value.
        """
        # Handle optional vmax override
        original_max_value = None
        if vmax is not None:
            original_max_value = self.max_value
            self.max_value = float(vmax) if not isinstance(vmax, torch.Tensor) else float(vmax.item())
            # Update anscombe_max if using Anscombe
            if self.norm_type == 'anscombe':
                self.anscombe_max = torch.tensor(
                    2.0 * (self.max_value + 0.375) ** 0.5,
                    dtype=torch.float32, device=pred.device
                )

        try:
            # Convert to count domain
            pred_count = self.inverse_from_norm_tensor(pred)
            target_count = self.inverse_from_norm_tensor(target)

            # Compute Poisson NLL
            loss = self.loss_weight * poisson_nll_loss_count(
                pred_count, target_count,
                weight=weight,
                eps=self.eps,
                reduction=self.reduction,
                full=True  # Add Stirling approximation for stable loss values
            )
        finally:
            # Restore original max_value if it was overridden
            if original_max_value is not None:
                self.max_value = original_max_value
                if self.norm_type == 'anscombe':
                    self.anscombe_max = torch.tensor(
                        2.0 * (self.max_value + 0.375) ** 0.5,
                        dtype=torch.float32, device=pred.device
                    )

        return loss


@LOSS_REGISTRY.register()
class PoissonNLLLossSimple(nn.Module):
    """Simplified Poisson NLL loss assuming linear normalization.

    This is a faster version that assumes:
    - Data is linearly normalized: y = x / max_value
    - max_value is known and constant

    Use this when you know you're using linear normalization.

    Args:
        loss_weight: Loss weight
        reduction: Reduction method
        eps: Small constant for stability
        max_value: Maximum count value for normalization
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
        eps: float = 1e-8,
        max_value: float = 100.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = float(eps)
        self.max_value = float(max_value)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # Convert to count domain (simple linear scaling)
        pred_count = pred * self.max_value
        target_count = target * self.max_value

        # Use the common function
        loss = poisson_nll_loss_count(
            pred_count, target_count,
            weight=weight,
            eps=self.eps,
            reduction=self.reduction,
            full=True
        )

        return self.loss_weight * loss
