from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """Gradient loss for edge preservation."""

    def __init__(self, loss_weight=1.0, reduction="mean", use_input_as_ref=False):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. Supported ones are: mean | sum")
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.use_input_as_ref = use_input_as_ref

    def compute_gradient(self, img):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x.repeat(img.size(1), 1, 1, 1), padding=1, groups=img.size(1))
        grad_y = F.conv2d(img, sobel_y.repeat(img.size(1), 1, 1, 1), padding=1, groups=img.size(1))
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

    def forward(self, pred, target=None, input_ref=None):
        pred_grad = self.compute_gradient(pred)

        if self.use_input_as_ref:
            if input_ref is None:
                raise ValueError("use_input_as_ref=True requires input_ref to be provided")
            ref_grad = self.compute_gradient(input_ref)
        else:
            if target is None:
                raise ValueError("use_input_as_ref=False requires target to be provided")
            ref_grad = self.compute_gradient(target)

        loss = F.l1_loss(pred_grad, ref_grad, reduction="none")
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss
