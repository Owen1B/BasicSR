from __future__ import annotations

import functools
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class Conv3dCircularDepth(nn.Module):
    """Conv3d with optional circular padding ONLY on depth (D) dimension.

    We keep spatial (H/W) padding as standard zero-padding inside Conv3d.
    This is useful for sinogram-style inputs where view dimension is periodic.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        bias: bool = True,
        circular_depth: bool = True,
    ) -> None:
        super().__init__()
        kd, kh, kw = kernel_size
        assert kd % 2 == 1 and kh % 2 == 1 and kw % 2 == 1, "Only odd kernels are supported."
        self.pad_d = kd // 2
        self.pad_hw = (kh // 2, kw // 2)
        self.circular_depth = bool(circular_depth)
        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=(0, self.pad_hw[0], self.pad_hw[1]),  # depth padding handled manually
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,D,H,W)
        if self.pad_d > 0:
            if self.circular_depth:
                x = F.pad(x, (0, 0, 0, 0, self.pad_d, self.pad_d), mode="circular")
            else:
                x = F.pad(x, (0, 0, 0, 0, self.pad_d, self.pad_d), mode="constant", value=0.0)
        return self.conv(x)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int, bias: bool = True, act: str = "relu", circular_depth: bool = True) -> None:
        super().__init__()
        if act == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif act == "lrelu":
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported act: {act}")

        self.c1 = Conv3dCircularDepth(ch, ch, kernel_size=(3, 3, 3), bias=bias, circular_depth=circular_depth)
        self.a1 = act_fn
        self.c2 = Conv3dCircularDepth(ch, ch, kernel_size=(3, 3, 3), bias=bias, circular_depth=circular_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.a1(self.c1(x)))


def downsample_strideconv_3d(in_ch: int, out_ch: int, bias: bool = True) -> nn.Module:
    # Only downsample H/W, keep D unchanged
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=(1, 2, 2), padding=1, bias=bias)


def upsample_convtranspose_3d(in_ch: int, out_ch: int, bias: bool = True) -> nn.Module:
    # Only upsample H/W, keep D unchanged
    return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, bias=bias)


@ARCH_REGISTRY.register()
class UNet3DRes(nn.Module):
    """3D UNetRes variant for sinogram volumes.

    - Input:  (N, C_in, D, H, W) e.g. (N,3,60,128,128)
    - Output: (N, C_out, D, H, W) e.g. (N,1,60,128,128)
    - Down/Up sampling only in H/W, not in D.
    - Optional circular padding on D for periodic view dimension.
    """

    def __init__(
        self,
        in_nc: int = 1,
        out_nc: int = 1,
        nc: Sequence[int] = (32, 64, 128, 256),
        nb: int = 2,
        bias: bool = True,
        global_residual: bool = True,
        circular_depth: bool = True,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.global_residual = bool(global_residual)
        self.circular_depth = bool(circular_depth)

        self.skip = None
        if self.global_residual and in_nc != out_nc:
            self.skip = nn.Conv3d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=bias)

        head = functools.partial(Conv3dCircularDepth, bias=bias, circular_depth=self.circular_depth)
        resblk = functools.partial(ResBlock3D, bias=bias, circular_depth=self.circular_depth, act=act)

        self.m_head = head(in_nc, int(nc[0]), kernel_size=(3, 3, 3))

        self.m_down1 = nn.Sequential(
            *[resblk(int(nc[0])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[0]), int(nc[1]), bias=bias),
        )
        self.m_down2 = nn.Sequential(
            *[resblk(int(nc[1])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[1]), int(nc[2]), bias=bias),
        )
        self.m_down3 = nn.Sequential(
            *[resblk(int(nc[2])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[2]), int(nc[3]), bias=bias),
        )

        self.m_body = nn.Sequential(*[resblk(int(nc[3])) for _ in range(int(nb))])

        self.m_up3 = nn.Sequential(
            upsample_convtranspose_3d(int(nc[3]), int(nc[2]), bias=bias),
            *[resblk(int(nc[2])) for _ in range(int(nb))],
        )
        self.m_up2 = nn.Sequential(
            upsample_convtranspose_3d(int(nc[2]), int(nc[1]), bias=bias),
            *[resblk(int(nc[1])) for _ in range(int(nb))],
        )
        self.m_up1 = nn.Sequential(
            upsample_convtranspose_3d(int(nc[1]), int(nc[0]), bias=bias),
            *[resblk(int(nc[0])) for _ in range(int(nb))],
        )

        self.m_tail = head(int(nc[0]), out_nc, kernel_size=(3, 3, 3))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        if self.global_residual:
            skip = self.skip(x0) if self.skip is not None else x0
            x = x + skip
        return x


