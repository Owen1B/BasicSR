from __future__ import annotations

"""
Residual U-Net (DRUNet-style) backbone adapted for BasicSR.

Original citation:
  Kai Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", 2020.

This file is self-contained (implements the minimal `basicblock` utilities used
by the original code), so you can use it directly as:

  network_g:
    type: UNetRes
    in_nc: 2
    out_nc: 2
    ...
"""

import functools
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


def sequential(*args: Iterable[nn.Module]) -> nn.Sequential:
    modules: List[nn.Module] = []
    for a in args:
        if a is None:
            continue
        if isinstance(a, nn.Sequential):
            modules.extend(list(a.children()))
        elif isinstance(a, (list, tuple)):
            modules.extend(list(a))
        else:
            modules.append(a)
    return nn.Sequential(*modules)


def conv(in_ch: int, out_ch: int, bias: bool = True, mode: str = 'C') -> nn.Module:
    """Minimal conv factory.

    mode currently supports only 'C' used by the provided UNetRes code.
    """
    assert mode == 'C'
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)


class ResBlock(nn.Module):
    """A simple residual block matching `mode='C'+act_mode+'C'` usage."""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True, mode: str = 'CRC'):
        super().__init__()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        else:
            self.skip = None

        # Parse a minimal subset of mode string: C (conv3x3) and R (ReLU)
        layers: List[nn.Module] = []
        cur_in = in_ch
        for ch in [out_ch, out_ch]:
            layers.append(nn.Conv2d(cur_in, ch, 3, 1, 1, bias=bias))
            cur_in = ch
            # add activation after first conv if requested
            if 'R' in mode:
                layers.append(nn.ReLU(inplace=True))
                mode = mode.replace('R', '', 1)
        # remove trailing activation if accidentally added
        if len(layers) > 0 and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        skip = self.skip(x) if self.skip is not None else x
        return out + skip


def downsample_strideconv(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Downsample by stride-2 conv."""
    assert mode == '2'
    return nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=bias)


def downsample_avgpool(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    assert mode == '2'
    return sequential(nn.AvgPool2d(kernel_size=2, stride=2), nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias))


def downsample_maxpool(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    assert mode == '2'
    return sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias))


def upsample_convtranspose(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Upsample by 2x conv-transpose."""
    assert mode == '2'
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=bias)


def upsample_upconv(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Upsample by nearest + conv."""
    assert mode == '2'
    return sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias))


def upsample_pixelshuffle(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Upsample by pixel shuffle (2x)."""
    assert mode == '2'
    return sequential(nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=bias), nn.PixelShuffle(2))


@ARCH_REGISTRY.register()
class UNetRes(nn.Module):
    """DRUNet architecture with optional global residual (recommended for denoising).

    When global_residual=True: output = skip(x) + f(x), so the network starts near identity.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: Sequence[int] = (64, 128, 256, 512),
        nb: int = 4,
        act_mode: str = 'R',
        downsample_mode: str = 'strideconv',
        upsample_mode: str = 'convtranspose',
        bias: bool = True,
        global_residual: bool = True,
    ):
        super().__init__()
        self.global_residual = bool(global_residual)
        self.skip = None
        if self.global_residual and in_nc != out_nc:
            self.skip = nn.Conv2d(in_nc, out_nc, 1, 1, 0, bias=bias)

        self.m_head = conv(in_nc, nc[0], bias=bias, mode='C')

        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(f'downsample mode [{downsample_mode}] is not found')

        resblocks = functools.partial(ResBlock, bias=bias, mode='C' + act_mode + 'C')

        self.m_down1 = sequential(*[resblocks(nc[0], nc[0]) for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = sequential(*[resblocks(nc[1], nc[1]) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = sequential(*[resblocks(nc[2], nc[2]) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body = sequential(*[resblocks(nc[3], nc[3]) for _ in range(nb)])

        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f'upsample mode [{upsample_mode}] is not found')

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[resblocks(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[resblocks(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[resblocks(nc[0], nc[0]) for _ in range(nb)])

        self.m_tail = conv(nc[0], out_nc, bias=bias, mode='C')

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
