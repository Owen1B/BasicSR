"""UNet with Dropout for Self2Self training.

Based on UNetRes (DRUNet) architecture with added Dropout layers after convolutions.
The dropout acts as a stochastic mask for Self2Self self-supervised training.

Reference:
- Self2Self paper: uses Dropout as a natural masking mechanism
- Implementation: self2self_ref/network/Punet.py
"""
from __future__ import annotations

import functools
from typing import Iterable, List

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


def sequential(*args: Iterable[nn.Module]) -> nn.Sequential:
    """Build sequential from nested lists/tuples."""
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


class ResBlockDropout(nn.Module):
    """Residual block with Dropout after each convolution."""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        else:
            self.skip = None

        # Conv -> Dropout -> ReLU -> Conv -> Dropout
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=bias),
            nn.Dropout2d(p=dropout_rate)
        )

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
    return nn.ConvTranspose2d(in_ch, out_ch, 2, 2, 0, bias=bias)


def upsample_upconv(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Upsample by pixel-shuffle."""
    assert mode == '2'
    return sequential(nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=bias), nn.PixelShuffle(2))


def upsample_pixelshuffle(in_ch: int, out_ch: int, bias: bool = True, mode: str = '2') -> nn.Module:
    """Upsample by pixel-shuffle."""
    assert mode == '2'
    return sequential(nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=bias), nn.PixelShuffle(2))


@ARCH_REGISTRY.register()
class UNetDropout(nn.Module):
    """U-Net with Dropout for Self2Self training.

    Architecture similar to UNetRes but with Dropout layers added after convolutions.
    Dropout serves as a stochastic masking mechanism for self-supervised learning.

    Args:
        in_nc (int): number of input channels
        out_nc (int): number of output channels
        nc (list[int]): number of channels at each level, e.g. [64, 128, 256, 512]
        nb (int): number of residual blocks per level
        dropout_rate (float): dropout probability (0.2-0.3 recommended)
        act_mode (str): activation mode, 'R' for ReLU
        downsample_mode (str): 'strideconv' | 'avgpool' | 'maxpool'
        upsample_mode (str): 'convtranspose' | 'upconv' | 'pixelshuffle'
        bias (bool): use bias in convolutions
        global_residual (bool): add global skip connection from input to output
    """

    def __init__(
        self,
        in_nc: int = 2,
        out_nc: int = 2,
        nc: list = None,
        nb: int = 4,
        dropout_rate: float = 0.3,
        act_mode: str = 'R',
        downsample_mode: str = 'strideconv',
        upsample_mode: str = 'convtranspose',
        bias: bool = False,
        global_residual: bool = True,
    ):
        super().__init__()
        if nc is None:
            nc = [64, 128, 256, 512]

        self.global_residual = global_residual
        self.depth = len(nc)
        self.dropout_rate = dropout_rate

        # Downsample function
        if downsample_mode == 'avgpool':
            downsample_fn = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_fn = downsample_maxpool
        else:  # 'strideconv'
            downsample_fn = downsample_strideconv

        # Upsample function
        if upsample_mode == 'upconv':
            upsample_fn = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_fn = upsample_pixelshuffle
        else:  # 'convtranspose'
            upsample_fn = upsample_convtranspose

        # Input conv
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=bias)

        # Encoder (downsampling path)
        self.m_down_res = nn.ModuleList()  # Residual blocks
        self.m_down_sample = nn.ModuleList()  # Downsampling
        for i in range(self.depth):
            # Residual blocks
            blocks = nn.Sequential(*[ResBlockDropout(nc[i], nc[i], bias=bias, dropout_rate=dropout_rate) for _ in range(nb)])
            self.m_down_res.append(blocks)
            # Downsampling (except for last level)
            if i < self.depth - 1:
                self.m_down_sample.append(downsample_fn(nc[i], nc[i + 1], bias=bias))
            else:
                self.m_down_sample.append(None)  # No downsampling at bottleneck

        # Decoder (upsampling path)
        self.m_up = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            # Upsampling
            up = upsample_fn(nc[i], nc[i - 1], bias=bias)
            # Residual blocks (input is concatenated features)
            # First block takes concatenated channels, rest take normal channels
            blocks = []
            blocks.append(ResBlockDropout(nc[i - 1] * 2, nc[i - 1], bias=bias, dropout_rate=dropout_rate))
            for _ in range(nb - 1):
                blocks.append(ResBlockDropout(nc[i - 1], nc[i - 1], bias=bias, dropout_rate=dropout_rate))
            self.m_up.append(nn.ModuleList([up, nn.Sequential(*blocks)]))

        # Output conv
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Dropout enabled."""
        h, w = x.size()[-2:]
        paddingBottom = int(torch.ceil(torch.tensor(h / (2 ** self.depth))) * (2 ** self.depth) - h)
        paddingRight = int(torch.ceil(torch.tensor(w / (2 ** self.depth))) * (2 ** self.depth) - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        # Store input for global residual
        x_input = x

        # Head
        x = self.m_head(x)

        # Encoder
        enc_features = []
        for i in range(self.depth):
            # Apply residual blocks
            x = self.m_down_res[i](x)
            # Save feature BEFORE downsampling (for skip connections)
            if i < self.depth - 1:
                enc_features.append(x)
                # Apply downsampling
                x = self.m_down_sample[i](x)

        # Decoder
        for i, (up, blocks) in enumerate(self.m_up):
            x = up(x)
            # Concatenate with encoder features
            x = torch.cat([x, enc_features[-(i + 1)]], dim=1)
            x = blocks(x)

        # Tail
        x = self.m_tail(x)

        # Crop padding
        x = x[..., :h, :w]

        # Global residual
        if self.global_residual:
            x_input = x_input[..., :h, :w]
            x = x + x_input

        return x

    def set_dropout_mode(self, training: bool):
        """Explicitly control dropout mode.

        Args:
            training (bool): if True, enable dropout; if False, disable dropout
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout2d):
                module.train(training)

