"""
Dilated Blind-Spot Network (DBSNl) for Noise2Void style self-supervised denoising.

Based on: "AP-BSN: Self-Supervised Denoising for Real-World Images via
Asymmetric PD and Blind-Spot Network" (CVPR 2022)

Original implementation: https://github.com/wooseoklee4/AP-BSN

Key features:
- CentralMaskedConv2d: Masks the center weight to prevent the network from
  seeing the input pixel at the output location (blind-spot property)
- Dilated convolutions: Expand receptive field while maintaining blind-spot
- Two parallel branches with different dilation rates for multi-scale features

This enables true blind-spot self-supervised denoising without data augmentation tricks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


class CentralMaskedConv2d(nn.Conv2d):
    """Convolution with center weight masked to zero.

    This ensures that the output at position (i,j) does not see the input at (i,j),
    which is the key requirement for blind-spot networks used in Noise2Void.

    The mask is applied by multiplying weights with a mask tensor that has
    zero at the center position and ones elsewhere.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create mask buffer (same shape as weight)
        self.register_buffer('mask', torch.ones_like(self.weight.data))

        # Set center position to zero
        _, _, kH, kW = self.weight.size()
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x):
        # Apply mask to weights before convolution
        # Note: We multiply weight.data (not weight) to avoid gradient issues
        masked_weight = self.weight * self.mask
        return nn.functional.conv2d(
            x, masked_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with residual connection.

    Uses dilated convolution to expand receptive field while maintaining
    the blind-spot property established by CentralMaskedConv2d.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.body(x)


class DCBranch(nn.Module):
    """Dilated Convolution Branch.

    One branch of the blind-spot network with a specific dilation rate.
    Starts with CentralMaskedConv2d to establish blind-spot, then uses
    dilated convolutions to build features.

    Args:
        dilation: Dilation rate (also determines kernel size of masked conv)
        channels: Number of channels
        num_blocks: Number of dilated conv blocks
    """

    def __init__(self, dilation: int, channels: int, num_blocks: int):
        super().__init__()

        # Kernel size for masked conv: 2*dilation - 1
        # This ensures proper padding for the given dilation
        kernel_size = 2 * dilation - 1
        padding = dilation - 1

        layers = []

        # Initial masked convolution (establishes blind-spot)
        layers.append(CentralMaskedConv2d(
            channels, channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ))
        layers.append(nn.ReLU(inplace=True))

        # 1x1 convs for channel mixing
        layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        # Dilated conv blocks with residual connections
        for _ in range(num_blocks):
            layers.append(DilatedConvBlock(channels, dilation))

        # Final 1x1 conv
        layers.append(nn.Conv2d(channels, channels, kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class DBSNl(nn.Module):
    """Dilated Blind-Spot Network (light version).

    A blind-spot network for self-supervised denoising. Uses two parallel
    branches with different dilation rates to capture multi-scale features
    while maintaining the blind-spot property.

    Args:
        in_nc: Number of input channels (default: 1 for grayscale, 2 for dual-view SPECT)
        out_nc: Number of output channels (should match in_nc)
        base_ch: Base number of channels (default: 128)
        num_module: Number of dilated conv blocks per branch (default: 9)
        global_residual: Whether to add global residual connection (default: True)

    Example config:
        network_g:
            type: DBSNl
            in_nc: 2
            out_nc: 2
            base_ch: 128
            num_module: 9
    """

    def __init__(
        self,
        in_nc: int = 1,
        out_nc: int = 1,
        base_ch: int = 128,
        num_module: int = 9,
        global_residual: bool = True,
    ):
        super().__init__()

        assert base_ch % 2 == 0, "base_ch should be divisible by 2"

        self.global_residual = global_residual

        # Head: project input to base channels
        self.head = nn.Sequential(
            nn.Conv2d(in_nc, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Two parallel branches with different dilation rates
        # Branch 1: dilation=2 (kernel_size=3 for masked conv)
        # Branch 2: dilation=3 (kernel_size=5 for masked conv)
        self.branch1 = DCBranch(dilation=2, channels=base_ch, num_blocks=num_module)
        self.branch2 = DCBranch(dilation=3, channels=base_ch, num_blocks=num_module)

        # Tail: merge branches and project to output channels
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_nc, kernel_size=1),
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        identity = x

        # Project to base channels
        x = self.head(x)

        # Two parallel branches
        br1 = self.branch1(x)
        br2 = self.branch2(x)

        # Concatenate and project to output
        x = torch.cat([br1, br2], dim=1)
        out = self.tail(x)

        # Global residual connection
        if self.global_residual:
            out = out + identity

        return out

    def _initialize_weights(self):
        """Initialize weights following the original AP-BSN implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming-like initialization
                nn.init.normal_(m.weight, 0, (2 / (9.0 * 64)) ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


@ARCH_REGISTRY.register()
class DBSNlLight(nn.Module):
    """Lighter version of DBSNl with fewer parameters.

    Suitable for smaller datasets or when computational resources are limited.
    Uses smaller base channels and fewer modules.

    Args:
        in_nc: Number of input channels
        out_nc: Number of output channels
        base_ch: Base number of channels (default: 64)
        num_module: Number of dilated conv blocks per branch (default: 5)
        global_residual: Whether to add global residual connection
    """

    def __init__(
        self,
        in_nc: int = 1,
        out_nc: int = 1,
        base_ch: int = 64,
        num_module: int = 5,
        global_residual: bool = True,
    ):
        super().__init__()

        assert base_ch % 2 == 0, "base_ch should be divisible by 2"

        self.global_residual = global_residual

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_nc, base_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Single branch for lighter model (dilation=2)
        self.branch = DCBranch(dilation=2, channels=base_ch, num_blocks=num_module)

        # Tail
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, out_nc, kernel_size=1),
        )

        self._initialize_weights()

    def forward(self, x):
        identity = x

        x = self.head(x)
        x = self.branch(x)
        out = self.tail(x)

        if self.global_residual:
            out = out + identity

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, (2 / (9.0 * 64)) ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

