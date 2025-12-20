"""
UNet with FPN-style Multi-scale Feature Fusion for dual-view SPECT denoising.

Architecture:
- Two separate encoder branches (one for each view: anterior and posterior)
- FPN-style feature fusion at each encoder level
- Shared decoder that uses fused multi-scale features

This allows independent feature extraction for each view while enabling
information exchange at multiple scales, similar to Feature Pyramid Networks.

Reference:
- FPN: "Feature Pyramid Networks for Object Detection" (CVPR 2017)
- PANet: "Path Aggregation Network for Instance Segmentation" (CVPR 2018)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class ConvBlock(nn.Module):
    """Basic convolution block: Conv + BN + ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, channels: int, use_bn: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class EncoderBlock(nn.Module):
    """Encoder block: multiple residual blocks + optional downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_res_blocks: int = 2,
        downsample: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()

        # Channel projection if needed
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.proj = nn.Identity()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(out_ch, use_bn) for _ in range(num_res_blocks)]
        )

        # Downsampling
        if downsample:
            self.downsample = nn.Conv2d(out_ch, out_ch, 3, 2, 1)
        else:
            self.downsample = None

    def forward(self, x):
        x = self.proj(x)
        feat = self.res_blocks(x)

        if self.downsample is not None:
            x_down = self.downsample(feat)
            return feat, x_down
        return feat, None


class FPNFusionModule(nn.Module):
    """FPN-style feature fusion module.

    Fuses features from two views at the same scale using:
    1. Concatenation
    2. Channel attention (SE-style)
    3. Spatial attention
    4. Residual refinement
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        # Channel attention (Squeeze-and-Excitation style)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid(),
        )

        # Reduce concatenated features
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid(),
        )

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, feat_ant: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        """Fuse features from anterior and posterior views.

        Args:
            feat_ant: Features from anterior view, shape (B, C, H, W)
            feat_post: Features from posterior view, shape (B, C, H, W)

        Returns:
            Fused features, shape (B, C, H, W)
        """
        # Concatenate
        concat = torch.cat([feat_ant, feat_post], dim=1)  # (B, 2C, H, W)

        # Channel attention
        ca = self.channel_attention(concat)  # (B, 2C, 1, 1)
        concat = concat * ca

        # Reduce channels
        fused = self.reduce(concat)  # (B, C, H, W)

        # Spatial attention
        sa = self.spatial_attention(fused)  # (B, 1, H, W)
        fused = fused * sa

        # Refinement with residual
        fused = fused + self.refine(fused)

        return fused


class DecoderBlock(nn.Module):
    """Decoder block: upsample + skip connection + residual blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_res_blocks: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()

        # Upsample
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1)

        # Combine upsampled features with skip connection
        self.combine = nn.Conv2d(in_ch + skip_ch, out_ch, 1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(out_ch, use_bn) for _ in range(num_res_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.combine(x)
        x = self.res_blocks(x)
        return x


class DualViewEncoder(nn.Module):
    """Encoder for a single view with multi-scale feature outputs."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        num_levels: int = 4,
        num_res_blocks: int = 2,
        use_bn: bool = True,
    ):
        super().__init__()

        self.num_levels = num_levels

        # Initial projection
        self.init_conv = ConvBlock(in_ch, base_ch, use_bn=use_bn)

        # Calculate channel sizes for each level
        # Level 0: base_ch, Level 1: base_ch*2, ..., last level: same as previous
        self.level_channels = []
        for i in range(num_levels):
            if i < num_levels - 1:
                self.level_channels.append(min(base_ch * (2 ** i), 512))
            else:
                # Last level keeps same channels
                self.level_channels.append(self.level_channels[-1] if self.level_channels else base_ch)

        # Encoder levels
        self.encoders = nn.ModuleList()
        for i in range(num_levels):
            in_channels = self.level_channels[i]
            out_channels = self.level_channels[i]  # Output same as level channels
            downsample = (i < num_levels - 1)

            # Input to first level is base_ch (from init_conv)
            if i == 0:
                in_channels = base_ch
            else:
                # Input from previous level's downsampled output
                in_channels = self.level_channels[i - 1]

            self.encoders.append(
                EncoderBlock(in_channels, out_channels, num_res_blocks, downsample, use_bn)
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Returns:
            List of features from each level, from finest to coarsest.
        """
        x = self.init_conv(x)

        features = []
        for encoder in self.encoders:
            feat, x_down = encoder(x)
            features.append(feat)
            if x_down is not None:
                x = x_down

        return features


@ARCH_REGISTRY.register()
class UNetFPNFusion(nn.Module):
    """UNet with FPN-style Multi-scale Feature Fusion for dual-view denoising.

    Architecture:
    1. Split input (B, 2, H, W) into anterior (B, 1, H, W) and posterior (B, 1, H, W)
    2. Two separate encoders extract multi-scale features for each view
    3. FPN fusion modules combine features from both views at each scale
    4. Shared decoder uses fused features to generate output
    5. Output (B, 2, H, W) with denoised anterior and posterior views

    Args:
        in_nc: Number of input channels (must be 2 for dual-view)
        out_nc: Number of output channels (must be 2)
        base_ch: Base number of channels (default: 64)
        num_levels: Number of encoder/decoder levels (default: 4)
        num_res_blocks: Number of residual blocks per level (default: 2)
        use_bn: Whether to use batch normalization (default: True)
        global_residual: Whether to add global residual connection (default: True)

    Example config:
        network_g:
            type: UNetFPNFusion
            in_nc: 2
            out_nc: 2
            base_ch: 64
            num_levels: 4
    """

    def __init__(
        self,
        in_nc: int = 2,
        out_nc: int = 2,
        base_ch: int = 64,
        num_levels: int = 4,
        num_res_blocks: int = 2,
        use_bn: bool = True,
        global_residual: bool = True,
    ):
        super().__init__()

        assert in_nc == 2 and out_nc == 2, "UNetFPNFusion requires in_nc=2, out_nc=2 for dual-view"

        self.num_levels = num_levels
        self.global_residual = global_residual

        # Two separate encoders for each view
        self.encoder_ant = DualViewEncoder(1, base_ch, num_levels, num_res_blocks, use_bn)
        self.encoder_post = DualViewEncoder(1, base_ch, num_levels, num_res_blocks, use_bn)

        # Calculate channel sizes for each level (must match encoder output)
        # Level 0: base_ch, Level 1: base_ch*2, ..., Level n-1: stays at previous
        self.level_channels = []
        ch = base_ch
        for i in range(num_levels):
            if i < num_levels - 1:
                self.level_channels.append(min(ch * (2 ** i), 512))
            else:
                # Last level keeps same channels as previous level
                self.level_channels.append(self.level_channels[-1] if self.level_channels else ch)

        # FPN fusion modules for each level
        self.fusion_modules = nn.ModuleList()
        for i in range(num_levels):
            self.fusion_modules.append(FPNFusionModule(self.level_channels[i]))

        # Shared decoder
        self.decoders = nn.ModuleList()

        for i in range(num_levels - 1, 0, -1):
            in_ch = self.level_channels[i]
            skip_ch = self.level_channels[i - 1]
            out_ch = self.level_channels[i - 1]
            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, out_ch, num_res_blocks, use_bn)
            )

        # Output projection: from base_ch to 2 channels (both views)
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_nc, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, 2, H, W)

        Returns:
            Output tensor, shape (B, 2, H, W)
        """
        identity = x

        # Split into two views
        x_ant = x[:, 0:1, :, :]   # (B, 1, H, W)
        x_post = x[:, 1:2, :, :]  # (B, 1, H, W)

        # Extract multi-scale features
        feats_ant = self.encoder_ant(x_ant)    # List of features
        feats_post = self.encoder_post(x_post)  # List of features

        # FPN fusion at each level
        fused_feats = []
        for i, (f_ant, f_post) in enumerate(zip(feats_ant, feats_post)):
            fused = self.fusion_modules[i](f_ant, f_post)
            fused_feats.append(fused)

        # Decoder: from coarsest to finest
        x = fused_feats[-1]  # Start from coarsest
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.num_levels - 2 - i
            skip = fused_feats[skip_idx]
            x = decoder(x, skip)

        # Output projection
        out = self.output_conv(x)

        # Global residual
        if self.global_residual:
            out = out + identity

        return out


@ARCH_REGISTRY.register()
class UNetFPNFusionLight(nn.Module):
    """Lighter version of UNetFPNFusion with fewer parameters.

    Uses smaller base channels and fewer residual blocks for faster training.

    Args:
        in_nc: Number of input channels (must be 2)
        out_nc: Number of output channels (must be 2)
        base_ch: Base number of channels (default: 32)
        num_levels: Number of encoder/decoder levels (default: 3)
        global_residual: Whether to add global residual connection
    """

    def __init__(
        self,
        in_nc: int = 2,
        out_nc: int = 2,
        base_ch: int = 32,
        num_levels: int = 3,
        global_residual: bool = True,
    ):
        super().__init__()

        # Use the full version with lighter settings
        self.model = UNetFPNFusion(
            in_nc=in_nc,
            out_nc=out_nc,
            base_ch=base_ch,
            num_levels=num_levels,
            num_res_blocks=1,
            use_bn=False,  # Skip BN for lighter model
            global_residual=global_residual,
        )

    def forward(self, x):
        return self.model(x)

