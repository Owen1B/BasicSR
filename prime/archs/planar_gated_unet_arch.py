from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .unet2d_res_arch import (
    ResBlock,
    conv,
    downsample_avgpool,
    downsample_maxpool,
    downsample_strideconv,
    sequential,
    upsample_convtranspose,
    upsample_pixelshuffle,
    upsample_upconv,
)


class CrossViewGatedFusion(nn.Module):
    """Late fusion block that keeps the target view in control.

    We optionally summarize the assist view with pooled cross-attention, then
    inject only a gated residual into the target path. The residual branch and
    gate start near zero so the network initially behaves like a self-only
    shared U-Net.
    """

    def __init__(
        self,
        channels: int,
        *,
        attention_heads: int = 4,
        attention_pool: tuple[int, int] = (32, 16),
        bias: bool = True,
        gate_bias: float = -2.0,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_attention = bool(use_attention)
        self.attention_pool = (int(attention_pool[0]), int(attention_pool[1]))

        if self.use_attention:
            self.q_proj = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)
            self.k_proj = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)
            self.v_proj = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)
            self.attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=max(1, int(attention_heads)),
                batch_first=True,
                bias=bias,
            )
            self.attn_out = nn.Conv2d(channels, channels, 1, 1, 0, bias=bias)
            nn.init.zeros_(self.attn_out.weight)
            if self.attn_out.bias is not None:
                nn.init.zeros_(self.attn_out.bias)

        hidden = max(channels // 2, 16)
        fuse_in = channels * 4
        self.delta_proj = nn.Sequential(
            nn.Conv2d(fuse_in, hidden, 1, 1, 0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, 1, 1, bias=bias),
        )
        self.confidence_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1, 0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
        )
        self.gate_proj = nn.Conv2d(fuse_in + 1, channels, 1, 1, 0, bias=True)

        nn.init.zeros_(self.delta_proj[-1].weight)
        if self.delta_proj[-1].bias is not None:
            nn.init.zeros_(self.delta_proj[-1].bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_bias))

    def _attention_context(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if not self.use_attention:
            return source

        pool_h = max(1, min(int(target.shape[-2]), self.attention_pool[0]))
        pool_w = max(1, min(int(target.shape[-1]), self.attention_pool[1]))
        target_pool = F.adaptive_avg_pool2d(target, output_size=(pool_h, pool_w))
        source_pool = F.adaptive_avg_pool2d(source, output_size=(pool_h, pool_w))

        query = self.q_proj(target_pool).flatten(2).transpose(1, 2)
        key = self.k_proj(source_pool).flatten(2).transpose(1, 2)
        value = self.v_proj(source_pool).flatten(2).transpose(1, 2)
        attended, _ = self.attn(query, key, value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(target.shape[0], self.channels, pool_h, pool_w)
        attended = self.attn_out(attended)
        return F.interpolate(attended, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_context = self._attention_context(target, source)
        confidence = torch.sigmoid(self.confidence_proj(source))
        features = torch.cat([target, source, source_context, target - source], dim=1)
        delta = self.delta_proj(features)
        gate = torch.sigmoid(self.gate_proj(torch.cat([features, confidence], dim=1))) * confidence
        return target + gate * delta, gate.mean(dim=1, keepdim=True)


@ARCH_REGISTRY.register()
class PlanarGatedUNetRes(nn.Module):
    """Shared-view U-Net with late gated cross-view fusion for AP planar denoising."""

    supports_fusion_controls = True

    def __init__(
        self,
        in_nc: int = 2,
        out_nc: int = 2,
        nc: Sequence[int] = (64, 128, 256, 512),
        nb: int = 4,
        act_mode: str = "R",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
        bias: bool = True,
        global_residual: bool = True,
        fusion_levels: Sequence[str] = ("body", "up3", "up2"),
        fusion_attention_heads: Sequence[int] = (8, 4, 4),
        fusion_attention_pool: Sequence[int] = (32, 16),
        gate_bias: float = -2.0,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if int(in_nc) != 2 or int(out_nc) != 2:
            raise ValueError("PlanarGatedUNetRes expects in_nc=2 and out_nc=2 for AP paired planar input.")

        self.global_residual = bool(global_residual)
        self.fusion_levels = {str(level).lower() for level in fusion_levels}

        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(f"downsample mode [{downsample_mode}] is not found")

        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode}] is not found")

        resblocks = lambda channels: [ResBlock(channels, channels, bias=bias, mode="C" + act_mode + "C") for _ in range(nb)]

        # Shared self-only backbone. Key names intentionally mirror legacy UNetRes
        # so a legacy 2-channel checkpoint can warm-start most layers.
        self.m_head = conv(1, nc[0], bias=bias, mode="C")
        self.m_down1 = sequential(*resblocks(nc[0]), downsample_block(nc[0], nc[1], bias=bias, mode="2"))
        self.m_down2 = sequential(*resblocks(nc[1]), downsample_block(nc[1], nc[2], bias=bias, mode="2"))
        self.m_down3 = sequential(*resblocks(nc[2]), downsample_block(nc[2], nc[3], bias=bias, mode="2"))
        self.m_body = sequential(*resblocks(nc[3]))
        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=bias, mode="2"), *resblocks(nc[2]))
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=bias, mode="2"), *resblocks(nc[1]))
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=bias, mode="2"), *resblocks(nc[0]))
        self.m_tail = conv(nc[0], 1, bias=bias, mode="C")

        pool = (int(fusion_attention_pool[0]), int(fusion_attention_pool[1]))
        heads = list(fusion_attention_heads)
        while len(heads) < 3:
            heads.append(heads[-1] if heads else 4)

        self.fuse_body = CrossViewGatedFusion(
            nc[3],
            attention_heads=heads[0],
            attention_pool=pool,
            bias=bias,
            gate_bias=gate_bias,
            use_attention=use_attention,
        )
        self.fuse_up3 = CrossViewGatedFusion(
            nc[2],
            attention_heads=heads[1],
            attention_pool=pool,
            bias=bias,
            gate_bias=gate_bias,
            use_attention=use_attention,
        )
        self.fuse_up2 = CrossViewGatedFusion(
            nc[1],
            attention_heads=heads[2],
            attention_pool=pool,
            bias=bias,
            gate_bias=gate_bias,
            use_attention=use_attention,
        )

    def _encode_view(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        xb = self.m_body(x4)
        return x1, x2, x3, x4, xb

    def forward(
        self,
        x0: torch.Tensor,
        *,
        disable_fusion: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x0.ndim != 4 or int(x0.shape[1]) != 2:
            raise ValueError(f"PlanarGatedUNetRes expects BCHW with C=2, got {tuple(x0.shape)}")

        xa = x0[:, 0:1]
        xp = x0[:, 1:2]

        a1, a2, a3, a4, ab = self._encode_view(xa)
        p1, p2, p3, p4, pb = self._encode_view(xp)

        gate_maps: list[torch.Tensor] = []
        if disable_fusion or "body" not in self.fusion_levels:
            ab_fused, pb_fused = ab, pb
        else:
            ab_fused, gate_a_body = self.fuse_body(ab, pb)
            pb_fused, gate_p_body = self.fuse_body(pb, ab)
            gate_maps.extend([gate_a_body, gate_p_body])

        a_up3 = self.m_up3(ab_fused + a4)
        p_up3 = self.m_up3(pb_fused + p4)

        if not disable_fusion and "up3" in self.fusion_levels:
            a_up3_src = a_up3
            p_up3_src = p_up3
            a_up3, gate_a_up3 = self.fuse_up3(a_up3_src, p_up3_src)
            p_up3, gate_p_up3 = self.fuse_up3(p_up3_src, a_up3_src)
            gate_maps.extend([gate_a_up3, gate_p_up3])

        a_up2 = self.m_up2(a_up3 + a3)
        p_up2 = self.m_up2(p_up3 + p3)
        if not disable_fusion and "up2" in self.fusion_levels:
            a_up2_src = a_up2
            p_up2_src = p_up2
            a_up2, gate_a_up2 = self.fuse_up2(a_up2_src, p_up2_src)
            p_up2, gate_p_up2 = self.fuse_up2(p_up2_src, a_up2_src)
            gate_maps.extend([gate_a_up2, gate_p_up2])

        a = self.m_up1(a_up2 + a2)
        p = self.m_up1(p_up2 + p2)
        ao = self.m_tail(a + a1)
        po = self.m_tail(p + p1)

        if self.global_residual:
            ao = ao + xa
            po = po + xp

        output = torch.cat([ao, po], dim=1)
        if not return_aux:
            return output

        gate_mean = None
        if gate_maps:
            gate_mean = torch.stack([g.mean() for g in gate_maps]).mean()
        aux = {
            "gate_maps": gate_maps,
            "gate_mean": gate_mean,
        }
        return output, aux

    def warmstart_from_legacy_unetres(
        self,
        load_path: str | Path,
        *,
        param_key: str = "params_ema",
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Warm-start from the old early-fusion 2-channel UNetRes checkpoint."""

        path = str(load_path)
        state = torch.load(path, map_location=map_location)
        if param_key is not None:
            if param_key not in state and "params" in state:
                param_key = "params"
            state = state[param_key]

        cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in deepcopy(state).items():
            if key.startswith("module."):
                cleaned[key[7:]] = value
            else:
                cleaned[key] = value

        target_state = self.state_dict()
        loadable: OrderedDict[str, torch.Tensor] = OrderedDict()
        reused, skipped = 0, []
        for key, value in cleaned.items():
            if key.startswith("skip."):
                skipped.append(key)
                continue
            if key not in target_state:
                skipped.append(key)
                continue

            target_value = target_state[key]
            if key == "m_head.weight" and value.ndim == 4 and value.shape[1] == 2 and target_value.shape[1] == 1:
                loadable[key] = value.mean(dim=1, keepdim=True)
                reused += 1
                continue
            if key == "m_tail.weight" and value.ndim == 4 and value.shape[0] == 2 and target_value.shape[0] == 1:
                loadable[key] = value.mean(dim=0, keepdim=True)
                reused += 1
                continue
            if key == "m_tail.bias" and value.ndim == 1 and value.shape[0] == 2 and target_value.shape[0] == 1:
                loadable[key] = value.mean(dim=0, keepdim=True)
                reused += 1
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                skipped.append(key)
                continue

            loadable[key] = value
            reused += 1

        self.load_state_dict(loadable, strict=False)
        return {
            "path": path,
            "param_key": param_key,
            "reused_keys": reused,
            "skipped_keys": len(skipped),
            "sample_skipped": skipped[:10],
        }
