from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from prime.runtime.support_conditioning import build_support_condition_tensors

from .unet3d_res_arch import Conv3dCircularDepth, ResBlock3D, downsample_strideconv_3d, upsample_convtranspose_3d


@ARCH_REGISTRY.register()
class SupportAwareUNet3DRes(nn.Module):
    """3D residual U-Net with explicit support conditioning and gated residual output."""

    supports_aux_outputs = True

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 1,
        obs_nc: int = 1,
        nc: Sequence[int] = (16, 32, 64, 128),
        nb: int = 2,
        bias: bool = True,
        circular_depth: bool = True,
        act: str = "relu",
        hard_support_output: bool = True,
        nonnegative_output: bool = True,
        gate_bias: float = 4.0,
        support_condition: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(obs_nc) <= 0 or int(obs_nc) >= int(in_nc):
            raise ValueError(f"SupportAwareUNet3DRes expects 0 < obs_nc < in_nc, got obs_nc={obs_nc}, in_nc={in_nc}")

        self.in_nc = int(in_nc)
        self.out_nc = int(out_nc)
        self.obs_nc = int(obs_nc)
        self.circular_depth = bool(circular_depth)
        self.hard_support_output = bool(hard_support_output)
        self.nonnegative_output = bool(nonnegative_output)
        self.support_condition = dict(support_condition or {})

        head = lambda cin, cout: Conv3dCircularDepth(cin, cout, kernel_size=(3, 3, 3), bias=bias, circular_depth=self.circular_depth)
        resblk = lambda channels: ResBlock3D(channels, bias=bias, circular_depth=self.circular_depth, act=act)

        self.m_head = head(self.in_nc, int(nc[0]))
        self.m_down1 = nn.Sequential(
            *[resblk(int(nc[0])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[0]), int(nc[1]), bias=bias, circular_depth=self.circular_depth),
        )
        self.m_down2 = nn.Sequential(
            *[resblk(int(nc[1])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[1]), int(nc[2]), bias=bias, circular_depth=self.circular_depth),
        )
        self.m_down3 = nn.Sequential(
            *[resblk(int(nc[2])) for _ in range(int(nb))],
            downsample_strideconv_3d(int(nc[2]), int(nc[3]), bias=bias, circular_depth=self.circular_depth),
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

        self.m_delta = head(int(nc[0]), self.out_nc)
        self.m_gate = head(int(nc[0]) + self.in_nc, self.out_nc)
        nn.init.zeros_(self.m_gate.conv.weight)
        if self.m_gate.conv.bias is not None:
            nn.init.constant_(self.m_gate.conv.bias, float(gate_bias))

    def _split_input(self, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = x0[:, : self.obs_nc]
        support_mask = x0[:, self.obs_nc : self.obs_nc + 1]
        return obs, support_mask

    def forward(
        self,
        x0: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x0.ndim != 5 or int(x0.shape[1]) != self.in_nc:
            raise ValueError(f"SupportAwareUNet3DRes expects B,C,D,H,W with C={self.in_nc}, got {tuple(x0.shape)}")

        obs, support_mask = self._split_input(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        feat = x + x1

        delta = self.m_delta(feat)
        gate = torch.sigmoid(self.m_gate(torch.cat([feat, x0], dim=1)))
        out = obs + gate * delta
        if self.hard_support_output:
            out = out * support_mask
        if self.nonnegative_output:
            out = F.relu(out)
            if self.hard_support_output:
                out = out * support_mask

        if not return_aux:
            return out

        aux = {
            "gate_mean": gate.mean(),
            "gate_map": gate,
        }
        return out, aux

    def build_condition_input(self, proj_count: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(proj_count, torch.Tensor):
            arr = proj_count.detach().cpu().numpy()
        else:
            arr = np.asarray(proj_count)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 4 and int(arr.shape[0]) == self.in_nc:
            obs = arr[0]
            cond = arr[1:]
            obs = obs[None, ...]
            return np.concatenate([obs, cond.astype(np.float32, copy=False)], axis=0)
        if arr.ndim != 3:
            raise ValueError(f"build_condition_input expects (V,H,W) or (C,V,H,W), got {arr.shape}")

        cond = build_support_condition_tensors(arr, self.support_condition)
        channels = [arr.astype(np.float32, copy=False), cond["mask"]]
        if "distance" in cond:
            channels.append(cond["distance"])
        return np.stack(channels, axis=0).astype(np.float32, copy=False)

    def prepare_inference_input(self, proj_count: np.ndarray, max_value: float) -> np.ndarray:
        x = self.build_condition_input(proj_count)
        x = np.asarray(x, dtype=np.float32)
        x_norm = x.copy()
        x_norm[: self.obs_nc] = x_norm[: self.obs_nc] / float(max_value)
        return x_norm

    def warmstart_from_legacy_unetres(
        self,
        load_path: str | Path,
        *,
        param_key: str = "params_ema",
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
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
        reused = 0
        skipped: list[str] = []

        for key, value in cleaned.items():
            if key == "m_head.conv.weight":
                target_value = target_state[key]
                if value.ndim == 5 and target_value.ndim == 5:
                    merged = torch.zeros_like(target_value)
                    copy_channels = min(int(value.shape[1]), int(self.obs_nc), int(target_value.shape[1]))
                    merged[:, :copy_channels] = value[:, :copy_channels]
                    loadable[key] = merged
                    reused += 1
                    continue
            if key == "m_tail.conv.weight":
                target_key = "m_delta.conv.weight"
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    loadable[target_key] = value
                    reused += 1
                    continue
            if key == "m_tail.conv.bias":
                target_key = "m_delta.conv.bias"
                if target_key in target_state and target_state[target_key].shape == value.shape:
                    loadable[target_key] = value
                    reused += 1
                    continue
            if key in target_state and target_state[key].shape == value.shape:
                loadable[key] = value
                reused += 1
                continue
            skipped.append(key)

        missing, unexpected = self.load_state_dict(loadable, strict=False)
        skipped.extend([f"missing:{key}" for key in missing])
        skipped.extend([f"unexpected:{key}" for key in unexpected])
        return {
            "path": path,
            "param_key": param_key,
            "reused_keys": reused,
            "skipped_keys": skipped,
        }
