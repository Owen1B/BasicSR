from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class LoadedNet:
    net: torch.nn.Module
    used_key: Literal["params_ema", "params", "raw"]


def _resolve_device(device: str) -> str:
    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def _is_3d_net(net: torch.nn.Module) -> bool:
    """Heuristic: treat as 3D net if any Conv3d exists."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d):
            return True
    return False


def _first_conv2d_in_channels(net: torch.nn.Module) -> int | None:
    """Return in_channels of the first Conv2d layer, or None if absent."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            return int(m.in_channels)
    return None


def _first_conv3d_in_channels(net: torch.nn.Module) -> int | None:
    """Return in_channels of the first Conv3d layer, or None if absent."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d):
            return int(m.in_channels)
    return None


def load_basicsr_net(
    config_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    *,
    require_ema: bool = True,
) -> LoadedNet:
    """统一的 BasicSR ckpt/config 加载逻辑。

    - 默认 require_ema=True：只允许加载 `params_ema`（避免误用非 EMA 或随机权重）
    - 若 require_ema=False：按优先级 params_ema -> params -> raw
    """
    # 延迟导入，避免 `--help` 依赖可选依赖（cv2 等）
    from prime import register_all  # type: ignore
    from basicsr.archs import build_network  # type: ignore
    from basicsr.utils.options import yaml_load  # type: ignore

    # Ensure recipe modules override same-name symbols in registry.
    register_all()

    cfg = yaml_load(str(config_path))
    net = build_network(cfg["network_g"])

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    used_key: Literal["params_ema", "params", "raw"]
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        net.load_state_dict(ckpt["params_ema"], strict=True)
        used_key = "params_ema"
    elif bool(require_ema):
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<raw_state_dict>"]
        raise RuntimeError(
            "require_ema=True but checkpoint has no 'params_ema'. "
            f"ckpt={checkpoint_path} keys={keys}"
        )
    elif isinstance(ckpt, dict) and "params" in ckpt:
        net.load_state_dict(ckpt["params"], strict=True)
        used_key = "params"
    else:
        net.load_state_dict(ckpt, strict=True)
        used_key = "raw"

    d = _resolve_device(device)
    net.eval()
    net.to(d)
    return LoadedNet(net=net, used_key=used_key)


@torch.no_grad()
def denoise_views(
    net: torch.nn.Module,
    proj_count: np.ndarray,
    max_value: float,
    device: str = "cuda",
) -> np.ndarray:
    """推理入口：支持 2D(单视角/对位双视角/多视角) 与 3D(整块 volume) 网络。

    输入支持:
      - (V,H,W): 单输入通道
      - (C,V,H,W): 多输入通道（如 obs + aux）
    输出统一为 count domain 的 (V,H,W)（默认取输出第一个通道）。
    """
    d = _resolve_device(device)
    net.eval()
    net.to(d)

    x_in = np.asarray(proj_count, dtype=np.float32)
    if x_in.ndim == 3:
        c = 1
        v, h, w = x_in.shape
    elif x_in.ndim == 4:
        c, v, h, w = x_in.shape
    else:
        raise ValueError(f"proj_count must be 3D or 4D, got shape={x_in.shape}")

    mv = float(max_value)

    if _is_3d_net(net):
        prepare_input = getattr(net, "prepare_inference_input", None)
        if callable(prepare_input):
            x = np.asarray(prepare_input(x_in, max_value=float(mv)), dtype=np.float32)
        else:
            x = x_in / mv

        conv3d_in_channels = _first_conv3d_in_channels(net)
        if conv3d_in_channels is not None and x.ndim == 3 and int(conv3d_in_channels) != 1:
            raise ValueError(
                f"3D net input channels mismatch: net expects {conv3d_in_channels}, but proj_count has shape {x_in.shape}. "
                "Provide a network-specific prepare_inference_input or pass an explicit (C,V,H,W) input."
            )

        # 3D net expects (B,C,D,H,W) where D=views.
        if x.ndim == 3:
            xt = torch.from_numpy(x[None, None, ...]).to(device=d, dtype=torch.float32)  # (1,1,V,H,W)
        else:
            xt = torch.from_numpy(x[None, ...]).to(device=d, dtype=torch.float32)  # (1,C,V,H,W)
        yt = net(xt)
        yt = torch.clamp(yt, min=0.0)
        y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (Cout,V,H,W) or (V,H,W)
        if y.ndim == 4:
            # For denoising task we use first output channel as target projection.
            y = y[0]
        return np.asarray(y, dtype=np.float32) * mv

    # 2D net: choose stacked-channel mode, multi-view mode, or view-by-view mode.
    in_channels = _first_conv2d_in_channels(net)
    if in_channels is None:
        raise RuntimeError("Cannot infer Conv2d input channels from network.")

    if x_in.ndim == 4:
        # Multi-input-channel case (e.g., obs+aux), run per-view inference.
        if int(in_channels) != int(c):
            raise ValueError(
                f"2D net input channels mismatch: net expects {in_channels}, but proj_count has C={c}."
            )
        out = np.empty((v, h, w), dtype=np.float32)
        for i in range(v):
            x = x_in[:, i].astype(np.float32, copy=False) / mv  # (C,H,W)
            xt = torch.from_numpy(x[None, ...]).to(device=d, dtype=torch.float32)  # (1,C,H,W)
            yt = net(xt)
            yt = torch.clamp(yt, min=0.0)
            y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            if y.ndim == 2:
                out[i] = y * mv
            elif y.ndim == 3:
                out[i] = y[0] * mv
            else:
                raise ValueError(f"Unexpected 2D net output shape in multi-channel mode: {y.shape}")
        return out

    if in_channels == v:
        # Treat (V,H,W) as a C-channel input tensor once.
        x = x_in.astype(np.float32, copy=False) / mv
        xt = torch.from_numpy(x[None, ...]).to(device=d, dtype=torch.float32)  # (1,C,H,W)
        yt = net(xt)
        yt = torch.clamp(yt, min=0.0)
        y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (C_out,H,W)
        return y * mv

    # Multi-view 2D mode: in_channels = K where 1 < K < V.
    # - K=2: treat as opposite-view pair (offset=V/2) for SPECT angular symmetry.
    # - K>2: run circular sliding window and take center-channel prediction.
    if 1 < in_channels < v:
        k = int(in_channels)
        out = np.empty((v, h, w), dtype=np.float32)

        if k == 2:
            if v % 2 != 0:
                raise ValueError(f"Opposite-pair inference requires even views, got v={v}.")
            offset = v // 2
            for i in range(v):
                idx = [i, int((i + offset) % v)]
                x = x_in[idx].astype(np.float32, copy=False) / mv  # (2,H,W)
                xt = torch.from_numpy(x[None, ...]).to(device=d, dtype=torch.float32)  # (1,2,H,W)
                yt = net(xt)
                yt = torch.clamp(yt, min=0.0)
                y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                if y.ndim == 2:
                    y = y[None, ...]

                if int(y.shape[0]) == 1:
                    out[i] = y[0] * mv
                elif int(y.shape[0]) == 2:
                    # Input order is [i, i+offset], so channel-0 corresponds to target view i.
                    out[i] = y[0] * mv
                else:
                    raise ValueError(
                        f"Unsupported opposite-pair output channels: out_nc={int(y.shape[0])}, "
                        "expected 1 or 2."
                    )
            return out

        center = k // 2
        for i in range(v):
            idx = [int((i - center + t) % v) for t in range(k)]
            x = x_in[idx].astype(np.float32, copy=False) / mv  # (K,H,W)
            xt = torch.from_numpy(x[None, ...]).to(device=d, dtype=torch.float32)  # (1,K,H,W)
            yt = net(xt)
            yt = torch.clamp(yt, min=0.0)
            y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            if y.ndim == 2:
                y = y[None, ...]

            if int(y.shape[0]) == 1:
                out[i] = y[0] * mv
            elif int(y.shape[0]) == k:
                out[i] = y[center] * mv
            else:
                raise ValueError(
                    f"Unsupported multi-view output channels: out_nc={int(y.shape[0])}, "
                    f"expected 1 or in_nc({k})."
                )
        return out

    if in_channels != 1:
        raise ValueError(
            f"Unsupported 2D net input channels: {in_channels}. "
            f"Expected 1 (view-by-view), 2 (opposite-pair), "
            f"K with 2<K<{v} (multi-view sliding window), "
            f"or {v} (stacked channels for current input)."
        )

    # View-by-view mode for single-channel 2D nets.
    out = np.empty((v, h, w), dtype=np.float32)
    for i in range(v):
        x = x_in[i].astype(np.float32, copy=False) / mv
        xt = torch.from_numpy(x[None, None, ...]).to(device=d, dtype=torch.float32)
        yt = net(xt)
        yt = torch.clamp(yt, min=0.0)
        y = yt.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        out[i] = y * mv
    return out


def proj_total_counts_round_clip(proj: np.ndarray) -> float:
    x = np.round(proj.astype(np.float64, copy=False))
    x = np.clip(x, 0.0, None)
    return float(np.sum(x))


def proj_total_counts_clip(proj: np.ndarray) -> float:
    """Total counts for lambda-like fp32 projections (no rounding; clip>=0)."""
    x = np.clip(proj.astype(np.float64, copy=False), 0.0, None)
    return float(np.sum(x))


def recon_total_counts_clip(recon: np.ndarray) -> float:
    x = np.clip(recon.astype(np.float64, copy=False), 0.0, None)
    return float(np.sum(x))




