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


def load_basicsr_net(config_path: Path, checkpoint_path: Path, device: str = "cuda") -> LoadedNet:
    """统一的 BasicSR ckpt/config 加载逻辑（优先 params_ema）。"""
    # 延迟导入，避免 `--help` 依赖可选依赖（cv2 等）
    from basicsr.archs import build_network  # type: ignore
    from basicsr.utils.options import yaml_load  # type: ignore

    cfg = yaml_load(str(config_path))
    net = build_network(cfg["network_g"])

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    used_key: Literal["params_ema", "params", "raw"]
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        net.load_state_dict(ckpt["params_ema"], strict=True)
        used_key = "params_ema"
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
    """对 (views,h,w) 逐视角推理，输入/输出均为 count domain。"""
    d = _resolve_device(device)
    net.eval()
    net.to(d)

    v, h, w = proj_count.shape
    out = np.empty((v, h, w), dtype=np.float32)
    mv = float(max_value)

    for i in range(v):
        x = proj_count[i].astype(np.float32, copy=False) / mv
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


def recon_total_counts_clip(recon: np.ndarray) -> float:
    x = np.clip(recon.astype(np.float64, copy=False), 0.0, None)
    return float(np.sum(x))











