from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY


def _as_tensor_01(img: np.ndarray, input_order: str) -> torch.Tensor:
    """Convert ndarray to torch tensor in [0,1], shape (1,C,H,W)."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'input_order must be HWC or CHW, got {input_order}')
    x = img
    if x.ndim == 2:
        x = x[..., None]
    if input_order == 'HWC':
        x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x).astype(np.float32, copy=False)
    t = torch.from_numpy(x).unsqueeze(0)
    return t


def _to_3ch(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 3-channel. Input shape (1,C,H,W)."""
    c = int(t.shape[1])
    if c == 3:
        return t
    if c == 1:
        return t.repeat(1, 3, 1, 1)
    # For multi-channel medical inputs, compute LPIPS per-channel and average externally.
    raise ValueError(f'LPIPS metric expects 1 or 3 channels per call, got {c}')


_LPIPS_CACHE: dict[tuple[str, str], "torch.nn.Module"] = {}


def _get_lpips(net: str, device: torch.device):
    """Get cached LPIPS model to avoid re-loading weights for every image."""
    key = (net, str(device))
    m = _LPIPS_CACHE.get(key)
    if m is not None:
        return m
    import lpips  # type: ignore
    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()
    _LPIPS_CACHE[key] = loss_fn
    return loss_fn


@METRIC_REGISTRY.register()
def calculate_lpips(
    img,
    img2,
    input_order: str = 'HWC',
    net: str = 'alex',
    device: str = 'cuda',
    channel_mode: str = 'avg',  # avg | first
    **kwargs,
):
    """LPIPS metric (optional dependency).

    Notes for medical/count data:
    - LPIPS is trained on natural RGB images. Interpret with caution.
    - For 1-channel data we replicate to 3 channels.
    - For 2-channel data you can set channel_mode=avg to average per-channel LPIPS.
    """
    try:
        import lpips  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError('LPIPS metric requires `pip install lpips`.') from e

    if isinstance(img, torch.Tensor):
        # expect (N,C,H,W) in [0,1]
        pred = img.detach().float()
        ref = img2.detach().float()
    else:
        pred = _as_tensor_01(img, input_order=input_order)
        ref = _as_tensor_01(img2, input_order=input_order)

    # handle 2-ch: compute per-channel and average
    c = int(pred.shape[1])
    if c == 2:
        if channel_mode == 'first':
            pred = pred[:, 0:1, ...]
            ref = ref[:, 0:1, ...]
            c = 1
        elif channel_mode == 'avg':
            v0 = calculate_lpips(pred[:, 0:1, ...], ref[:, 0:1, ...], input_order='CHW', net=net, device=device)
            v1 = calculate_lpips(pred[:, 1:2, ...], ref[:, 1:2, ...], input_order='CHW', net=net, device=device)
            return float((v0 + v1) * 0.5)
        else:
            raise ValueError(f'Unknown channel_mode: {channel_mode}')

    pred3 = _to_3ch(pred)
    ref3 = _to_3ch(ref)

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    loss_fn = _get_lpips(net=net, device=dev)

    pred3 = pred3.to(dev)
    ref3 = ref3.to(dev)

    # LPIPS expects [-1,1]
    pred3 = pred3 * 2.0 - 1.0
    ref3 = ref3 * 2.0 - 1.0

    with torch.no_grad():
        val = loss_fn(pred3, ref3)
    return float(val.mean().item())


