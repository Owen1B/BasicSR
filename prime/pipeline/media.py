from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio  # type: ignore
import numpy as np


def _to_rgb_array(frame) -> np.ndarray:
    # PIL Image
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"))
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def write_mp4_frames(
    out_path: Path | str,
    frames: Iterable,
    *,
    fps: float = 10.0,
    codec: str = "libx264",
    crf: int | None = 18,
    pixelformat: str | None = "yuv420p",
    macro_block_size: int | None = 1,
    quality: int | None = None,
    extra_ffmpeg_params: list[str] | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer_kwargs: dict = {
        "format": "FFMPEG",
        "fps": float(fps) if float(fps) > 0 else 10.0,
        "codec": str(codec),
        "macro_block_size": macro_block_size,
    }
    if pixelformat is not None:
        writer_kwargs["pixelformat"] = pixelformat
    if quality is not None:
        writer_kwargs["quality"] = int(quality)

    ffmpeg_params: list[str] = []
    if crf is not None:
        ffmpeg_params.extend(["-crf", str(int(crf))])
    if extra_ffmpeg_params:
        ffmpeg_params.extend(extra_ffmpeg_params)
    if ffmpeg_params:
        writer_kwargs["ffmpeg_params"] = ffmpeg_params

    with imageio.get_writer(str(out_path), **writer_kwargs) as w:
        for fr in frames:
            w.append_data(_to_rgb_array(fr))
    return out_path

