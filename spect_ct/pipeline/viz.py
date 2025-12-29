from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from scipy.ndimage import rotate as scipy_rotate


def get_colormap(name: str):
    """获取 matplotlib colormap（兼容不同版本）"""
    try:
        return matplotlib.colormaps[name]
    except (KeyError, AttributeError):
        return cm.get_cmap(name)


class Log1pNormalize(Normalize):
    """支持 log1p 的归一化类"""

    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip)
        result = np.log1p(9.0 * result) / math.log1p(9.0)
        return result


def create_subplot_with_colorbar(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = "gray",
    title: Optional[str] = None,
    use_log1p: bool = False,
    figsize: Tuple[float, float] = (2.2, 1.8),
    dpi: int = 150,
    fontsize: int = 7,
) -> Image.Image:
    """创建带 colorbar 的子图并返回 PIL Image。"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0.05, 0.75, 0.75])
    cax = fig.add_axes([0.78, 0.1, 0.05, 0.7])

    norm = Log1pNormalize(vmin=vmin, vmax=vmax) if use_log1p else Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, cmap=cmap_name, norm=norm, aspect="equal", interpolation="nearest")
    ax.axis("off")

    if title:
        ax.text(
            0.5,
            1.12,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    try:
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = buf.reshape(h, w, 4)[:, :, :3]
    except AttributeError:
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = buf.reshape(h, w, 3)

    plt.close(fig)
    return Image.fromarray(img_array, mode="RGB")


def rotate_volume_around_x(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    """围绕 X 轴旋转体积（axes=(1,2)）。"""
    return scipy_rotate(volume, angle_deg, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0)


def compute_mip(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    return np.max(volume, axis=axis)


def calculate_vmax_percentile(volumes: list[np.ndarray], percentile: float = 99.95) -> float:
    vmaxs = [float(np.percentile(vol, percentile)) for vol in volumes]
    return float(max(vmaxs))


def pxx_nonzero(v: np.ndarray, q: float) -> float:
    if np.any(v > 0):
        return float(np.percentile(v[v > 0], q))
    return float(v.max())


def orth_slices(vol: np.ndarray, mid: int) -> List[np.ndarray]:
    axial = np.flipud(vol[mid, :, :])
    coronal = np.flipud(vol[:, mid, :])
    sagittal = np.flipud(vol[:, :, mid])
    return [axial, coronal, sagittal]


def draw_label(img: Image.Image, text: str) -> Image.Image:
    """Draw label on an RGB image (top-left)."""
    im = img.convert("RGBA")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([pad - 3, pad - 3, pad + tw + 3, pad + th + 3], fill=(0, 0, 0, 160))
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
    return im.convert("RGB")


def merge_gifs_row(gif_paths: List[Path], labels: List[str], out_gif: Path) -> None:
    """把多个 GIF 按列拼成一张大 GIF（1 x N）。"""
    if len(gif_paths) == 0:
        raise ValueError("gif_paths is empty")
    if len(gif_paths) != len(labels):
        raise ValueError("gif_paths and labels length mismatch")

    gifs = [Image.open(p) for p in gif_paths]
    try:
        frames_list: List[List[Image.Image]] = []
        durations: List[int] = []
        for g, label in zip(gifs, labels):
            durations.append(int(g.info.get("duration", 100)))
            frames = [draw_label(fr.convert("RGB"), label) for fr in ImageSequence.Iterator(g)]
            frames_list.append(frames)

        n_frames = min(len(fr) for fr in frames_list)
        if n_frames <= 0:
            raise ValueError("No frames found in input GIFs")

        tw, th = frames_list[0][0].size
        for k in range(len(frames_list)):
            frames_list[k] = [
                (fr.resize((tw, th), resample=Image.BILINEAR) if fr.size != (tw, th) else fr) for fr in frames_list[k][:n_frames]
            ]

        out_frames: List[Image.Image] = []
        for i in range(n_frames):
            canvas = Image.new("RGB", (tw * len(frames_list), th))
            for j, frames in enumerate(frames_list):
                canvas.paste(frames[i], (j * tw, 0))
            out_frames.append(canvas)

        out_gif.parent.mkdir(parents=True, exist_ok=True)
        out_frames[0].save(
            out_gif,
            save_all=True,
            append_images=out_frames[1:],
            duration=int(np.median(durations)) if durations else 100,
            loop=0,
            optimize=False,
        )
    finally:
        for g in gifs:
            try:
                g.close()
            except Exception:
                pass











