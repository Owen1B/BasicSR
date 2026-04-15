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


def scalar_to_u8_cmap(arr: np.ndarray, *, vmin: float, vmax: float, cmap_name: str = "hot") -> np.ndarray:
    """Map scalar image to RGB uint8 with a matplotlib-like colormap."""
    vmin_f = float(vmin)
    vmax_f = float(vmax)
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-6
    normed = np.clip((arr.astype(np.float32, copy=False) - vmin_f) / (vmax_f - vmin_f), 0.0, 1.0)
    try:
        cmap = get_colormap(str(cmap_name))
        return (cmap(normed)[..., :3] * 255.0).round().astype(np.uint8)
    except Exception:
        u = (normed * 255.0).round().astype(np.uint8)
        return np.repeat(u[:, :, None], 3, axis=2)


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
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0.05, 0.75, 0.75])
    cax = fig.add_axes([0.78, 0.1, 0.05, 0.7])
    ax.set_facecolor("black")
    cax.set_facecolor("black")

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
            color="white",
        )

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5, colors="white", labelcolor="white", length=2, width=0.6)
    try:
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.6)
    except Exception:
        pass

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


def render_2x4_nogap_png(
    *,
    proj_imgs: list[np.ndarray],
    recon_imgs: list[np.ndarray],
    proj_texts: list[str] | None = None,
    recon_texts: list[str] | None = None,
    out_png: Path,
    proj_vmin: float,
    proj_vmax: float,
    recon_vmin: float,
    recon_vmax: float,
    use_log1p: bool = False,
    dpi: int = 150,
) -> Path:
    """渲染 2x4 网格到单张 PNG：

    - 子图之间无间隔（wspace/hspace=0）
    - 每一行仅最右侧一个 colorbar

    proj_imgs/recon_imgs: 每行 4 张 2D 图（灰度）。长度必须为 4。
    """
    if len(proj_imgs) != 4 or len(recon_imgs) != 4:
        raise ValueError(f"expected 4 proj + 4 recon, got proj={len(proj_imgs)} recon={len(recon_imgs)}")
    if proj_texts is not None and len(proj_texts) != 4:
        raise ValueError(f"expected 4 proj_texts, got {len(proj_texts)}")
    if recon_texts is not None and len(recon_texts) != 4:
        raise ValueError(f"expected 4 recon_texts, got {len(recon_texts)}")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    norm_proj = Log1pNormalize(vmin=float(proj_vmin), vmax=float(proj_vmax)) if use_log1p else Normalize(vmin=float(proj_vmin), vmax=float(proj_vmax))
    norm_recon = Log1pNormalize(vmin=float(recon_vmin), vmax=float(recon_vmax)) if use_log1p else Normalize(vmin=float(recon_vmin), vmax=float(recon_vmax))

    img = render_2x4_nogap_frame(
        proj_imgs=proj_imgs,
        recon_imgs=recon_imgs,
        proj_texts=proj_texts,
        recon_texts=recon_texts,
        proj_norm=norm_proj,
        recon_norm=norm_recon,
        dpi=dpi,
    )
    img.save(str(out_png))
    return out_png


def render_2x4_nogap_frame(
    *,
    proj_imgs: list[np.ndarray],
    recon_imgs: list[np.ndarray],
    proj_texts: list[str] | None = None,
    recon_texts: list[str] | None = None,
    proj_norm: Normalize,
    recon_norm: Normalize,
    dpi: int = 150,
    figsize: tuple[float, float] = (8.2, 4.4),
) -> Image.Image:
    """Render a single 2x4 no-gap frame to a PIL Image (RGB), with one colorbar per row."""
    if len(proj_imgs) != 4 or len(recon_imgs) != 4:
        raise ValueError(f"expected 4 proj + 4 recon, got proj={len(proj_imgs)} recon={len(recon_imgs)}")
    if proj_texts is not None and len(proj_texts) != 4:
        raise ValueError(f"expected 4 proj_texts, got {len(proj_texts)}")
    if recon_texts is not None and len(recon_texts) != 4:
        raise ValueError(f"expected 4 recon_texts, got {len(recon_texts)}")

    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    fig.patch.set_facecolor("black")

    gs = fig.add_gridspec(
        2,
        6,
        # 4 tiles + spacer + cbar (cbar wider; also reserve right margin for tick labels)
        width_ratios=[1, 1, 1, 1, 0.01, 0.045],
        height_ratios=[1, 1],
        wspace=0.0,
        hspace=0.0,
        left=0.0,
        right=0.97,
        bottom=0.0,
        top=1.0,
    )

    axes_top = [fig.add_subplot(gs[0, j]) for j in range(4)]
    axes_bot = [fig.add_subplot(gs[1, j]) for j in range(4)]
    cax_top_slot = fig.add_subplot(gs[0, 5])
    cax_bot_slot = fig.add_subplot(gs[1, 5])

    ims_top = []
    for ax, img in zip(axes_top, proj_imgs):
        im = ax.imshow(img, cmap="gray", norm=proj_norm, aspect="auto", interpolation="nearest")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ims_top.append(im)

    ims_bot = []
    for ax, img in zip(axes_bot, recon_imgs):
        im = ax.imshow(img, cmap="hot", norm=recon_norm, aspect="auto", interpolation="nearest")
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ims_bot.append(im)

    # per-tile stats text
    def _annot(ax, text: str) -> None:
        if not text:
            return
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6,
            color="white",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.55), ec="none"),
        )

    if proj_texts is not None:
        for ax, t in zip(axes_top, proj_texts):
            _annot(ax, t)
    if recon_texts is not None:
        for ax, t in zip(axes_bot, recon_texts):
            _annot(ax, t)

    # Make colorbars slightly shorter (leave some vertical padding).
    def _shrink_cax(slot_ax):
        bb = slot_ax.get_position()
        h = bb.height * 0.84
        y0 = bb.y0 + (bb.height - h) * 0.5
        slot_ax.set_position([bb.x0, y0, bb.width, h])

    _shrink_cax(cax_top_slot)
    _shrink_cax(cax_bot_slot)

    cb1 = fig.colorbar(ims_top[0], cax=cax_top_slot)
    cb1.ax.set_facecolor("black")
    cb1.ax.tick_params(labelsize=5, length=2, colors="white", labelcolor="white", width=0.6)
    try:
        cb1.outline.set_edgecolor("white")
        cb1.outline.set_linewidth(0.6)
    except Exception:
        pass

    cb2 = fig.colorbar(ims_bot[0], cax=cax_bot_slot)
    cb2.ax.set_facecolor("black")
    cb2.ax.tick_params(labelsize=5, length=2, colors="white", labelcolor="white", width=0.6)
    try:
        cb2.outline.set_edgecolor("white")
        cb2.outline.set_linewidth(0.6)
    except Exception:
        pass

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










