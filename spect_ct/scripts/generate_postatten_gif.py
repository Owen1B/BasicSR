#!/usr/bin/env python3
"""
为衰减图 PostAtten.dat 生成旋转 MIP GIF（单通道 1x1 布局）

用法示例：
    cd /home/owen/code/BasicSR
    python3 spect_ct/scripts/generate_postatten_gif.py \
        --atten-path datasets/bone_20230327/BianChengMing/par/PostAtten.dat \
        --output-dir spect_ct/results/atten_gifs \
        --frames 120 \
        --log1p

默认假设 PostAtten.dat 为 float32，尺寸为 128x128x128（与重建一致）。
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import rotate as scipy_rotate  # noqa: E402


def get_colormap(name):
    """获取 matplotlib colormap"""
    try:
        return matplotlib.colormaps[name]
    except (KeyError, AttributeError):
        return cm.get_cmap(name)


class Log1pNormalize(Normalize):
    """支持 log1p 的归一化类（和 generate_2x3_comparison_gif 中保持一致）"""

    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # 先线性归一化
        result = super().__call__(value, clip)
        # 应用 log1p 变换
        result = np.log1p(9.0 * result) / math.log1p(9.0)
        return result


def create_subplot_with_colorbar(data, vmin, vmax, cmap_name="gray", title=None, use_log1p=False):
    """创建带 colorbar 的子图（单幅图像，布局/风格与 2x3 脚本一致）"""
    fig = plt.figure(figsize=(2.2, 1.8), dpi=150)
    ax = fig.add_axes([0, 0.05, 0.75, 0.75])
    cax = fig.add_axes([0.78, 0.1, 0.05, 0.7])

    if use_log1p:
        norm = Log1pNormalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = get_colormap(cmap_name)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax.axis("off")

    if title:
        ax.text(
            0.5,
            1.12,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
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


def rotate_volume_around_x(volume, angle_deg):
    """围绕 X 轴旋转体积（与 2x3 脚本保持一致的实现）"""
    rotated = scipy_rotate(
        volume,
        angle_deg,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )
    return rotated


def load_attenuation_volume(filepath):
    """加载衰减图（默认 float32，优先尝试 128x128x128）"""
    filepath = Path(filepath)
    data = np.fromfile(filepath, dtype=np.float32)
    n = data.size

    # 常见候选尺寸
    candidates = [(128, 128, 128), (256, 256, 32)]
    for sx, sy, sz in candidates:
        if sx * sy * sz == n:
            vol = data.reshape(sz, sy, sx)  # [z, y, x]
            print(f"   使用体尺寸: {sz}x{sy}x{sx} (总元素 {n})")
            print(f"   统计: min={vol.min():.6f}, max={vol.max():.6f}, mean={vol.mean():.6f}")
            return vol

    raise ValueError(f"无法匹配体尺寸，元素数量={n}，请手动检查。")


def generate_atten_gif(atten_vol, output_path, num_frames=120, use_log1p=False):
    """对衰减图做旋转 MIP，生成 GIF（1x1 单幅）"""
    z, y, x = atten_vol.shape
    print(f"   衰减体尺寸: {z}x{y}x{x}")

    vmax = float(np.percentile(atten_vol[atten_vol > 0], 99.9)) if np.any(atten_vol > 0) else float(atten_vol.max())
    vmin = 0.0
    print(f"   可视化范围: vmin={vmin:.6f}, vmax={vmax:.6f} (p99.9)")

    frames = []
    # 与 2x3 脚本类似，做一圈 360° 旋转
    angle_start = 90.0  # 起始角度与重建 MIP 保持一致
    angle_step = 360.0 / num_frames

    for i in range(num_frames):
        angle = angle_start + i * angle_step
        rotated = rotate_volume_around_x(atten_vol, angle)
        mip = np.max(rotated, axis=2)  # [z, y, x] → 沿 x 方向做 MIP

        # 上下翻转，使解剖方向与重建 MIP 一致
        mip = np.flipud(mip)

        img = create_subplot_with_colorbar(
            mip,
            vmin,
            vmax,
            cmap_name="gray",
            title=f"PostAtten MIP ({angle:.1f}°)",
            use_log1p=use_log1p,
        )
        frames.append(img)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),  # 10 fps
        loop=0,
        optimize=False,
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="为衰减图 PostAtten.dat 生成旋转 MIP GIF")
    parser.add_argument(
        "--atten-path",
        type=str,
        default="datasets/bone_20230327/BianChengMing/par/PostAtten.dat",
        help="衰减图 .dat 路径（float32）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spect_ct/results/atten_gifs",
        help="输出目录",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="GIF 帧数（默认 120）",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="是否使用 log1p 对比度增强（推荐打开，和 2x3 脚本风格一致）",
    )
    args = parser.parse_args()

    atten_path = Path(args.atten_path)
    if not atten_path.exists():
        raise FileNotFoundError(f"找不到衰减文件: {atten_path}")

    print("=" * 60)
    print(f"衰减图路径: {atten_path}")
    print(f"输出目录:   {args.output_dir}")
    print(f"帧数:       {args.frames}")
    print(f"log1p 增强: {'开启' if args.log1p else '关闭'}")
    print("=" * 60)

    atten_vol = load_attenuation_volume(atten_path)

    patient_name = atten_path.parent.parent.name if atten_path.parent.parent.name else "patient"
    output_dir = Path(args.output_dir)
    output_filename = output_dir / f"{patient_name}_PostAtten_MIP.gif"

    generate_atten_gif(atten_vol, output_filename, num_frames=args.frames, use_log1p=args.log1p)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()


