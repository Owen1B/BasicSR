#!/usr/bin/env python3
"""
生成 2x2 布局的重建 MIP vs 衰减 MIP 对比 GIF

布局（每帧）：
  第一行：左 = 重建 MIP（沿 X 方向 MIP，hot），右 = 衰减 MIP（沿 X 方向 MIP，gray）
  第二行：左 = 重建 MIP（沿 Y 方向 MIP，hot），右 = 衰减 MIP（沿 Y 方向 MIP，gray）

旋转方式：
  - 重建和衰减体使用相同角度、相同旋转轴，同时旋转，确保解剖对齐。

默认假设：
  - 重建体：float32，尺寸 128x128x128，对应 RCAC-20s-1.img
  - 衰减体：float32，尺寸 128x128x128，对应 PostAtten.dat

用法示例（单个病人）：
  cd /home/owen/code/BasicSR
  python3 spect_ct/scripts/generate_recon_atten_2x2_gif.py --patient FanCuiLing

如果不指定 --patient，后续可以扩展为遍历所有病人，这里先按单病人用法为主。
"""

import argparse
import math
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
    """支持 log1p 的归一化类（备用，目前默认关闭）"""

    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip)
        result = np.log1p(9.0 * result) / math.log1p(9.0)
        return result


def create_subplot_with_colorbar(data, vmin, vmax, cmap_name="gray", title=None, use_log1p=False):
    """创建带 colorbar 的子图（风格与其他脚本统一）"""
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
    """围绕 X 轴旋转体积"""
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


def load_recon_volume(path: Path) -> np.ndarray:
    """加载重建体（float32，128x128x128）"""
    data = np.fromfile(path, dtype=np.float32)
    if data.size != 128 * 128 * 128:
        raise ValueError(f"重建大小不匹配: {data.size}, 期待 128x128x128")
    vol = data.reshape(128, 128, 128)  # [z,y,x]
    print(f"   重建体尺寸: {vol.shape}, min={vol.min():.4f}, max={vol.max():.4f}, mean={vol.mean():.4f}")
    return vol


def load_atten_volume(path: Path) -> np.ndarray:
    """加载衰减体（float32，优先尝试 128x128x128）"""
    data = np.fromfile(path, dtype=np.float32)
    n = data.size
    candidates = [(128, 128, 128), (256, 256, 32)]
    for sx, sy, sz in candidates:
        if sx * sy * sz == n:
            vol = data.reshape(sz, sy, sx)  # [z,y,x]
            print(f"   衰减体尺寸: {vol.shape}, min={vol.min():.6f}, max={vol.max():.6f}, mean={vol.mean():.6f}")
            return vol
    raise ValueError(f"衰减体大小不匹配: 元素数={n}，无法匹配常见尺寸")


def generate_2x2_gif(patient, recon_vol, atten_vol, output_path, num_frames=120, use_log1p=False):
    """生成 2x2 布局 GIF：重建/衰减 MIP 同步旋转"""
    z, y, x = recon_vol.shape
    assert atten_vol.shape == recon_vol.shape, "重建与衰减体尺寸不一致"

    # 统一 vmax 范围：重建和衰减分别各自确定
    vmax_recon = float(np.percentile(recon_vol[recon_vol > 0], 99.95)) if np.any(recon_vol > 0) else float(
        recon_vol.max()
    )
    vmax_atten = float(np.percentile(atten_vol[atten_vol > 0], 99.9)) if np.any(atten_vol > 0) else float(
        atten_vol.max()
    )
    vmin_recon = 0.0
    vmin_atten = 0.0

    print(f"   重建可视化范围: [{vmin_recon:.4f}, {vmax_recon:.4f}] (p99.95)")
    print(f"   衰减可视化范围: [{vmin_atten:.6f}, {vmax_atten:.6f}] (p99.9)")

    frames = []
    angle_start = 90.0
    angle_step = 360.0 / num_frames

    for i in range(num_frames):
        angle = angle_start + i * angle_step

        # 旋转
        recon_rot = rotate_volume_around_x(recon_vol, angle)
        atten_rot = rotate_volume_around_x(atten_vol, angle)

        # 第一行：沿 X 方向 MIP（axis=2）
        recon_mip_x = np.max(recon_rot, axis=2)
        atten_mip_x = np.max(atten_rot, axis=2)

        # 第二行：沿 Y 方向 MIP（axis=1）
        recon_mip_y = np.max(recon_rot, axis=1)
        atten_mip_y = np.max(atten_rot, axis=1)

        # 上下翻转，使解剖方向更自然
        recon_mip_x = np.flipud(recon_mip_x)
        atten_mip_x = np.flipud(atten_mip_x)
        recon_mip_y = np.flipud(recon_mip_y)
        atten_mip_y = np.flipud(atten_mip_y)

        # 子图
        img_recon_x = create_subplot_with_colorbar(
            recon_mip_x,
            vmin_recon,
            vmax_recon,
            cmap_name="hot",
            title=f"Recon MIP-X ({angle:.1f}°)",
            use_log1p=use_log1p,
        )
        img_atten_x = create_subplot_with_colorbar(
            atten_mip_x,
            vmin_atten,
            vmax_atten,
            cmap_name="gray",
            title=f"Atten MIP-X ({angle:.1f}°)",
            use_log1p=False,
        )
        img_recon_y = create_subplot_with_colorbar(
            recon_mip_y,
            vmin_recon,
            vmax_recon,
            cmap_name="hot",
            title=f"Recon MIP-Y ({angle:.1f}°)",
            use_log1p=use_log1p,
        )
        img_atten_y = create_subplot_with_colorbar(
            atten_mip_y,
            vmin_atten,
            vmax_atten,
            cmap_name="gray",
            title=f"Atten MIP-Y ({angle:.1f}°)",
            use_log1p=False,
        )

        # 拼 2x2 画布
        w, h = img_recon_x.width, img_recon_x.height
        canvas = Image.new("RGB", (w * 2, h * 2))

        canvas.paste(img_recon_x, (0, 0))
        canvas.paste(img_atten_x, (w, 0))
        canvas.paste(img_recon_y, (0, h))
        canvas.paste(img_atten_y, (w, h))

        frames.append(canvas)

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
    parser = argparse.ArgumentParser(description="生成重建 MIP vs 衰减 MIP 的 2x2 对比 GIF")
    parser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="病人名称（如 BianChengMing）",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/bone_20230327",
        help="bone 数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spect_ct/results/recon_atten_2x2_gifs",
        help="输出目录",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="GIF 帧数",
    )
    parser.add_argument(
        "--log1p-recon",
        action="store_true",
        help="是否对重建使用 log1p 增强（衰减始终线性）",
    )
    args = parser.parse_args()

    patient = args.patient
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    recon_path = data_root / patient / "recon" / "RCAC-20s-1.img"
    atten_path = data_root / patient / "par" / "PostAtten.dat"

    print("=" * 60)
    print(f"病人:       {patient}")
    print(f"重建路径:   {recon_path}")
    print(f"衰减路径:   {atten_path}")
    print(f"输出目录:   {output_dir}")
    print(f"帧数:       {args.frames}")
    print(f"重建 log1p: {'开启' if args.log1p_recon else '关闭'}")
    print("=" * 60)

    if not recon_path.exists():
        raise FileNotFoundError(f"找不到重建文件: {recon_path}")
    if not atten_path.exists():
        raise FileNotFoundError(f"找不到衰减文件: {atten_path}")

    recon_vol = load_recon_volume(recon_path)
    atten_vol = load_atten_volume(atten_path)

    output_path = output_dir / f"{patient}_ReconAtten_2x2.gif"
    generate_2x2_gif(patient, recon_vol, atten_vol, output_path, num_frames=args.frames, use_log1p=args.log1p_recon)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()


