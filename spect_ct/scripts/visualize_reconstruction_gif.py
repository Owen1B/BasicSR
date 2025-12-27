#!/usr/bin/env python3
"""
生成重建图像的旋转 MIP GIF

使用与训练时相同的可视化风格：
- PIL 渲染（轻量、快速）
- log1p + gamma 增强
- 统一 vmax 归一化
- 简洁标签
"""

import numpy as np
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate as scipy_rotate


def load_reconstruction(filepath, dtype='float32'):
    """加载重建文件"""
    filepath = Path(filepath)
    if dtype == 'float32':
        data = np.fromfile(filepath, dtype=np.float32)
    else:  # uint16
        data = np.fromfile(filepath, dtype=np.uint16).astype(np.float32)

    if data.size == 128 * 128 * 128:
        return data.reshape(128, 128, 128)
    else:
        raise ValueError(f"文件大小不匹配！期望 128×128×128，实际 {data.size} 个值")


def normalize_to_u8(frame, vmin, vmax, gamma=0.8, log1p=True):
    """归一化到 0-255，使用 log1p + gamma 增强"""
    f = frame.astype(np.float32)
    f = np.clip(f, vmin, vmax)
    f = (f - vmin) / max(1e-8, (vmax - vmin))
    if log1p:
        # log1p 增强：增强低值区域的可见性
        f = np.log1p(9.0 * f) / math.log1p(9.0)
    if gamma != 1.0:
        f = np.power(np.clip(f, 0.0, 1.0), gamma)
    return (f * 255.0 + 0.5).astype(np.uint8)


def apply_hot_colormap(data_u8):
    """应用 hot colormap（黑 → 红 → 黄 → 白）"""
    # 简化版 hot colormap
    # R: 0-85 → 0-255, 85-255 → 255
    # G: 0-85 → 0, 85-170 → 0-255, 170-255 → 255
    # B: 0-170 → 0, 170-255 → 0-255
    h, w = data_u8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Red channel
    rgb[:, :, 0] = np.clip(data_u8 * 3, 0, 255).astype(np.uint8)

    # Green channel
    rgb[:, :, 1] = np.clip((data_u8.astype(np.int32) - 85) * 3, 0, 255).astype(np.uint8)

    # Blue channel
    rgb[:, :, 2] = np.clip((data_u8.astype(np.int32) - 170) * 3, 0, 255).astype(np.uint8)

    return rgb


def draw_label(img, text, position='top-left', font_size=12):
    """在图像上绘制标签"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if position == 'top-left':
        x0, y0 = pad, pad
    elif position == 'bottom-center':
        x0 = (img.width - tw) // 2
        y0 = img.height - th - pad - 4
    else:
        x0, y0 = pad, pad

    # 半透明背景
    draw.rectangle([x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2], fill=(0, 0, 0, 180))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
    return img


def compute_mip(volume, axis=2):
    """计算最大强度投影 (MIP)"""
    return np.max(volume, axis=axis)


def rotate_volume_around_x(volume, angle_deg):
    """围绕 X 轴旋转体积（在 YZ 平面上旋转）"""
    # 对整个体积进行旋转
    rotated = scipy_rotate(
        volume,
        angle_deg,
        axes=(1, 2),  # Y-Z plane
        reshape=False,
        order=1,
        mode='constant',
        cval=0
    )
    return rotated


def generate_single_volume_gif(volume, title, output_path, num_frames=60):
    """生成单个体积的旋转 MIP GIF"""
    print(f"   生成 {num_frames} 帧旋转 GIF...")

    # 计算 vmax（使用 99.9 百分位）
    if np.any(volume > 0):
        vmax = np.percentile(volume[volume > 0], 99.9)
    else:
        vmax = volume.max()
    vmin = 0

    print(f"   vmax: {vmax:.2f}, 范围: [{volume.min():.2f}, {volume.max():.2f}]")

    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     帧 {i+1}/{num_frames} (角度: {angle:.0f}°)")

        # 围绕 X 轴旋转
        rotated_vol = rotate_volume_around_x(volume, angle)

        # MIP 投影（沿 Z 轴）
        mip = compute_mip(rotated_vol, axis=2)

        # 归一化并应用 colormap
        mip_u8 = normalize_to_u8(mip, vmin, vmax, gamma=0.8, log1p=True)
        mip_rgb = apply_hot_colormap(mip_u8)

        # 创建 PIL 图像
        img = Image.fromarray(mip_rgb, mode='RGB')

        # 添加标签
        img = draw_label(img, f"{title}")
        img = draw_label(img, f"Angle: {angle:.0f}°", position='bottom-center')

        frames.append(img)

    # 保存 GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),  # 10 fps
        loop=0,
        optimize=False
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(output_path)


def generate_dual_volume_gif(volume1, volume2, title1, title2, output_path, num_frames=60):
    """生成两个体积并排对比的旋转 MIP GIF"""
    print(f"   生成 {num_frames} 帧双视图对比 GIF...")

    # 计算统一的 vmax
    def get_vmax(vol):
        if np.any(vol > 0):
            return np.percentile(vol[vol > 0], 99.9)
            return vol.max()

    vmax = max(get_vmax(volume1), get_vmax(volume2))
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    print(f"     - {title1}: [{volume1.min():.2f}, {volume1.max():.2f}]")
    print(f"     - {title2}: [{volume2.min():.2f}, {volume2.max():.2f}]")

    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     帧 {i+1}/{num_frames} (角度: {angle:.0f}°)")

        # 旋转两个体积
        rotated1 = rotate_volume_around_x(volume1, angle)
        rotated2 = rotate_volume_around_x(volume2, angle)

        # MIP 投影
        mip1 = compute_mip(rotated1, axis=2)
        mip2 = compute_mip(rotated2, axis=2)

        # 归一化并应用 colormap
        mip1_u8 = normalize_to_u8(mip1, vmin, vmax, gamma=0.8, log1p=True)
        mip2_u8 = normalize_to_u8(mip2, vmin, vmax, gamma=0.8, log1p=True)

        mip1_rgb = apply_hot_colormap(mip1_u8)
        mip2_rgb = apply_hot_colormap(mip2_u8)

        # 创建 PIL 图像
        img1 = Image.fromarray(mip1_rgb, mode='RGB')
        img2 = Image.fromarray(mip2_rgb, mode='RGB')

        # 添加标签
        img1 = draw_label(img1, title1)
        img2 = draw_label(img2, title2)

        # 合并为画布（2 列）
        canvas = Image.new('RGB', (img1.width * 2, img1.height))
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (img1.width, 0))

        # 添加角度信息
        canvas = draw_label(canvas, f"Angle: {angle:.0f}°", position='bottom-center')

        frames.append(canvas)

    # 保存 GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),  # 10 fps
        loop=0,
        optimize=False
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(output_path)


def generate_triple_volume_gif(volumes, titles, output_path, num_frames=60):
    """生成三个体积并排对比的旋转 MIP GIF"""
    print(f"   生成 {num_frames} 帧三视图对比 GIF...")

    # 计算统一的 vmax
    def get_vmax(vol):
        if np.any(vol > 0):
            return np.percentile(vol[vol > 0], 99.9)
            return vol.max()

    vmax = max(get_vmax(v) for v in volumes)
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    for title, vol in zip(titles, volumes):
        print(f"     - {title}: [{vol.min():.2f}, {vol.max():.2f}]")

    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     帧 {i+1}/{num_frames} (角度: {angle:.0f}°)")

        imgs = []
        for vol, title in zip(volumes, titles):
            rotated = rotate_volume_around_x(vol, angle)
            mip = compute_mip(rotated, axis=2)
            mip_u8 = normalize_to_u8(mip, vmin, vmax, gamma=0.8, log1p=True)
            mip_rgb = apply_hot_colormap(mip_u8)
            img = Image.fromarray(mip_rgb, mode='RGB')
            img = draw_label(img, title)
            imgs.append(img)

        # 合并为画布（3 列）
        canvas = Image.new('RGB', (imgs[0].width * 3, imgs[0].height))
        for j, img in enumerate(imgs):
            canvas.paste(img, (img.width * j, 0))

        canvas = draw_label(canvas, f"Angle: {angle:.0f}°", position='bottom-center')
        frames.append(canvas)

    # 保存 GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),
        loop=0,
        optimize=False
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(output_path)


def generate_quad_volume_gif(volumes, titles, output_path, num_frames=60):
    """生成四个体积并排对比的旋转 MIP GIF（2x2 布局）"""
    print(f"   生成 {num_frames} 帧四视图对比 GIF (2x2)...")

    # 计算统一的 vmax
    def get_vmax(vol):
        if np.any(vol > 0):
            return np.percentile(vol[vol > 0], 99.9)
            return vol.max()

    vmax = max(get_vmax(v) for v in volumes)
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    for title, vol in zip(titles, volumes):
        print(f"     - {title}: [{vol.min():.2f}, {vol.max():.2f}]")

    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     帧 {i+1}/{num_frames} (角度: {angle:.0f}°)")

        imgs = []
        for vol, title in zip(volumes, titles):
            rotated = rotate_volume_around_x(vol, angle)
            mip = compute_mip(rotated, axis=2)
            mip_u8 = normalize_to_u8(mip, vmin, vmax, gamma=0.8, log1p=True)
            mip_rgb = apply_hot_colormap(mip_u8)
            img = Image.fromarray(mip_rgb, mode='RGB')
            img = draw_label(img, title)
            imgs.append(img)

        # 2x2 布局
        w, h = imgs[0].width, imgs[0].height
        canvas = Image.new('RGB', (w * 2, h * 2))
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (w, 0))
        canvas.paste(imgs[2], (0, h))
        canvas.paste(imgs[3], (w, h))

        canvas = draw_label(canvas, f"Angle: {angle:.0f}°", position='bottom-center')
        frames.append(canvas)

    # 保存 GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),
        loop=0,
        optimize=False
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='生成重建图像的旋转 MIP GIF')
    parser.add_argument('--input', type=str, nargs='+', default=None,
                        help='输入文件路径（可指定多个）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 GIF 路径')
    parser.add_argument('--output-dir', type=str, default='spect_ct/results',
                        help='输出目录（当未指定 --output 时使用）')
    parser.add_argument('--titles', type=str, nargs='+', default=None,
                        help='每个输入文件的标题')
    parser.add_argument('--frames', type=int, default=60,
                       help='GIF 帧数（默认：60）')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'uint16'],
                        help='数据类型')
    args = parser.parse_args()

    # 如果没有指定输入，使用默认数据目录
    if args.input is None:
        data_dir = Path('spect_ct/data')
        files = sorted(data_dir.glob('*.dat'))
        if not files:
            print("❌ 未找到 .dat 文件！")
            return
            else:
        files = [Path(f) for f in args.input]

    # 检查文件是否存在
    for f in files:
        if not f.exists():
            print(f"❌ 文件不存在: {f}")
            return

    # 设置标题
    if args.titles:
        titles = args.titles
            else:
        titles = [f.stem for f in files]

    # 确保标题数量与文件数量匹配
    if len(titles) < len(files):
        titles.extend([f.stem for f in files[len(titles):]])

    # 加载数据
    print(f"加载 {len(files)} 个重建文件...")
        volumes = []
    for filepath, title in zip(files, titles):
        print(f"  📁 {filepath.name}")
        vol = load_reconstruction(filepath, dtype=args.dtype)
        volumes.append(vol)
        print(f"     形状: {vol.shape}, 范围: [{vol.min():.2f}, {vol.max():.2f}]")

    # 设置输出路径
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        # 根据文件数量生成输出文件名
        if len(files) == 1:
            output_path = output_dir / f"{files[0].stem}_rotating.gif"
        else:
            output_path = output_dir / f"{'_'.join(titles)}_comparison.gif"

            # 生成 GIF
    print("=" * 60)
    if len(volumes) == 1:
        generate_single_volume_gif(volumes[0], titles[0], output_path, args.frames)
    elif len(volumes) == 2:
        generate_dual_volume_gif(volumes[0], volumes[1], titles[0], titles[1], output_path, args.frames)
    elif len(volumes) == 3:
        generate_triple_volume_gif(volumes, titles, output_path, args.frames)
    elif len(volumes) == 4:
        generate_quad_volume_gif(volumes, titles, output_path, args.frames)
            else:
        print(f"⚠️ 支持 1-4 个文件，当前有 {len(volumes)} 个")
        # 只处理前 4 个
        generate_quad_volume_gif(volumes[:4], titles[:4], output_path, args.frames)

    print("=" * 60)
    print("✅ 完成！")


if __name__ == '__main__':
    main()
