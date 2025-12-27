#!/usr/bin/env python3
"""
生成 2x2 OSEM 重建对比 GIF

布局：
左上：Original (Iter 10) | 右上：Original (Iter 30)
左下：Denoised (Iter 10) | 右下：Denoised (Iter 30)

所有重建使用相同的 vmax (99.95分位数)
"""

import numpy as np
from pathlib import Path
import argparse
import math
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate as scipy_rotate
from tqdm import tqdm


def get_colormap(name):
    """获取 matplotlib colormap"""
    try:
        return matplotlib.colormaps[name]
    except (KeyError, AttributeError):
        return cm.get_cmap(name)


class Log1pNormalize(Normalize):
    """支持 log1p 的归一化类"""

    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # 先线性归一化
        result = super().__call__(value, clip)
        # 应用 log1p 变换
        result = np.log1p(9.0 * result) / math.log1p(9.0)
        return result


def create_subplot_with_colorbar(data, vmin, vmax, cmap_name='gray', title=None, use_log1p=False):
    """创建带 colorbar 的子图"""
    fig = plt.figure(figsize=(2.2, 1.8), dpi=150)
    ax = fig.add_axes([0, 0.05, 0.75, 0.75])
    cax = fig.add_axes([0.78, 0.1, 0.05, 0.7])

    # 设置归一化方式
    if use_log1p:
        norm = Log1pNormalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # 显示图像
    im = ax.imshow(data, cmap=cmap_name, norm=norm, aspect='equal', interpolation='nearest')
    ax.axis('off')

    # 添加标题（缩小字体）
    if title:
        ax.text(
            0.5,
            1.12,
            title,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold',
        )

    # 添加 colorbar
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)

    # 渲染为图像
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    try:
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = buf.reshape(h, w, 4)[:, :, :3]
    except AttributeError:
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape(h, w, 3)

    plt.close(fig)
    return Image.fromarray(img_array, mode='RGB')


def rotate_volume_around_x(volume, angle_deg):
    """围绕 X 轴旋转体积"""
    rotated = scipy_rotate(
        volume,
        angle_deg,
        axes=(1, 2),
        reshape=False,
        order=1,
        mode='constant',
        cval=0
    )
    return rotated


def compute_mip(volume, axis=2):
    """计算最大强度投影（MIP）"""
    return np.max(volume, axis=axis)


def load_dat_file(file_path, shape=(128, 128, 128), dtype=np.float32):
    """读取 .dat 文件"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = np.fromfile(file_path, dtype=dtype)
    expected_size = np.prod(shape)

    if data.size != expected_size:
        raise ValueError(
            f"数据大小不匹配: 期望 {expected_size}, 实际 {data.size}\n"
            f"文件: {file_path}"
        )

    return data.reshape(shape)


def generate_comparison_gif(
    original_iter10_file,
    original_iter30_file,
    denoised_iter10_file,
    denoised_iter30_file,
    output_file,
    vmax=None,
    use_log1p=True,
    fps=10,
    shape=(128, 128, 128),
):
    """
    生成 2x2 对比 GIF

    Args:
        original_iter10_file: 原始投影重建文件（Iter 10）
        original_iter30_file: 原始投影重建文件（Iter 30）
        denoised_iter10_file: 降噪投影重建文件（Iter 10）
        denoised_iter30_file: 降噪投影重建文件（Iter 30）
        output_file: 输出 GIF 文件路径
        vmax: 显示最大值（None 则使用 99.95分位数）
        use_log1p: 是否使用 log1p 增强
        fps: 帧率
        shape: 数据形状
    """
    print(f"读取重建文件...")
    vol_orig_10 = load_dat_file(original_iter10_file, shape)
    vol_orig_30 = load_dat_file(original_iter30_file, shape)
    vol_den_10 = load_dat_file(denoised_iter10_file, shape)
    vol_den_30 = load_dat_file(denoised_iter30_file, shape)

    # 计算总活度（所有体素值的总和）
    total_orig_10 = np.sum(vol_orig_10)
    total_orig_30 = np.sum(vol_orig_30)
    total_den_10 = np.sum(vol_den_10)
    total_den_30 = np.sum(vol_den_30)

    # 计算降噪前后的相对差异（百分比）
    diff_iter10 = ((total_den_10 - total_orig_10) / total_orig_10) * 100
    diff_iter30 = ((total_den_30 - total_orig_30) / total_orig_30) * 100

    print(f"\n📊 总活度统计：")
    print(f"   Original Iter10:  {total_orig_10:.2e}  (基准)")
    print(f"   Original Iter30:  {total_orig_30:.2e}")
    print(f"   Denoised Iter10:  {total_den_10:.2e}  ({diff_iter10:+.2f}%)")
    print(f"   Denoised Iter30:  {total_den_30:.2e}  ({diff_iter30:+.2f}%)")
    print(f"   降噪影响 (Iter10): {diff_iter10:+.2f}%")
    print(f"   降噪影响 (Iter30): {diff_iter30:+.2f}%")

    # 计算 vmax (99.95分位数)
    if vmax is None:
        vmax = max(
            np.percentile(vol_orig_10, 99.95),
            np.percentile(vol_orig_30, 99.95),
            np.percentile(vol_den_10, 99.95),
            np.percentile(vol_den_30, 99.95)
        )

    print(f"\n使用 vmax = {vmax:.4f} (99.95分位数)")

    # 生成旋转角度
    num_frames = 120
    recon_angle_start = 90.0
    recon_angle_step = 360.0 / num_frames

    frames = []

    for i in tqdm(range(num_frames), desc="生成帧"):
        angle = recon_angle_start + i * recon_angle_step

        # 旋转体数据
        vol_rot_orig_10 = rotate_volume_around_x(vol_orig_10, angle)
        vol_rot_orig_30 = rotate_volume_around_x(vol_orig_30, angle)
        vol_rot_den_10 = rotate_volume_around_x(vol_den_10, angle)
        vol_rot_den_30 = rotate_volume_around_x(vol_den_30, angle)

        # 计算 MIP
        mip_orig_10 = compute_mip(vol_rot_orig_10, axis=2)
        mip_orig_30 = compute_mip(vol_rot_orig_30, axis=2)
        mip_den_10 = compute_mip(vol_rot_den_10, axis=2)
        mip_den_30 = compute_mip(vol_rot_den_30, axis=2)

        # 上下翻转
        mip_orig_10 = np.flipud(mip_orig_10)
        mip_orig_30 = np.flipud(mip_orig_30)
        mip_den_10 = np.flipud(mip_den_10)
        mip_den_30 = np.flipud(mip_den_30)

        # 创建子图（2x2）
        img_orig_10 = create_subplot_with_colorbar(
            mip_orig_10, 0, vmax, 'hot',
            'Original\n(Iter 10)', use_log1p
        )
        img_orig_30 = create_subplot_with_colorbar(
            mip_orig_30, 0, vmax, 'hot',
            'Original\n(Iter 30)', use_log1p
        )
        img_den_10 = create_subplot_with_colorbar(
            mip_den_10, 0, vmax, 'hot',
            'Denoised\n(Iter 10)', use_log1p
        )
        img_den_30 = create_subplot_with_colorbar(
            mip_den_30, 0, vmax, 'hot',
            'Denoised\n(Iter 30)', use_log1p
        )

        # 拼接成 2x2 布局
        w, h = img_orig_10.width, img_orig_10.height

        # 添加顶部文本区域来显示总活度统计
        text_height = 35
        canvas = Image.new('RGB', (w * 2, h * 2 + text_height), color=(255, 255, 255))

        # 添加统计文本
        draw = ImageDraw.Draw(canvas)
        try:
            # 尝试使用 DejaVu Sans 字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except:
            # 如果失败，使用默认字体
            font = ImageFont.load_default()

        stats_text = f"Total Activity Change: Iter10={diff_iter10:+.2f}%  |  Iter30={diff_iter30:+.2f}%"
        # 计算文本位置（居中）
        bbox = draw.textbbox((0, 0), stats_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (w * 2 - text_width) // 2
        draw.text((text_x, 10), stats_text, fill=(0, 0, 0), font=font)

        # 粘贴 4 个子图
        canvas.paste(img_orig_10, (0, text_height))
        canvas.paste(img_orig_30, (w, text_height))
        canvas.paste(img_den_10, (0, h + text_height))
        canvas.paste(img_den_30, (w, h + text_height))

        frames.append(canvas)

    # 保存 GIF
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    print(f"✅ GIF 已保存: {output_path}")
    print(f"   尺寸: {frames[0].size}")
    print(f"   帧数: {len(frames)}")


def main():
    parser = argparse.ArgumentParser(description="生成 2x2 OSEM 重建对比 GIF")
    parser.add_argument('--original-iter10', required=True, help='原始投影重建文件（Iter 10）')
    parser.add_argument('--original-iter30', required=True, help='原始投影重建文件（Iter 30）')
    parser.add_argument('--denoised-iter10', required=True, help='降噪投影重建文件（Iter 10）')
    parser.add_argument('--denoised-iter30', required=True, help='降噪投影重建文件（Iter 30）')
    parser.add_argument('--output', required=True, help='输出 GIF 文件')
    parser.add_argument('--vmax', type=float, default=None, help='显示最大值（默认使用99.95分位数）')
    parser.add_argument('--no-log1p', action='store_true', help='不使用 log1p 增强')
    parser.add_argument('--fps', type=int, default=10, help='帧率（默认 10）')

    args = parser.parse_args()

    generate_comparison_gif(
        args.original_iter10,
        args.original_iter30,
        args.denoised_iter10,
        args.denoised_iter30,
        args.output,
        vmax=args.vmax,
        use_log1p=not args.no_log1p,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()

