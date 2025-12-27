#!/usr/bin/env python3
"""
生成 1x3 OSEM 重建对比 GIF

布局：
左边：原始投影重建（Iter 10）| 中间：降噪投影重建（Iter 10）| 右边：降噪投影重建（Iter 30）

所有重建使用相同的 vmax
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
from PIL import Image
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

    # 添加标题
    if title:
        ax.text(
            0.5,
            1.12,
            title,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=9,
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
        img_array = buf.reshape(h, w, 4)[:, :, :3]  # 去掉 alpha 通道
    except AttributeError:
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape(h, w, 3)

    plt.close(fig)
    return Image.fromarray(img_array, mode='RGB')


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


def generate_comparison_gif(
    original_file,
    denoised_file,
    denoised_iter30_file,
    output_file,
    vmax=None,
    use_log1p=True,
    fps=10,
    shape=(128, 128, 128),
):
    """
    生成 1x3 对比 GIF

    Args:
        original_file: 原始投影重建文件路径
        denoised_file: 降噪投影重建文件路径（10次迭代）
        denoised_iter30_file: 降噪投影重建文件路径（30次迭代）
        output_file: 输出 GIF 文件路径
        vmax: 显示最大值（None 则自动计算）
        use_log1p: 是否使用 log1p 增强
        fps: 帧率
        shape: 数据形状
    """
    print(f"读取重建文件...")
    vol_original = load_dat_file(original_file, shape)
    vol_denoised = load_dat_file(denoised_file, shape)
    vol_denoised_iter30 = load_dat_file(denoised_iter30_file, shape)

    # 计算 vmax（使用 99.95 分位数）
    if vmax is None:
        vmax = max(
            np.percentile(vol_original, 99.95),
            np.percentile(vol_denoised, 99.95),
            np.percentile(vol_denoised_iter30, 99.95)
        )
    
    print(f"使用 vmax = {vmax:.4f} (99.95分位数)")

    # 生成旋转角度（参考 2x3 脚本）
    num_frames = 120
    recon_angle_start = 90.0  # 起始角度
    recon_angle_step = 360.0 / num_frames
    
    frames = []
    
    for i in tqdm(range(num_frames), desc="生成帧"):
        angle = recon_angle_start + i * recon_angle_step
        
        # 旋转体数据（围绕 X 轴）
        vol_rot_orig = rotate_volume_around_x(vol_original, angle)
        vol_rot_den = rotate_volume_around_x(vol_denoised, angle)
        vol_rot_den30 = rotate_volume_around_x(vol_denoised_iter30, angle)
        
        # 计算 MIP
        mip_orig = compute_mip(vol_rot_orig, axis=2)
        mip_den = compute_mip(vol_rot_den, axis=2)
        mip_den30 = compute_mip(vol_rot_den30, axis=2)
        
        # 上下翻转（参考 2x3 脚本）
        mip_orig = np.flipud(mip_orig)
        mip_den = np.flipud(mip_den)
        mip_den30 = np.flipud(mip_den30)

        # 创建子图（使用 hot colormap，参考 2x3 脚本）
        img_orig = create_subplot_with_colorbar(
            mip_orig, 0, vmax, 'hot', 
            'Original\n(Iter 10)', use_log1p
        )
        img_den = create_subplot_with_colorbar(
            mip_den, 0, vmax, 'hot', 
            'Denoised\n(Iter 10)', use_log1p
        )
        img_den30 = create_subplot_with_colorbar(
            mip_den30, 0, vmax, 'hot', 
            'Denoised\n(Iter 30)', use_log1p
        )

        # 拼接成 1x3 布局
        w, h = img_orig.width, img_orig.height
        canvas = Image.new('RGB', (w * 3, h))
        canvas.paste(img_orig, (0, 0))
        canvas.paste(img_den, (w, 0))
        canvas.paste(img_den30, (w * 2, 0))
        
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
    parser = argparse.ArgumentParser(description="生成 1x3 OSEM 重建对比 GIF")
    parser.add_argument('--original', required=True, help='原始投影重建文件')
    parser.add_argument('--denoised', required=True, help='降噪投影重建文件（Iter 10）')
    parser.add_argument('--denoised-iter30', required=True, help='降噪投影重建文件（Iter 30）')
    parser.add_argument('--output', required=True, help='输出 GIF 文件')
    parser.add_argument('--vmax', type=float, default=None, help='显示最大值（默认自动计算）')
    parser.add_argument('--no-log1p', action='store_true', help='不使用 log1p 增强')
    parser.add_argument('--fps', type=int, default=10, help='帧率（默认 10）')

    args = parser.parse_args()

    generate_comparison_gif(
        args.original,
        args.denoised,
        args.denoised_iter30,
        args.output,
        vmax=args.vmax,
        use_log1p=not args.no_log1p,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()

