#!/usr/bin/env python3
"""
生成旋转的 3D 体积渲染 GIF
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
from PIL import Image
import imageio

def load_reconstruction(filepath, dtype='float32'):
    """加载重建文件"""
    if dtype == 'float32':
        data = np.fromfile(filepath, dtype=np.float32)
    else:  # uint16
        data = np.fromfile(filepath, dtype=np.uint16).astype(np.float32)

    if data.size == 128 * 128 * 128:
        return data.reshape(128, 128, 128)
    else:
        raise ValueError(f"文件大小不匹配！期望 128×128×128，实际 {data.size} 个值")

def create_volume_rendering(volume, angle, threshold=None, opacity=0.8):
    """创建体积渲染（MIP 投影，带旋转）"""
    # 旋转体积（绕 Z 轴）
    from scipy.ndimage import rotate
    # 简化：使用 MIP 投影 + 旋转
    # 先做 MIP 投影到 XY 平面
    mip = np.max(volume, axis=2)  # Z 方向投影

    # 旋转 MIP 图像
    rotated = rotate(mip, angle, reshape=False, order=1, mode='constant', cval=0)

    return rotated

def generate_rotating_gif(volume, title, output_path, num_frames=60, threshold=None):
    """生成旋转 GIF"""
    print(f"   生成 {num_frames} 帧旋转 GIF...")

    # 计算每帧的角度
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    # 准备数据：使用 MIP 投影
    # 为了更好的 3D 效果，我们使用多个角度的投影
    frames = []

    vmax = np.percentile(volume[volume > 0], 99.9) if np.any(volume > 0) else volume.max()
    vmin = 0

    for i, angle in enumerate(angles):
        if (i + 1) % 10 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{title} - Rotation {angle:.0f}°', fontsize=14, fontweight='bold')

        # 1. XY 平面 MIP（从 Z 方向看）
        mip_xy = np.max(volume, axis=2)
        axes[0].imshow(mip_xy, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('MIP (XY plane)', fontweight='bold')
        axes[0].axis('off')

        # 2. XZ 平面 MIP（从 Y 方向看，旋转）
        mip_xz = np.max(volume, axis=1)
        # 旋转这个投影
        from scipy.ndimage import rotate
        mip_xz_rotated = rotate(mip_xz, angle, reshape=False, order=1, mode='constant', cval=0)
        axes[1].imshow(mip_xz_rotated, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'MIP (XZ plane, rotated {angle:.0f}°)', fontweight='bold')
        axes[1].axis('off')

        # 3. YZ 平面 MIP（从 X 方向看，旋转）
        mip_yz = np.max(volume, axis=0)
        mip_yz_rotated = rotate(mip_yz, angle, reshape=False, order=1, mode='constant', cval=0)
        axes[2].imshow(mip_yz_rotated, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'MIP (YZ plane, rotated {angle:.0f}°)', fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        # 转换为数组
        fig.canvas.draw()
        # 兼容新旧 matplotlib API
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]  # 去掉 alpha 通道
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(output_path, frames, duration=0.1, loop=0)
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧)")

def generate_simple_rotating_gif(volume, title, output_path, num_frames=60):
    """生成简单的旋转 GIF（单视图，更快）"""
    print(f"   生成 {num_frames} 帧旋转 GIF（单视图）...")

    # 使用 MIP 投影
    mip = np.max(volume, axis=2)  # Z 方向投影

    vmax = np.percentile(volume[volume > 0], 99.9) if np.any(volume > 0) else volume.max()
    vmin = 0

    angles = np.linspace(0, 360, num_frames, endpoint=False)
    frames = []

    from scipy.ndimage import rotate

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 旋转 MIP
        rotated = rotate(mip, angle, reshape=False, order=1, mode='constant', cval=0)

        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rotated, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title} - Rotation {angle:.0f}°', fontsize=14, fontweight='bold')
        ax.axis('off')

        # 添加统计信息
        stats_text = f"Min: {volume.min():.2f}  Max: {volume.max():.2f}  Mean: {volume.mean():.2f}"
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 转换为数组
        fig.canvas.draw()
        # 兼容新旧 matplotlib API
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]  # 去掉 alpha 通道
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(output_path, frames, duration=0.1, loop=0)
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧, {output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def rotate_volume_around_x_axis(volume, angle_deg):
    """围绕X轴旋转3D体积（角度：0°=轴向视图，90°=冠状视图）"""
    from scipy.ndimage import rotate

    # 围绕X轴旋转：在YZ平面上旋转
    # angle_deg: 0° = 轴向视图（从上往下看，投影沿Z轴）
    #            90° = 冠状视图（从前往后看，投影沿Y轴）

    # 对于每个X切片，在YZ平面上旋转
    rotated_volume = np.zeros_like(volume)

    for x in range(volume.shape[0]):
        yz_slice = volume[x, :, :]  # (Y, Z)
        rotated_slice = rotate(yz_slice, angle_deg, axes=(0, 1), reshape=False,
                             order=1, mode='constant', cval=0)
        rotated_volume[x, :, :] = rotated_slice

    return rotated_volume

def project_volume(volume, projection_axis='z'):
    """对体积进行MIP投影"""
    if projection_axis == 'z':  # 轴向视图
        return np.max(volume, axis=2)
    elif projection_axis == 'y':  # 冠状视图
        return np.max(volume, axis=1)
    elif projection_axis == 'x':  # 矢状视图
        return np.max(volume, axis=0)
    else:
        raise ValueError(f"Unknown projection axis: {projection_axis}")

def generate_axial_to_coronal_rotating_gif(volume, title, output_path, num_frames=120):
    """围绕X轴（轴向和冠状视图的交线）旋转360度观察重建图像"""
    print(f"   生成 {num_frames} 帧旋转 GIF（围绕X轴旋转360°，单一视图投影）...")

    # 使用更高的 vmax，让高亮区域更容易区分
    # 直接使用最大值，或者使用更高的百分位数
    if np.any(volume > 0):
        # 使用 99.95 百分位数，如果还是太小就使用最大值
        vmax_p99_95 = np.percentile(volume[volume > 0], 99.95)
        vmax_actual = volume.max()
        # 使用两者中较大的，确保高亮区域有足够的动态范围
        vmax = max(vmax_p99_95, vmax_actual * 0.8)  # 至少是最大值的 80%
    else:
        vmax = volume.max()
    vmin = 0

    print(f"   vmax: {vmax:.2f} (实际最大值: {volume.max():.2f})")

    # 生成旋转角度：0°到360°
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 围绕X轴旋转体积
        rotated_vol = rotate_volume_around_x_axis(volume, angle)

        # 始终沿Z方向进行MIP投影（从上往下看，单一视图）
        # 这样旋转后，可以看到不同角度的侧面效果
        projection = np.max(rotated_vol, axis=2)  # Z方向投影 (X, Y)

        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(projection, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title} - Rotation around X-axis: {angle:.0f}°',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # 添加统计信息
        stats_text = f"Min: {volume.min():.2f}  Max: {volume.max():.2f}  Mean: {volume.mean():.2f}"
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes,
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # 转换为数组
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF（优化压缩）
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(
        output_path,
        frames,
        duration=0.1,  # 稍微增加帧间隔
        loop=0
    )
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧, {output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def generate_triple_rotating_gif(volume1, volume2, volume3, title1, title2, title3,
                                  output_path, num_frames=120):
    """生成并排显示三个体积旋转的 GIF，使用统一的 vmax"""
    print(f"   生成 {num_frames} 帧三视图旋转 GIF（围绕X轴旋转360°）...")

    # 计算统一的 vmax
    def calc_vmax(vol):
        if np.any(vol > 0):
            vmax_p99_95 = np.percentile(vol[vol > 0], 99.95)
            vmax_actual = vol.max()
            return max(vmax_p99_95, vmax_actual * 0.8)
        else:
            return vol.max()

    vmax1 = calc_vmax(volume1)
    vmax2 = calc_vmax(volume2)
    vmax3 = calc_vmax(volume3)
    vmax = max(vmax1, vmax2, vmax3)  # 使用统一的 vmax
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    print(f"     - {title1}: 范围 [{volume1.min():.2f}, {volume1.max():.2f}]")
    print(f"     - {title2}: 范围 [{volume2.min():.2f}, {volume2.max():.2f}]")
    print(f"     - {title3}: 范围 [{volume3.min():.2f}, {volume3.max():.2f}]")

    # 生成旋转角度：0°到360°
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 围绕X轴旋转三个体积
        rotated_vol1 = rotate_volume_around_x_axis(volume1, angle)
        rotated_vol2 = rotate_volume_around_x_axis(volume2, angle)
        rotated_vol3 = rotate_volume_around_x_axis(volume3, angle)

        # 始终沿Z方向进行MIP投影
        projection1 = np.max(rotated_vol1, axis=2)
        projection2 = np.max(rotated_vol2, axis=2)
        projection3 = np.max(rotated_vol3, axis=2)

        # 创建三视图并排图形（减小尺寸以降低文件大小）
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=80)

        # 左侧：OSEM1
        im1 = axes[0].imshow(projection1, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title1} - Rotation: {angle:.0f}°',
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 中间：OSEM2
        im2 = axes[1].imshow(projection2, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title2} - Rotation: {angle:.0f}°',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 右侧：OSEM3
        im3 = axes[2].imshow(projection3, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'{title3} - Rotation: {angle:.0f}°',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # 添加统一的颜色条
        cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.05)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1)

        # 转换为数组
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF（优化压缩）
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(
        output_path,
        frames,
        duration=0.1,  # 稍微增加帧间隔
        loop=0
    )
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧, {output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def generate_quad_rotating_gif(volume1, volume2, volume3, volume4, title1, title2, title3, title4,
                                output_path, num_frames=120):
    """生成并排显示四个体积旋转的 GIF，使用统一的 vmax"""
    print(f"   生成 {num_frames} 帧四视图旋转 GIF（围绕X轴旋转360°）...")

    # 计算统一的 vmax
    def calc_vmax(vol):
        if np.any(vol > 0):
            vmax_p99_95 = np.percentile(vol[vol > 0], 99.95)
            vmax_actual = vol.max()
            return max(vmax_p99_95, vmax_actual * 0.8)
        else:
            return vol.max()

    vmax1 = calc_vmax(volume1)
    vmax2 = calc_vmax(volume2)
    vmax3 = calc_vmax(volume3)
    vmax4 = calc_vmax(volume4)
    vmax = max(vmax1, vmax2, vmax3, vmax4)  # 使用统一的 vmax
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    print(f"     - {title1}: 范围 [{volume1.min():.2f}, {volume1.max():.2f}]")
    print(f"     - {title2}: 范围 [{volume2.min():.2f}, {volume2.max():.2f}]")
    print(f"     - {title3}: 范围 [{volume3.min():.2f}, {volume3.max():.2f}]")
    print(f"     - {title4}: 范围 [{volume4.min():.2f}, {volume4.max():.2f}]")

    # 生成旋转角度：0°到360°
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 围绕X轴旋转四个体积
        rotated_vol1 = rotate_volume_around_x_axis(volume1, angle)
        rotated_vol2 = rotate_volume_around_x_axis(volume2, angle)
        rotated_vol3 = rotate_volume_around_x_axis(volume3, angle)
        rotated_vol4 = rotate_volume_around_x_axis(volume4, angle)

        # 始终沿Z方向进行MIP投影
        projection1 = np.max(rotated_vol1, axis=2)
        projection2 = np.max(rotated_vol2, axis=2)
        projection3 = np.max(rotated_vol3, axis=2)
        projection4 = np.max(rotated_vol4, axis=2)

        # 创建四视图并排图形（减小尺寸以降低文件大小）
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=80)

        # OSEM1
        im1 = axes[0].imshow(projection1, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title1} - {angle:.0f}°', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # OSEM2
        im2 = axes[1].imshow(projection2, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title2} - {angle:.0f}°', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # OSEM3
        im3 = axes[2].imshow(projection3, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'{title3} - {angle:.0f}°', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # OSEM4
        im4 = axes[3].imshow(projection4, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[3].set_title(f'{title4} - {angle:.0f}°', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # 添加统一的颜色条
        cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.05)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1)

        # 转换为数组
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF（优化压缩）
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(
        output_path,
        frames,
        duration=0.1,  # 稍微增加帧间隔
        loop=0
    )
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧, {output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def generate_dual_rotating_gif(volume1, volume2, title1, title2, output_path, num_frames=120):
    """生成并排显示两个体积旋转的 GIF，使用统一的 vmax"""
    print(f"   生成 {num_frames} 帧双视图旋转 GIF（围绕X轴旋转360°）...")

    # 计算统一的 vmax（使用两个体积中较大的值）
    def calc_vmax(vol):
        if np.any(vol > 0):
            vmax_p99_95 = np.percentile(vol[vol > 0], 99.95)
            vmax_actual = vol.max()
            return max(vmax_p99_95, vmax_actual * 0.8)
        else:
            return vol.max()

    vmax1 = calc_vmax(volume1)
    vmax2 = calc_vmax(volume2)
    vmax = max(vmax1, vmax2)  # 使用统一的 vmax
    vmin = 0

    print(f"   统一 vmax: {vmax:.2f}")
    print(f"     - {title1}: 范围 [{volume1.min():.2f}, {volume1.max():.2f}]")
    print(f"     - {title2}: 范围 [{volume2.min():.2f}, {volume2.max():.2f}]")

    # 生成旋转角度：0°到360°
    angles = np.linspace(0, 360, num_frames, endpoint=False)

    frames = []

    for i, angle in enumerate(angles):
        if (i + 1) % 15 == 0:
            print(f"     处理帧 {i+1}/{num_frames} (角度: {angle:.1f}°)")

        # 围绕X轴旋转两个体积
        rotated_vol1 = rotate_volume_around_x_axis(volume1, angle)
        rotated_vol2 = rotate_volume_around_x_axis(volume2, angle)

        # 始终沿Z方向进行MIP投影
        projection1 = np.max(rotated_vol1, axis=2)
        projection2 = np.max(rotated_vol2, axis=2)

        # 创建并排图形（减小尺寸以降低文件大小）
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=80)

        # 左侧：OSEM1
        im1 = axes[0].imshow(projection1, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title1} - Rotation: {angle:.0f}°',
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 右侧：OSEM2
        im2 = axes[1].imshow(projection2, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title2} - Rotation: {angle:.0f}°',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 添加统一的颜色条
        cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, fraction=0.05)

        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)

        # 转换为数组
        fig.canvas.draw()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            buf = buf[:, :, :3]
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)

        plt.close(fig)

    # 保存为 GIF（优化压缩）
    print(f"   保存 GIF 到: {output_path}")
    imageio.mimsave(
        output_path,
        frames,
        duration=0.1,  # 稍微增加帧间隔
        loop=0
    )
    print(f"   ✅ GIF 已保存 ({len(frames)} 帧, {output_path.stat().st_size / 1024 / 1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description='生成旋转的 3D 体积渲染 GIF')
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件路径（默认：可视化 data 目录下所有文件）')
    parser.add_argument('--output_dir', type=str, default='spect_ct/results',
                       help='输出目录')
    parser.add_argument('--dtype', type=str, default='auto',
                       choices=['auto', 'float32', 'uint16'],
                       help='数据类型（auto 自动检测）')
    parser.add_argument('--frames', type=int, default=60,
                       help='GIF 帧数（默认：60）')
    parser.add_argument('--simple', action='store_true',
                       help='使用简单模式（单视图，更快）')
    parser.add_argument('--axial-to-coronal', action='store_true',
                       help='从轴向视图旋转到冠状视图，然后在冠状视图上旋转')
    parser.add_argument('--dual', action='store_true',
                       help='生成并排对比 GIF（需要 OSEM1.dat 和 OSEM2.dat）')
    parser.add_argument('--triple', action='store_true',
                       help='生成三视图对比 GIF（OSEM1, OSEM2, OSEM3）')
    parser.add_argument('--quad', action='store_true',
                       help='生成四视图对比 GIF（OSEM1, OSEM2, OSEM3, OSEM4）')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果指定了四视图模式
    if args.quad:
        data_dir = Path('spect_ct/data')

        file1_path = data_dir / 'OSEM1.dat'
        file2_path = data_dir / 'OSEM2.dat'
        file3_path = data_dir / 'OSEM3.dat'
        file4_path = data_dir / 'OSEM4.dat'

        if not all(p.exists() for p in [file1_path, file2_path, file3_path, file4_path]):
            print("❌ 未找到 OSEM1.dat、OSEM2.dat、OSEM3.dat 或 OSEM4.dat！")
            for i, filepath in enumerate([file1_path, file2_path, file3_path, file4_path], 1):
                if not filepath.exists():
                    print(f"   缺少: {filepath}")
            return

        print("找到 OSEM1.dat, OSEM2.dat, OSEM3.dat 和 OSEM4.dat，生成四视图对比 GIF...")
        print("=" * 70)

        # 加载四个文件
        volumes = []
        titles = ['OSEM1', 'OSEM2', 'OSEM3', 'OSEM4']
        for filepath, title in zip([file1_path, file2_path, file3_path, file4_path], titles):
            file_size = filepath.stat().st_size
            expected_float32 = 128 * 128 * 128 * 4

            if abs(file_size - expected_float32) < 1000:
                dtype = 'float32'
            else:
                print(f"⚠️  文件大小不匹配: {file_size} 字节")
                return

            volume = load_reconstruction(filepath, dtype=dtype)
            volumes.append(volume)
            print(f"  ✅ {title}: 形状 {volume.shape}, 范围 [{volume.min():.2f}, {volume.max():.2f}]")

        output_path = output_dir / "OSEM1_OSEM2_OSEM3_OSEM4_quad_rotating.gif"
        generate_quad_rotating_gif(volumes[0], volumes[1], volumes[2], volumes[3],
                                    titles[0], titles[1], titles[2], titles[3],
                                    output_path, args.frames)
        print("\n" + "=" * 70)
        print("✅ 四视图对比 GIF 生成完成！")
        print(f"   文件: {output_path}")
        return

    # 如果指定了三视图模式
    if args.triple:
        data_dir = Path('spect_ct/data')

        file1_path = data_dir / 'OSEM1.dat'
        file2_path = data_dir / 'OSEM2.dat'
        file3_path = data_dir / 'OSEM3.dat'

        if not file1_path.exists() or not file2_path.exists() or not file3_path.exists():
            print("❌ 未找到 OSEM1.dat、OSEM2.dat 或 OSEM3.dat！")
            if not file1_path.exists():
                print(f"   缺少: {file1_path}")
            if not file2_path.exists():
                print(f"   缺少: {file2_path}")
            if not file3_path.exists():
                print(f"   缺少: {file3_path}")
            return

        print("找到 OSEM1.dat, OSEM2.dat 和 OSEM3.dat，生成三视图对比 GIF...")
        print("=" * 70)

        # 加载三个文件
        volumes = []
        titles = ['OSEM1', 'OSEM2', 'OSEM3']
        for filepath, title in zip([file1_path, file2_path, file3_path], titles):
            file_size = filepath.stat().st_size
            expected_float32 = 128 * 128 * 128 * 4

            if abs(file_size - expected_float32) < 1000:
                dtype = 'float32'
            else:
                print(f"⚠️  文件大小不匹配: {file_size} 字节")
                return

            volume = load_reconstruction(filepath, dtype=dtype)
            volumes.append(volume)
            print(f"   {title}: 形状 {volume.shape}, 范围 [{volume.min():.2f}, {volume.max():.2f}]")

        # 生成三视图 GIF
        output_path = output_dir / "OSEM1_OSEM2_OSEM3_triple_rotating.gif"
        generate_triple_rotating_gif(volumes[0], volumes[1], volumes[2],
                                     titles[0], titles[1], titles[2],
                                     output_path, args.frames)
        print("\n" + "=" * 70)
        print("✅ 三视图对比 GIF 生成完成！")
        return

    # 如果指定了双视图模式
    if args.dual:
        data_dir = Path('spect_ct/data')
        file1_path = data_dir / 'OSEM1.dat'
        file2_path = data_dir / 'OSEM2.dat'

        if not file1_path.exists() or not file2_path.exists():
            print("❌ 未找到 OSEM1.dat 或 OSEM2.dat！")
            return

        print("找到 OSEM1.dat 和 OSEM2.dat，生成并排对比 GIF...")
        print("=" * 70)

        # 加载两个文件
        volumes = []
        for filepath in [file1_path, file2_path]:
            file_size = filepath.stat().st_size
            expected_float32 = 128 * 128 * 128 * 4

            if abs(file_size - expected_float32) < 1000:
                dtype = 'float32'
            else:
                print(f"⚠️  文件大小不匹配: {file_size} 字节")
                return

            volume = load_reconstruction(filepath, dtype=dtype)
            volumes.append(volume)
            print(f"   {filepath.name}: 形状 {volume.shape}, 范围 [{volume.min():.2f}, {volume.max():.2f}]")

        # 生成并排 GIF
        output_path = output_dir / "OSEM1_OSEM2_dual_rotating.gif"
        generate_dual_rotating_gif(volumes[0], volumes[1], 'OSEM1', 'OSEM2',
                                   output_path, args.frames)
        print("\n" + "=" * 70)
        print("✅ 并排对比 GIF 生成完成！")
        return

    # 如果指定了双视图模式
    if args.dual:
        data_dir = Path('spect_ct/data')
        file1_path = data_dir / 'OSEM1.dat'
        file2_path = data_dir / 'OSEM2.dat'

        if not file1_path.exists() or not file2_path.exists():
            print("❌ 未找到 OSEM1.dat 或 OSEM2.dat！")
            return

        print("找到 OSEM1.dat 和 OSEM2.dat，生成并排对比 GIF...")
        print("=" * 70)

        # 加载两个文件
        volumes = []
        for filepath in [file1_path, file2_path]:
            file_size = filepath.stat().st_size
            expected_float32 = 128 * 128 * 128 * 4

            if abs(file_size - expected_float32) < 1000:
                dtype = 'float32'
            else:
                print(f"⚠️  文件大小不匹配: {file_size} 字节")
                return

            volume = load_reconstruction(filepath, dtype=dtype)
            volumes.append(volume)
            print(f"   {filepath.name}: 形状 {volume.shape}, 范围 [{volume.min():.2f}, {volume.max():.2f}]")

        # 生成并排 GIF
        output_path = output_dir / "OSEM1_OSEM2_dual_rotating.gif"
        generate_dual_rotating_gif(volumes[0], volumes[1], 'OSEM1', 'OSEM2',
                                   output_path, args.frames)
        print("\n" + "=" * 70)
        print("✅ 并排对比 GIF 生成完成！")
        return

    # 确定输入文件
    if args.input:
        files = [Path(args.input)]
    else:
        data_dir = Path('spect_ct/data')
        files = list(data_dir.glob('*.dat'))

    if not files:
        print("❌ 未找到重建文件！")
        return

    print(f"找到 {len(files)} 个重建文件，开始生成旋转 GIF...")
    print("=" * 70)

    for filepath in files:
        print(f"\n📁 处理: {filepath.name}")

        # 自动检测数据类型
        file_size = filepath.stat().st_size
        expected_uint16 = 128 * 128 * 128 * 2
        expected_float32 = 128 * 128 * 128 * 4

        if args.dtype == 'auto':
            if abs(file_size - expected_uint16) < 1000:
                dtype = 'uint16'
            elif abs(file_size - expected_float32) < 1000:
                dtype = 'float32'
            else:
                print(f"⚠️  文件大小不匹配: {file_size} 字节")
                continue
        else:
            dtype = args.dtype

        print(f"   格式: {dtype}, 大小: {file_size / 1024 / 1024:.2f} MB")

        try:
            # 加载数据
            volume = load_reconstruction(filepath, dtype=dtype)
            print(f"   形状: {volume.shape}")
            print(f"   范围: [{volume.min():.2f}, {volume.max():.2f}]")

            # 生成 GIF
            if args.axial_to_coronal:
                output_path = output_dir / f"{filepath.stem}_axial_to_coronal_rotating.gif"
                generate_axial_to_coronal_rotating_gif(volume, filepath.name, output_path, args.frames)
            elif args.simple:
                output_path = output_dir / f"{filepath.stem}_rotating.gif"
                generate_simple_rotating_gif(volume, filepath.name, output_path, args.frames)
            else:
                output_path = output_dir / f"{filepath.stem}_rotating.gif"
                generate_rotating_gif(volume, filepath.name, output_path, args.frames)

            print(f"   ✅ GIF 生成完成")

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 所有文件处理完成！")

if __name__ == '__main__':
    main()

