#!/usr/bin/env python3
"""
生成 2x3 布局的对比 GIF

第一行：原始投影 | 降噪投影 | 残差（红正蓝负）
第二行：原始重建 MIP | 降噪重建 MIP | 重建差异

支持 log1p 增强和不同的 vmax 模式
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
    """创建带 colorbar 的子图

    Args:
        data: 图像数据
        vmin, vmax: 数据范围
        cmap_name: colormap 名称
        title: 标题
        use_log1p: 是否使用 log1p 增强
    """
    fig = plt.figure(figsize=(2.2, 1.8), dpi=150)  # 增大尺寸提高像素：330x270 像素
    ax = fig.add_axes([0, 0.05, 0.75, 0.75])  # 左边 75% 给图像，bottom 降低，height 减小，为标题留空间
    cax = fig.add_axes([0.78, 0.1, 0.05, 0.7])  # 右边 colorbar，相应调整

    # 设置归一化方式
    if use_log1p:
        norm = Log1pNormalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # 显示图像
    im = ax.imshow(data, cmap=cmap_name, norm=norm, aspect='equal', interpolation='nearest')
    ax.axis('off')

    # 添加标题（放在 ax 上方）
    if title:
        ax.text(0.5, 1.12, title, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 添加 colorbar（字体更小）
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
        img_array = buf.reshape(h, w, 3)

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


def load_projection(filepath):
    """加载投影数据 (int16, 60x128x128)"""
    data = np.fromfile(filepath, dtype=np.int16).astype(np.float32)
    num_elements = data.size

    if num_elements == 128 * 128 * 60:
        return data.reshape(60, 128, 128)
    elif num_elements == 128 * 128 * 30:
        return data.reshape(30, 128, 128)
    else:
        raise ValueError(f"投影大小不匹配: {num_elements}")


def load_reconstruction(filepath):
    """加载重建数据 (float32, 128x128x128)"""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size == 128 * 128 * 128:
        return data.reshape(128, 128, 128)
    else:
        raise ValueError(f"重建大小不匹配: {data.size}")


def generate_2x3_gif(patient_name, orig_proj, denoised_proj, orig_recon, denoised_recon, output_path,
                     recon_vmax_mode='max75', proj_vmax_mode='p99.9', use_log1p=False):
    """生成 2x3 布局的对比 GIF

    Args:
        recon_vmax_mode: 重建 vmax 模式 - 'max' 使用最大值, 'max75' 使用 max×0.75, 'p9995' 使用 99.95 分位数
        proj_vmax_mode: 投影 vmax 模式 - 'max' 使用最大值, 'p99.9' 使用 99.9 分位数
        use_log1p: 是否使用 log1p 增强（投影和重建都使用）
    """

    # 投影帧数
    num_proj_frames = orig_proj.shape[0]  # 60 或 30
    # 重建使用更多帧（高帧率）
    num_recon_frames = 120

    # 投影的 vmax：根据模式选择
    if proj_vmax_mode == 'max':
        vmax_proj = max(orig_proj.max(), denoised_proj.max())
        proj_vmax_desc = "max"
    else:  # 'p99.9'
        vmax_proj = max(
            np.percentile(orig_proj[orig_proj > 0], 99.9) if np.any(orig_proj > 0) else orig_proj.max(),
            np.percentile(denoised_proj[denoised_proj > 0], 99.9) if np.any(denoised_proj > 0) else denoised_proj.max()
        )
        proj_vmax_desc = "p99.9"

    # 重建的 vmax：根据模式选择
    if recon_vmax_mode == 'max':
        vmax_recon = max(orig_recon.max(), denoised_recon.max())
        recon_vmax_desc = "max"
    elif recon_vmax_mode == 'p9995':
        vmax_recon = max(
            np.percentile(orig_recon[orig_recon > 0], 99.95) if np.any(orig_recon > 0) else orig_recon.max(),
            np.percentile(denoised_recon[denoised_recon > 0], 99.95) if np.any(denoised_recon > 0) else denoised_recon.max()
        )
        recon_vmax_desc = "p99.95"
    else:  # 'max75'
        vmax_recon = max(orig_recon.max(), denoised_recon.max()) * 0.75
        recon_vmax_desc = "max×0.75"

    # 计算残差范围（对称，投影残差使用 99.9 分位数）
    residual = denoised_proj - orig_proj
    vmax_residual = np.percentile(np.abs(residual), 99.9)
    vmin_residual = -vmax_residual

    # 计算重建差异范围（按照重建/投影的 vmax 比例缩放）
    recon_diff = denoised_recon - orig_recon
    scale_ratio = vmax_recon / vmax_proj  # 重建和投影的 vmax 比例
    vmax_recon_diff = vmax_residual * scale_ratio  # 按比例缩放投影残差范围
    vmin_recon_diff = -vmax_recon_diff

    print(f"   投影帧数: {num_proj_frames}, 重建帧数: {num_recon_frames}")
    print(f"   投影 vmax: {vmax_proj:.2f} ({proj_vmax_desc})")
    print(f"   重建 vmax: {vmax_recon:.4f} ({recon_vmax_desc})")
    print(f"   比例系数: {scale_ratio:.4f} (重建vmax/投影vmax)")
    if use_log1p:
        print(f"   对比度增强: log1p (投影和重建)")
    print(f"   投影残差范围: [{vmin_residual:.2f}, {vmax_residual:.2f}] (残差p99.9)")
    print(f"     - 实际范围: [{residual.min():.2f}, {residual.max():.2f}]")
    print(f"   重建差异范围: [{vmin_recon_diff:.4f}, {vmax_recon_diff:.4f}] (按比例缩放 ×{scale_ratio:.4f})")
    print(f"     - 实际范围: [{recon_diff.min():.4f}, {recon_diff.max():.4f}]")

    # 生成帧
    frames = []
    # 投影角度映射（与训练时相同）
    proj_angle_start = -180.0
    proj_angle_step = 6.0
    # 重建角度映射（旋转起始 +90+180=270 度，即 -90 度）
    recon_angle_start = 90.0  # 起始角度相比投影再偏移 270 度（即 -90+180=90）
    recon_angle_step = 360.0 / num_recon_frames

    for i in tqdm(range(num_recon_frames), desc="   生成帧"):
        # 投影帧索引（插值）
        proj_idx = int(i * num_proj_frames / num_recon_frames)
        proj_idx = min(proj_idx, num_proj_frames - 1)  # 确保不越界

        proj_angle = proj_angle_start + proj_idx * proj_angle_step
        recon_angle = recon_angle_start + i * recon_angle_step

        # === 第一行：投影 ===
        # 1. 原始投影（灰度/gray）
        proj_orig_i = orig_proj[proj_idx]  # (128, 128)
        img_proj_orig = create_subplot_with_colorbar(proj_orig_i, 0, vmax_proj, 'gray', title='Original Proj', use_log1p=use_log1p)

        # 2. 降噪投影（灰度/gray）
        proj_denoised_i = denoised_proj[proj_idx]  # (128, 128)
        img_proj_denoised = create_subplot_with_colorbar(proj_denoised_i, 0, vmax_proj, 'gray', title='Denoised Proj', use_log1p=use_log1p)

        # 3. 残差（RdBu_r）
        residual_i = residual[proj_idx]  # (128, 128)
        img_residual = create_subplot_with_colorbar(residual_i, vmin_residual, vmax_residual, 'RdBu_r', title='Residual')

        # === 第二行：重建 MIP ===
        # 旋转体积并计算 MIP
        rotated_orig = rotate_volume_around_x(orig_recon, recon_angle)
        rotated_denoised = rotate_volume_around_x(denoised_recon, recon_angle)
        rotated_diff = rotate_volume_around_x(recon_diff, recon_angle)

        mip_orig = np.max(rotated_orig, axis=2)
        mip_denoised = np.max(rotated_denoised, axis=2)
        # 对于差异，使用绝对值最大的投影（保留正负符号）
        mip_diff_pos = np.max(rotated_diff, axis=2)
        mip_diff_neg = np.min(rotated_diff, axis=2)
        mip_diff = np.where(np.abs(mip_diff_pos) > np.abs(mip_diff_neg), mip_diff_pos, mip_diff_neg)

        # 上下翻转重建图
        mip_orig = np.flipud(mip_orig)
        mip_denoised = np.flipud(mip_denoised)
        mip_diff = np.flipud(mip_diff)

        # 4. 原始重建 MIP（hot）
        img_recon_orig = create_subplot_with_colorbar(mip_orig, 0, vmax_recon, 'hot', title='Original Recon', use_log1p=use_log1p)

        # 5. 降噪重建 MIP（hot）
        img_recon_denoised = create_subplot_with_colorbar(mip_denoised, 0, vmax_recon, 'hot', title='Denoised Recon', use_log1p=use_log1p)

        # 6. 重建差异（RdBu_r）
        img_recon_diff = create_subplot_with_colorbar(mip_diff, vmin_recon_diff, vmax_recon_diff, 'RdBu_r', title='Recon Diff')

        # === 合并为 2x3 画布 ===
        # 每个子图现在是 160x128，包含了 colorbar
        w, h = img_proj_orig.width, img_proj_orig.height
        canvas = Image.new('RGB', (w * 3, h * 2))

        # 第一行
        canvas.paste(img_proj_orig, (0, 0))
        canvas.paste(img_proj_denoised, (w, 0))
        canvas.paste(img_residual, (w * 2, 0))

        # 第二行
        canvas.paste(img_recon_orig, (0, h))
        canvas.paste(img_recon_denoised, (w, h))
        canvas.paste(img_recon_diff, (w * 2, h))

        frames.append(canvas)

    # 保存 GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / 10.0),  # 10 fps（重建帧数增加，可以用更高帧率）
        loop=0,
        optimize=False
    )

    print(f"   ✅ GIF 已保存: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description='生成 2x3 布局的对比 GIF')
    parser.add_argument('--patient', type=str, default=None,
                        help='病人名称（默认处理所有病人）')
    parser.add_argument('--output-dir', type=str, default='spect_ct/results/2x3_comparison_gifs',
                        help='输出目录')
    args = parser.parse_args()

    # 路径设置（使用新的统一数据目录）
    data_root = Path('spect_ct/data')
    original_proj_root = data_root / 'original_projections'
    original_recon_root = data_root / 'original_reconstructions'
    denoised_proj_root = data_root / 'denoised_projections' / 'n2n_bone_proj_20s_singleview_sota_edge'
    denoised_recon_root = data_root / 'denoised_reconstructions'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有病人
    if args.patient:
        patients = [args.patient]
    else:
        # 从原始投影目录获取病人列表
        patients = sorted([f.stem.replace('_ProjectionImage1', '') for f in original_proj_root.glob('*_ProjectionImage1.dat')])

    print(f"找到 {len(patients)} 个病人")
    print("=" * 60)

    for patient in patients:
        print(f"\n📁 处理: {patient}")

        # 检查文件是否存在
        orig_proj_path = original_proj_root / f'{patient}_ProjectionImage1.dat'
        denoised_proj_path = denoised_proj_root / patient / 'ProjectionImage1_denoised_net_g_100000.dat'
        orig_recon_path = original_recon_root / f'{patient}_RCAC-20s-1.img'
        denoised_recon_path = denoised_recon_root / f'{patient}_OSEMReconed.dat'

        missing = []
        if not orig_proj_path.exists():
            missing.append(f"原始投影: {orig_proj_path}")
        if not denoised_proj_path.exists():
            missing.append(f"降噪投影: {denoised_proj_path}")
        if not orig_recon_path.exists():
            missing.append(f"原始重建: {orig_recon_path}")
        if not denoised_recon_path.exists():
            missing.append(f"降噪重建: {denoised_recon_path}")

        if missing:
            print(f"   ⚠️ 缺少文件:")
            for m in missing:
                print(f"      - {m}")
            continue

        try:
            # 加载数据
            print("   加载数据...")
            orig_proj = load_projection(orig_proj_path)
            denoised_proj = load_projection(denoised_proj_path)
            orig_recon = load_reconstruction(orig_recon_path)
            denoised_recon = load_reconstruction(denoised_recon_path)

            print(f"   原始投影: {orig_proj.shape}")
            print(f"   降噪投影: {denoised_proj.shape}")
            print(f"   原始重建: {orig_recon.shape}")
            print(f"   降噪重建: {denoised_recon.shape}")

            # 检查投影帧数是否匹配
            if orig_proj.shape[0] != denoised_proj.shape[0]:
                print(f"   ⚠️ 投影帧数不匹配: {orig_proj.shape[0]} vs {denoised_proj.shape[0]}")
                # 使用较小的帧数
                min_frames = min(orig_proj.shape[0], denoised_proj.shape[0])
                orig_proj = orig_proj[:min_frames]
                denoised_proj = denoised_proj[:min_frames]
                print(f"   使用前 {min_frames} 帧")

            # 生成新的 GIF：max + log1p
            print("\n   === 生成 GIF: max + log1p ===")
            output_path_max_log1p = output_dir / f'{patient}_2x3_comparison_max_log1p.gif'
            generate_2x3_gif(patient, orig_proj, denoised_proj, orig_recon, denoised_recon, output_path_max_log1p,
                             recon_vmax_mode='max', proj_vmax_mode='max', use_log1p=True)

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 完成！")


if __name__ == '__main__':
    main()
