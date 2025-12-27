#!/usr/bin/env python3
"""
生成重建切面对比图

2行3列布局：
- 第一行：降噪前的三个正交切面（轴位、冠状面、矢状面）
- 第二行：降噪后的三个正交切面

支持低计数对比度增强：
- log1p 变换：增强低值区域可见性
- gamma 校正：进一步增强低值对比度
- 分位数裁剪：使用 p99.9 避免极值影响
"""

import numpy as np
from pathlib import Path
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_reconstruction(filepath):
    """加载重建数据 (float32, 128x128x128)"""
    data = np.fromfile(filepath, dtype=np.float32)
    if data.size == 128 * 128 * 128:
        return data.reshape(128, 128, 128)
    else:
        raise ValueError(f"重建大小不匹配: {data.size}")


def apply_contrast_enhancement(data, vmin, vmax, gamma=0.8, log1p=True):
    """应用对比度增强：log1p + gamma 变换

    Args:
        data: 输入数据
        vmin, vmax: 数据范围
        gamma: gamma 校正参数（<1 增强低值）
        log1p: 是否使用 log1p 变换

    Returns:
        变换后的数据（归一化到 [0, 1]）
    """
    f = data.astype(np.float32)
    f = np.clip(f, vmin, vmax)
    f = (f - vmin) / max(1e-8, (vmax - vmin))

    if log1p:
        # log1p 增强：增强低值区域的可见性
        # log1p(9*x) / log1p(9) 将 [0,1] 映射到 [0,1]，但增强低值
        f = np.log1p(9.0 * f) / math.log1p(9.0)

    if gamma != 1.0:
        # gamma 校正：gamma < 1 增强低值，压缩高值
        f = np.power(np.clip(f, 0.0, 1.0), gamma)

    return f


class EnhancedNormalize(Normalize):
    """支持 log1p + gamma 的归一化类"""
    def __init__(self, vmin=None, vmax=None, gamma=0.8, log1p=True, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma
        self.log1p = log1p

    def __call__(self, value, clip=None):
        # 先线性归一化
        result = super().__call__(value, clip)

        # 应用 log1p 变换
        if self.log1p:
            result = np.log1p(9.0 * result) / math.log1p(9.0)

        # 应用 gamma 校正
        if self.gamma != 1.0:
            result = np.power(np.clip(result, 0.0, 1.0), self.gamma)

        return result


def generate_slice_comparison(patient_name, orig_recon, denoised_recon, output_path,
                               vmax_mode='p99.9', enhance_contrast=True, gamma=0.8):
    """生成2行3列的切面对比图

    Args:
        patient_name: 病人名称
        orig_recon: 原始重建数据
        denoised_recon: 降噪重建数据
        output_path: 输出路径
        vmax_mode: vmax 计算模式
            - 'max': 使用最大值（默认）
            - 'p99.9': 使用 99.9 分位数（避免极值影响）
            - 'p99.95': 使用 99.95 分位数
            - 'max75': 使用 max * 0.75
        enhance_contrast: 是否启用对比度增强（log1p + gamma）
        gamma: gamma 校正参数（默认 0.8，<1 增强低值）
    """

    # 中间切片索引
    mid_idx = 64

    # 计算统一的 vmax
    vmin = 0
    if vmax_mode == 'max':
        vmax = max(orig_recon.max(), denoised_recon.max())
        vmax_desc = "max"
    elif vmax_mode == 'p99.9':
        def get_p99_9(vol):
            if np.any(vol > 0):
                return np.percentile(vol[vol > 0], 99.9)
            return vol.max()
        vmax = max(get_p99_9(orig_recon), get_p99_9(denoised_recon))
        vmax_desc = "p99.9"
    elif vmax_mode == 'p99.95':
        def get_p99_95(vol):
            if np.any(vol > 0):
                return np.percentile(vol[vol > 0], 99.95)
            return vol.max()
        vmax = max(get_p99_95(orig_recon), get_p99_95(denoised_recon))
        vmax_desc = "p99.95"
    else:  # 'max75'
        vmax = max(orig_recon.max(), denoised_recon.max()) * 0.75
        vmax_desc = "max×0.75"

    print(f"   vmax: {vmax:.4f} ({vmax_desc})")
    print(f"   原始重建范围: [{orig_recon.min():.4f}, {orig_recon.max():.4f}]")
    print(f"   降噪重建范围: [{denoised_recon.min():.4f}, {denoised_recon.max():.4f}]")
    if enhance_contrast:
        print(f"   对比度增强: log1p + gamma={gamma}")

    # 创建图形：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 设置归一化方式
    if enhance_contrast:
        norm = EnhancedNormalize(vmin=vmin, vmax=vmax, gamma=gamma, log1p=True)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # 第一行：降噪前的切面
    # 轴位 (Axial) - XY平面
    axial_orig = np.flipud(orig_recon[mid_idx, :, :])
    axes[0, 0].imshow(axial_orig, cmap='hot', norm=norm, aspect='equal')
    axes[0, 0].set_title('Original - Axial (XY)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # 冠状面 (Coronal) - XZ平面
    coronal_orig = np.flipud(orig_recon[:, mid_idx, :])
    axes[0, 1].imshow(coronal_orig, cmap='hot', norm=norm, aspect='equal')
    axes[0, 1].set_title('Original - Coronal (XZ)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # 矢状面 (Sagittal) - YZ平面
    sagittal_orig = np.flipud(orig_recon[:, :, mid_idx])
    axes[0, 2].imshow(sagittal_orig, cmap='hot', norm=norm, aspect='equal')
    axes[0, 2].set_title('Original - Sagittal (YZ)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # 第二行：降噪后的切面
    # 轴位 (Axial) - XY平面
    axial_denoised = np.flipud(denoised_recon[mid_idx, :, :])
    axes[1, 0].imshow(axial_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 0].set_title('Denoised - Axial (XY)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # 冠状面 (Coronal) - XZ平面
    coronal_denoised = np.flipud(denoised_recon[:, mid_idx, :])
    axes[1, 1].imshow(coronal_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 1].set_title('Denoised - Coronal (XZ)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # 矢状面 (Sagittal) - YZ平面
    sagittal_denoised = np.flipud(denoised_recon[:, :, mid_idx])
    axes[1, 2].imshow(sagittal_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 2].set_title('Denoised - Sagittal (YZ)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    # 添加总标题
    fig.suptitle(f'{patient_name} - Reconstruction Slice Comparison', fontsize=16, fontweight='bold', y=0.98)

    # 添加统一的 colorbar
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.05)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    # colorbar 显示的是原始数据值（vmin 到 vmax），而不是变换后的值

    # 保存图像
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ PNG 已保存: {output_path} ({output_path.stat().st_size / 1024:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(description='生成重建切面对比图')
    parser.add_argument('--patient', type=str, default=None,
                        help='病人名称（默认处理所有病人）')
    parser.add_argument('--output-dir', type=str, default='spect_ct/results/slice_comparison',
                        help='输出目录')
    parser.add_argument('--vmax-mode', type=str, default='max',
                        choices=['max', 'p99.9', 'p99.95', 'max75'],
                        help='vmax 计算模式: max (默认), p99.9, p99.95, max75')
    parser.add_argument('--no-enhance', action='store_true',
                        help='禁用对比度增强（使用线性显示）')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='gamma 校正参数（默认 0.8，<1 增强低值，>1 增强高值）')
    args = parser.parse_args()

    # 路径设置
    data_root = Path('spect_ct/data')
    original_recon_root = data_root / 'original_reconstructions'
    denoised_recon_root = data_root / 'denoised_reconstructions'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有病人
    if args.patient:
        patients = [args.patient]
    else:
        # 从原始重建目录获取病人列表
        patients = sorted([f.stem.replace('_RCAC-20s-1', '') for f in original_recon_root.glob('*_RCAC-20s-1.img')])

    print(f"找到 {len(patients)} 个病人")
    print("=" * 60)

    for patient in patients:
        print(f"\n📁 处理: {patient}")

        # 文件路径
        orig_recon_path = original_recon_root / f'{patient}_RCAC-20s-1.img'
        denoised_recon_path = denoised_recon_root / f'{patient}_OSEMReconed.dat'

        missing = []
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
            orig_recon = load_reconstruction(orig_recon_path)
            denoised_recon = load_reconstruction(denoised_recon_path)

            print(f"   原始重建: {orig_recon.shape}")
            print(f"   降噪重建: {denoised_recon.shape}")

            # 生成对比图
            output_path = output_dir / f'{patient}_slice_comparison.png'
            generate_slice_comparison(
                patient, orig_recon, denoised_recon, output_path,
                vmax_mode=args.vmax_mode,
                enhance_contrast=not args.no_enhance,
                gamma=args.gamma
            )

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 完成！")


if __name__ == '__main__':
    main()



