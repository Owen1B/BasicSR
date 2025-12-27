#!/usr/bin/env python3
"""
生成OSEM重建切面对比图（降噪前后）

2行3列布局：
- 第一行：降噪前的三个正交切面（轴位、冠状面、矢状面）
- 第二行：降噪后的三个正交切面

使用 Iter 10 的重建结果进行对比
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
        enhance_contrast: 是否启用对比度增强（log1p + gamma）
        gamma: gamma 校正参数（默认 0.8，<1 增强低值）
    """

    # 中间切片索引
    mid_idx = 64

    # 计算总活度差异
    total_orig = np.sum(orig_recon)
    total_denoised = np.sum(denoised_recon)
    activity_change = ((total_denoised - total_orig) / total_orig) * 100

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
    print(f"   原始重建范围: [{orig_recon.min():.4f}, {orig_recon.max():.4f}], 总活度: {total_orig:.2e}")
    print(f"   降噪重建范围: [{denoised_recon.min():.4f}, {denoised_recon.max():.4f}], 总活度: {total_denoised:.2e}")
    print(f"   总活度变化: {activity_change:+.2f}%")
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
    axes[0, 0].set_title('Original - Axial (XY)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 冠状面 (Coronal) - XZ平面
    coronal_orig = np.flipud(orig_recon[:, mid_idx, :])
    axes[0, 1].imshow(coronal_orig, cmap='hot', norm=norm, aspect='equal')
    axes[0, 1].set_title('Original - Coronal (XZ)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # 矢状面 (Sagittal) - YZ平面
    sagittal_orig = np.flipud(orig_recon[:, :, mid_idx])
    axes[0, 2].imshow(sagittal_orig, cmap='hot', norm=norm, aspect='equal')
    axes[0, 2].set_title('Original - Sagittal (YZ)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # 第二行：降噪后的切面
    # 轴位 (Axial) - XY平面
    axial_denoised = np.flipud(denoised_recon[mid_idx, :, :])
    axes[1, 0].imshow(axial_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 0].set_title('Denoised - Axial (XY)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # 冠状面 (Coronal) - XZ平面
    coronal_denoised = np.flipud(denoised_recon[:, mid_idx, :])
    axes[1, 1].imshow(coronal_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 1].set_title('Denoised - Coronal (XZ)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # 矢状面 (Sagittal) - YZ平面
    sagittal_denoised = np.flipud(denoised_recon[:, :, mid_idx])
    axes[1, 2].imshow(sagittal_denoised, cmap='hot', norm=norm, aspect='equal')
    axes[1, 2].set_title('Denoised - Sagittal (YZ)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    # 添加总标题（包含总活度变化信息）
    title_text = f'{patient_name} - Reconstruction Slice Comparison (Iter 10)\nTotal Activity Change: {activity_change:+.2f}%'
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)

    # 添加统一的 colorbar
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.05)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)

    # 保存图像
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ PNG 已保存: {output_path} ({output_path.stat().st_size / 1024:.2f} KB)")


def main():
    parser = argparse.ArgumentParser(description='生成OSEM重建切面对比图')
    parser.add_argument('--patients', nargs='+', default=None,
                        help='病人名称列表（不指定则处理所有病人）')
    parser.add_argument('--top-n', type=int, default=None,
                        help='只处理前N个病人')
    parser.add_argument('--output-dir', type=str, default='spect_ct/results/osem_slice_comparison',
                        help='输出目录')
    parser.add_argument('--vmax-mode', type=str, default='p99.9',
                        choices=['max', 'p99.9', 'p99.95', 'max75'],
                        help='vmax 计算模式: max, p99.9 (默认), p99.95, max75')
    parser.add_argument('--no-enhance', action='store_true',
                        help='禁用对比度增强（使用线性显示）')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='gamma 校正参数（默认 0.8）')
    args = parser.parse_args()

    # 路径设置
    base_dir = Path('spect_ct')
    data_root = base_dir / 'data'
    original_recon_root = data_root / 'original' / 'reconstructions'
    denoised_recon_root = data_root / 'denoised' / 'reconstructions'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取病人列表
    if args.patients:
        patients = args.patients
    else:
        # 从原始重建目录获取病人列表
        patients = sorted([d.name for d in original_recon_root.iterdir() if d.is_dir()])

    # 只处理前N个
    if args.top_n:
        patients = patients[:args.top_n]

    print(f"找到 {len(patients)} 个病人")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for patient in patients:
        print(f"\n📁 处理: {patient}")

        # 文件路径（使用 Iter 10 的重建）
        orig_recon_path = original_recon_root / patient / f'{patient}_original_OSEMReconed.dat'
        denoised_recon_path = denoised_recon_root / patient / f'{patient}_denoised_OSEMReconed.dat'

        missing = []
        if not orig_recon_path.exists():
            missing.append(f"原始重建: {orig_recon_path}")
        if not denoised_recon_path.exists():
            missing.append(f"降噪重建: {denoised_recon_path}")

        if missing:
            print(f"   ⚠️ 缺少文件:")
            for m in missing:
                print(f"      - {m}")
            fail_count += 1
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
            success_count += 1

        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"✅ 完成！成功: {success_count}, 失败: {fail_count}")


if __name__ == '__main__':
    main()


