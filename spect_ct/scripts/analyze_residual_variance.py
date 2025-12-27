#!/usr/bin/env python3
"""
分析降噪残差方差与像素值的关系

对于泊松噪声：Var(y) = E[y] = λ
理论上残差方差应该与原始像素值成正比
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "spect_ct" / "data"
OUTPUT_DIR = PROJECT_ROOT / "spect_ct" / "results"


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


def analyze_residual_variance_by_intensity(original_proj_root, denoised_proj_root, patients):
    """分析残差方差与像素值的关系"""

    print("分析残差方差与像素值的关系...")
    print("=" * 60)

    # 设置分桶范围
    bin_edges = np.arange(0, 151, 1)  # 0-150，每1个像素值一个bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 存储所有病人的统计数据
    all_bin_means = []  # 每个bin的平均像素值
    all_bin_variances = []  # 每个bin的残差方差
    all_bin_counts = []  # 每个bin的像素数量

    for patient in tqdm(patients, desc="处理病人"):
        # 加载数据
        orig_proj_path = original_proj_root / f'{patient}_ProjectionImage1.dat'
        denoised_proj_path = denoised_proj_root / patient / 'ProjectionImage1_denoised_net_g_100000.dat'

        if not orig_proj_path.exists() or not denoised_proj_path.exists():
            continue

        try:
            orig_proj = load_projection(orig_proj_path)
            denoised_proj = load_projection(denoised_proj_path)

            # 计算残差
            residual = denoised_proj - orig_proj

            # 展平数据
            orig_flat = orig_proj.flatten()
            denoised_flat = denoised_proj.flatten()
            residual_flat = residual.flatten()

            # 按降噪后的像素值分桶统计残差方差
            bin_variances = []
            bin_means = []
            bin_counts = []

            for i in range(len(bin_edges) - 1):
                # 找到该bin范围内的像素（按降噪后的值）
                mask = (denoised_flat >= bin_edges[i]) & (denoised_flat < bin_edges[i+1])

                if np.sum(mask) >= 10:  # 至少10个像素才统计
                    bin_residuals = residual_flat[mask]
                    bin_denoised_values = denoised_flat[mask]

                    bin_means.append(np.mean(bin_denoised_values))
                    bin_variances.append(np.var(bin_residuals))
                    bin_counts.append(np.sum(mask))
                else:
                    bin_means.append(np.nan)
                    bin_variances.append(np.nan)
                    bin_counts.append(0)

            all_bin_means.append(bin_means)
            all_bin_variances.append(bin_variances)
            all_bin_counts.append(bin_counts)

        except Exception as e:
            print(f"处理 {patient} 失败: {e}")
            continue

    # 转换为numpy数组
    all_bin_means = np.array(all_bin_means)  # (n_patients, n_bins)
    all_bin_variances = np.array(all_bin_variances)
    all_bin_counts = np.array(all_bin_counts)

    # 计算所有病人的平均统计
    mean_bin_means = np.nanmean(all_bin_means, axis=0)
    mean_bin_variances = np.nanmean(all_bin_variances, axis=0)
    std_bin_variances = np.nanstd(all_bin_variances, axis=0)

    return {
        'bin_centers': bin_centers,
        'mean_bin_means': mean_bin_means,
        'mean_bin_variances': mean_bin_variances,
        'std_bin_variances': std_bin_variances,
        'all_bin_means': all_bin_means,
        'all_bin_variances': all_bin_variances,
        'all_bin_counts': all_bin_counts,
    }


def visualize_residual_variance(stats, output_path):
    """可视化残差方差与像素值的关系"""

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # ========== 1. 散点图：所有病人的数据点 ==========
    ax1 = axes[0, 0]

    # 绘制所有病人的散点（半透明）
    for i in range(len(stats['all_bin_means'])):
        valid_mask = ~np.isnan(stats['all_bin_means'][i]) & ~np.isnan(stats['all_bin_variances'][i])
        if np.any(valid_mask):
            ax1.scatter(stats['all_bin_means'][i][valid_mask],
                       stats['all_bin_variances'][i][valid_mask],
                       alpha=0.1, s=20, color='steelblue')

    ax1.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residual Variance', fontsize=12, fontweight='bold')
    ax1.set_title('Residual Variance vs Denoised Pixel Value (All Patients)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 150)

    # ========== 2. 平均曲线 + 误差带 ==========
    ax2 = axes[0, 1]

    valid_mask = ~np.isnan(stats['mean_bin_variances'])
    x = stats['mean_bin_means'][valid_mask]
    y = stats['mean_bin_variances'][valid_mask]
    y_std = stats['std_bin_variances'][valid_mask]

    # 绘制平均曲线
    ax2.plot(x, y, 'o-', color='darkred', linewidth=2, markersize=4, label='Mean Variance')

    # 绘制误差带
    ax2.fill_between(x, y - y_std, y + y_std, alpha=0.3, color='coral', label='±1 Std Dev')

    # 理论泊松线（残差方差应该接近原始值）
    # 对于泊松噪声：Var(noise) ≈ λ，所以 Var(residual) ≈ λ (如果降噪完美去除噪声)
    ax2.plot([0, 150], [0, 150], 'k--', linewidth=2, alpha=0.5, label='Theoretical Poisson (Var=λ)')

    ax2.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Residual Variance vs Denoised Pixel Value', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 150)

    # ========== 3. 标准差曲线 ==========
    ax3 = axes[1, 0]

    # 残差的标准差（方差的平方根）
    y_std_residual = np.sqrt(y)

    ax3.plot(x, y_std_residual, 'o-', color='darkgreen', linewidth=2, markersize=4, label='Residual Std Dev')

    # 理论泊松标准差（sqrt(λ)）
    ax3.plot(x, np.sqrt(x), 'k--', linewidth=2, alpha=0.5, label='Theoretical Poisson (σ=√λ)')

    ax3.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residual Standard Deviation', fontsize=12, fontweight='bold')
    ax3.set_title('Residual Std Dev vs Denoised Pixel Value', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 150)

    # ========== 4. 归一化方差（方差/均值）==========
    ax4 = axes[1, 1]

    # 归一化方差（应该接近1，表示泊松特性）
    normalized_var = y / x
    normalized_var = normalized_var[x > 10]  # 只看计数>10的区域
    x_filtered = x[x > 10]

    ax4.plot(x_filtered, normalized_var, 'o-', color='purple', linewidth=2, markersize=4, label='Var/Mean')
    ax4.axhline(y=1, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal Poisson (Var/Mean=1)')

    ax4.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized Variance (Var/Mean)', fontsize=12, fontweight='bold')
    ax4.set_title('Normalized Residual Variance vs Denoised Pixel Value', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(10, 150)
    ax4.set_ylim(0, 3)

    # ========== 5. 每个bin的样本数量（上半部分）==========
    ax5 = axes[0, 2]

    # 计算所有病人的平均样本数量
    mean_bin_counts = np.mean(stats['all_bin_counts'], axis=0)
    valid_mask_counts = mean_bin_counts > 0

    ax5.bar(stats['bin_centers'][valid_mask_counts], mean_bin_counts[valid_mask_counts],
            width=1.0, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.7)

    ax5.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average Sample Count per Bin', fontsize=12, fontweight='bold')
    ax5.set_title('Sample Distribution per Pixel Value', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 150)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_yscale('log')  # 使用对数刻度，因为样本数量差异可能很大

    # ========== 6. 累积样本数量（下半部分）==========
    ax6 = axes[1, 2]

    # 计算累积样本数量
    cumulative_counts = np.cumsum(mean_bin_counts[valid_mask_counts])
    total_samples = cumulative_counts[-1]
    cumulative_percentage = (cumulative_counts / total_samples) * 100

    ax6.plot(stats['bin_centers'][valid_mask_counts], cumulative_percentage,
             'o-', color='darkblue', linewidth=2, markersize=4)
    ax6.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% of Samples')
    ax6.axhline(y=90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='90% of Samples')

    ax6.set_xlabel('Denoised Pixel Value (counts)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Cumulative Sample Distribution', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 150)
    ax6.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("残差方差与像素值关系分析")
    print("=" * 60)

    # 路径配置
    original_proj_root = DATA_ROOT / 'original_projections'
    denoised_proj_root = DATA_ROOT / 'denoised_projections' / 'n2n_bone_proj_20s_singleview_sota_edge'

    # 获取病人列表
    patients = sorted([f.stem.replace('_ProjectionImage1', '')
                      for f in original_proj_root.glob('*_ProjectionImage1.dat')])

    print(f"\n找到 {len(patients)} 个病人\n")

    # 分析残差方差
    stats = analyze_residual_variance_by_intensity(original_proj_root, denoised_proj_root, patients)

    # 可视化
    output_path = OUTPUT_DIR / "residual_variance_analysis.png"
    visualize_residual_variance(stats, output_path)

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

