#!/usr/bin/env python3
"""
统计投影降噪前后的总计数变化和重建后的总活度变化（新OSEM结构）

分析内容：
1. 投影降噪前后的总计数变化
2. 重建后的总活度变化（使用Iter 10结果）

数据路径：
- 原始投影：spect_ct/data/original/projections/{patient}/ProjectionImage1.dat
- 降噪投影：spect_ct/data/denoised/projections/{patient}/ProjectionImage1_denoised_net_g_100000.dat
- 原始重建：spect_ct/data/original/reconstructions/{patient}/{patient}_original_OSEMReconed.dat (Iter 10)
- 降噪重建：spect_ct/data/denoised/reconstructions/{patient}/{patient}_denoised_OSEMReconed.dat (Iter 10)

输出：CSV 表格和可视化图表
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径（新的OSEM重建结构）
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "spect_ct" / "data"
ORIGINAL_PROJ_ROOT = DATA_ROOT / "original" / "projections"
ORIGINAL_RECON_ROOT = DATA_ROOT / "original" / "reconstructions"
DENOISED_PROJ_ROOT = DATA_ROOT / "denoised" / "projections"
DENOISED_RECON_ROOT = DATA_ROOT / "denoised" / "reconstructions"
OUTPUT_DIR = PROJECT_ROOT / "spect_ct" / "results"

# 数据尺寸
PROJ_SHAPE = (60, 128, 128)  # Views, H, W (60帧双视图)
RECON_SHAPE = (128, 128, 128)  # 重建体积


def load_projection_dat(filepath: Path, is_original: bool = True) -> np.ndarray:
    """加载投影数据 (.dat)

    Args:
        filepath: 数据文件路径
        is_original: True表示原始数据(int16)，False表示降噪数据(int16)

    Returns:
        投影数据，形状 (60, 128, 128)
    """
    # 原始和降噪数据现在都是 int16 格式
    data = np.fromfile(filepath, dtype=np.int16)
    # 投影数据: (Views, H, W) = (60, 128, 128)
    return data.reshape(60, 128, 128)


def load_recon_img(filepath: Path) -> np.ndarray:
    """加载重建数据 (.img, float32)"""
    data = np.fromfile(filepath, dtype=np.float32)
    return data.reshape(RECON_SHAPE)


def load_recon_dat(filepath: Path) -> np.ndarray:
    """加载重建数据 (.dat, float32)"""
    data = np.fromfile(filepath, dtype=np.float32)
    return data.reshape(RECON_SHAPE)


def get_patient_list() -> list:
    """获取所有病人名称"""
    patients = []
    # 从原始投影目录获取病人列表（新结构：每个病人一个子目录）
    for patient_dir in sorted(ORIGINAL_PROJ_ROOT.iterdir()):
        if patient_dir.is_dir():
            patients.append(patient_dir.name)
    return patients


def analyze_projection_counts(patients: list) -> pd.DataFrame:
    """分析投影降噪前后的总计数变化"""
    results = []

    for patient in patients:
        # 原始投影（新结构）
        orig_proj_path = ORIGINAL_PROJ_ROOT / patient / "ProjectionImage1.dat"
        # 降噪后投影（新结构）
        denoised_proj_path = DENOISED_PROJ_ROOT / patient / "ProjectionImage1_denoised_net_g_100000.dat"

        if not orig_proj_path.exists():
            print(f"⚠️  原始投影不存在: {patient}")
            continue
        if not denoised_proj_path.exists():
            print(f"⚠️  降噪后投影不存在: {patient}")
            continue

        # 加载数据
        orig_proj = load_projection_dat(orig_proj_path, is_original=True)
        denoised_proj = load_projection_dat(denoised_proj_path, is_original=False)

        # 计算总计数（全部）
        orig_total = orig_proj.sum()
        denoised_total = denoised_proj.sum()

        # 计算变化
        abs_change = denoised_total - orig_total
        rel_change = (abs_change / orig_total) * 100 if orig_total > 0 else 0

        # 计算非零区域像素个数
        orig_nonzero_pixels = np.count_nonzero(orig_proj)
        denoised_nonzero_pixels = np.count_nonzero(denoised_proj)
        nonzero_pixels_change = denoised_nonzero_pixels - orig_nonzero_pixels
        nonzero_pixels_rel_change = (nonzero_pixels_change / orig_nonzero_pixels) * 100 if orig_nonzero_pixels > 0 else 0

        # 计算排除低值噪声的总计数（使用原始数据的5%分位数作为阈值）
        threshold = np.percentile(orig_proj[orig_proj > 0], 5) if np.any(orig_proj > 0) else 0
        orig_total_excl_low = orig_proj[orig_proj > threshold].sum()
        denoised_total_excl_low = denoised_proj[denoised_proj > threshold].sum()
        abs_change_excl_low = denoised_total_excl_low - orig_total_excl_low
        rel_change_excl_low = (abs_change_excl_low / orig_total_excl_low) * 100 if orig_total_excl_low > 0 else 0

        results.append({
            'Patient': patient,
            'Orig_Proj_Count': orig_total,
            'Denoised_Proj_Count': denoised_total,
            'Proj_Abs_Change': abs_change,
            'Proj_Rel_Change_%': rel_change,
            'Orig_Proj_NonZero_Pixels': orig_nonzero_pixels,
            'Denoised_Proj_NonZero_Pixels': denoised_nonzero_pixels,
            'Proj_NonZero_Pixels_Change': nonzero_pixels_change,
            'Proj_NonZero_Pixels_Rel_Change_%': nonzero_pixels_rel_change,
            'Orig_Proj_Count_ExclLow': orig_total_excl_low,
            'Denoised_Proj_Count_ExclLow': denoised_total_excl_low,
            'Proj_Abs_Change_ExclLow': abs_change_excl_low,
            'Proj_Rel_Change_ExclLow_%': rel_change_excl_low,
            'Proj_Threshold': threshold
        })

        print(f"✓ {patient}: 原始={orig_total:.0f}, 降噪后={denoised_total:.0f}, 变化={rel_change:+.2f}%")
        print(f"           非零像素: 原始={orig_nonzero_pixels}, 降噪后={denoised_nonzero_pixels}, 变化={nonzero_pixels_rel_change:+.2f}%")
        print(f"           排除低值(阈值={threshold:.1f}): 原始={orig_total_excl_low:.0f}, 降噪后={denoised_total_excl_low:.0f}, 变化={rel_change_excl_low:+.2f}%")

    return pd.DataFrame(results)


def analyze_recon_activity(patients: list) -> pd.DataFrame:
    """分析重建后的总活度变化（使用Iter 10结果）"""
    results = []

    for patient in patients:
        # 原始投影重建（新结构，Iter 10）
        orig_recon_path = ORIGINAL_RECON_ROOT / patient / f"{patient}_original_OSEMReconed.dat"
        # 降噪后投影重建（新结构，Iter 10）
        denoised_recon_path = DENOISED_RECON_ROOT / patient / f"{patient}_denoised_OSEMReconed.dat"

        if not orig_recon_path.exists():
            print(f"⚠️  原始重建不存在: {patient}")
            continue
        if not denoised_recon_path.exists():
            print(f"⚠️  降噪后重建不存在: {patient}")
            continue

        # 加载数据（两个都是.dat格式，float32）
        orig_recon = load_recon_dat(orig_recon_path)
        denoised_recon = load_recon_dat(denoised_recon_path)

        # 计算总活度（所有体素值之和）
        orig_total = orig_recon.sum()
        denoised_total = denoised_recon.sum()

        # 计算变化
        abs_change = denoised_total - orig_total
        rel_change = (abs_change / orig_total) * 100 if orig_total > 0 else 0

        # 计算非零区域体素个数
        orig_nonzero_voxels = np.count_nonzero(orig_recon)
        denoised_nonzero_voxels = np.count_nonzero(denoised_recon)
        nonzero_voxels_change = denoised_nonzero_voxels - orig_nonzero_voxels
        nonzero_voxels_rel_change = (nonzero_voxels_change / orig_nonzero_voxels) * 100 if orig_nonzero_voxels > 0 else 0

        # 计算排除体素值=1的总活度（排除低值噪声）
        # 注意：重建数据通常是浮点数，这里使用 > 1.0 作为阈值
        orig_total_excl1 = orig_recon[orig_recon > 1.0].sum()
        denoised_total_excl1 = denoised_recon[denoised_recon > 1.0].sum()
        abs_change_excl1 = denoised_total_excl1 - orig_total_excl1
        rel_change_excl1 = (abs_change_excl1 / orig_total_excl1) * 100 if orig_total_excl1 > 0 else 0

        results.append({
            'Patient': patient,
            'Orig_Recon_Activity': orig_total,
            'Denoised_Recon_Activity': denoised_total,
            'Recon_Abs_Change': abs_change,
            'Recon_Rel_Change_%': rel_change,
            'Orig_Recon_NonZero_Voxels': orig_nonzero_voxels,
            'Denoised_Recon_NonZero_Voxels': denoised_nonzero_voxels,
            'Recon_NonZero_Voxels_Change': nonzero_voxels_change,
            'Recon_NonZero_Voxels_Rel_Change_%': nonzero_voxels_rel_change,
            'Orig_Recon_Activity_Excl1': orig_total_excl1,
            'Denoised_Recon_Activity_Excl1': denoised_total_excl1,
            'Recon_Abs_Change_Excl1': abs_change_excl1,
            'Recon_Rel_Change_Excl1_%': rel_change_excl1
        })

        print(f"✓ {patient}: 原始={orig_total:.0f}, 降噪后={denoised_total:.0f}, 变化={rel_change:+.2f}%")
        print(f"           非零体素: 原始={orig_nonzero_voxels}, 降噪后={denoised_nonzero_voxels}, 变化={nonzero_voxels_rel_change:+.2f}%")
        print(f"           排除值>1: 原始={orig_total_excl1:.0f}, 降噪后={denoised_total_excl1:.0f}, 变化={rel_change_excl1:+.2f}%")

    return pd.DataFrame(results)


def plot_results(df_proj: pd.DataFrame, df_recon: pd.DataFrame, output_dir: Path):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 投影计数变化百分比
    ax1 = axes[0, 0]
    patients = df_proj['Patient'].tolist()
    changes = df_proj['Proj_Rel_Change_%'].tolist()
    colors = ['green' if c >= 0 else 'red' for c in changes]
    ax1.barh(patients, changes, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('变化百分比 (%)')
    ax1.set_title('投影降噪前后总计数变化')
    ax1.tick_params(axis='y', labelsize=8)

    # 2. 重建活度变化百分比
    ax2 = axes[0, 1]
    if not df_recon.empty:
        patients_r = df_recon['Patient'].tolist()
        changes_r = df_recon['Recon_Rel_Change_%'].tolist()
        colors_r = ['green' if c >= 0 else 'red' for c in changes_r]
        ax2.barh(patients_r, changes_r, color=colors_r, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('变化百分比 (%)')
    ax2.set_title('重建降噪前后总活度变化')
    ax2.tick_params(axis='y', labelsize=8)

    # 3. 投影计数对比
    ax3 = axes[1, 0]
    x = np.arange(len(df_proj))
    width = 0.35
    ax3.bar(x - width/2, df_proj['Orig_Proj_Count'] / 1e6, width, label='原始', alpha=0.7)
    ax3.bar(x + width/2, df_proj['Denoised_Proj_Count'] / 1e6, width, label='降噪后', alpha=0.7)
    ax3.set_xlabel('病人')
    ax3.set_ylabel('总计数 (×10⁶)')
    ax3.set_title('投影总计数对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_proj['Patient'], rotation=90, fontsize=6)
    ax3.legend()

    # 4. 重建活度对比
    ax4 = axes[1, 1]
    if not df_recon.empty:
        x = np.arange(len(df_recon))
        ax4.bar(x - width/2, df_recon['Orig_Recon_Activity'] / 1e6, width, label='原始', alpha=0.7)
        ax4.bar(x + width/2, df_recon['Denoised_Recon_Activity'] / 1e6, width, label='降噪后', alpha=0.7)
        ax4.set_xlabel('病人')
        ax4.set_ylabel('总活度 (×10⁶)')
        ax4.set_title('重建总活度对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df_recon['Patient'], rotation=90, fontsize=6)
        ax4.legend()

    plt.tight_layout()
    output_path = output_dir / "osem_count_activity_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("投影降噪前后计数变化 & 重建活度变化分析")
    print("=" * 60)

    # 获取病人列表
    patients = get_patient_list()
    print(f"\n找到 {len(patients)} 个病人")

    # 分析投影计数
    print("\n" + "=" * 40)
    print("1. 投影降噪前后总计数变化")
    print("=" * 40)
    df_proj = analyze_projection_counts(patients)

    # 分析重建活度
    print("\n" + "=" * 40)
    print("2. 重建降噪前后总活度变化")
    print("=" * 40)
    df_recon = analyze_recon_activity(patients)

    # 合并结果
    if not df_proj.empty and not df_recon.empty:
        df_merged = pd.merge(df_proj, df_recon, on='Patient', how='outer')
    else:
        df_merged = df_proj if not df_proj.empty else df_recon

    # 保存 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "osem_count_activity_analysis.csv"
    df_merged.to_csv(csv_path, index=False)
    print(f"\n📄 CSV 已保存: {csv_path}")

    # 打印统计摘要
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)

    if not df_proj.empty:
        print(f"\n投影计数变化:")
        print(f"  平均变化: {df_proj['Proj_Rel_Change_%'].mean():+.2f}%")
        print(f"  最小变化: {df_proj['Proj_Rel_Change_%'].min():+.2f}%")
        print(f"  最大变化: {df_proj['Proj_Rel_Change_%'].max():+.2f}%")
        print(f"  标准差:   {df_proj['Proj_Rel_Change_%'].std():.2f}%")

    if not df_recon.empty:
        print(f"\n重建活度变化:")
        print(f"  平均变化: {df_recon['Recon_Rel_Change_%'].mean():+.2f}%")
        print(f"  最小变化: {df_recon['Recon_Rel_Change_%'].min():+.2f}%")
        print(f"  最大变化: {df_recon['Recon_Rel_Change_%'].max():+.2f}%")
        print(f"  标准差:   {df_recon['Recon_Rel_Change_%'].std():.2f}%")

    # 绘制图表
    if not df_proj.empty or not df_recon.empty:
        plot_results(df_proj, df_recon, OUTPUT_DIR)

    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()


