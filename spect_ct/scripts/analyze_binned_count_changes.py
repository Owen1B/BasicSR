#!/usr/bin/env python3
"""
分析不同像素值分桶的计数变化

分桶：0, 1, 2-10, 11-20, 21-30, >30
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "bone_20230327"
DENOISED_PROJ_ROOT = PROJECT_ROOT / "spect_ct" / "results" / "denoised_projections_all_patients" / "n2n_bone_proj_20s_singleview_sota_edge"
OUTPUT_DIR = PROJECT_ROOT / "spect_ct" / "results"


def load_original_projection(filepath: Path) -> np.ndarray:
    """加载原始投影数据 (int16格式)"""
    data = np.fromfile(filepath, dtype=np.int16).astype(np.float32)
    # 原始数据是60帧（双视图）
    return data.reshape(60, 128, 128)


def load_denoised_projection(filepath: Path) -> np.ndarray:
    """加载降噪后投影数据 (int16格式) - 需要先找到对应的原始int16文件"""
    # 降噪后的数据是30帧（单视图）
    data = np.fromfile(filepath, dtype=np.int16).astype(np.float32)
    return data.reshape(30, 128, 128)


def get_patient_list() -> list:
    """获取所有病人名称"""
    patients = []
    for patient_dir in sorted(DATASET_ROOT.iterdir()):
        if patient_dir.is_dir():
            patients.append(patient_dir.name)
    return patients


def analyze_binned_counts(patients: list) -> pd.DataFrame:
    """分析不同像素值分桶的计数变化"""
    results = []

    # 定义分桶
    bins = [
        (0, 0, '==0'),
        (1, 1, '==1'),
        (2, 10, '2-10'),
        (11, 20, '11-20'),
        (21, 30, '21-30'),
        (31, np.inf, '>30')
    ]

    for patient in patients:
        # 原始投影 (int16)
        orig_proj_path = DATASET_ROOT / patient / "20s-1" / "ProjectionImage1.dat"
        # 降噪后投影 - 需要找对应的int16文件
        # 注意：降噪后的.dat文件可能是float32格式，需要检查
        denoised_proj_path_float = DENOISED_PROJ_ROOT / f"{patient}_ProjectionImage1_denoised_net_g_100000.dat"

        if not orig_proj_path.exists():
            print(f"⚠️  原始投影不存在: {patient}")
            continue
        if not denoised_proj_path_float.exists():
            print(f"⚠️  降噪后投影不存在: {patient}")
            continue

        # 加载原始投影 (int16, 60帧)
        orig_proj = load_original_projection(orig_proj_path)

        # 加载降噪后投影 (float32归一化数据，30帧) - 需要反归一化
        # 由于降噪模型输出的是归一化数据，我们需要检查是否有对应的int16版本
        # 如果没有，我们可能需要跳过或者使用其他方法

        # 先尝试加载float32格式并检查
        denoised_data_float = np.fromfile(denoised_proj_path_float, dtype=np.float32)

        # 检查数据是否已归一化（值很小）
        if denoised_data_float.max() < 1.0:
            print(f"⚠️  {patient}: 降噪数据已归一化，无法直接分桶统计")
            print(f"    原始投影统计:")

            # 只统计原始投影的分桶
            result = {'Patient': patient}
            for bin_min, bin_max, bin_name in bins:
                if bin_max == np.inf:
                    mask = orig_proj > bin_min
                elif bin_min == bin_max:
                    mask = orig_proj == bin_min
                else:
                    mask = (orig_proj >= bin_min) & (orig_proj <= bin_max)

                count = orig_proj[mask].sum()
                pixel_count = np.sum(mask)
                result[f'Orig_Bin_{bin_name}_Count'] = count
                result[f'Orig_Bin_{bin_name}_Pixels'] = pixel_count
                print(f"      {bin_name}: count={count:.0f}, pixels={pixel_count}")

            results.append(result)
            continue

    return pd.DataFrame(results)


def visualize_binned_changes(df: pd.DataFrame, output_dir: Path):
    """可视化分桶统计结果"""
    if df.empty:
        print("No data to visualize")
        return

    # 提取分桶列
    bin_names = ['==0', '==1', '2-10', '11-20', '21-30', '>30']
    colors = ['gray', 'lightcoral', 'steelblue', 'mediumseagreen', 'orange', 'purple']

    # 创建总览图：所有病人的分桶计数堆叠柱状图
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 堆叠柱状图（左上大图，占2x1）
    ax_main = fig.add_subplot(gs[0:2, 0])

    patients = df['Patient'].tolist()
    x = np.arange(len(patients))
    width = 0.8

    # 准备堆叠数据
    bottom = np.zeros(len(patients))
    for idx, bin_name in enumerate(bin_names):
        count_col = f'Orig_Bin_{bin_name}_Count'
        if count_col in df.columns:
            data = df[count_col].fillna(0).values
            ax_main.bar(x, data, width, label=f'Pixel {bin_name}',
                       bottom=bottom, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data

    ax_main.set_xlabel('Patient', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Total Count', fontsize=12, fontweight='bold')
    ax_main.set_title('Original Projection Count Distribution by Pixel Value', fontsize=14, fontweight='bold')
    ax_main.set_xticks(x[::2])  # 每隔一个显示
    ax_main.set_xticklabels(patients[::2], rotation=45, ha='right', fontsize=8)
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(True, alpha=0.3, axis='y')

    # 2. 各分桶的平均计数（右上）
    ax_avg = fig.add_subplot(gs[0, 1])
    avg_counts = []
    for bin_name in bin_names:
        count_col = f'Orig_Bin_{bin_name}_Count'
        if count_col in df.columns:
            avg_counts.append(df[count_col].mean())
        else:
            avg_counts.append(0)

    bars = ax_avg.bar(range(len(bin_names)), avg_counts, color=colors, alpha=0.8, edgecolor='black')
    ax_avg.set_xlabel('Pixel Value Bin', fontsize=10)
    ax_avg.set_ylabel('Average Count', fontsize=10)
    ax_avg.set_title('Average Count per Bin', fontsize=12, fontweight='bold')
    ax_avg.set_xticks(range(len(bin_names)))
    ax_avg.set_xticklabels(bin_names, fontsize=9)
    ax_avg.grid(True, alpha=0.3, axis='y')

    # 3. 各分桶的平均像素数（右中）
    ax_pixels = fig.add_subplot(gs[1, 1])
    avg_pixels = []
    for bin_name in bin_names:
        pixel_col = f'Orig_Bin_{bin_name}_Pixels'
        if pixel_col in df.columns:
            avg_pixels.append(df[pixel_col].mean())
        else:
            avg_pixels.append(0)

    bars = ax_pixels.bar(range(len(bin_names)), avg_pixels, color=colors, alpha=0.8, edgecolor='black')
    ax_pixels.set_xlabel('Pixel Value Bin', fontsize=10)
    ax_pixels.set_ylabel('Average Pixel Count', fontsize=10)
    ax_pixels.set_title('Average Pixel Count per Bin', fontsize=12, fontweight='bold')
    ax_pixels.set_xticks(range(len(bin_names)))
    ax_pixels.set_xticklabels(bin_names, fontsize=9)
    ax_pixels.grid(True, alpha=0.3, axis='y')

    # 4. 统计摘要表格（底部跨2列）
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # 准备表格数据
    table_data = [['Metric'] + bin_names]

    # 平均计数
    table_data.append(['Avg Count'] + [f'{c:.0f}' for c in avg_counts])

    # 平均像素数
    table_data.append(['Avg Pixels'] + [f'{p:.0f}' for p in avg_pixels])

    # 计数范围
    count_ranges = []
    for bin_name in bin_names:
        count_col = f'Orig_Bin_{bin_name}_Count'
        if count_col in df.columns:
            min_val = df[count_col].min()
            max_val = df[count_col].max()
            count_ranges.append(f'[{min_val:.0f}, {max_val:.0f}]')
        else:
            count_ranges.append('[0, 0]')
    table_data.append(['Count Range'] + count_ranges)

    # 创建表格
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # 设置表格样式
    for key in table.get_celld().keys():
        cell = table.get_celld()[key]
        if key[0] == 0:  # 表头
            cell.set_facecolor('lightgray')
            cell.set_text_props(weight='bold')
            cell.set_height(0.15)
        elif key[1] == 0:  # 第一列
            cell.set_facecolor('lightblue')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white')
        cell.set_linewidth(1.5)
        cell.set_edgecolor('black')

    plt.savefig(output_dir / "original_projection_bin_distribution.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 原始投影分桶分布图已保存: {output_dir / 'original_projection_bin_distribution.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("像素值分桶计数统计")
    print("=" * 60)

    # 获取病人列表
    patients = get_patient_list()
    print(f"\n找到 {len(patients)} 个病人\n")

    # 分析分桶计数
    df = analyze_binned_counts(patients)

    # 保存CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "binned_count_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📄 CSV 已保存: {csv_path}")

    # 可视化
    visualize_binned_changes(df, OUTPUT_DIR)

    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()

