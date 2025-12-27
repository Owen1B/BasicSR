#!/usr/bin/env python3
"""
增强版可视化：投影和重建的变化分析（新OSEM结构）

使用新的OSEM重建数据（Iter 10）进行分析

生成更直观的总览图，包括：
1. 变化分布直方图
2. 投影 vs 重建变化的散点图（相关性分析）
3. 盒图（箱线图）显示统计分布
4. 综合总结图
"""

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置字体（使用英文，避免字体警告）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "spect_ct" / "results"


def create_enhanced_visualization(csv_path: Path):
    """创建增强版可视化"""
    # 读取数据
    df = pd.read_csv(csv_path)

    # 创建大图：3行2列
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # ========== 1. 变化分布直方图 (左上) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    proj_changes = df['Proj_Rel_Change_%'].dropna()
    ax1.hist(proj_changes, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Change')
    ax1.axvline(x=proj_changes.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean={proj_changes.mean():.2f}%')
    ax1.set_xlabel('Projection Count Change (%)', fontsize=12)
    ax1.set_ylabel('Number of Patients', fontsize=12)
    ax1.set_title('Distribution of Projection Count Change', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ========== 2. 重建活度变化分布 (右上) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    recon_changes = df['Recon_Rel_Change_%'].dropna()
    ax2.hist(recon_changes, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Change')
    ax2.axvline(x=recon_changes.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean={recon_changes.mean():.2f}%')
    ax2.set_xlabel('Reconstruction Activity Change (%)', fontsize=12)
    ax2.set_ylabel('Number of Patients', fontsize=12)
    ax2.set_title('Distribution of Reconstruction Activity Change', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ========== 3. 散点图：投影 vs 重建变化 (中间跨两列) ==========
    ax3 = fig.add_subplot(gs[1, :])
    valid_data = df.dropna(subset=['Proj_Rel_Change_%', 'Recon_Rel_Change_%'])
    proj_change = valid_data['Proj_Rel_Change_%']
    recon_change = valid_data['Recon_Rel_Change_%']

    # 散点图
    scatter = ax3.scatter(proj_change, recon_change, c=np.abs(recon_change), cmap='YlOrRd', s=100, alpha=0.7, edgecolors='black', linewidth=1)

    # 添加参考线
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # 标注病人名称
    for idx, row in valid_data.iterrows():
        ax3.annotate(row['Patient'], (row['Proj_Rel_Change_%'], row['Recon_Rel_Change_%']),
                    fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    ax3.set_xlabel('Projection Count Change (%)', fontsize=12)
    ax3.set_ylabel('Reconstruction Activity Change (%)', fontsize=12)
    ax3.set_title('Projection Change vs Reconstruction Change (Correlation Analysis)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('|Recon Change| (%)', fontsize=10)

    # ========== 4. 箱线图对比 (左下) ==========
    ax4 = fig.add_subplot(gs[2, 0])
    data_to_plot = [proj_changes, recon_changes]
    bp = ax4.boxplot(data_to_plot, tick_labels=['Projection\nCount Change', 'Reconstruction\nActivity Change'],
                     patch_artist=True, showmeans=True, meanline=True)

    # 设置颜色
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_ylabel('Change (%)', fontsize=12)
    ax4.set_title('Change Distribution Statistics (Boxplot)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # ========== 5. 统计摘要表格 (右下) - LaTeX 三线表风格 ==========
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # 准备表格数据（三线表）- 两列：投影、重建
    table_data = [
        ['Metric', 'Projection\nCount Change', 'Reconstruction\nActivity Change'],
        ['N', f'{len(proj_changes)}', f'{len(recon_changes)}'],
        ['Mean (%)', f'{proj_changes.mean():+.2f}', f'{recon_changes.mean():+.2f}'],
        ['Min (%)', f'{proj_changes.min():+.2f}', f'{recon_changes.min():+.2f}'],
        ['Max (%)', f'{proj_changes.max():+.2f}', f'{recon_changes.max():+.2f}'],
    ]

    # 创建表格（3列：Metric + 2列数据）
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.32, 0.8, 0.60])  # 调整宽度以容纳3列

    # 设置表格样式（三线表）
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # 字体可以稍大一些

    # 遍历所有单元格 - 传统三线表样式
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]

        # 表头行（第一行）- 白色背景，黑色加粗文字
        if key[0] == 0:
            cell.set_facecolor('white')
            cell.set_text_props(weight='bold', color='black', fontsize=9)
            cell.set_height(0.20)
            # 顶线和表头下线（适中粗细）
            cell.set_linewidth(1.5)
            cell.set_edgecolor('black')
            cell.visible_edges = 'TB'  # 显示顶部和底部边框
        # 数据行 - 白色背景，黑色文字
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black', fontsize=8)
            cell.set_height(0.18)
            # 最后一行：底线（适中粗细）
            if key[0] == len(table_data) - 1:
                cell.set_linewidth(1.5)
                cell.set_edgecolor('black')
                cell.visible_edges = 'B'  # 只显示底部边框
            else:
                cell.set_linewidth(0)
                cell.visible_edges = ''  # 不显示边框（中间行无横线）

    # 保存图表
    output_path = OUTPUT_DIR / "osem_count_activity_analysis_enhanced.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Enhanced visualization saved: {output_path}")
    plt.close()

    # ========== 创建一个简洁的总结图 ==========
    create_summary_figure(df)


def create_summary_figure(df):
    """创建简洁的总结图（单张大图）"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # 准备数据
    patients = df['Patient'].tolist()
    proj_changes = df['Proj_Rel_Change_%'].tolist()
    recon_changes = df['Recon_Rel_Change_%'].tolist()

    x = np.arange(len(patients))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, proj_changes, width, label='Projection Count Change',
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, recon_changes, width, label='Reconstruction Activity Change',
                   color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)

    # 添加零线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Change')

    # 添加均值线
    proj_mean = np.mean([c for c in proj_changes if not np.isnan(c)])
    recon_mean = np.mean([c for c in recon_changes if not np.isnan(c)])
    ax.axhline(y=proj_mean, color='blue', linestyle=':', linewidth=2, alpha=0.5, label=f'Proj Mean={proj_mean:+.2f}%')
    ax.axhline(y=recon_mean, color='red', linestyle=':', linewidth=2, alpha=0.5, label=f'Recon Mean={recon_mean:+.2f}%')

    # 设置标签
    ax.set_xlabel('Patient', fontsize=14, fontweight='bold')
    ax.set_ylabel('Change (%)', fontsize=14, fontweight='bold')
    ax.set_title('Projection Count and Reconstruction Activity Change Overview', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # 设置y轴范围
    all_changes = proj_changes + recon_changes
    y_min = min([c for c in all_changes if not np.isnan(c)])
    y_max = max([c for c in all_changes if not np.isnan(c)])
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.2)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "osem_count_activity_summary.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Summary figure saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Enhanced Visualization: OSEM Projection and Reconstruction Changes")
    print("=" * 60)

    # 读取 CSV
    csv_path = OUTPUT_DIR / "osem_count_activity_analysis.csv"
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("Please run analyze_count_activity_changes.py first")
        return

    print(f"\n📂 Reading data: {csv_path}")

    # 创建增强版可视化
    create_enhanced_visualization(csv_path)

    print("\n✅ All visualizations generated!")
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print("   - osem_count_activity_analysis_enhanced.png (detailed)")
    print("   - osem_count_activity_summary.png (summary)")


if __name__ == "__main__":
    main()

