#!/usr/bin/env python3
"""
批量为所有病人生成降噪投影数据

使用训练好的模型对所有病人进行降噪，保存为 int16 格式（四舍五入）
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def get_patient_list(dataset_root):
    """获取所有病人列表"""
    dataset_root = Path(dataset_root)
    patients = []

    for patient_dir in sorted(dataset_root.iterdir()):
        if patient_dir.is_dir():
            proj_file = patient_dir / "20s-1" / "ProjectionImage1.dat"
            if proj_file.exists():
                patients.append(patient_dir.name)

    return patients


def denoise_single_patient(patient_name, config_path, checkpoint_path, dataset_root, output_dir, device='cuda'):
    """为单个病人生成降噪投影"""
    input_file = Path(dataset_root) / patient_name / "20s-1" / "ProjectionImage1.dat"

    cmd = [
        'python3', 'spect_ct/scripts/denoise_projection_for_recon.py',
        '--config', config_path,
        '--checkpoint', checkpoint_path,
        '--input', str(input_file),
        '--output_dir', output_dir,
        '--device', device
    ]

    # 运行命令，捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False, result.stderr
    else:
        return True, None


def main():
    parser = argparse.ArgumentParser(description='批量为所有病人生成降噪投影')
    parser.add_argument(
        '--config',
        type=str,
        default='options/train/bone_real/n2n_bone_proj_20s_singleview_sota_edge_patch128.yml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='experiments/n2n_bone_proj_20s_singleview_sota_edge_patch128/models/net_g_100000.pth',
        help='模型检查点路径'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='datasets/bone_20230327',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='spect_ct/results/denoised_projections_all_patients',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda 或 cpu)'
    )
    args = parser.parse_args()

    # 获取病人列表
    patients = get_patient_list(args.dataset_root)
    print(f"找到 {len(patients)} 个病人")
    print("=" * 80)

    # 批量处理
    success_count = 0
    failed_patients = []

    for patient in tqdm(patients, desc="处理进度"):
        success, error = denoise_single_patient(
            patient,
            args.config,
            args.checkpoint,
            args.dataset_root,
            args.output_dir,
            args.device
        )

        if success:
            success_count += 1
        else:
            failed_patients.append((patient, error))
            tqdm.write(f"⚠️  {patient} 处理失败")

    # 统计结果
    print()
    print("=" * 80)
    print(f"✅ 成功: {success_count}/{len(patients)}")

    if failed_patients:
        print(f"❌ 失败: {len(failed_patients)}")
        print("\n失败的病人:")
        for patient, error in failed_patients:
            print(f"  - {patient}")
            if error:
                print(f"    错误: {error[:200]}")  # 只显示前200个字符

    # 输出统计
    output_root = Path(args.output_dir)
    if output_root.exists():
        exp_dirs = list(output_root.glob('*/'))
        if exp_dirs:
            exp_dir = exp_dirs[0]
            patient_dirs = list(exp_dir.glob('*/'))
            total_size = sum(f.stat().st_size for d in patient_dirs for f in d.glob('*.dat'))
            print(f"\n📊 总数据大小: {total_size / 1024 / 1024:.2f} MB")
            print(f"📁 输出目录: {output_root}")


if __name__ == '__main__':
    main()

