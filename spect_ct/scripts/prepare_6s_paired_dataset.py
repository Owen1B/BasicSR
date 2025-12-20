#!/usr/bin/env python3
"""
将 6s-3 投影数据两两配对，用于 Noise2Noise 训练

每个病人有 3 个 6s 投影：
- ProjectionImage1.dat
- ProjectionImage2.dat
- ProjectionImage3.dat

配对方式（两两配对）：
- (1, 2) -> pair_001_002.dat
- (1, 3) -> pair_001_003.dat
- (2, 3) -> pair_002_003.dat

每个病人生成 3 对，30 个病人共 90 对。

输出格式：.dat float32 (2, 128, 128)
- 第一个通道：anterior view
- 第二个通道：posterior view
"""

import numpy as np
from pathlib import Path
import argparse


def load_projection(filepath):
    """加载投影数据 (60, 128, 128) uint16"""
    data = np.fromfile(filepath, dtype=np.uint16)
    proj = data.reshape(60, 128, 128).astype(np.float32)
    return proj


def extract_dual_view(proj):
    """
    从投影数据中提取前后视角（anterior + posterior）

    Args:
        proj: (60, 128, 128) - 60 个视角

    Returns:
        anterior: (128, 128) - 前 30 个视角的平均
        posterior: (128, 128) - 后 30 个视角的平均
    """
    # 前 30 个视角是 anterior，后 30 个视角是 posterior
    anterior_views = proj[:30]  # (30, 128, 128)
    posterior_views = proj[30:]  # (30, 128, 128)

    # 取平均（或者可以取中位数，这里用平均）
    anterior = anterior_views.mean(axis=0)  # (128, 128)
    posterior = posterior_views.mean(axis=0)  # (128, 128)

    return anterior, posterior


def create_pair(proj1, proj2):
    """
    创建配对数据：将两个投影的前后视角堆叠

    Args:
        proj1: (60, 128, 128) - 第一个投影
        proj2: (60, 128, 128) - 第二个投影

    Returns:
        pair: (2, 128, 128) float32 - 配对数据
    """
    ant1, post1 = extract_dual_view(proj1)
    ant2, post2 = extract_dual_view(proj2)

    # 堆叠为双通道：第一个投影作为 lq，第二个作为 gt
    # 或者可以混合：ant1+ant2, post1+post2
    # 这里使用：lq 用 proj1，gt 用 proj2
    pair = np.stack([ant1, post1], axis=0)  # (2, 128, 128) - lq
    # 注意：对于 N2N，lq 和 gt 都是噪声的，所以这里只保存 lq
    # gt 会在另一个文件中

    return pair.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Prepare 6s paired dataset for N2N training')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Root directory containing patient folders (e.g., datasets/bone_20230327)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for paired .dat files')
    parser.add_argument('--pattern', type=str, default='6s-3',
                       help='Subdirectory pattern (default: 6s-3)')

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有病人目录
    patient_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories")

    total_pairs = 0
    pair_info = []

    for patient_dir in patient_dirs:
        patient_name = patient_dir.name
        proj_dir = patient_dir / args.pattern

        if not proj_dir.exists():
            print(f"⚠️  Skipping {patient_name}: {proj_dir} not found")
            continue

        # 加载 3 个投影文件
        proj_files = [
            proj_dir / 'ProjectionImage1.dat',
            proj_dir / 'ProjectionImage2.dat',
            proj_dir / 'ProjectionImage3.dat'
        ]

        # 检查文件是否存在
        if not all(f.exists() for f in proj_files):
            print(f"⚠️  Skipping {patient_name}: missing projection files")
            continue

        # 加载投影数据
        projs = []
        for f in proj_files:
            proj = load_projection(f)
            projs.append(proj)

        # 两两配对：(1,2), (1,3), (2,3)
        pairs = [
            (1, 2, projs[0], projs[1]),
            (1, 3, projs[0], projs[2]),
            (2, 3, projs[1], projs[2])
        ]

        for idx1, idx2, proj1, proj2 in pairs:
            # 创建配对：proj1 作为 lq，proj2 作为 gt
            pair_lq = create_pair(proj1, proj1)  # lq: 用 proj1
            pair_gt = create_pair(proj2, proj2)  # gt: 用 proj2

            # 保存为两个文件（paired mode 需要 lq 和 gt 分开）
            # 命名：patient_p001_p002.dat (lq 和 gt 同名，但放在不同目录)
            basename = f"{patient_name}_p{idx1:03d}_p{idx2:03d}"

            # 创建 lq 和 gt 子目录
            lq_dir = output_dir / 'lq'
            gt_dir = output_dir / 'gt'
            lq_dir.mkdir(exist_ok=True)
            gt_dir.mkdir(exist_ok=True)

            lq_file = lq_dir / f"{basename}.dat"
            gt_file = gt_dir / f"{basename}.dat"

            pair_lq.tofile(lq_file)
            pair_gt.tofile(gt_file)

            total_pairs += 1
            pair_info.append({
                'patient': patient_name,
                'pair': f"{idx1}-{idx2}",
                'lq': str(lq_file.relative_to(output_dir)),
                'gt': str(gt_file.relative_to(output_dir))
            })

    print(f"\n✅ Created {total_pairs} pairs from {len(patient_dirs)} patients")
    print(f"   Output directory: {output_dir}")
    print(f"   Expected: 30 patients × 3 pairs = 90 pairs")

    # 保存配对信息（可选）
    info_file = output_dir / 'pair_info.json'
    import json
    with open(info_file, 'w') as f:
        json.dump(pair_info, f, indent=2)
    print(f"   Pair info saved to: {info_file}")


if __name__ == '__main__':
    main()

