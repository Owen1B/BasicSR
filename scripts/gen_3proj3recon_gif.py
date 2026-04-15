#!/usr/bin/env python3
import argparse
"""
单病人生成 2x3 GIF + counts.txt（wrapper）。

核心实现已抽到 `prime/pipeline/threeproj3recon.py`，便于复用与减少重复代码。
"""

from pathlib import Path

from prime.pipeline.threeproj3recon import process_patient_3proj3recon


def main():
    parser = argparse.ArgumentParser(description='生成三个投影和三个重建的 2x3 对比 GIF')
    parser.add_argument('--patient', type=str, required=True, help='病人名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--input-proj', type=str, required=True, help='输入投影文件路径')
    parser.add_argument('--par-file', type=str, required=True, help='par 文件路径')
    parser.add_argument('--orbit-file', type=str, required=True, help='orbit.orb 文件路径')
    parser.add_argument('--atten-file', type=str, required=True, help='PostAtten.dat 文件路径')
    parser.add_argument('--results-root', type=str, default='outputs', help='统一结果根目录')
    parser.add_argument('--exp-name', type=str, required=True, help='实验名（将输出到 results/<exp-name>/...）')
    parser.add_argument('--output-dir', type=str, default=None, help='可选：手动指定输出目录（覆盖默认 results-root/exp-name）')
    parser.add_argument('--max-value', type=float, default=150.0, help='归一化最大值')
    parser.add_argument('--iterations', type=int, default=10, help='OSEM 重建迭代次数（默认10）')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='如果该病人的最终 GIF 与 counts.txt 已存在，则直接跳过（用于断点续跑）'
    )
    parser.add_argument('--skip-denoise', action='store_true', help='跳过降噪步骤（使用已有文件）')
    parser.add_argument('--skip-recon', action='store_true', help='跳过重建步骤（使用已有文件）')
    # Visualization only (does NOT change saved .dat projections / recon outputs)
    parser.add_argument('--log1p', dest='use_log1p', action='store_true', help='可视化时使用 log1p 增强对比度（默认开启）')
    parser.add_argument('--no-log1p', dest='use_log1p', action='store_false', help='关闭 log1p（恢复线性显示）')
    parser.set_defaults(use_log1p=True)
    # 已移除：recon-atten 2x2 GIF 逻辑

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.results_root) / str(args.exp_name) / "3proj_3recon")

    process_patient_3proj3recon(
        patient=args.patient,
        checkpoint=Path(args.checkpoint),
        config=Path(args.config),
        input_proj=Path(args.input_proj),
        par_file=Path(args.par_file),
        orbit_file=Path(args.orbit_file),
        atten_file=Path(args.atten_file),
        output_dir=output_dir,
        max_value=float(args.max_value),
        iterations=int(args.iterations),
        skip_existing=bool(args.skip_existing),
        skip_denoise=bool(args.skip_denoise),
        skip_recon=bool(args.skip_recon),
        use_log1p=bool(args.use_log1p),
        device="cuda",
    )


if __name__ == '__main__':
    main()

