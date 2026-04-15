#!/usr/bin/env python3
"""NEMA 数据重建和可视化脚本

功能：
1. 对原始投影和降噪投影运行 OSEM 重建
2. 加载所有重建结果（包括已有的 Filtered 和 OSEM_RCACSC）
3. 绘制 4 个重建结果的 MIP 旋转 GIF
"""

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径

from prime.pipeline.osem import ensure_osem_reconstruction
from prime.pipeline.io import load_recon_f32
from prime.pipeline.viz import compute_mip, rotate_volume_around_x, scalar_to_u8_cmap


def _load_recon_dat(path: Path, shape=(256, 256, 256)) -> np.ndarray:
    """加载重建 .dat 文件（float32）"""
    return load_recon_f32(path, shape=shape)


def run_reconstructions(
    nema_dir: Path,
    osem_dir: Path,
    original_par: Path,
    denoised_par: Path,
    orbit_file: Path,
    atten_file: Path,
    original_proj: Path,
    denoised_proj: Path,
    output_dir: Path,
    iterations: int = 10,
    timeout_sec: int = 600,
    force_rerun: bool = False,
) -> tuple[Path, Path]:
    """运行两次 OSEM 重建

    Returns:
        (original_recon_path, denoised_recon_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预期输出路径（固定命名，避免脚本参数变化造成缓存失配）
    original_output = output_dir / f"NEMA_original_OSEM_iter{int(iterations)}.dat"
    denoised_output = output_dir / f"NEMA_denoised_OSEM_iter{int(iterations)}.dat"

    # 1. 原始投影重建
    original_exists = original_output.exists()
    if force_rerun or not original_exists:
        print(f"\n🚀 运行原始投影 OSEM 重建...")
        print(f"   投影: {original_proj.name}")
        print(f"   迭代: {iterations}")
    original_recon = ensure_osem_reconstruction(
        proj_file=original_proj,
        patient_name="NEMA",
        par_file=original_par,
        orbit_file=orbit_file,
        atten_file=atten_file,
        output_name="original",
        iterations=iterations,
        prj_data_type=1,
        osem_dir=osem_dir,
        final_output_dir=output_dir,
        output_filename=original_output.name,
        timeout_sec=timeout_sec,
        overwrite=bool(force_rerun),
    )
    if force_rerun or not original_exists:
        print(f"✅ 原始投影重建完成: {original_recon}")
    else:
        print(f"✅ 原始投影重建已存在: {original_recon}")

    # 2. 降噪投影重建
    denoised_exists = denoised_output.exists()
    if force_rerun or not denoised_exists:
        print(f"\n🚀 运行降噪投影 OSEM 重建...")
        print(f"   投影: {denoised_proj.name}")
        print(f"   迭代: {iterations}")
    denoised_recon = ensure_osem_reconstruction(
        proj_file=denoised_proj,
        patient_name="NEMA",
        par_file=denoised_par,
        orbit_file=orbit_file,
        atten_file=atten_file,
        output_name="denoised",
        iterations=iterations,
        prj_data_type=1,
        osem_dir=osem_dir,
        final_output_dir=output_dir,
        output_filename=denoised_output.name,
        timeout_sec=timeout_sec,
        overwrite=bool(force_rerun),
    )
    if force_rerun or not denoised_exists:
        print(f"✅ 降噪投影重建完成: {denoised_recon}")
    else:
        print(f"✅ 降噪投影重建已存在: {denoised_recon}")

    return original_recon, denoised_recon


def create_mip_rotation_gif(
    recon_files: dict[str, Path],
    output_gif: Path,
    vmax_percentile: float = 100.0,
    no_log1p: bool = False,
    num_angles: int = 60,
    fps: int = 10,
    cmap_name: str = "hot",
    figsize_per_col: tuple[float, float] = (4, 4),
) -> None:
    """创建 4 个重建结果的 MIP 旋转 GIF

    Args:
        recon_files: {"label": recon_path, ...}
        output_gif: 输出 GIF 路径
        vmax_percentile: vmax 计算百分位数（100.0 = 全局最大值）
        no_log1p: 不使用 log1p 变换
        num_angles: 旋转角度数
        fps: 帧率
        cmap_name: colormap 名称
        figsize_per_col: 每列的图形大小
    """
    print(f"\n🎬 创建 MIP 旋转 GIF...")
    print(f"   重建结果数: {len(recon_files)}")
    print(f"   旋转角度数: {num_angles}")
    print(f"   帧率: {fps} fps")

    # 加载所有重建结果
    recons = {}
    for label, path in recon_files.items():
        print(f"   加载 {label}: {path.name}")
        recons[label] = _load_recon_dat(path)

    # 计算全局 vmax（所有重建结果的联合百分位数）
    all_data = np.concatenate([r.ravel() for r in recons.values()])
    if vmax_percentile >= 100.0:
        vmax_recon = float(np.max(all_data))
        vmax_str = f"max={vmax_recon:.2f}"
    else:
        vmax_recon = float(np.percentile(all_data, vmax_percentile))
        vmax_str = f"p{vmax_percentile:.1f}={vmax_recon:.2f}"

    print(f"   全局 vmax: {vmax_str}")
    del all_data

    # 设置 matplotlib
    ncols = len(recon_files)
    fig, axes = plt.subplots(1, ncols, figsize=(figsize_per_col[0] * ncols, figsize_per_col[1]))
    if ncols == 1:
        axes = [axes]

    fig.tight_layout(pad=2.0)

    # 旋转角度
    angles = np.linspace(0, 360, num_angles, endpoint=False)

    frames = []
    for i, angle in enumerate(angles):
        print(f"\r   渲染帧 {i+1}/{num_angles} (角度={angle:.1f}°)", end="", flush=True)

        for ax, (label, recon) in zip(axes, recons.items()):
            ax.clear()

            # 旋转体素
            rotated = rotate_volume_around_x(recon, angle)

            # MIP
            mip = compute_mip(rotated, axis=1)  # Y轴投影

            # 应用变换
            if no_log1p:
                mip_vis = mip
            else:
                mip_vis = np.log1p(mip)

            # 归一化并应用 colormap
            mip_rgb = scalar_to_u8_cmap(
                mip_vis,
                vmin=0.0,
                vmax=(vmax_recon if no_log1p else np.log1p(vmax_recon)),
                cmap_name=cmap_name,
            )

            # 绘制
            ax.imshow(mip_rgb, aspect="equal")
            ax.set_title(f"{label}\n(angle={angle:.0f}°)", fontsize=12)
            ax.axis("off")

        # 保存帧
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[..., :3])  # 只保留 RGB

    print()  # 换行

    # 保存 GIF
    print(f"   保存 GIF: {output_gif}")
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_gif, frames, fps=fps, loop=0)

    plt.close(fig)
    print(f"✅ GIF 创建完成: {output_gif}")
    print(f"   大小: {output_gif.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="NEMA 数据重建和可视化")
    parser.add_argument("--nema-dir", type=Path, default=Path("datasets/NEMA"),
                        help="NEMA 数据目录")
    parser.add_argument("--osem-dir", type=Path, default=None,
                        help="OSEM 可执行文件目录（默认 osemreocnexe）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/NEMA"),
                        help="输出目录")
    parser.add_argument("--iterations", type=int, default=10,
                        help="OSEM 迭代次数")
    parser.add_argument("--timeout", type=int, default=600,
                        help="OSEM 超时时间（秒）")
    parser.add_argument("--force-rerun", action="store_true",
                        help="强制重新运行 OSEM（即使重建结果已存在）")
    parser.add_argument("--skip-recon", action="store_true",
                        help="跳过重建，仅生成 GIF")
    parser.add_argument("--vmax-percentile", type=float, default=99.5,
                        help="vmax 百分位数（100.0 = 全局最大值）")
    parser.add_argument("--no-log1p", action="store_true",
                        help="不使用 log1p 变换")
    parser.add_argument("--num-angles", type=int, default=60,
                        help="旋转角度数")
    parser.add_argument("--fps", type=int, default=10,
                        help="GIF 帧率")
    parser.add_argument("--cmap", type=str, default="hot",
                        help="colormap 名称")

    args = parser.parse_args()

    # 检查输入文件
    nema_dir = Path(args.nema_dir)
    if not nema_dir.exists():
        raise FileNotFoundError(f"NEMA 目录不存在: {nema_dir}")

    # 必需文件
    original_proj = nema_dir / "energy_window1_full_rotation_360.dat"
    denoised_proj = nema_dir / "denoised_projection_120views.dat"
    filtered_recon = nema_dir / "Filtered_10-7-7-7_Bq_ml-1.dat"
    osem_rcacsc_recon = nema_dir / "OSEM_RCACSC_Bq_ml-1.dat"

    original_par = nema_dir / "ParamFile_original_recon.par"
    denoised_par = nema_dir / "ParamFile_denoised_recon.par"
    orbit_file = nema_dir / "orbit_459255658.orb"
    atten_file = nema_dir / "PostAtten_1E005A1FFEA282E76C61AE2B22E34917D09F33E146BCD1DA96C7D0136879877C.dat"

    for f in [original_proj, denoised_proj, filtered_recon, osem_rcacsc_recon,
              original_par, denoised_par, orbit_file, atten_file]:
        if not f.exists():
            raise FileNotFoundError(f"必需文件不存在: {f}")

    # OSEM 目录
    if args.osem_dir is None:
        osem_dir = Path(__file__).resolve().parents[1] / "osemreocnexe"
    else:
        osem_dir = Path(args.osem_dir)

    output_dir = Path(args.output_dir)
    recon_dir = output_dir / "reconstructions"

    # Step 1: 运行 OSEM 重建
    if not args.skip_recon:
        original_recon, denoised_recon = run_reconstructions(
            nema_dir=nema_dir,
            osem_dir=osem_dir,
            original_par=original_par,
            denoised_par=denoised_par,
            orbit_file=orbit_file,
            atten_file=atten_file,
            original_proj=original_proj,
            denoised_proj=denoised_proj,
            output_dir=recon_dir,
            iterations=args.iterations,
            timeout_sec=args.timeout,
            force_rerun=args.force_rerun,
        )
    else:
        original_recon = recon_dir / "NEMA_original_OSEM_iter10.dat"
        denoised_recon = recon_dir / "NEMA_denoised_OSEM_iter10.dat"
        print("⏭️  跳过重建步骤")

    # Step 2: 创建 MIP 旋转 GIF
    recon_files = {
        "Original OSEM": original_recon,
        "Denoised OSEM": denoised_recon,
        "Filtered": filtered_recon,
        "OSEM RCACSC": osem_rcacsc_recon,
    }

    # GIF 文件名
    log_str = "nolog1p" if args.no_log1p else "log1p"
    if args.vmax_percentile >= 100.0:
        vmax_str = "vmax"
    else:
        vmax_str = f"v{args.vmax_percentile:.1f}".replace(".", "")

    gif_filename = f"nema_4recon_mip_rotation_{log_str}_{vmax_str}.gif"
    output_gif = output_dir / gif_filename

    create_mip_rotation_gif(
        recon_files=recon_files,
        output_gif=output_gif,
        vmax_percentile=args.vmax_percentile,
        no_log1p=args.no_log1p,
        num_angles=args.num_angles,
        fps=args.fps,
        cmap_name=args.cmap,
    )

    print(f"\n🎉 全部完成！")
    print(f"   重建结果: {recon_dir}")
    print(f"   GIF 输出: {output_gif}")


if __name__ == "__main__":
    main()
