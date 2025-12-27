#!/usr/bin/env python3
"""
批量 OSEM 重建脚本
处理原始投影和去噪投影的 OSEM 重建
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
import csv
from datetime import datetime

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OSEM_DIR = BASE_DIR / "osemreocnexe"
OSEM_EXE = OSEM_DIR / "osemrecon.exe"
TEMPLATE_PAR = OSEM_DIR / "ParamFile4OSEMRecon.par"

# 获取所有病人列表
def get_patient_list():
    """从 original/projections 目录获取病人列表"""
    proj_dir = DATA_DIR / "original" / "projections"
    if not proj_dir.exists():
        return []
    return sorted([d.name for d in proj_dir.iterdir() if d.is_dir()])


def prepare_par_file(patient_name, projection_type, iterations=10):
    """
    准备 par 配置文件（将所有文件复制到 osemreocnexe 目录）

    Args:
        patient_name: 病人姓名
        projection_type: 'original' or 'denoised'
        iterations: 迭代次数（默认 10）

    Returns:
        临时 par 文件路径, 输出文件路径（在 osemreocnexe 中）, 最终输出路径
    """
    # 读取病人特定的 par 文件
    par_dir = DATA_DIR / projection_type / "par" / patient_name
    par_files = list(par_dir.glob("ParamFile4OSEMRecon*.par"))

    if not par_files:
        raise FileNotFoundError(f"No par file found for {patient_name} in {par_dir}")

    patient_par = par_files[0]

    # 读取 par 文件内容
    with open(patient_par, 'r', encoding='utf-8', errors='ignore') as f:
        par_content = f.read()

    # 设置源文件路径
    proj_dir = DATA_DIR / projection_type / "projections" / patient_name
    par_path = DATA_DIR / projection_type / "par" / patient_name

    # 确定投影文件名
    if projection_type == "original":
        proj_file = proj_dir / "ProjectionImage1.dat"
        temp_proj_name = "ProjectionImage1.dat"
    else:  # denoised
        proj_file = proj_dir / "ProjectionImage1_denoised_net_g_100000.dat"
        temp_proj_name = "ProjectionImage1_denoised.dat"

    if not proj_file.exists():
        raise FileNotFoundError(f"Projection file not found: {proj_file}")

    orbit_file = par_path / "orbit.orb"
    atten_file = par_path / "PostAtten.dat"

    # 复制文件到 osemreocnexe 目录
    temp_proj_file = OSEM_DIR / temp_proj_name
    temp_orbit_file = OSEM_DIR / "orbit.orb"
    temp_atten_file = OSEM_DIR / "PostAtten.dat"

    shutil.copy2(proj_file, temp_proj_file)
    shutil.copy2(orbit_file, temp_orbit_file)
    shutil.copy2(atten_file, temp_atten_file)

    # 输出文件名（在 osemreocnexe 目录）
    if iterations == 10:
        temp_output_file = OSEM_DIR / f"{patient_name}_{projection_type}_OSEMReconed.dat"
        final_output_file = DATA_DIR / projection_type / "reconstructions" / patient_name / f"{patient_name}_{projection_type}_OSEMReconed.dat"
    else:
        temp_output_file = OSEM_DIR / f"{patient_name}_{projection_type}_OSEMReconed_Iter{iterations}.dat"
        final_output_file = DATA_DIR / projection_type / "reconstructions" / patient_name / f"{patient_name}_{projection_type}_OSEMReconed_Iter{iterations}.dat"

    # 修改路径参数（使用相对路径，都在 osemreocnexe 目录下）
    par_content = re.sub(
        r'(ProjectionFileName=)[^;]+(;)',
        f'\\1{temp_proj_name}\\2',
        par_content
    )
    par_content = re.sub(
        r'(CpOrbitFileName=)[^;]+(;)',
        f'\\1orbit.orb\\2',
        par_content
    )
    par_content = re.sub(
        r'(AtnMap_File_Name=)[^;]+(;)',
        f'\\1PostAtten.dat\\2',
        par_content
    )
    par_content = re.sub(
        r'(OutputImageFileName=)[^;]+(;)',
        f'\\1{temp_output_file.name}\\2',
        par_content
    )

    # 修改或添加迭代次数
    if re.search(r'Iterations=', par_content):
        # 如果存在，则替换
        par_content = re.sub(
            r'(Iterations=)[0-9]+(;)',
            rf'\g<1>{iterations}\g<2>',
            par_content
        )
    else:
        # 如果不存在，在 Image_Pixel_Height 后面添加
        par_content = re.sub(
            r'(Image_Pixel_Height=[0-9.]+;)',
            rf'\g<1>\nIterations={iterations};',
            par_content
        )

    # 保存临时 par 文件到 OSEM 目录
    temp_par = OSEM_DIR / "ParamFile4OSEMRecon.par"
    with open(temp_par, 'w', encoding='utf-8') as f:
        f.write(par_content)

    return temp_par, temp_output_file, final_output_file


def run_osem_recon(patient_name, projection_type, iterations=10, skip_if_exists=True):
    """
    执行 OSEM 重建

    Args:
        patient_name: 病人姓名
        projection_type: 'original' or 'denoised'
        iterations: 迭代次数（默认 10）
        skip_if_exists: 如果输出文件已存在，是否跳过

    Returns:
        (success, duration, message, final_output_file)
    """
    temp_proj_file = None
    temp_output_file = None

    try:
        # 准备 par 文件（复制文件到 osemreocnexe）
        temp_par, temp_output_file, final_output_file = prepare_par_file(
            patient_name, projection_type, iterations
        )

        # 记录临时投影文件名，用于清理
        if projection_type == "original":
            temp_proj_file = OSEM_DIR / "ProjectionImage1.dat"
        else:
            temp_proj_file = OSEM_DIR / "ProjectionImage1_denoised.dat"

        # 检查最终输出文件是否已存在
        if skip_if_exists and final_output_file.exists():
            print(f"  ⏭️  Output already exists: {final_output_file.name}")
            # 清理临时文件
            cleanup_temp_files(temp_proj_file)
            return True, 0.0, "Skipped (already exists)", final_output_file

        # 运行 OSEM 重建
        print(f"  🔄 Running OSEM reconstruction (Iter {iterations})...")
        print(f"     开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = time.time()

        # 实时输出 exe 运行结果
        process = subprocess.Popen(
            [str(OSEM_EXE.absolute())],
            cwd=str(OSEM_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 实时打印输出
        output_lines = []
        try:
            for line in process.stdout:
                print(f"     [EXE] {line.rstrip()}")
                output_lines.append(line)

            process.wait(timeout=600)  # 10 分钟超时
            result_returncode = process.returncode

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print(f"  ❌ Timeout after 10 minutes")
            cleanup_temp_files(temp_proj_file)
            return False, 600.0, "Timeout", None

        duration = time.time() - start_time
        print(f"     结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     耗时: {duration:.1f} 秒")

        # 检查临时输出文件
        if temp_output_file.exists():
            file_size = temp_output_file.stat().st_size / (1024 * 1024)  # MB

            # 移动到最终位置
            final_output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_output_file), str(final_output_file))

            print(f"  ✅ Success!")
            print(f"     Duration: {duration:.1f}s ({duration/60:.2f} min)")
            print(f"     Size: {file_size:.1f} MB")
            print(f"     Saved to: {final_output_file.relative_to(BASE_DIR)}")

            # 清理临时文件
            cleanup_temp_files(temp_proj_file)

            return True, duration, f"Success ({file_size:.1f} MB, {duration:.1f}s)", final_output_file
        else:
            print(f"  ❌ Failed! Output file not created (exit code: {result_returncode})")
            # 清理临时文件
            cleanup_temp_files(temp_proj_file)
            return False, duration, f"Failed (exit code: {result_returncode}, {duration:.1f}s)", None

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        cleanup_temp_files(temp_proj_file)
        return False, 0.0, str(e), None


def cleanup_temp_files(temp_proj_file):
    """清理 osemreocnexe 目录中的临时文件"""
    try:
        # 清理投影文件
        if temp_proj_file and temp_proj_file.exists():
            temp_proj_file.unlink()

        # 清理 orbit 和 atten 文件
        temp_orbit = OSEM_DIR / "orbit.orb"
        temp_atten = OSEM_DIR / "PostAtten.dat"

        if temp_orbit.exists():
            temp_orbit.unlink()
        if temp_atten.exists():
            temp_atten.unlink()

    except Exception as e:
        print(f"  ⚠️  Warning: Failed to cleanup temp files: {e}")


def generate_comparison_gif(patient_name, original_iter10_file, original_iter30_file, denoised_iter10_file, denoised_iter30_file, skip_if_exists=True):
    """
    生成对比 GIF (2x2布局)

    Args:
        patient_name: 病人姓名
        original_iter10_file: 原始重建文件（Iter 10）
        original_iter30_file: 原始重建文件（Iter 30）
        denoised_iter10_file: 降噪重建文件（Iter 10）
        denoised_iter30_file: 降噪重建文件（Iter 30）
        skip_if_exists: 是否跳过已存在的 GIF
    """
    try:
        gif_dir = BASE_DIR / "results" / "osem_comparison_gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

        gif_file = gif_dir / f"{patient_name}_osem_comparison.gif"

        # 检查 GIF 是否已存在
        if skip_if_exists and gif_file.exists():
            print(f"  ⏭️  GIF already exists: {gif_file.name}")
            return True, "Skipped (already exists)"

        print(f"  🎬 Generating comparison GIF...")

        # 调用 gif 生成脚本
        gif_script = BASE_DIR / "scripts" / "generate_2x2_osem_comparison.py"

        result = subprocess.run(
            [
                "python3",
                str(gif_script),
                "--original-iter10", str(original_iter10_file),
                "--original-iter30", str(original_iter30_file),
                "--denoised-iter10", str(denoised_iter10_file),
                "--denoised-iter30", str(denoised_iter30_file),
                "--output", str(gif_file),
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0 and gif_file.exists():
            print(f"  ✅ GIF generated: {gif_file.name}")
            return True, "Success"
        else:
            print(f"  ❌ GIF generation failed")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False, "Failed"

    except Exception as e:
        print(f"  ❌ Error generating GIF: {str(e)}")
        return False, str(e)


def batch_reconstruct(patient_list=None, skip_if_exists=True, generate_gif=True):
    """
    批量重建（每个病人生成 3 个重建 + GIF）

    Args:
        patient_list: 病人列表，None 表示所有病人
        skip_if_exists: 是否跳过已存在的输出文件
        generate_gif: 是否生成对比 GIF
    """
    if patient_list is None:
        patient_list = get_patient_list()

    print(f"\n{'='*70}")
    print(f"OSEM 批量重建 + GIF 生成")
    print(f"{'='*70}")
    print(f"病人数量: {len(patient_list)}")
    print(f"每个病人: Original (Iter 10/30) + Denoised (Iter 10/30)")
    print(f"总任务数: {len(patient_list)} 病人 × 4 重建 = {len(patient_list) * 4} 重建")
    if generate_gif:
        print(f"           + {len(patient_list)} GIF (2×2)")
    print(f"{'='*70}\n")

    # 记录结果
    results = []

    for idx, patient in enumerate(patient_list, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(patient_list)}] 📋 Patient: {patient}")
        print(f"{'='*70}")

        patient_results = {'patient': patient}
        output_files = {}

        # 1. Original reconstruction (Iter 10)
        print(f"\n[1/4] Original projection (Iter 10):")
        success, duration, message, output_file = run_osem_recon(
            patient, 'original', iterations=10, skip_if_exists=skip_if_exists
        )
        patient_results['original_iter10_success'] = success
        patient_results['original_iter10_duration'] = duration
        patient_results['original_iter10_message'] = message
        if output_file:
            output_files['original_iter10'] = output_file

        # 2. Original reconstruction (Iter 30)
        print(f"\n[2/4] Original projection (Iter 30):")
        success, duration, message, output_file = run_osem_recon(
            patient, 'original', iterations=30, skip_if_exists=skip_if_exists
        )
        patient_results['original_iter30_success'] = success
        patient_results['original_iter30_duration'] = duration
        patient_results['original_iter30_message'] = message
        if output_file:
            output_files['original_iter30'] = output_file

        # 3. Denoised reconstruction (Iter 10)
        print(f"\n[3/4] Denoised projection (Iter 10):")
        success, duration, message, output_file = run_osem_recon(
            patient, 'denoised', iterations=10, skip_if_exists=skip_if_exists
        )
        patient_results['denoised_iter10_success'] = success
        patient_results['denoised_iter10_duration'] = duration
        patient_results['denoised_iter10_message'] = message
        if output_file:
            output_files['denoised_iter10'] = output_file

        # 4. Denoised reconstruction (Iter 30)
        print(f"\n[4/4] Denoised projection (Iter 30):")
        success, duration, message, output_file = run_osem_recon(
            patient, 'denoised', iterations=30, skip_if_exists=skip_if_exists
        )
        patient_results['denoised_iter30_success'] = success
        patient_results['denoised_iter30_duration'] = duration
        patient_results['denoised_iter30_message'] = message
        if output_file:
            output_files['denoised_iter30'] = output_file

        # 5. Generate comparison GIF
        if generate_gif and len(output_files) == 4:
            print(f"\n[5/5] Generating comparison GIF:")
            gif_success, gif_message = generate_comparison_gif(
                patient,
                output_files['original_iter10'],
                output_files['original_iter30'],
                output_files['denoised_iter10'],
                output_files['denoised_iter30'],
                skip_if_exists=skip_if_exists
            )
            patient_results['gif_success'] = gif_success
            patient_results['gif_message'] = gif_message

        results.append(patient_results)

        print(f"\n{'='*70}")
        print(f"✅ Patient {patient} completed!")
        print(f"{'='*70}\n")

    # 保存报告
    save_report(results)

    # 打印总结
    print_summary(results)


def save_report(results):
    """保存重建报告为 CSV"""
    report_dir = BASE_DIR / "results"
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"osem_recon_report_{timestamp}.csv"

    with open(report_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊 Report saved to: {report_file}")


def print_summary(results):
    """打印重建总结"""
    print(f"\n{'='*70}")
    print(f"重建总结")
    print(f"{'='*70}")

    total = len(results)
    original_success = sum(1 for r in results if r.get('original_success', False))
    denoised_success = sum(1 for r in results if r.get('denoised_success', False))
    denoised_iter30_success = sum(1 for r in results if r.get('denoised_iter30_success', False))
    gif_success = sum(1 for r in results if r.get('gif_success', False))

    total_original_time = sum(r.get('original_duration', 0) for r in results)
    total_denoised_time = sum(r.get('denoised_duration', 0) for r in results)
    total_denoised_iter30_time = sum(r.get('denoised_iter30_duration', 0) for r in results)

    print(f"总病人数: {total}")
    print(f"\n重建成功率:")
    print(f"  Original (Iter 10):  {original_success}/{total} ({original_success/total*100:.1f}%)")
    print(f"  Denoised (Iter 10):  {denoised_success}/{total} ({denoised_success/total*100:.1f}%)")
    print(f"  Denoised (Iter 30):  {denoised_iter30_success}/{total} ({denoised_iter30_success/total*100:.1f}%)")
    if any('gif_success' in r for r in results):
        print(f"  GIF 生成:            {gif_success}/{total} ({gif_success/total*100:.1f}%)")

    print(f"\n时间统计:")
    print(f"  Original 总时间:     {total_original_time/60:.1f} 分钟")
    print(f"  Denoised 总时间:     {total_denoised_time/60:.1f} 分钟")
    print(f"  Denoised(30) 总时间: {total_denoised_iter30_time/60:.1f} 分钟")
    total_time = total_original_time + total_denoised_time + total_denoised_iter30_time
    print(f"  总时间:              {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")

    if total > 0:
        avg_time_per_patient = total_time / total
        print(f"  平均每病人:          {avg_time_per_patient:.1f} 秒")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OSEM 批量重建脚本（自动生成对比 GIF）")
    parser.add_argument(
        '--patients',
        nargs='+',
        help='指定病人列表（默认：所有病人）'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重建，即使输出文件已存在'
    )
    parser.add_argument(
        '--no-gif',
        action='store_true',
        help='不生成对比 GIF'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='测试模式：只处理第一个病人'
    )

    args = parser.parse_args()

    # 获取病人列表
    if args.test:
        patient_list = get_patient_list()[:1]
        print(f"🧪 测试模式：只处理病人 {patient_list}")
    elif args.patients:
        patient_list = args.patients
    else:
        patient_list = None

    # 执行批量重建
    batch_reconstruct(
        patient_list=patient_list,
        skip_if_exists=not args.force,
        generate_gif=not args.no_gif
    )

