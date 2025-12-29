from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def run_osem_reconstruction(
    proj_file: Path,
    patient_name: str,
    par_file: Path,
    orbit_file: Path,
    atten_file: Path,
    output_name: str,
    iterations: int = 10,
    osem_dir: Path | None = None,
    final_output_dir: Path | None = None,
    timeout_sec: int = 600,
) -> Path:
    """运行外部 `osemrecon.exe`，并把输出移动到最终目录。

    约定（与旧脚本一致）：
    - osem 程序只识别固定 par 名：`ParamFile4OSEMRecon.par`
    - 投影/轨道/衰减也拷贝到固定文件名，避免 par 里路径不一致
    """
    if osem_dir is None:
        osem_dir = Path(__file__).resolve().parents[1] / "osemreocnexe"

    if final_output_dir is None:
        final_output_dir = proj_file.parent.parent / "reconstructions" / patient_name

    osem_dir = Path(osem_dir)
    final_output_dir = Path(final_output_dir)

    osem_exe = osem_dir / "osemrecon.exe"
    if not osem_exe.exists():
        raise FileNotFoundError(f"OSEM 可执行文件不存在: {osem_exe}")

    # 固定文件名（程序/旧 par 逻辑）
    temp_proj = osem_dir / "ProjectionImage1.dat"
    temp_par = osem_dir / "ParamFile4OSEMRecon.par"
    temp_orbit = osem_dir / "orbit.orb"
    temp_atten = osem_dir / "PostAtten.dat"

    shutil.copy2(proj_file, temp_proj)
    shutil.copy2(par_file, temp_par)
    shutil.copy2(orbit_file, temp_orbit)
    shutil.copy2(atten_file, temp_atten)

    # 仅修改 par 内路径相关字段（其余保持病人自带 par 不变）
    par_content = temp_par.read_text(encoding="utf-8", errors="ignore")
    par_content = re.sub(r"(ProjectionFileName=)[^;]+(;)", r"\1ProjectionImage1.dat\2", par_content)
    par_content = re.sub(r"(CpOrbitFileName=)[^;]+(;)", r"\1orbit.orb\2", par_content)
    par_content = re.sub(r"(AtnMap_File_Name=)[^;]+(;)", r"\1PostAtten.dat\2", par_content)
    output_filename = f"{patient_name}_{output_name}_OSEMReconed_Iter{iterations}.dat"
    par_content = re.sub(r"(OutputImageFileName=)[^;]+(;)", rf"\1{output_filename}\2", par_content)
    temp_par.write_text(par_content, encoding="utf-8")

    # 运行并记录日志
    final_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = final_output_dir / f"{patient_name}_{output_name}_osem_log.txt"
    result = subprocess.run(
        [str(osem_exe.absolute())],
        cwd=str(osem_dir),
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )

    log_file.write_text(
        "\n".join(
            [
                "=" * 80,
                "OSEM Reconstruction Log",
                f"Patient: {patient_name}",
                f"Output Name: {output_name}",
                f"Iterations: {iterations}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 80,
                "",
                "Command:",
                f"  {osem_exe.absolute()}",
                f"  Working Directory: {osem_dir}",
                "",
                "=" * 80,
                "STDOUT:",
                "=" * 80,
                result.stdout if result.stdout else "(empty)",
                "",
                "=" * 80,
                "STDERR:",
                "=" * 80,
                result.stderr if result.stderr else "(empty)",
                "",
                "=" * 80,
                f"Exit Code: {result.returncode}",
                "=" * 80,
                "",
            ]
        ),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(f"OSEM 重建失败 (exit code: {result.returncode}), 日志: {log_file}")

    output_file = osem_dir / output_filename
    if not output_file.exists():
        raise FileNotFoundError(f"重建输出文件不存在: {output_file}")

    final_output = final_output_dir / output_filename
    shutil.move(str(output_file), str(final_output))

    # 清理临时文件（par 保留给下次覆盖）
    for temp_file in [temp_proj, temp_orbit, temp_atten]:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass

    return final_output











