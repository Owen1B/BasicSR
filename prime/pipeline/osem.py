from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _is_wsl() -> bool:
    """Return True if running inside WSL (Windows Subsystem for Linux)."""
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        # Common markers:
        # - /proc/version contains "Microsoft"
        # - /proc/sys/kernel/osrelease contains "microsoft"
        v = Path("/proc/version")
        if v.exists() and "microsoft" in v.read_text(encoding="utf-8", errors="ignore").lower():
            return True
        r = Path("/proc/sys/kernel/osrelease")
        if r.exists() and "microsoft" in r.read_text(encoding="utf-8", errors="ignore").lower():
            return True
    except Exception:
        pass
    return False


def _copy2_if_different(src: Path, dst: Path) -> bool:
    """shutil.copy2 but ignore SameFileError when src and dst point to the same file.

    Returns True if a copy was performed, False if skipped due to same-file.
    """
    src_p = Path(src)
    dst_p = Path(dst)
    try:
        if src_p.resolve() == dst_p.resolve():
            # If user points src at the same file but it doesn't exist, fail loudly.
            if not src_p.exists():
                raise FileNotFoundError(f"required file missing: {src_p}")
            return False
    except Exception:
        # If resolve fails for any reason, fall back to best-effort copy.
        pass
    shutil.copy2(src_p, dst_p)
    return True


def run_osem_reconstruction(
    proj_file: Path,
    patient_name: str,
    par_file: Path,
    orbit_file: Path,
    atten_file: Path,
    output_name: str,
    iterations: int = 10,
    views_per_subset: int | None = None,
    prj_data_type: int | None = None,
    osem_dir: Path | None = None,
    final_output_dir: Path | None = None,
    output_filename: str | None = None,
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

    # NOTE:
    # - On Windows: run directly.
    # - On WSL: Windows interop can run *.exe directly (preferred; avoids wine DLL issues).
    # - On other Linux: require wine/wine64.
    cmd: list[str]
    if os.name == "nt":
        cmd = [str(osem_exe.absolute())]
    elif _is_wsl():
        # WSL interop: calling the .exe directly typically works.
        cmd = [str(osem_exe.absolute())]
    else:
        # Prefer the 64-bit wine loader for this PE32+ (x86-64) executable.
        # On some distros, `wine` may route to a 32-bit loader and fail with kernel32.dll errors.
        wine = (
            shutil.which("wine64")
            or ("/usr/lib/wine/wine64" if Path("/usr/lib/wine/wine64").exists() else None)
            or shutil.which("wine")
        )
        if not wine:
            raise RuntimeError(
                "当前环境是非 Windows 且不是 WSL，无法直接运行 osemrecon.exe。\n"
                "可选方案：\n"
                "1) 在 Windows/WSL（启用 Windows interop）环境跑重建；\n"
                "2) 安装并配置 wine，然后再运行（不保证 DLL/依赖兼容）。\n"
                "提示：只做推理/投影域指标可继续，重建阶段需要可运行的 OSEM。"
            )
        cmd = [wine, str(osem_exe.absolute())]

    # 固定文件名（程序/旧 par 逻辑）
    temp_proj = osem_dir / "ProjectionImage1.dat"
    temp_par = osem_dir / "ParamFile4OSEMRecon.par"
    temp_orbit = osem_dir / "orbit.orb"
    temp_atten = osem_dir / "PostAtten.dat"

    copied_proj = _copy2_if_different(proj_file, temp_proj)
    _copy2_if_different(par_file, temp_par)
    copied_orbit = _copy2_if_different(orbit_file, temp_orbit)
    copied_atten = _copy2_if_different(atten_file, temp_atten)

    # 仅修改 par 内路径相关字段（其余保持病人自带 par 不变）
    par_content = temp_par.read_text(encoding="utf-8", errors="ignore")
    # Use \g<1>...\g<2> form to avoid any ambiguity with backslashes in replacement strings.
    par_content = re.sub(r"(ProjectionFileName=)[^;]+(;)", r"\g<1>ProjectionImage1.dat\g<2>", par_content)
    par_content = re.sub(r"(CpOrbitFileName=)[^;]+(;)", r"\g<1>orbit.orb\g<2>", par_content)
    par_content = re.sub(r"(AtnMap_File_Name=)[^;]+(;)", r"\g<1>PostAtten.dat\g<2>", par_content)
    if prj_data_type is not None:
        # PrjDataType mapping (empirically verified with this OSEM build):
        # - 2: int16 projections (count domain integers)
        # - 1: float32 projections (supports fractional counts; NOT rounded internally)
        par_content2, n = re.subn(
            r"(PrjDataType=)[^;]+(;)",
            rf"\g<1>{int(prj_data_type)}\g<2>",
            par_content,
        )
        if n == 0:
            raise RuntimeError("PrjDataType field not found in par file; cannot set projection dtype")
        par_content = par_content2
    # Ensure reconstruction params match caller intent (do not rely on whatever is in patient's .par).
    par_content = re.sub(r"(Iterations=)[^;]+(;)", rf"\g<1>{int(iterations)}\g<2>", par_content)
    if views_per_subset is not None:
        par_content = re.sub(r"(Number_of_Views_Per_Subset=)[^;]+(;)", rf"\g<1>{int(views_per_subset)}\g<2>", par_content)
    if output_filename is None:
        output_filename = f"{patient_name}_{output_name}_OSEMReconed_Iter{iterations}.dat"
    par_content = re.sub(r"(OutputImageFileName=)[^;]+(;)", rf"\g<1>{output_filename}\g<2>", par_content)
    temp_par.write_text(par_content, encoding="utf-8")

    # 运行并记录日志
    final_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = final_output_dir / f"{patient_name}_{output_name}_osem_log.txt"
    result = subprocess.run(
        cmd,
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
                f"  {' '.join(cmd)}",
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
    # IMPORTANT:
    # - orbit.orb may be a shared static file under osem_dir; do NOT delete it when src==dst.
    # - same for other resources if caller points src to the same path.
    to_cleanup: list[Path] = []
    if bool(copied_proj):
        to_cleanup.append(temp_proj)
    if bool(copied_orbit):
        to_cleanup.append(temp_orbit)
    if bool(copied_atten):
        to_cleanup.append(temp_atten)
    for temp_file in to_cleanup:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass

    return final_output


def resolve_osem_output_filename(
    *,
    patient_name: str,
    output_name: str,
    iterations: int,
    output_filename: str | None = None,
) -> str:
    """Resolve deterministic OSEM output filename from arguments."""
    if output_filename is not None:
        return str(output_filename)
    return f"{patient_name}_{output_name}_OSEMReconed_Iter{int(iterations)}.dat"


def ensure_osem_reconstruction(
    *,
    proj_file: Path,
    patient_name: str,
    par_file: Path,
    orbit_file: Path,
    atten_file: Path,
    output_name: str,
    iterations: int = 10,
    views_per_subset: int | None = None,
    prj_data_type: int | None = None,
    osem_dir: Path | None = None,
    final_output_dir: Path | None = None,
    output_filename: str | None = None,
    timeout_sec: int = 600,
    overwrite: bool = False,
) -> Path:
    """Return cached reconstruction if present; otherwise run OSEM reconstruction."""
    proj_file = Path(proj_file)
    final_output_dir = Path(final_output_dir) if final_output_dir is not None else (proj_file.parent.parent / "reconstructions" / patient_name)
    out_name = resolve_osem_output_filename(
        patient_name=str(patient_name),
        output_name=str(output_name),
        iterations=int(iterations),
        output_filename=output_filename,
    )
    out_path = final_output_dir / out_name
    if out_path.exists() and not bool(overwrite):
        return out_path
    return run_osem_reconstruction(
        proj_file=proj_file,
        patient_name=str(patient_name),
        par_file=Path(par_file),
        orbit_file=Path(orbit_file),
        atten_file=Path(atten_file),
        output_name=str(output_name),
        iterations=int(iterations),
        views_per_subset=views_per_subset,
        prj_data_type=prj_data_type,
        osem_dir=osem_dir,
        final_output_dir=final_output_dir,
        output_filename=out_name,
        timeout_sec=int(timeout_sec),
    )










