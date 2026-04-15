#!/usr/bin/env python3
"""
病人级一站式可视化/对比脚本（SPECT229）

功能：
1) 指定 checkpoint + config，对选中的病人执行：
   - 投影降噪（按视角）
   - Poisson sample（对降噪投影）
   - 三套投影 OSEM 重建
   - 生成 2x3（3proj + 3recon MIP）GIF，并写 counts.txt
2) 生成中间切片图（重建域，三正交切面）
3) 生成衰减图切片（可选：也生成衰减旋转 GIF）
4) 支持多个模型 ckpt 的对比：每个模型各自产物 + 额外输出“多模型 denoised 重建切片对比图”

说明：
- 本脚本尽量复用已有通用脚本：
  - 2x3 GIF + counts：`generate_3projections_3recons_gif.py`
  - 衰减旋转 GIF：`generate_postatten_gif.py`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import matplotlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from prime.pipeline.io import load_atten_f32, load_recon_f32
from prime.pipeline.viz import merge_gifs_row
from prime.pipeline.threeproj3recon import process_patient_3proj3recon

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _load_recon(path: Path, shape=(128, 128, 128)) -> np.ndarray:
    return load_recon_f32(path, shape=shape)


def _load_atten(path: Path) -> np.ndarray:
    return load_atten_f32(path)


def _p99(v: np.ndarray, q: float = 99.9) -> float:
    if np.any(v > 0):
        return float(np.percentile(v[v > 0], q))
    return float(v.max())


def _orth_slices(vol: np.ndarray, mid: int = 64) -> List[np.ndarray]:
    # [z,y,x]
    axial = np.flipud(vol[mid, :, :])
    coronal = np.flipud(vol[:, mid, :])
    sagittal = np.flipud(vol[:, :, mid])
    return [axial, coronal, sagittal]


def save_recon_mid_slices_png(
    recon_paths: List[Path],
    row_titles: List[str],
    out_png: Path,
    vmax_mode: str = "p99.9",
    use_log1p: bool = True,
    gamma: float = 0.8,
) -> None:
    """rows = recon types, cols = axial/coronal/sagittal"""
    vols = [_load_recon(p) for p in recon_paths]
    if vmax_mode == "max":
        vmax = max(float(v.max()) for v in vols)
        vmax_desc = "max"
    else:
        vmax = max(_p99(v, 99.9) for v in vols)
        vmax_desc = "p99.9"
    vmin = 0.0

    # 归一化（可选 log1p + gamma），用于显示
    def _norm_show(x: np.ndarray) -> np.ndarray:
        f = np.clip(x.astype(np.float32, copy=False), vmin, vmax)
        f = (f - vmin) / max(1e-8, (vmax - vmin))
        if use_log1p:
            f = np.log1p(9.0 * f) / np.log1p(9.0)
        if gamma != 1.0:
            f = np.power(np.clip(f, 0.0, 1.0), float(gamma))
        return f

    fig, axes = plt.subplots(len(vols), 3, figsize=(12, 4 * len(vols)))
    if len(vols) == 1:
        axes = np.expand_dims(axes, 0)

    for r, (vol, title) in enumerate(zip(vols, row_titles)):
        slices = _orth_slices(vol)
        for c, sl in enumerate(slices):
            axes[r, c].imshow(_norm_show(sl), cmap="hot", norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(["Axial", "Coronal", "Sagittal"][c], fontsize=12, fontweight="bold")
        axes[r, 0].text(
            -0.02,
            0.5,
            title,
            transform=axes[r, 0].transAxes,
            va="center",
            ha="right",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(f"Recon mid-slices (vmax={vmax_desc}, log1p={use_log1p}, gamma={gamma})", fontsize=13, y=0.99)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_atten_mid_slices_png(atten_path: Path, out_png: Path, use_log1p: bool = False) -> None:
    vol = _load_atten(atten_path)
    vmax = _p99(vol, 99.9)
    vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for c, sl in enumerate(_orth_slices(vol, mid=vol.shape[0] // 2)):
        if use_log1p:
            f = np.clip(sl, vmin, vmax)
            f = (f - vmin) / max(1e-8, (vmax - vmin))
            f = np.log1p(9.0 * f) / np.log1p(9.0)
            axes[c].imshow(f, cmap="gray", norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
        else:
            axes[c].imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[c].axis("off")
        axes[c].set_title(["Axial", "Coronal", "Sagittal"][c], fontsize=12, fontweight="bold")
    fig.suptitle(f"Atten mid-slices (vmax=p99.9, log1p={use_log1p})", fontsize=13, y=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def merge_2x3_gifs(gif_paths: List[Path], labels: List[str], out_gif: Path, layout: str = "row") -> None:
    if layout != "row":
        raise ValueError(f"Unsupported layout: {layout}")
    merge_gifs_row(gif_paths=gif_paths, labels=labels, out_gif=out_gif)


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    checkpoint: Path
    config: Path


def _run_recon_job(
    patient: str,
    tag: str,
    checkpoint: str,
    config: str,
    spect229_dir: str,
    orbit_file: str,
    out_root: str,
    max_value: float,
    iterations: int,
    skip_existing: bool,
) -> str:
    """CPU-side heavy work: OSEM recon + GIF + counts. Safe to run in multiprocessing."""
    spect229_dir_p = Path(spect229_dir)
    orbit_file_p = Path(orbit_file)
    out_root_p = Path(out_root)
    spec = ModelSpec(tag=tag, checkpoint=Path(checkpoint), config=Path(config))
    _run_3proj3recon(
        patient=patient,
        spec=spec,
        spect229_dir=spect229_dir_p,
        orbit_file=orbit_file_p,
        out_root=out_root_p,
        max_value=float(max_value),
        iterations=int(iterations),
        skip_existing=bool(skip_existing),
    )
    return f"{tag}:{patient}"


def _run_gif_job(
    patient: str,
    tag: str,
    checkpoint: str,
    config: str,
    spect229_dir: str,
    orbit_file: str,
    out_root: str,
    max_value: float,
    iterations: int,
    skip_existing: bool,
) -> str:
    """CPU-side plot work only: load recon/proj and generate counts+GIF."""
    spect229_dir_p = Path(spect229_dir)
    orbit_file_p = Path(orbit_file)
    out_root_p = Path(out_root)
    input_proj = spect229_dir_p / patient / f"{patient}_Proj4Filter.dat"
    par_file = spect229_dir_p / patient / f"{patient}_ParamFile.par"
    atten_file = spect229_dir_p / patient / f"{patient}_PostAtten.dat"
    out_dir = out_root_p / tag
    process_patient_3proj3recon(
        patient=patient,
        checkpoint=Path(checkpoint),
        config=Path(config),
        input_proj=input_proj,
        par_file=par_file,
        orbit_file=orbit_file_p,
        atten_file=atten_file,
        output_dir=out_dir,
        max_value=float(max_value),
        iterations=int(iterations),
        skip_existing=bool(skip_existing),
        skip_denoise=True,
        skip_recon=True,
        use_log1p=True,
        device="cpu",
        stage="gif_counts",
    )
    return f"{tag}:{patient}"


def _read_patients_file(p: Path) -> List[str]:
    out: List[str] = []
    for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.split()[0])
    return out


def _list_patients_all(spect229_dir: Path) -> List[str]:
    # 只要目录下存在对应投影文件，就认为是有效病人
    patients: List[str] = []
    for d in sorted(spect229_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d.name
        if (spect229_dir / p / f"{p}_Proj4Filter.dat").exists():
            patients.append(p)
    return patients


def _resolve_patients(args: argparse.Namespace, spect229_dir: Path) -> List[str]:
    if args.patients_all:
        patients = _list_patients_all(spect229_dir)
    elif args.patients_file:
        patients = _read_patients_file(Path(args.patients_file))
    elif args.patients and len(args.patients) > 0:
        patients = list(args.patients)
    else:
        raise ValueError("必须提供 --patients / --patients-file / --patients-all 之一")

    start = max(0, int(args.start_index))
    if start > 0:
        patients = patients[start:]
    if args.max_patients is not None:
        patients = patients[: max(0, int(args.max_patients))]
    return patients


def _parse_counts_txt(path: Path) -> Dict[str, float]:
    """解析 counts.txt（由 generate_3projections_3recons_gif.py 写入）。"""
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: Dict[str, float] = {}
    for line in txt:
        line = line.strip()
        if line.startswith("patient:"):
            out["patient"] = line.split(":", 1)[1].strip()

    def _extract_pct(s: str) -> Optional[float]:
        if "(" not in s or "%" not in s:
            return None
        try:
            inner = s.split("(", 1)[1]
            pct_s = inner.split("%", 1)[0]
            return float(pct_s)
        except Exception:
            return None

    in_proj = False
    in_recon = False
    for line in txt:
        line = line.strip()
        if line.startswith("[projection_domain_total_counts]"):
            in_proj = True
            in_recon = False
            continue
        if line.startswith("[reconstruction_domain_total_counts]"):
            in_proj = False
            in_recon = True
            continue

        if in_proj:
            if line.startswith("denoised:"):
                v = _extract_pct(line)
                if v is not None:
                    out["proj_denoised_pct"] = v
            if line.startswith("denoised_poisson:"):
                v = _extract_pct(line)
                if v is not None:
                    out["proj_denoised_poisson_pct"] = v
            if line.startswith("original:"):
                try:
                    out["proj_original"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass

        if in_recon:
            if line.startswith("denoised:"):
                v = _extract_pct(line)
                if v is not None:
                    out["recon_denoised_pct"] = v
            if line.startswith("denoised_poisson:"):
                v = _extract_pct(line)
                if v is not None:
                    out["recon_denoised_poisson_pct"] = v
            if line.startswith("original:"):
                try:
                    out["recon_original"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
    return out


def _write_counts_summary_for_model(out_dir: Path, patients: List[str], out_summary_dir: Path) -> None:
    """只统计指定 patients（不会扫全目录）。"""
    # lazy import：只有需要 summary 时才引入 pandas
    import pandas as pd  # type: ignore

    rows: List[Dict[str, float]] = []
    for p in patients:
        counts = out_dir / p / "counts.txt"
        if not counts.exists():
            continue
        d = _parse_counts_txt(counts)
        if "patient" not in d:
            d["patient"] = p
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    out_summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_summary_dir / "counts_activity_analysis.csv"
    df.to_csv(csv_path, index=False)

    # plots（复用原 analyze 脚本风格）
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    try:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    except Exception:
        pass

    proj = df["proj_denoised_pct"].dropna()
    recon = df["recon_denoised_pct"].dropna()

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(proj, bins=25, color="steelblue", alpha=0.75, edgecolor="black")
    ax1.axvline(0, color="red", linestyle="--", linewidth=2)
    ax1.axvline(proj.mean(), color="orange", linestyle="-", linewidth=2, label=f"mean={proj.mean():+.2f}%")
    ax1.set_title("Projection total-count change (denoised vs original)", fontweight="bold")
    ax1.set_xlabel("Change (%)")
    ax1.set_ylabel("Patients")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(recon, bins=25, color="coral", alpha=0.75, edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.axvline(recon.mean(), color="orange", linestyle="-", linewidth=2, label=f"mean={recon.mean():+.2f}%")
    ax2.set_title("Reconstruction total-activity change (denoised vs original)", fontweight="bold")
    ax2.set_xlabel("Change (%)")
    ax2.set_ylabel("Patients")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, :])
    valid = df.dropna(subset=["proj_denoised_pct", "recon_denoised_pct"])
    sc = ax3.scatter(
        valid["proj_denoised_pct"],
        valid["recon_denoised_pct"],
        c=np.abs(valid["recon_denoised_pct"]),
        cmap="YlOrRd",
        s=60,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.axhline(0, color="gray", linewidth=1, alpha=0.6)
    ax3.axvline(0, color="gray", linewidth=1, alpha=0.6)
    ax3.set_xlabel("Projection change (%)")
    ax3.set_ylabel("Recon change (%)")
    ax3.set_title("Projection change vs Reconstruction change", fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("|Recon change| (%)")

    ax4 = fig.add_subplot(gs[2, 0])
    data_to_plot = [proj, recon]
    bp = ax4.boxplot(
        data_to_plot,
        tick_labels=["Proj change", "Recon change"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    for patch, color in zip(bp["boxes"], ["steelblue", "coral"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.axhline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax4.set_ylabel("Change (%)")
    ax4.set_title("Distribution summary (boxplot)", fontweight="bold")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    table_data = [
        ["Metric", "Proj change", "Recon change"],
        ["N", f"{len(proj)}", f"{len(recon)}"],
        ["Mean (%)", f"{proj.mean():+.2f}", f"{recon.mean():+.2f}"],
        ["Min (%)", f"{proj.min():+.2f}", f"{recon.min():+.2f}"],
        ["Max (%)", f"{proj.max():+.2f}", f"{recon.max():+.2f}"],
    ]
    table = ax5.table(cellText=table_data, cellLoc="center", loc="center", bbox=[0.05, 0.35, 0.9, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.5)
        elif key[0] == len(table_data) - 1:
            cell.visible_edges = "B"
            cell.set_linewidth(1.5)
        else:
            cell.visible_edges = ""

    fig.tight_layout()
    fig.savefig(out_summary_dir / "counts_activity_analysis_enhanced.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # summary bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    valid2 = df.dropna(subset=["proj_denoised_pct", "recon_denoised_pct"])
    x = np.arange(len(valid2))
    width = 0.35
    ax.bar(x - width / 2, valid2["proj_denoised_pct"], width, label="Proj", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, valid2["recon_denoised_pct"], width, label="Recon", color="coral", alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(valid2["patient"].tolist(), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Change (%)")
    ax.set_title("Counts/Activity change per patient (denoised vs original)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_summary_dir / "counts_activity_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_3proj3recon(
    patient: str,
    spec: ModelSpec,
    spect229_dir: Path,
    orbit_file: Path,
    out_root: Path,
    max_value: float,
    iterations: int,
    skip_existing: bool,
) -> Path:
    input_proj = spect229_dir / patient / f"{patient}_Proj4Filter.dat"
    par_file = spect229_dir / patient / f"{patient}_ParamFile.par"
    atten_file = spect229_dir / patient / f"{patient}_PostAtten.dat"
    out_dir = out_root / spec.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 直接调用 pipeline 实现（避免 subprocess，逻辑只维护一份）
    process_patient_3proj3recon(
        patient=patient,
        checkpoint=spec.checkpoint,
        config=spec.config,
        input_proj=input_proj,
        par_file=par_file,
        orbit_file=orbit_file,
        atten_file=atten_file,
        output_dir=out_dir,
        max_value=float(max_value),
        iterations=int(iterations),
        skip_existing=bool(skip_existing),
        skip_denoise=False,
        skip_recon=False,
        use_log1p=True,
        device="cuda",
    )
    return out_dir


def _parse_models(models: List[str]) -> List[ModelSpec]:
    """
    --model 形如：
      tag=/abs/ckpt.pth:/abs/config.yml
    例：
      --model cons=experiments/.../net_g_500000.pth:experiments/.../xxx.yml
    """
    out: List[ModelSpec] = []
    for s in models:
        if "=" not in s or ":" not in s:
            raise ValueError(f"非法 --model: {s}，期望 tag=ckpt:config")
        tag, rest = s.split("=", 1)
        ckpt_s, cfg_s = rest.split(":", 1)
        out.append(ModelSpec(tag=tag, checkpoint=Path(ckpt_s), config=Path(cfg_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SPECT229 病人级报告：2x3 GIF + 中间切片 + 衰减图 + 多模型对比")
    parser.add_argument("--patients", nargs="*", default=None, help="病人名列表（如 AnYufeng BaiYukun ...）")
    parser.add_argument("--patients-file", type=str, default=None, help="从文件读取病人列表（每行一个，#开头为注释）")
    parser.add_argument("--patients-all", action="store_true", help="自动扫描 spect229-dir 下所有病人（以 *_Proj4Filter.dat 存在为准）")
    parser.add_argument("--start-index", type=int, default=0, help="从排序后的病人列表第 start-index 个开始跑（断点分段）")
    parser.add_argument("--max-patients", type=int, default=None, help="最多跑 N 个病人（用于分段/调试）")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="模型规格：tag=ckpt:config（可多次传入以对比多个模型）",
    )
    parser.add_argument("--spect229-dir", type=str, default="datasets/SPECT229", help="SPECT229 根目录")
    parser.add_argument("--orbit-file", type=str, required=True, help="orbit.orb（可统一用同一个）")
    parser.add_argument("--results-root", type=str, default="outputs", help="统一结果根目录")
    parser.add_argument("--exp-name", type=str, required=True, help="实验名（将输出到 results/<exp-name>/...）")
    parser.add_argument("--out-root", type=str, default=None, help="可选：手动指定输出根目录（覆盖默认 results-root/exp-name）")
    parser.add_argument("--max-value", type=float, default=150.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true", help="断点续跑：若最终 GIF+counts.txt 已存在则跳过该病人")
    parser.add_argument("--slice-vmax", type=str, default="p99.9", choices=["p99.9", "max"])
    parser.add_argument("--slice-log1p", action="store_true", default=True, help="重建切片图使用 log1p（默认开）")
    parser.add_argument("--slice-no-log1p", dest="slice_log1p", action="store_false", help="关闭重建切片 log1p（线性显示）")
    parser.add_argument("--slice-gamma", type=float, default=0.8)
    parser.add_argument("--atten-log1p", action="store_true", help="衰减切片使用 log1p（默认线性）")
    parser.add_argument("--pbar", action="store_true", default=True, help="显示 tqdm 进度条（默认开启）")
    parser.add_argument("--no-pbar", dest="pbar", action="store_false", help="关闭 tqdm 进度条")
    parser.add_argument(
        "--gif-workers",
        type=int,
        default=1,
        help="并行跑绘图/GIF 的进程数（重建用 GPU：保持串行；仅把 GIF+counts 的 CPU 绘图放进进程池）",
    )
    # atten gif 已不再做（可视化/对比的核心是 recon/2x3；需要时再加回去）
    # 已移除：recon-atten 2x2 GIF 逻辑
    parser.add_argument(
        "--merge-2x3-gif",
        action="store_true",
        help="若传入多个 --model，则把各模型的 2x3 GIF 拼成一个大对比 GIF（每列一个模型）",
    )
    parser.add_argument("--summary", action="store_true", help="在本次处理的病人范围内，汇总 counts.txt 并输出 CSV+PNG")
    args = parser.parse_args()

    spect229_dir = Path(args.spect229_dir)
    orbit_file = Path(args.orbit_file)
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = Path(args.results_root) / str(args.exp_name) / "patient_reports"
    out_root.mkdir(parents=True, exist_ok=True)

    models = _parse_models(args.model)
    patients = _resolve_patients(args, spect229_dir=spect229_dir)

    # 逐病人逐模型执行 2x3 + counts
    total_jobs = len(patients) * len(models)
    use_pbar = bool(args.pbar) and tqdm is not None and total_jobs > 0
    job_iter = (
        tqdm(total=total_jobs, desc="run_patient_report", unit="job", dynamic_ncols=True)
        if use_pbar
        else None
    )
    gif_workers = max(1, int(getattr(args, "gif_workers", 1)))
    parallel_gif = gif_workers > 1
    executor = None
    futures = []

    try:
        if parallel_gif:
            executor = ProcessPoolExecutor(max_workers=gif_workers)

        for patient in patients:
            for spec in models:
                # Pipeline mode (recommended when OSEM uses GPU):
                # - GPU: denoise + poisson + recon runs sequentially in main process.
                # - CPU: GIF+counts generation runs in a process pool, overlapping with next GPU job.
                if parallel_gif:
                    input_proj = spect229_dir / patient / f"{patient}_Proj4Filter.dat"
                    par_file = spect229_dir / patient / f"{patient}_ParamFile.par"
                    atten_file = spect229_dir / patient / f"{patient}_PostAtten.dat"
                    out_dir = out_root / spec.tag
                    process_patient_3proj3recon(
                        patient=patient,
                        checkpoint=spec.checkpoint,
                        config=spec.config,
                        input_proj=input_proj,
                        par_file=par_file,
                        orbit_file=orbit_file,
                        atten_file=atten_file,
                        output_dir=out_dir,
                        max_value=float(args.max_value),
                        iterations=int(args.iterations),
                        skip_existing=bool(args.skip_existing),
                        skip_denoise=False,
                        skip_recon=False,
                        use_log1p=True,
                        device="cuda",
                        stage="recon_only",
                    )

                    fut = executor.submit(
                        _run_gif_job,
                        patient,
                        spec.tag,
                        str(spec.checkpoint),
                        str(spec.config),
                        str(spect229_dir),
                        str(orbit_file),
                        str(out_root),
                        float(args.max_value),
                        int(args.iterations),
                        bool(args.skip_existing),
                    )
                    futures.append(fut)
                    continue

                # Non-parallel (original behavior): all in one go
                out_dir = _run_3proj3recon(
                    patient=patient,
                    spec=spec,
                    spect229_dir=spect229_dir,
                    orbit_file=orbit_file,
                    out_root=out_root,
                    max_value=float(args.max_value),
                    iterations=int(args.iterations),
                    skip_existing=bool(args.skip_existing),
                )

                # 生成切片图：orig / denoised / denoised_poisson 三行
                recon_dir = out_dir / patient / "reconstructions"
                recon_paths = [
                    recon_dir / f"{patient}_original_OSEMReconed_Iter{args.iterations}.dat",
                    recon_dir / f"{patient}_denoised_OSEMReconed_Iter{args.iterations}.dat",
                    recon_dir / f"{patient}_denoised_poisson_OSEMReconed_Iter{args.iterations}.dat",
                ]
                slice_png = out_dir / patient / f"{patient}_recon_mid_slices_{spec.tag}.png"
                save_recon_mid_slices_png(
                    recon_paths=recon_paths,
                    row_titles=["original", "denoised", "denoised_poisson"],
                    out_png=slice_png,
                    vmax_mode=args.slice_vmax,
                    use_log1p=bool(args.slice_log1p),
                    gamma=float(args.slice_gamma),
                )

                # 衰减切片 + 可选 GIF
                atten_path = spect229_dir / patient / f"{patient}_PostAtten.dat"
                atten_png = out_dir / patient / f"{patient}_atten_mid_slices.png"
                save_atten_mid_slices_png(atten_path, atten_png, use_log1p=bool(args.atten_log1p))

                if job_iter is not None:
                    job_iter.update(1)

            # 多模型对比：仅对比 “denoised 重建”三正交切片（每个模型一行）
            if len(models) >= 2:
                compare_paths: List[Path] = []
                row_titles: List[str] = []
                for spec in models:
                    out_dir = out_root / spec.tag
                    recon_dir = out_dir / patient / "reconstructions"
                    compare_paths.append(recon_dir / f"{patient}_denoised_OSEMReconed_Iter{args.iterations}.dat")
                    row_titles.append(f"{spec.tag} (denoised)")
                compare_png = out_root / patient / f"{patient}_compare_models_denoised_recon_mid_slices.png"
                save_recon_mid_slices_png(
                    recon_paths=compare_paths,
                    row_titles=row_titles,
                    out_png=compare_png,
                    vmax_mode=args.slice_vmax,
                    use_log1p=bool(args.slice_log1p),
                    gamma=float(args.slice_gamma),
                )

            # 多模型对比：拼 2x3 GIF（每列一个模型）
            if bool(args.merge_2x3_gif) and len(models) >= 2:
                gif_paths: List[Path] = []
                labels: List[str] = []
                for spec in models:
                    gif_p = out_root / spec.tag / patient / f"{patient}_3proj_3recon_2x3.gif"
                    if gif_p.exists():
                        gif_paths.append(gif_p)
                        labels.append(spec.tag)
                if len(gif_paths) >= 2:
                    out_gif = out_root / patient / f"{patient}_compare_models_2x3.gif"
                    merge_2x3_gifs(gif_paths, labels, out_gif, layout="row")

        # If parallel gif: wait and advance progress bar as jobs finish.
        if parallel_gif and futures:
            for fut in as_completed(futures):
                _ = fut.result()
                if job_iter is not None:
                    job_iter.update(1)

        # In pipeline mode, generate cross-model comparisons after GIFs are ready.
        if parallel_gif:
            for patient in patients:
                if len(models) >= 2:
                    compare_paths = []
                    row_titles = []
                    for spec in models:
                        out_dir = out_root / spec.tag
                        recon_dir = out_dir / patient / "reconstructions"
                        compare_paths.append(recon_dir / f"{patient}_denoised_OSEMReconed_Iter{args.iterations}.dat")
                        row_titles.append(f"{spec.tag} (denoised)")
                    compare_png = out_root / patient / f"{patient}_compare_models_denoised_recon_mid_slices.png"
                    save_recon_mid_slices_png(
                        recon_paths=compare_paths,
                        row_titles=row_titles,
                        out_png=compare_png,
                        vmax_mode=args.slice_vmax,
                        use_log1p=bool(args.slice_log1p),
                        gamma=float(args.slice_gamma),
                    )

                if bool(args.merge_2x3_gif) and len(models) >= 2:
                    gif_paths = []
                    labels = []
                    for spec in models:
                        gif_p = out_root / spec.tag / patient / f"{patient}_3proj_3recon_2x3.gif"
                        if gif_p.exists():
                            gif_paths.append(gif_p)
                            labels.append(spec.tag)
                    if len(gif_paths) >= 2:
                        out_gif = out_root / patient / f"{patient}_compare_models_2x3.gif"
                        merge_2x3_gifs(gif_paths, labels, out_gif, layout="row")
    finally:
        if job_iter is not None:
            job_iter.close()
        if executor is not None:
            executor.shutdown(wait=True)

    # 汇总统计（只统计这次选择的 patients）
    if bool(args.summary):
        for spec in models:
            model_out_dir = out_root / spec.tag
            summary_dir = out_root / "_summary" / spec.tag
            _write_counts_summary_for_model(model_out_dir, patients=patients, out_summary_dir=summary_dir)


if __name__ == "__main__":
    main()


