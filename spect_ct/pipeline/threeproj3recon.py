from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from spect_ct.pipeline.inference import (
    denoise_views,
    load_basicsr_net,
    proj_total_counts_round_clip,
    recon_total_counts_clip,
)
from spect_ct.pipeline.io import load_projection_i16, load_recon_f32, save_projection_i16_round_clip
from spect_ct.pipeline.osem import run_osem_reconstruction
from spect_ct.pipeline.poisson import poisson_sample
from spect_ct.pipeline.viz import compute_mip, create_subplot_with_colorbar, rotate_volume_around_x


@dataclass(frozen=True)
class Patient3Proj3ReconOutputs:
    patient: str
    output_dir: Path
    gif_path: Path
    counts_txt: Path
    proj_dir: Path
    recon_dir: Path


def generate_2x3_gif(
    orig_proj: np.ndarray,
    denoised_proj: np.ndarray,
    denoised_poisson_proj: np.ndarray,
    orig_recon: np.ndarray,
    denoised_recon: np.ndarray,
    denoised_poisson_recon: np.ndarray,
    output_path: Path,
    proj_vmax_mode: str = "p99.9",
    recon_vmax_mode: str = "max75",
    use_log1p: bool = True,
    fps: float = 10.0,
    num_recon_frames: int = 120,
) -> None:
    """生成 2x3 布局 GIF：第一行投影、第二行重建 MIP。"""
    num_proj_frames = int(orig_proj.shape[0])  # 60

    # 投影 vmax
    if proj_vmax_mode == "max":
        vmax_proj = float(max(orig_proj.max(), denoised_proj.max(), denoised_poisson_proj.max()))
    else:
        def _p999(x: np.ndarray) -> float:
            return float(np.percentile(x[x > 0], 99.9)) if np.any(x > 0) else float(x.max())

        vmax_proj = max(_p999(orig_proj), _p999(denoised_proj), _p999(denoised_poisson_proj))

    # 重建 vmax
    if recon_vmax_mode == "max":
        vmax_recon = float(max(orig_recon.max(), denoised_recon.max(), denoised_poisson_recon.max()))
    elif recon_vmax_mode == "p9995":
        def _p9995(x: np.ndarray) -> float:
            return float(np.percentile(x[x > 0], 99.95)) if np.any(x > 0) else float(x.max())

        vmax_recon = max(_p9995(orig_recon), _p9995(denoised_recon), _p9995(denoised_poisson_recon))
    else:
        vmax_recon = float(max(orig_recon.max(), denoised_recon.max(), denoised_poisson_recon.max()) * 0.75)

    frames: list[Image.Image] = []
    recon_angle_start = 90.0
    recon_angle_step = 360.0 / float(num_recon_frames)

    for i in range(int(num_recon_frames)):
        proj_idx = int(i * num_proj_frames / num_recon_frames)
        proj_idx = min(proj_idx, num_proj_frames - 1)
        recon_angle = recon_angle_start + i * recon_angle_step

        img_proj_orig = create_subplot_with_colorbar(orig_proj[proj_idx], 0.0, vmax_proj, "gray", title="Original Proj", use_log1p=use_log1p)
        img_proj_den = create_subplot_with_colorbar(denoised_proj[proj_idx], 0.0, vmax_proj, "gray", title="Denoised Proj", use_log1p=use_log1p)
        img_proj_poi = create_subplot_with_colorbar(
            denoised_poisson_proj[proj_idx], 0.0, vmax_proj, "gray", title="Denoised+Poisson Proj", use_log1p=use_log1p
        )

        rot_o = rotate_volume_around_x(orig_recon, recon_angle)
        rot_d = rotate_volume_around_x(denoised_recon, recon_angle)
        rot_p = rotate_volume_around_x(denoised_poisson_recon, recon_angle)

        mip_o = np.flipud(compute_mip(rot_o, axis=2))
        mip_d = np.flipud(compute_mip(rot_d, axis=2))
        mip_p = np.flipud(compute_mip(rot_p, axis=2))

        img_rec_o = create_subplot_with_colorbar(mip_o, 0.0, vmax_recon, "hot", title="Original Recon", use_log1p=use_log1p)
        img_rec_d = create_subplot_with_colorbar(mip_d, 0.0, vmax_recon, "hot", title="Denoised Recon", use_log1p=use_log1p)
        img_rec_p = create_subplot_with_colorbar(mip_p, 0.0, vmax_recon, "hot", title="Denoised+Poisson Recon", use_log1p=use_log1p)

        w, h = img_proj_orig.width, img_proj_orig.height
        canvas = Image.new("RGB", (w * 3, h * 2))
        canvas.paste(img_proj_orig, (0, 0))
        canvas.paste(img_proj_den, (w, 0))
        canvas.paste(img_proj_poi, (w * 2, 0))
        canvas.paste(img_rec_o, (0, h))
        canvas.paste(img_rec_d, (w, h))
        canvas.paste(img_rec_p, (w * 2, h))
        frames.append(canvas)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000.0 / max(float(fps), 1e-6)),
        loop=0,
        optimize=False,
    )


def process_patient_3proj3recon(
    *,
    patient: str,
    checkpoint: Path,
    config: Path,
    input_proj: Path,
    par_file: Path,
    orbit_file: Path,
    atten_file: Path,
    output_dir: Path,
    max_value: float = 150.0,
    iterations: int = 10,
    skip_existing: bool = False,
    skip_denoise: bool = False,
    skip_recon: bool = False,
    use_log1p: bool = True,
    device: str = "cuda",
    stage: str = "all",  # all | denoise | recon_only | gif_counts
) -> Patient3Proj3ReconOutputs:
    """单病人：生成 2x3 GIF + counts.txt（以及中间 dat/projection/recon 文件）。"""
    patient = str(patient)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_dir = output_dir / patient / "projections"
    recon_dir = output_dir / patient / "reconstructions"
    proj_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    gif_path = output_dir / patient / f"{patient}_3proj_3recon_2x3.gif"
    counts_txt = output_dir / patient / "counts.txt"

    if skip_existing and gif_path.exists() and counts_txt.exists():
        return Patient3Proj3ReconOutputs(
            patient=patient,
            output_dir=output_dir,
            gif_path=gif_path,
            counts_txt=counts_txt,
            proj_dir=proj_dir,
            recon_dir=recon_dir,
        )

    stage = str(stage).lower().strip()
    if stage not in ["all", "denoise", "recon_only", "gif_counts"]:
        raise ValueError(f"Unsupported stage: {stage}. Use all|denoise|recon_only|gif_counts")

    # 1) load original projection
    orig_proj = load_projection_i16(input_proj)
    orig_proj_file = proj_dir / "original_projection.dat"
    save_projection_i16_round_clip(orig_proj_file, orig_proj)

    # 2) denoise
    denoised_proj_file = proj_dir / "denoised_projection.dat"
    if not skip_denoise:
        loaded = load_basicsr_net(config, checkpoint, device=device)
        denoised_proj = denoise_views(loaded.net, orig_proj, max_value=float(max_value), device=device)
        save_projection_i16_round_clip(denoised_proj_file, denoised_proj)
    else:
        denoised_proj = load_projection_i16(denoised_proj_file)

    # 3) poisson sample
    denoised_poisson_proj = poisson_sample(denoised_proj)
    denoised_poisson_proj_file = proj_dir / "denoised_poisson_projection.dat"
    save_projection_i16_round_clip(denoised_poisson_proj_file, denoised_poisson_proj)

    # Stage gate: allow preparing projections first, then run recon+gif in parallel.
    if stage == "denoise":
        return Patient3Proj3ReconOutputs(
            patient=patient,
            output_dir=output_dir,
            gif_path=gif_path,
            counts_txt=counts_txt,
            proj_dir=proj_dir,
            recon_dir=recon_dir,
        )

    # In gif_counts stage, we should not require GPU.
    # Ensure projections & recon outputs exist (they should if stage='recon_only' ran previously).
    if stage == "gif_counts":
        orig_proj = load_projection_i16(orig_proj_file)
        denoised_proj = load_projection_i16(denoised_proj_file)
        denoised_poisson_proj = load_projection_i16(denoised_poisson_proj_file)

    # 4) reconstructions
    recon_paths: dict[str, Path] = {}
    for proj_file, recon_name in [
        (orig_proj_file, "original"),
        (denoised_proj_file, "denoised"),
        (denoised_poisson_proj_file, "denoised_poisson"),
    ]:
        recon_out = recon_dir / f"{patient}_{recon_name}_OSEMReconed_Iter{iterations}.dat"
        if skip_recon and recon_out.exists():
            recon_paths[recon_name] = recon_out
        else:
            recon_out = run_osem_reconstruction(
                proj_file=proj_file,
                patient_name=patient,
                par_file=par_file,
                orbit_file=orbit_file,
                atten_file=atten_file,
                output_name=recon_name,
                iterations=int(iterations),
                final_output_dir=recon_dir,
            )
            recon_paths[recon_name] = recon_out

    if stage == "recon_only":
        return Patient3Proj3ReconOutputs(
            patient=patient,
            output_dir=output_dir,
            gif_path=gif_path,
            counts_txt=counts_txt,
            proj_dir=proj_dir,
            recon_dir=recon_dir,
        )

    # 5) load recon volumes
    orig_recon = load_recon_f32(recon_paths["original"], shape=(128, 128, 128))
    denoised_recon = load_recon_f32(recon_paths["denoised"], shape=(128, 128, 128))
    denoised_poisson_recon = load_recon_f32(recon_paths["denoised_poisson"], shape=(128, 128, 128))

    # 6) counts.txt
    def _safe_pct(new: float, base: float) -> float:
        if abs(base) < 1e-12:
            return float("nan")
        return (new - base) / base * 100.0

    proj_counts = {
        "original": proj_total_counts_round_clip(orig_proj),
        "denoised": proj_total_counts_round_clip(denoised_proj),
        "denoised_poisson": proj_total_counts_round_clip(denoised_poisson_proj),
    }
    recon_counts = {
        "original": recon_total_counts_clip(orig_recon),
        "denoised": recon_total_counts_clip(denoised_recon),
        "denoised_poisson": recon_total_counts_clip(denoised_poisson_recon),
    }

    lines = [
        f"patient: {patient}",
        f"checkpoint: {Path(checkpoint)}",
        f"config: {Path(config)}",
        f"max_value: {float(max_value)}",
        f"iterations: {int(iterations)}",
        "",
        "[projection_domain_total_counts]  (sum over views/pixels, using round()+clip>=0)",
        f"original: {proj_counts['original']:.0f}",
        f"denoised: {proj_counts['denoised']:.0f}  ({_safe_pct(proj_counts['denoised'], proj_counts['original']):+.4f}%)",
        f"denoised_poisson: {proj_counts['denoised_poisson']:.0f}  ({_safe_pct(proj_counts['denoised_poisson'], proj_counts['original']):+.4f}%)",
        "",
        "[reconstruction_domain_total_counts]  (sum over voxels, clip>=0)",
        f"original: {recon_counts['original']:.6f}",
        f"denoised: {recon_counts['denoised']:.6f}  ({_safe_pct(recon_counts['denoised'], recon_counts['original']):+.4f}%)",
        f"denoised_poisson: {recon_counts['denoised_poisson']:.6f}  ({_safe_pct(recon_counts['denoised_poisson'], recon_counts['original']):+.4f}%)",
        "",
    ]
    counts_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 7) 2x3 gif
    generate_2x3_gif(
        orig_proj=orig_proj,
        denoised_proj=denoised_proj,
        denoised_poisson_proj=denoised_poisson_proj,
        orig_recon=orig_recon,
        denoised_recon=denoised_recon,
        denoised_poisson_recon=denoised_poisson_recon,
        output_path=gif_path,
        proj_vmax_mode="p99.9",
        recon_vmax_mode="max75",
        use_log1p=bool(use_log1p),
    )

    return Patient3Proj3ReconOutputs(
        patient=patient,
        output_dir=output_dir,
        gif_path=gif_path,
        counts_txt=counts_txt,
        proj_dir=proj_dir,
        recon_dir=recon_dir,
    )



