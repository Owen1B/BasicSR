from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spect_ct.pipeline.inference import (
    denoise_views,
    load_basicsr_net,
    proj_total_counts_clip,
    proj_total_counts_round_clip,
    recon_total_counts_clip,
)
from spect_ct.pipeline.io import (
    load_projection_f32,
    load_projection_i16,
    load_recon_f32,
    save_projection_f32,
)
from spect_ct.pipeline.osem import run_osem_reconstruction
from spect_ct.pipeline.poisson import poisson_sample
from spect_ct.pipeline.viz import compute_mip, create_subplot_with_colorbar, render_2x4_nogap_frame, rotate_volume_around_x


@dataclass(frozen=True)
class Patient3Proj3ReconOutputs:
    patient: str
    output_dir: Path
    gif_path: Path
    counts_txt: Path
    proj_dir: Path
    recon_dir: Path


def generate_gif_counts_from_existing(
    *,
    patient: str,
    patient_dir: Path,
    iterations: int = 10,
    max_value: float = 150.0,
    use_log1p: bool = True,
    skip_existing: bool = True,
    mp4: bool = False,
    gif: bool = True,
) -> tuple[Path, Path]:
    """仅基于已存在的 projections + reconstructions 生成 2x3 GIF + counts.txt。

    需求场景：
    - 已经有投影（original/denoised/denoised_poisson）以及重建（original/denoised/denoised_poisson）
    - 不想再跑推理/重建，只想补生成 GIF/统计

    Required files under patient_dir (fp32-only layout):
      projections/original_projection_f32.dat
      projections/denoised_projection_f32.dat
      projections/denoised_poisson_projection_f32.dat
      reconstructions/<patient>_{original,denoised,denoised_poisson}_OSEMReconed_Iter{iterations}.dat
    """
    patient = str(patient)
    patient_dir = Path(patient_dir)
    proj_dir = patient_dir / "projections"
    recon_dir = patient_dir / "reconstructions"

    # Put all GIFs directly under <patient>/gifs/ (no extra nested folder).
    gif_dir = patient_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{patient}_4proj_4recon_2x4.gif"
    mp4_path = gif_dir / f"{patient}_4proj_4recon_2x4.mp4"
    counts_txt = patient_dir / "counts.txt"
    if bool(skip_existing) and counts_txt.exists():
        ok_gif = (not bool(gif)) or gif_path.exists()
        ok_mp4 = (not bool(mp4)) or mp4_path.exists()
        if ok_gif and ok_mp4:
            return gif_path, counts_txt

    # ---- projections ----
    orig_proj_file = proj_dir / "original_projection_f32.dat"
    den_f32_file = proj_dir / "denoised_projection_f32.dat"
    den_p_file = proj_dir / "denoised_poisson_projection_f32.dat"
    missing_proj = [str(p) for p in [orig_proj_file, den_f32_file, den_p_file] if not p.exists()]
    if missing_proj:
        raise FileNotFoundError(f"[{patient}] missing projections: {missing_proj}")

    orig_proj = load_projection_f32(orig_proj_file)
    denoised_fp32_proj = load_projection_f32(den_f32_file)
    denoised_u16_proj = denoised_fp32_proj  # placeholder for legacy layout; no int16 anymore
    denoised_poisson_proj = load_projection_f32(den_p_file)

    # ---- reconstructions ----
    it = int(iterations)
    r_o = recon_dir / f"{patient}_original_OSEMReconed_Iter{it}.dat"
    r_u = recon_dir / f"{patient}_denoised_OSEMReconed_Iter{it}.dat"
    r_f = recon_dir / f"{patient}_denoised_fp32_OSEMReconed_Iter{it}.dat"
    r_p = recon_dir / f"{patient}_denoised_poisson_OSEMReconed_Iter{it}.dat"
    missing_recon = [str(p) for p in [r_o, r_u, r_f, r_p] if not p.exists()]
    if missing_recon:
        raise FileNotFoundError(f"[{patient}] missing reconstructions: {missing_recon}")

    orig_recon = load_recon_f32(r_o, shape=(128, 128, 128))
    denoised_u16_recon = load_recon_f32(r_u, shape=(128, 128, 128))
    denoised_fp32_recon = load_recon_f32(r_f, shape=(128, 128, 128))
    denoised_poisson_recon = load_recon_f32(r_p, shape=(128, 128, 128))

    # ---- counts.txt ----
    def _safe_pct(new: float, base: float) -> float:
        if abs(base) < 1e-12:
            return float("nan")
        return (new - base) / base * 100.0

    proj_counts = {
        "original": proj_total_counts_round_clip(orig_proj),
        "denoised_u16": proj_total_counts_round_clip(denoised_u16_proj),
        # fp32 projection is lambda-like; use float sum (no rounding) to match validation display.
        "denoised_fp32": proj_total_counts_clip(denoised_fp32_proj),
        "denoised_fp32_round": proj_total_counts_round_clip(denoised_fp32_proj),
        "denoised_poisson": proj_total_counts_round_clip(denoised_poisson_proj),
    }
    recon_counts = {
        "original": recon_total_counts_clip(orig_recon),
        "denoised_u16": recon_total_counts_clip(denoised_u16_recon),
        "denoised_fp32": recon_total_counts_clip(denoised_fp32_recon),
        "denoised_poisson": recon_total_counts_clip(denoised_poisson_recon),
    }
    lines = [
        f"patient: {patient}",
        "source: existing_outputs_only",
        f"max_value: {float(max_value)}",
        f"iterations: {int(iterations)}",
        "",
        "[projection_domain_total_counts]  (sum over views/pixels, using round()+clip>=0)",
        f"original: {proj_counts['original']:.0f}",
        f"denoised_u16: {proj_counts['denoised_u16']:.0f}  ({_safe_pct(proj_counts['denoised_u16'], proj_counts['original']):+.4f}%)",
        f"denoised_fp32: {proj_counts['denoised_fp32']:.3f}  ({_safe_pct(proj_counts['denoised_fp32'], proj_counts['original']):+.4f}%)",
        f"denoised_fp32_round: {proj_counts['denoised_fp32_round']:.0f}  ({_safe_pct(proj_counts['denoised_fp32_round'], proj_counts['original']):+.4f}%)",
        f"denoised_poisson: {proj_counts['denoised_poisson']:.0f}  ({_safe_pct(proj_counts['denoised_poisson'], proj_counts['original']):+.4f}%)",
        "",
        "[reconstruction_domain_total_counts]  (sum over voxels, clip>=0)",
        f"original: {recon_counts['original']:.6f}",
        f"denoised_u16: {recon_counts['denoised_u16']:.6f}  ({_safe_pct(recon_counts['denoised_u16'], recon_counts['original']):+.4f}%)",
        f"denoised_fp32: {recon_counts['denoised_fp32']:.6f}  ({_safe_pct(recon_counts['denoised_fp32'], recon_counts['original']):+.4f}%)",
        f"denoised_poisson: {recon_counts['denoised_poisson']:.6f}  ({_safe_pct(recon_counts['denoised_poisson'], recon_counts['original']):+.4f}%)",
        "",
    ]
    counts_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 2x3 gif ----
    generate_2x4_gif(
        orig_proj=orig_proj,
        denoised_u16_proj=denoised_u16_proj,
        denoised_fp32_proj=denoised_fp32_proj,
        denoised_poisson_proj=denoised_poisson_proj,
        orig_recon=orig_recon,
        denoised_u16_recon=denoised_u16_recon,
        denoised_fp32_recon=denoised_fp32_recon,
        denoised_poisson_recon=denoised_poisson_recon,
        output_path=(gif_path if bool(gif) else None),
        mp4_output_path=(mp4_path if bool(mp4) else None),
        proj_vmax_mode="p99.9",
        recon_vmax_mode="max75",
        use_log1p=bool(use_log1p),
        proj_metric_max_value=float(max_value),
        lpips_nets=("alex", "vgg"),
        recon_turns=(1.5 if bool(mp4) else 1.0),
    )
    return gif_path, counts_txt


def _normalize_to_u8(x: np.ndarray, *, vmax: float, use_log1p: bool) -> np.ndarray:
    vmax = float(vmax)
    if vmax < 1e-6:
        vmax = 1.0
    y = np.clip(x.astype(np.float32, copy=False), 0.0, vmax)
    if bool(use_log1p):
        y = np.log1p(y) / np.log1p(vmax)
    else:
        y = y / vmax
    return (y * 255.0).round().astype(np.uint8)


def _pick_sample_dir(gif_root: Path, base_name: str) -> Path:
    """Pick a non-conflicting sample directory like <base>_000, <base>_001, ..."""
    gif_root = Path(gif_root)
    gif_root.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        d = gif_root / f"{base_name}_{i:03d}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            return d
    # fallback: reuse 999 if many exist
    d = gif_root / f"{base_name}_{999:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_1x3_proj_gif(
    *,
    orig_proj: np.ndarray,
    denoised_proj: np.ndarray,
    denoised_proj_q: np.ndarray | None = None,
    denoised_poisson_proj: np.ndarray,
    output_path: Path,
    use_log1p: bool = True,
    fps: float = 10.0,
    metric_max_value: float | None = None,
) -> None:
    """生成投影 GIF。

    - denoised_proj_q is None: 1x3 = Original | Denoised(fp32) | Denoised+Poisson
    - denoised_proj_q provided: 1x4 = Original | Denoised(fp32) | Denoised(uint16 quantized) | Denoised+Poisson
    """
    num_frames = int(orig_proj.shape[0])
    vmax = float(np.max(orig_proj))  # display scaling follows BasicSR: per-sample max from ORIGINAL
    # Precompute counts (aligned with counts.txt: round()+clip>=0).
    tot_o = float(proj_total_counts_round_clip(orig_proj))
    tot_d = float(proj_total_counts_clip(denoised_proj))
    tot_q = float(proj_total_counts_round_clip(denoised_proj_q)) if denoised_proj_q is not None else float("nan")
    tot_p = float(proj_total_counts_round_clip(denoised_poisson_proj))

    def _pct(new: float, base: float) -> float:
        if abs(base) < 1e-12:
            return float("nan")
        return (new - base) / base * 100.0

    def _per_view_counts_round_clip(x: np.ndarray) -> np.ndarray:
        y = np.round(x.astype(np.float64, copy=False))
        y = np.clip(y, 0.0, None)
        return y.sum(axis=(1, 2)).astype(np.float64, copy=False)

    # Precompute metrics vs the first column (Original). We show view-averaged metrics (not per-view).
    mv = float(metric_max_value) if metric_max_value is not None else float(vmax)
    if mv <= 0:
        mv = 1.0

    def _psnr_ssim_views_mean(pred: np.ndarray, ref: np.ndarray, data_range: float) -> tuple[float, float]:
        # Prefer BasicSR metrics (cv2-based) in KAIR env; fallback otherwise.
        try:
            from basicsr.metrics import calculate_psnr_range, calculate_ssim_range  # type: ignore
            _use_basicsr = True
        except Exception:
            _use_basicsr = False
            from spect_ct.pipeline.simple_metrics import psnr_range, ssim_range

        psnrs = []
        ssims = []
        dr = float(data_range)
        for i in range(int(ref.shape[0])):
            r = np.clip(ref[i].astype(np.float32, copy=False), 0.0, dr)
            p = np.clip(pred[i].astype(np.float32, copy=False), 0.0, dr)
            if _use_basicsr:
                psnrs.append(float(calculate_psnr_range(p[:, :, None], r[:, :, None], crop_border=0, data_range=float(dr), input_order="HWC")))
                ssims.append(float(calculate_ssim_range(p[:, :, None], r[:, :, None], crop_border=0, data_range=float(dr), input_order="HWC")))
            else:
                psnrs.append(float(psnr_range(p, r, data_range=dr)))
                ssims.append(float(ssim_range(p, r, data_range=dr)))
        return float(np.mean(psnrs)) if psnrs else float("nan"), float(np.mean(ssims)) if ssims else float("nan")

    psnr_d, ssim_d = float("nan"), float("nan")
    psnr_q, ssim_q = float("nan"), float("nan")
    psnr_p, ssim_p = float("nan"), float("nan")
    try:
        psnr_d, ssim_d = _psnr_ssim_views_mean(denoised_proj, orig_proj, mv)
        if denoised_proj_q is not None:
            psnr_q, ssim_q = _psnr_ssim_views_mean(denoised_proj_q, orig_proj, mv)
        psnr_p, ssim_p = _psnr_ssim_views_mean(denoised_poisson_proj, orig_proj, mv)
    except Exception:
        pass

    lp_d_alex = float("nan")
    lp_p_alex = float("nan")
    lp_d_vgg = float("nan")
    lp_p_vgg = float("nan")
    lp_q_alex = float("nan")
    lp_q_vgg = float("nan")
    try:
        from spect_ct.pipeline.metrics import lpips_views_mean
        lp_d = lpips_views_mean(pred=denoised_proj, ref=orig_proj, max_value=mv, nets=("alex", "vgg"))
        lp_p = lpips_views_mean(pred=denoised_poisson_proj, ref=orig_proj, max_value=mv, nets=("alex", "vgg"))
        lp_d_alex = float(lp_d.get("alex", float("nan")))
        lp_p_alex = float(lp_p.get("alex", float("nan")))
        lp_d_vgg = float(lp_d.get("vgg", float("nan")))
        lp_p_vgg = float(lp_p.get("vgg", float("nan")))
        if denoised_proj_q is not None:
            lp_q = lpips_views_mean(pred=denoised_proj_q, ref=orig_proj, max_value=mv, nets=("alex", "vgg"))
            lp_q_alex = float(lp_q.get("alex", float("nan")))
            lp_q_vgg = float(lp_q.get("vgg", float("nan")))
    except Exception:
        pass

    def _load_small_font(size: int = 10) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    font = _load_small_font(10)
    text_h = 82  # a bit more space: total + 4 metric lines (PSNR/SSIM/LPIPS alex/vgg)
    frames: list[Image.Image] = []
    for i in range(num_frames):
        a_u8 = _normalize_to_u8(orig_proj[i], vmax=vmax, use_log1p=use_log1p)
        b_u8 = _normalize_to_u8(denoised_proj[i], vmax=vmax, use_log1p=use_log1p)
        q_u8 = _normalize_to_u8(denoised_proj_q[i], vmax=vmax, use_log1p=use_log1p) if denoised_proj_q is not None else None
        c_u8 = _normalize_to_u8(denoised_poisson_proj[i], vmax=vmax, use_log1p=use_log1p)
        a = Image.fromarray(a_u8, mode="L")
        b = Image.fromarray(b_u8, mode="L")
        q = Image.fromarray(q_u8, mode="L") if q_u8 is not None else None
        c = Image.fromarray(c_u8, mode="L")

        w, h = a.width, a.height
        ncols = 4 if denoised_proj_q is not None else 3
        canvas = Image.new("L", (w * ncols, h + text_h), color=0)
        canvas.paste(a, (0, 0))
        canvas.paste(b, (w, 0))
        if q is not None:
            canvas.paste(q, (w * 2, 0))
            canvas.paste(c, (w * 3, 0))
        else:
            canvas.paste(c, (w * 2, 0))

        # Put stats under each subplot (small font), with a bit of bottom padding.
        draw = ImageDraw.Draw(canvas)
        pad_x = 3
        pad_y = 2
        y_txt = h + pad_y

        # Original panel (baseline)
        txt_o = "\n".join(
            [
                f"T {tot_o:.0f} (+0.000%)",
                "PSNR -  SSIM -",
                "LP A -  V -",
            ]
        )
        # Denoised panel (vs original)
        txt_d = "\n".join(
            [
                f"T {tot_d:.0f} ({_pct(tot_d, tot_o):+.3f}%)",
                f"PSNR {psnr_d:.2f}  SSIM {ssim_d:.4f}",
                f"LP A {lp_d_alex:.4f}  V {lp_d_vgg:.4f}",
            ]
        )
        txt_q = None
        if denoised_proj_q is not None:
            txt_q = "\n".join(
                [
                    f"T {tot_q:.0f} ({_pct(tot_q, tot_o):+.3f}%)",
                    f"PSNR {psnr_q:.2f}  SSIM {ssim_q:.4f}",
                    f"LP A {lp_q_alex:.4f}  V {lp_q_vgg:.4f}",
                ]
            )
        # Denoised+Poisson panel (vs original)
        txt_p = "\n".join(
            [
                f"T {tot_p:.0f} ({_pct(tot_p, tot_o):+.3f}%)",
                f"PSNR {psnr_p:.2f}  SSIM {ssim_p:.4f}",
                f"LP A {lp_p_alex:.4f}  V {lp_p_vgg:.4f}",
            ]
        )

        draw.multiline_text((0 * w + pad_x, y_txt), txt_o, fill=255, font=font, spacing=1)
        draw.multiline_text((1 * w + pad_x, y_txt), txt_d, fill=255, font=font, spacing=1)
        if txt_q is not None:
            draw.multiline_text((2 * w + pad_x, y_txt), txt_q, fill=255, font=font, spacing=1)
            draw.multiline_text((3 * w + pad_x, y_txt), txt_p, fill=255, font=font, spacing=1)
        else:
            draw.multiline_text((2 * w + pad_x, y_txt), txt_p, fill=255, font=font, spacing=1)

        frames.append(canvas)
    if not frames:
        return
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


def generate_2x4_gif(
    orig_proj: np.ndarray,
    denoised_u16_proj: np.ndarray,
    denoised_fp32_proj: np.ndarray,
    denoised_poisson_proj: np.ndarray,
    orig_recon: np.ndarray,
    denoised_u16_recon: np.ndarray,
    denoised_fp32_recon: np.ndarray,
    denoised_poisson_recon: np.ndarray,
    output_path: Path | None,
    proj_vmax_mode: str = "global_max",
    recon_vmax_mode: str = "global_max",
    use_log1p: bool = True,
    fps: float = 10.0,
    num_recon_frames: int = 120,
    recon_turns: float = 1.0,
    proj_metric_max_value: float | None = None,
    lpips_nets: tuple[str, str] = ("alex", "vgg"),
    recon_lpips_slice_stride: int = 1,
    gif_fixed_palette: bool = True,
    mp4_output_path: Path | None = None,
    mp4_bottom_pad_px: int = 96,
) -> None:
    """生成 2x4 布局 GIF：第一行投影、第二行重建 MIP。

    Columns:
      1) Original
      2) Denoised (uint16)
      3) Denoised (fp32)
      4) Denoised+Poisson (uint16)
    """
    num_proj_frames = int(orig_proj.shape[0])  # 60

    # Per-patient global max (stable across frames) for visualization
    proj_mode = str(proj_vmax_mode).lower().strip()
    recon_mode = str(recon_vmax_mode).lower().strip()

    if proj_mode in ["global_max", "max"]:
        vmax_proj = float(max(orig_proj.max(), denoised_u16_proj.max(), denoised_fp32_proj.max(), denoised_poisson_proj.max()))
    else:
        # legacy: robust percentile
        def _p999(x: np.ndarray) -> float:
            return float(np.percentile(x[x > 0], 99.9)) if np.any(x > 0) else float(x.max())

        vmax_proj = max(_p999(orig_proj), _p999(denoised_u16_proj), _p999(denoised_fp32_proj), _p999(denoised_poisson_proj))

    if recon_mode in ["global_max", "max"]:
        vmax_recon = float(max(orig_recon.max(), denoised_u16_recon.max(), denoised_fp32_recon.max(), denoised_poisson_recon.max()))
    elif recon_mode == "p9995":
        def _p9995(x: np.ndarray) -> float:
            return float(np.percentile(x[x > 0], 99.95)) if np.any(x > 0) else float(x.max())

        vmax_recon = max(_p9995(orig_recon), _p9995(denoised_u16_recon), _p9995(denoised_fp32_recon), _p9995(denoised_poisson_recon))
    else:
        # legacy: max75
        vmax_recon = float(max(orig_recon.max(), denoised_u16_recon.max(), denoised_fp32_recon.max(), denoised_poisson_recon.max()) * 0.75)

    # ---- metrics (constant across frames) ----
    mv_proj = float(proj_metric_max_value) if proj_metric_max_value is not None else float(np.max(orig_proj))
    if mv_proj <= 0:
        mv_proj = 1.0

    def _psnr_ssim_views_mean(pred: np.ndarray, ref: np.ndarray, data_range: float) -> tuple[float, float]:
        from spect_ct.pipeline.simple_metrics import psnr_range, ssim_range

        psnrs = []
        ssims = []
        dr = float(data_range)
        for i in range(int(ref.shape[0])):
            r = np.clip(ref[i].astype(np.float32, copy=False), 0.0, dr)
            p = np.clip(pred[i].astype(np.float32, copy=False), 0.0, dr)
            psnrs.append(float(psnr_range(p, r, data_range=dr)))
            ssims.append(float(ssim_range(p, r, data_range=dr)))
        return float(np.mean(psnrs)) if psnrs else float("nan"), float(np.mean(ssims)) if ssims else float("nan")

    psnr_pu16, ssim_pu16 = float("nan"), float("nan")
    psnr_pf32, ssim_pf32 = float("nan"), float("nan")
    psnr_pp, ssim_pp = float("nan"), float("nan")
    try:
        psnr_pu16, ssim_pu16 = _psnr_ssim_views_mean(denoised_u16_proj, orig_proj, mv_proj)
        psnr_pf32, ssim_pf32 = _psnr_ssim_views_mean(denoised_fp32_proj, orig_proj, mv_proj)
        psnr_pp, ssim_pp = _psnr_ssim_views_mean(denoised_poisson_proj, orig_proj, mv_proj)
    except Exception:
        pass

    lp_pu16: dict[str, float] = {lpips_nets[0]: float("nan"), lpips_nets[1]: float("nan")}
    lp_pf32: dict[str, float] = {lpips_nets[0]: float("nan"), lpips_nets[1]: float("nan")}
    lp_pp: dict[str, float] = {lpips_nets[0]: float("nan"), lpips_nets[1]: float("nan")}
    try:
        from spect_ct.pipeline.metrics import lpips_views_mean
        lp_pu16 = lpips_views_mean(pred=denoised_u16_proj, ref=orig_proj, max_value=mv_proj, nets=lpips_nets)
        lp_pf32 = lpips_views_mean(pred=denoised_fp32_proj, ref=orig_proj, max_value=mv_proj, nets=lpips_nets)
        lp_pp = lpips_views_mean(pred=denoised_poisson_proj, ref=orig_proj, max_value=mv_proj, nets=lpips_nets)
    except Exception:
        pass

    # Recon-domain metrics computed on a deterministic (non-rotated) MIP for stability.
    mip_o0 = compute_mip(orig_recon, axis=2).astype(np.float32, copy=False)
    mip_u0 = compute_mip(denoised_u16_recon, axis=2).astype(np.float32, copy=False)
    mip_f0 = compute_mip(denoised_fp32_recon, axis=2).astype(np.float32, copy=False)
    mip_p0 = compute_mip(denoised_poisson_recon, axis=2).astype(np.float32, copy=False)
    dr_recon = float(np.max(mip_o0)) if np.isfinite(mip_o0).any() else 1.0
    if dr_recon <= 0:
        dr_recon = 1.0

    def _psnr_ssim_2d(pred2d: np.ndarray, ref2d: np.ndarray, data_range: float) -> tuple[float, float]:
        from spect_ct.pipeline.simple_metrics import psnr_range, ssim_range
        dr = float(data_range)
        r = np.clip(ref2d.astype(np.float32, copy=False), 0.0, dr)
        p = np.clip(pred2d.astype(np.float32, copy=False), 0.0, dr)
        psnr = float(psnr_range(p, r, data_range=dr))
        ssim = float(ssim_range(p, r, data_range=dr))
        return psnr, ssim

    psnr_ru16, ssim_ru16 = float("nan"), float("nan")
    psnr_rf32, ssim_rf32 = float("nan"), float("nan")
    psnr_rp, ssim_rp = float("nan"), float("nan")
    try:
        psnr_ru16, ssim_ru16 = _psnr_ssim_2d(mip_u0, mip_o0, dr_recon)
        psnr_rf32, ssim_rf32 = _psnr_ssim_2d(mip_f0, mip_o0, dr_recon)
        psnr_rp, ssim_rp = _psnr_ssim_2d(mip_p0, mip_o0, dr_recon)
    except Exception:
        pass

    # Recon-domain LPIPS:
    # Prefer slice-wise mean LPIPS over the full 3D volume (more faithful than a single MIP).
    lp_rd: dict[str, float] = {lpips_nets[0]: float("nan"), lpips_nets[1]: float("nan")}
    lp_rp: dict[str, float] = {lpips_nets[0]: float("nan"), lpips_nets[1]: float("nan")}
    try:
        from spect_ct.pipeline.metrics import lpips_recon_slices_mean
        lp_ru16 = lpips_recon_slices_mean(
            pred=denoised_u16_recon,
            ref=orig_recon,
            data_range=float(np.max(orig_recon)) if np.isfinite(orig_recon).any() else 1.0,
            axis=2,
            nets=lpips_nets,
            device=None,
            slice_stride=int(recon_lpips_slice_stride),
        )
        lp_rf32 = lpips_recon_slices_mean(
            pred=denoised_fp32_recon,
            ref=orig_recon,
            data_range=float(np.max(orig_recon)) if np.isfinite(orig_recon).any() else 1.0,
            axis=2,
            nets=lpips_nets,
            device=None,
            slice_stride=int(recon_lpips_slice_stride),
        )
        lp_rp = lpips_recon_slices_mean(
            pred=denoised_poisson_recon,
            ref=orig_recon,
            data_range=float(np.max(orig_recon)) if np.isfinite(orig_recon).any() else 1.0,
            axis=2,
            nets=lpips_nets,
            device=None,
            slice_stride=int(recon_lpips_slice_stride),
        )
    except Exception:
        # Keep NaN if LPIPS is unavailable (e.g., missing deps).
        pass

    frames: list[Image.Image] = []
    recon_angle_start = 90.0
    recon_angle_step = (360.0 * float(recon_turns)) / float(num_recon_frames)

    for i in range(int(num_recon_frames)):
        # Sync projection view index and recon rotation using the same "turns" phase.
        # phase in turns: 0 -> recon_turns
        phase_turns = (float(i) / float(num_recon_frames)) * float(recon_turns)
        proj_idx = int(phase_turns * float(num_proj_frames)) % int(num_proj_frames)
        recon_angle = recon_angle_start + phase_turns * 360.0

        rot_o = rotate_volume_around_x(orig_recon, recon_angle)
        rot_u = rotate_volume_around_x(denoised_u16_recon, recon_angle)
        rot_f = rotate_volume_around_x(denoised_fp32_recon, recon_angle)
        rot_p = rotate_volume_around_x(denoised_poisson_recon, recon_angle)

        mip_o = np.flipud(compute_mip(rot_o, axis=2))
        mip_u = np.flipud(compute_mip(rot_u, axis=2))
        mip_f = np.flipud(compute_mip(rot_f, axis=2))
        mip_p = np.flipud(compute_mip(rot_p, axis=2))

        # row1: projection metrics vs original projection
        tot_po = proj_total_counts_round_clip(orig_proj)
        tot_pu = proj_total_counts_round_clip(denoised_u16_proj)
        tot_pf = proj_total_counts_clip(denoised_fp32_proj)
        tot_pp = proj_total_counts_round_clip(denoised_poisson_proj)
        def _pct(new: float, base: float) -> float:
            if abs(base) < 1e-12:
                return float("nan")
            return (new - base) / base * 100.0

        txt_o = "\n".join([f"T {tot_po:.0f} (+0.000%)", "PSNR -  SSIM -", f"LP A -  V -"])
        txt_u = "\n".join(
            [
                f"T {tot_pu:.0f} ({_pct(tot_pu, tot_po):+.3f}%)",
                f"PSNR {psnr_pu16:.2f}  SSIM {ssim_pu16:.4f}",
                f"LP A {float(lp_pu16.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_pu16.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )
        txt_f = "\n".join(
            [
                f"T {tot_pf:.0f} ({_pct(tot_pf, tot_po):+.3f}%)",
                f"PSNR {psnr_pf32:.2f}  SSIM {ssim_pf32:.4f}",
                f"LP A {float(lp_pf32.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_pf32.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )
        txt_p = "\n".join(
            [
                f"T {tot_pp:.0f} ({_pct(tot_pp, tot_po):+.3f}%)",
                f"PSNR {psnr_pp:.2f}  SSIM {ssim_pp:.4f}",
                f"LP A {float(lp_pp.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_pp.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )

        # row2: recon metrics vs original recon (on deterministic MIP)
        tot_ro = recon_total_counts_clip(orig_recon)
        tot_ru = recon_total_counts_clip(denoised_u16_recon)
        tot_rf = recon_total_counts_clip(denoised_fp32_recon)
        tot_rp = recon_total_counts_clip(denoised_poisson_recon)

        txt_ro = "\n".join([f"T {tot_ro:.3f} (+0.000%)", "PSNR -  SSIM -", "LP A -  V -"])
        txt_ru = "\n".join(
            [
                f"T {tot_ru:.3f} ({_pct(tot_ru, tot_ro):+.3f}%)",
                f"PSNR {psnr_ru16:.2f}  SSIM {ssim_ru16:.4f}",
                f"LP A {float(lp_ru16.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_ru16.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )
        txt_rf = "\n".join(
            [
                f"T {tot_rf:.3f} ({_pct(tot_rf, tot_ro):+.3f}%)",
                f"PSNR {psnr_rf32:.2f}  SSIM {ssim_rf32:.4f}",
                f"LP A {float(lp_rf32.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_rf32.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )
        txt_rp = "\n".join(
            [
                f"T {tot_rp:.3f} ({_pct(tot_rp, tot_ro):+.3f}%)",
                f"PSNR {psnr_rp:.2f}  SSIM {ssim_rp:.4f}",
                f"LP A {float(lp_rp.get(lpips_nets[0], float('nan'))):.4f}  V {float(lp_rp.get(lpips_nets[1], float('nan'))):.4f}",
            ]
        )
        # Render single no-gap frame (same layout as preview PNG)
        from matplotlib.colors import Normalize
        from spect_ct.pipeline.viz import Log1pNormalize

        proj_norm = Log1pNormalize(vmin=0.0, vmax=vmax_proj) if use_log1p else Normalize(vmin=0.0, vmax=vmax_proj)
        recon_norm = Log1pNormalize(vmin=0.0, vmax=vmax_recon) if use_log1p else Normalize(vmin=0.0, vmax=vmax_recon)

        frame = render_2x4_nogap_frame(
            proj_imgs=[orig_proj[proj_idx], denoised_u16_proj[proj_idx], denoised_fp32_proj[proj_idx], denoised_poisson_proj[proj_idx]],
            recon_imgs=[mip_o, mip_u, mip_f, mip_p],
            proj_texts=[txt_o, txt_u, txt_f, txt_p],
            recon_texts=[txt_ro, txt_ru, txt_rf, txt_rp],
            proj_norm=proj_norm,
            recon_norm=recon_norm,
            dpi=160,
        )
        frames.append(frame)

    # Optional MP4 (truecolor, avoids GIF palette limitations)
    if mp4_output_path is not None and frames:
        try:
            import imageio.v2 as imageio
            import numpy as _np

            mp4_output_path = Path(mp4_output_path)
            mp4_output_path.parent.mkdir(parents=True, exist_ok=True)
            fps_f = float(fps) if float(fps) > 0 else 10.0
            pad_px = int(mp4_bottom_pad_px) if mp4_bottom_pad_px is not None else 0
            if pad_px < 0:
                pad_px = 0
            with imageio.get_writer(
                str(mp4_output_path),
                fps=fps_f,
                codec="libx264",
                quality=8,
                macro_block_size=None,  # do not force resize to multiples of 16
            ) as w:
                for fr in frames:
                    fr_rgb = fr.convert("RGB")
                    if pad_px > 0:
                        canvas = Image.new("RGB", (fr_rgb.width, fr_rgb.height + pad_px), (0, 0, 0))
                        canvas.paste(fr_rgb, (0, 0))
                        fr_rgb = canvas
                    w.append_data(_np.asarray(fr_rgb))
        except Exception:
            pass

    # Optional GIF
    if output_path is not None and frames:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: GIF uses a 256-color palette. If each frame is quantized independently,
        # gradients like colorbars can "shimmer"/jump across frames. To stabilize, we
        # optionally quantize all frames with a fixed palette from the first frame.
        if bool(gif_fixed_palette):
            try:
                pal_img = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
                frames_p = [pal_img] + [
                    fr.quantize(palette=pal_img, dither=Image.Dither.NONE).convert("P") for fr in frames[1:]
                ]
                frames = frames_p
            except Exception:
                pass

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000.0 / max(float(fps), 1e-6)),
            loop=0,
            optimize=False,
        )


def generate_2x3_gif(
    orig_proj: np.ndarray,
    denoised_proj: np.ndarray,
    denoised_poisson_proj: np.ndarray,
    orig_recon: np.ndarray,
    denoised_recon: np.ndarray,
    denoised_poisson_recon: np.ndarray,
    output_path: Path,
    proj_vmax_mode: str = "global_max",
    recon_vmax_mode: str = "global_max",
    use_log1p: bool = True,
    fps: float = 10.0,
    num_recon_frames: int = 120,
    proj_metric_max_value: float | None = None,
    lpips_nets: tuple[str, str] = ("alex", "vgg"),
) -> None:
    """Backward-compatible wrapper: treat the provided denoised as both u16 and fp32."""
    generate_2x4_gif(
        orig_proj=orig_proj,
        denoised_u16_proj=denoised_proj,
        denoised_fp32_proj=denoised_proj,
        denoised_poisson_proj=denoised_poisson_proj,
        orig_recon=orig_recon,
        denoised_u16_recon=denoised_recon,
        denoised_fp32_recon=denoised_recon,
        denoised_poisson_recon=denoised_poisson_recon,
        output_path=output_path,
        proj_vmax_mode=proj_vmax_mode,
        recon_vmax_mode=recon_vmax_mode,
        use_log1p=use_log1p,
        fps=fps,
        num_recon_frames=num_recon_frames,
        proj_metric_max_value=proj_metric_max_value,
        lpips_nets=lpips_nets,
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
    views_per_subset: int | None = None,
    skip_existing: bool = False,
    skip_denoise: bool = False,
    skip_recon: bool = False,
    use_log1p: bool = True,
    device: str = "cuda",
    stage: str = "all",  # all | denoise | recon_only | gif_counts
    original_recon_path: Path | None = None,
    overwrite: bool = False,
    mp4: bool = False,
    gif: bool = True,
) -> Patient3Proj3ReconOutputs:
    """单病人：生成 2x3 GIF + counts.txt（以及中间 dat/projection/recon 文件）。"""
    patient = str(patient)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_dir = output_dir / patient / "projections"
    recon_dir = output_dir / patient / "reconstructions"
    proj_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    # New GIF layout:
    #   <output_dir>/<patient>/gifs/...
    gif_dir = output_dir / patient / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{patient}_4proj_4recon_2x4.gif"
    mp4_path = gif_dir / f"{patient}_4proj_4recon_2x4.mp4"
    counts_txt = output_dir / patient / "counts.txt"

    if bool(skip_existing) and counts_txt.exists():
        ok_gif = (not bool(gif)) or gif_path.exists()
        ok_mp4 = (not bool(mp4)) or mp4_path.exists()
        if ok_gif and ok_mp4:
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

    # 1) load original projection (dataset is int16) but store as fp32 only.
    orig_proj = load_projection_i16(input_proj)
    orig_proj_file = proj_dir / "original_projection_f32.dat"
    save_projection_f32(orig_proj_file, orig_proj)

    # 2) denoise
    denoised_proj_f32_file = proj_dir / "denoised_projection_f32.dat"
    if not skip_denoise:
        print(f"[INFO][{patient}] loading net: ckpt={Path(checkpoint)} config={Path(config)}")
        # load_basicsr_net uses strict=True; if mismatch, it will raise (no silent fallback to random init).
        loaded = load_basicsr_net(config, checkpoint, device=device, require_ema=True)
        print(f"[INFO][{patient}] loaded weights key={loaded.used_key} (params_ema is aligned with training EMA)")
        denoised_proj = denoise_views(loaded.net, orig_proj, max_value=float(max_value), device=device)
        save_projection_f32(denoised_proj_f32_file, denoised_proj)
    else:
        if denoised_proj_f32_file.exists():
            print(f"[INFO][{patient}] skip_denoise=true, reuse(fp32): {denoised_proj_f32_file}")
            denoised_proj = load_projection_f32(denoised_proj_f32_file)
        else:
            # Backward-compat: older runs may have int16 file only.
            denoised_proj_file_legacy = proj_dir / "denoised_projection.dat"
            print(f"[INFO][{patient}] skip_denoise=true, reuse(legacy int16): {denoised_proj_file_legacy}")
            denoised_proj = load_projection_i16(denoised_proj_file_legacy)
            save_projection_f32(denoised_proj_f32_file, denoised_proj)

    # 3) poisson sample
    denoised_poisson_proj = poisson_sample(denoised_proj)
    denoised_poisson_proj_file = proj_dir / "denoised_poisson_projection_f32.dat"
    save_projection_f32(denoised_poisson_proj_file, denoised_poisson_proj)

    # Stage gate: allow preparing projections first, then run recon+gif in parallel.
    if stage == "denoise":
        # Also generate a projection-only GIF (similar naming/layout to BasicSR experiments GIFs).
        # IMPORTANT: respect `gif` flag (toolbox --no-gif), and default to 1x3 to avoid confusion
        # from low-dose fp32->int16 rounding drift in the optional quantized column.
        if bool(gif):
            try:
                exp_name = Path(config).stem
                ckpt_path = Path(checkpoint)
                # Parse iter from checkpoint name like net_g_98500.pth.
                # If checkpoint is net_g_latest.pth (no iter in name), infer from sibling net_g_*.pth.
                import re
                m = re.search(r"net_g[_-](\d+)\.pth", str(ckpt_path.name))
                it = int(m.group(1)) if m else None
                if it is None and ckpt_path.parent.is_dir():
                    best = 0
                    for p in ckpt_path.parent.glob("net_g_*.pth"):
                        mm = re.search(r"net_g_(\d+)\.pth", p.name)
                        if mm:
                            best = max(best, int(mm.group(1)))
                    it = best
                if it is None:
                    it = 0
                # We only support EMA model for this pipeline.
                net_tag = "ema"
                # Put projection-only GIF under the same per-patient gif_dir.
                gif_path = gif_dir / f"{exp_name}_iter{it}_{net_tag}.gif"
                generate_1x3_proj_gif(
                    orig_proj=orig_proj,
                    denoised_proj=denoised_proj,
                    denoised_proj_q=None,
                    denoised_poisson_proj=denoised_poisson_proj,
                    output_path=gif_path,
                    use_log1p=bool(use_log1p),
                    fps=10.0,
                    metric_max_value=float(max_value),
                )
            except Exception:
                # Don't fail denoise stage due to visualization issues.
                pass
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
        # fp32-only layout (backward compat: fall back to legacy int16 if needed)
        orig_proj = load_projection_f32(orig_proj_file) if orig_proj_file.name.endswith("_f32.dat") else load_projection_i16(orig_proj_file)
        denoised_fp32_proj = load_projection_f32(denoised_proj_f32_file)
        denoised_u16_proj = denoised_fp32_proj  # placeholder to satisfy downstream signatures; not used for saving.
        denoised_poisson_proj = load_projection_f32(denoised_poisson_proj_file) if denoised_poisson_proj_file.exists() else denoised_poisson_proj
    else:
        # For all other stages, we already have denoised_proj in memory (fp32 if computed).
        denoised_fp32_proj = denoised_proj
        denoised_u16_proj = denoised_fp32_proj  # no int16 path anymore

    # 4) reconstructions
    recon_paths: dict[str, Path] = {}
    # fp32-only projection files for recon (PrjDataType=1).
    recon_jobs: list[tuple[Path, str, int]] = [
        (orig_proj_file, "original", 1),
        (denoised_proj_f32_file, "denoised_fp32", 1),
        (denoised_poisson_proj_file, "denoised_poisson", 1),
    ]
    for proj_file, recon_name, prj_dtype in recon_jobs:
        # Original recon may come from a dataset-side cache (ensure_original_recon).
        # For consistency, we also copy it into this patient's recon_dir using the standard filename.
        if recon_name == "original" and original_recon_path is not None:
            cached = Path(original_recon_path)
            if not cached.exists():
                raise FileNotFoundError(f"[{patient}] original_recon_path not found: {cached}")
            recon_out = recon_dir / f"{patient}_{recon_name}_OSEMReconed_Iter{iterations}.dat"
            if not (bool(skip_recon) and recon_out.exists()):
                try:
                    import shutil
                    shutil.copy2(str(cached), str(recon_out))
                except Exception:
                    pass
            recon_paths[recon_name] = recon_out if recon_out.exists() else cached
            continue

        recon_out = recon_dir / f"{patient}_{recon_name}_OSEMReconed_Iter{iterations}.dat"
        if skip_recon and recon_out.exists():
            recon_paths[recon_name] = recon_out
            continue

        recon_out = run_osem_reconstruction(
            proj_file=proj_file,
            patient_name=patient,
            par_file=par_file,
            orbit_file=orbit_file,
            atten_file=atten_file,
            output_name=recon_name,
            iterations=int(iterations),
            views_per_subset=views_per_subset,
            prj_data_type=int(prj_dtype),
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
    denoised_u16_recon = load_recon_f32(recon_paths["denoised"], shape=(128, 128, 128))
    denoised_fp32_recon = load_recon_f32(recon_paths["denoised_fp32"], shape=(128, 128, 128))
    denoised_poisson_recon = load_recon_f32(recon_paths["denoised_poisson"], shape=(128, 128, 128))

    # 6) counts.txt
    def _safe_pct(new: float, base: float) -> float:
        if abs(base) < 1e-12:
            return float("nan")
        return (new - base) / base * 100.0

    proj_counts = {
        "original": proj_total_counts_round_clip(orig_proj),
        "denoised_u16": proj_total_counts_round_clip(denoised_u16_proj),
        # fp32 projection is lambda-like; use float sum (no rounding) to match validation display.
        "denoised_fp32": proj_total_counts_clip(denoised_fp32_proj),
        "denoised_fp32_round": proj_total_counts_round_clip(denoised_fp32_proj),
        "denoised_poisson": proj_total_counts_round_clip(denoised_poisson_proj),
    }
    recon_counts = {
        "original": recon_total_counts_clip(orig_recon),
        "denoised_u16": recon_total_counts_clip(denoised_u16_recon),
        "denoised_fp32": recon_total_counts_clip(denoised_fp32_recon),
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
        f"denoised_u16: {proj_counts['denoised_u16']:.0f}  ({_safe_pct(proj_counts['denoised_u16'], proj_counts['original']):+.4f}%)",
        f"denoised_fp32: {proj_counts['denoised_fp32']:.3f}  ({_safe_pct(proj_counts['denoised_fp32'], proj_counts['original']):+.4f}%)",
        f"denoised_fp32_round: {proj_counts['denoised_fp32_round']:.0f}  ({_safe_pct(proj_counts['denoised_fp32_round'], proj_counts['original']):+.4f}%)",
        f"denoised_poisson: {proj_counts['denoised_poisson']:.0f}  ({_safe_pct(proj_counts['denoised_poisson'], proj_counts['original']):+.4f}%)",
        "",
        "[reconstruction_domain_total_counts]  (sum over voxels, clip>=0)",
        f"original: {recon_counts['original']:.6f}",
        f"denoised_u16: {recon_counts['denoised_u16']:.6f}  ({_safe_pct(recon_counts['denoised_u16'], recon_counts['original']):+.4f}%)",
        f"denoised_fp32: {recon_counts['denoised_fp32']:.6f}  ({_safe_pct(recon_counts['denoised_fp32'], recon_counts['original']):+.4f}%)",
        f"denoised_poisson: {recon_counts['denoised_poisson']:.6f}  ({_safe_pct(recon_counts['denoised_poisson'], recon_counts['original']):+.4f}%)",
        "",
    ]
    counts_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 7) 2x3 gif
    generate_2x4_gif(
        orig_proj=orig_proj,
        denoised_u16_proj=denoised_u16_proj,
        denoised_fp32_proj=denoised_fp32_proj,
        denoised_poisson_proj=denoised_poisson_proj,
        orig_recon=orig_recon,
        denoised_u16_recon=denoised_u16_recon,
        denoised_fp32_recon=denoised_fp32_recon,
        denoised_poisson_recon=denoised_poisson_recon,
        output_path=(gif_path if bool(gif) else None),
        mp4_output_path=(mp4_path if bool(mp4) else None),
        proj_vmax_mode="p99.9",
        recon_vmax_mode="max75",
        use_log1p=bool(use_log1p),
        proj_metric_max_value=float(max_value),
        lpips_nets=("alex", "vgg"),
        recon_turns=(1.5 if bool(mp4) else 1.0),
    )

    return Patient3Proj3ReconOutputs(
        patient=patient,
        output_dir=output_dir,
        gif_path=gif_path,
        counts_txt=counts_txt,
        proj_dir=proj_dir,
        recon_dir=recon_dir,
    )



