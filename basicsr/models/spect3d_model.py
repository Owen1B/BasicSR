"""
3D SPECT Denoising Model with GIF generation during validation.

This model extends SRModel to support 3D data (e.g., projection sequences)
and automatically generates GIFs during validation for visual assessment.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SPECT3DModel(SRModel):
    """
    3D SPECT Denoising Model.

    Extends SRModel with:
    - Automatic GIF generation during validation for 3D data (projection sequences)
    - Separate handling for 2D vs 3D data validation

    Usage:
        Set model_type: SPECT3DModel in your config to enable GIF generation.
        In val section, add:
          generate_gif: true  # Enable GIF generation for 3D data
          gif_output_dir: experiments/{name}/visualization/gifs  # Output directory
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.logger = get_root_logger()

    def _is_3d_data(self, lq_path: str) -> bool:
        """
        Detect if the data is 3D (e.g., projection sequence).

        Heuristics:
        - Filename contains "ProjectionImage" or "Proj4Filter" (raw projection file)
        - Dataset type is SPECTProjectionDataset (designed for 3D data)
        """
        path_lower = lq_path.lower()
        # Check filename patterns (raw projection files)
        if 'projectionimage' in path_lower:
            return True
        if 'proj4filter' in path_lower:
            return True
        # Check dataset type from both val and datasets.val
        val_opt = self.opt.get('val', {})
        datasets_val_opt = self.opt.get('datasets', {}).get('val', {})
        if val_opt.get('type') == 'SPECTProjectionDataset':
            return True
        if datasets_val_opt.get('type') == 'SPECTProjectionDataset':
            return True
        # Allow forcing GIF path for paired 2ch validation.
        try:
            val_opt = self.opt.get('val', {}) or {}
            datasets_val_opt = self.opt.get('datasets', {}).get('val', {}) or {}
            infer_mode = str((datasets_val_opt.get('infer_mode') or val_opt.get('infer_mode') or '')).strip().lower()
            if infer_mode == 'paired2ch':
                return True
        except Exception:
            pass
        return False

    def _load_or_compute_bm3d(
        self,
        proj_u16: np.ndarray,
        cache_dir: Path,
        lq_path: str,
    ) -> np.ndarray:
        """
        Load BM3D denoised result from cache, or compute and save it.

        Args:
            proj_u16: Original projection data (60, 128, 128) float32
            cache_dir: Directory to cache BM3D results
            lq_path: Original file path (for cache key)

        Returns:
            BM3D denoised projection (60, 128, 128) float32
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename from input path
        # Use full relative path to avoid conflicts between different patients
        # e.g., "BianChengMing/20s-1/ProjectionImage1" or "FanCuiLing/20s-1/ProjectionImage1"
        lq_path_obj = Path(lq_path)

        # Try to extract patient name and subdirectory from path
        # Pattern: datasets/bone_20230327/{patient}/{subdir}/ProjectionImage1.dat
        parts = lq_path_obj.parts
        if 'bone_20230327' in parts:
            idx = parts.index('bone_20230327')
            if idx + 2 < len(parts):
                # Extract patient name and subdirectory (e.g., "BianChengMing/20s-1")
                patient_subdir = '/'.join(parts[idx + 1:idx + 3])
                # Replace / with _ for filename safety
                cache_key = f"{patient_subdir.replace('/', '_')}_{lq_path_obj.stem}"  # e.g., "BianChengMing_20s-1_ProjectionImage1"
            else:
                cache_key = lq_path_obj.stem
        else:
            # Fallback: use full path hash if pattern doesn't match
            import hashlib
            path_str = str(lq_path)
            cache_key = hashlib.md5(path_str.encode()).hexdigest()[:16]

        cache_file = cache_dir / f"{cache_key}_bm3d.npy"

        # Try to load from cache
        if cache_file.exists():
            # Load from cache (silent)
            return np.load(cache_file).astype(np.float32)

        # Compute BM3D denoising
        # Computing BM3D denoising (silent, may take a while)
        try:
            import bm3d
            from basicsr.utils.anscombe import anscombe_forward, anscombe_inverse_unbiased
        except ImportError:
            self.logger.warning("BM3D not available, skipping BM3D denoising")
            return proj_u16.copy()  # Return original if BM3D not available

        denoised_bm3d = np.zeros_like(proj_u16, dtype=np.float32)

        for i in range(proj_u16.shape[0]):
            # Anscombe transform
            proj_anscombe = anscombe_forward(proj_u16[i])

            # BM3D denoising (sigma_psd=1.0 for variance-stabilized domain)
            denoised_anscombe = bm3d.bm3d(
                proj_anscombe,
                sigma_psd=1.0,
                stage_arg=bm3d.BM3DStages.ALL_STAGES
            )

            # Inverse Anscombe transform
            denoised_bm3d[i] = anscombe_inverse_unbiased(denoised_anscombe)

        # Save to cache
        np.save(cache_file, denoised_bm3d)
        # Saved BM3D result to cache (silent)

        return denoised_bm3d

    def _generate_validation_gif(
        self,
        val_data: Dict,
        output_path: str,
        current_iter: int,
        net_tag: str = 'ema',
        tb_logger=None,
        val_opt: Optional[Dict] = None,
        poisson_calib_ctx: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Generate a 6-view GIF for 3D projection data validation.

        Views: Original | BM3D | Denoised(g) | Denoised(ema) | Poisson Sample | Residual | Residual Anscombe

        Args:
            val_data: Validation data dict containing 'lq' tensor and 'lq_path'
            output_path: Output GIF path
            net_tag: Network tag ('g' or 'ema')

        Returns:
            Output GIF path if successful, None otherwise
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import math
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.cm as cm
            import torch.nn.functional as F

            # Merge validation options (top-level val + datasets.val) if not provided by caller
            if val_opt is None:
                _val_opt = self.opt.get('val', {}) or {}
                _datasets_val_opt = self.opt.get('datasets', {}).get('val', {}) or {}
                val_opt = {**_val_opt, **_datasets_val_opt}

            infer_mode = str(val_opt.get('infer_mode', '')).strip().lower()

            # NOTE: For GIF inference, ALWAYS feed the network with the same normalization as training:
            # use a fixed max_value (e.g. 150.0). This keeps the input distribution consistent and avoids
            # scale-induced artifacts because the network is not strictly scale-invariant.
            #
            # Separately, for GIF DISPLAY (mapping to uint8), we can use a per-patient global max to avoid
            # "too flat/too gray" GIFs while staying stable across views.

            # Get projection data from validation data
            # For SPECTProjectionDataset, lq_path points directly to ProjectionImage*.dat
            lq_path = val_data.get('lq_path', [''])[0] if isinstance(val_data.get('lq_path'), list) else val_data.get('lq_path', '')

            # For SPECTProjectionDataset, lq/proj_sequence is already the full projection sequence
            # Check multiple possible keys (DataLoader may convert numpy to tensor)
            proj_data = None
            for key in ['proj_sequence', 'lq']:
                if key in val_data:
                    data = val_data[key]
                    # Handle both numpy array and torch tensor (DataLoader converts to tensor)
                    if isinstance(data, torch.Tensor):
                        # DataLoader adds batch dimension: [B, 60, H, W] -> [60, H, W]
                        if data.dim() == 4:  # [B, 60, H, W]
                            proj_data = data[0].cpu().numpy()  # Take first sample, remove batch dim
                        elif data.dim() == 3:  # [60, H, W] (no batch dim)
                            proj_data = data.cpu().numpy()
                        break
                    elif isinstance(data, np.ndarray):
                        proj_data = data
                        break

            if proj_data is not None:
                # Direct access:
                # - SPECTProjectionDataset returns (60, H, W)
                # - SPECTDatPairedDataset (paired2ch) returns (2, H, W)
                proj_u16 = proj_data.astype(np.float32)
                if infer_mode == 'paired2ch':
                    if not (proj_u16.ndim == 3 and proj_u16.shape[0] == 2):
                        self.logger.warning(f"paired2ch expects proj_u16 shape (2,H,W), got {proj_u16.shape}")
                        return None
                else:
                    if not (proj_u16.ndim == 3 and proj_u16.shape[0] == 60):
                        self.logger.warning(f"Invalid projection shape: {proj_u16.shape}, expected (60,H,W)")
                        return None
            elif lq_path and Path(lq_path).exists():
                # Fallback: load from file
                if infer_mode == 'paired2ch':
                    # Paired dataset files are float32 (2,H,W)
                    ds_train_opt0 = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
                    h = int((val_opt or {}).get('height', ds_train_opt0.get('height', 128)))
                    w = int((val_opt or {}).get('width', ds_train_opt0.get('width', 128)))
                    if h < 1 or w < 1:
                        self.logger.warning(f"Invalid height/width for paired2ch file loading: height={h}, width={w}")
                        return None
                    arr = np.fromfile(lq_path, dtype=np.float32)
                    if arr.size != 2 * h * w:
                        self.logger.warning(
                            f"Invalid paired2ch size: {lq_path} (expected 2×{h}×{w}, got {arr.size})"
                        )
                        return None
                    proj_u16 = arr.reshape(2, h, w).astype(np.float32)
                else:
                    ds_train_opt0 = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
                    h = int((val_opt or {}).get('height', ds_train_opt0.get('height', 128)))
                    w = int((val_opt or {}).get('width', ds_train_opt0.get('width', 128)))
                    if h < 1 or w < 1:
                        self.logger.warning(f"Invalid height/width for projection file loading: height={h}, width={w}")
                        return None
                    proj_u16 = np.fromfile(lq_path, dtype=np.uint16)
                    if proj_u16.size == 60 * h * w:
                        proj_u16 = proj_u16.reshape(60, h, w).astype(np.float32)
                    else:
                        self.logger.warning(
                            f"Invalid projection size: {lq_path} (expected 60×{h}×{w}, got {proj_u16.size})"
                        )
                        return None
            else:
                self.logger.warning(f"Cannot load projection data from: {lq_path}")
                return None

            # Auto-select infer mode if not explicitly set.
            # - If input is (2,H,W): treat it as paired A/P -> paired2ch
            # - If input is (60,H,W) AND network is 2ch A/P: use pair60_stitch to run 2ch net on 60-view data
            # - Otherwise: leave infer_mode as-is (default single-channel 60-view path)
            network_g_opt0 = self.opt.get('network_g', {}) or {}
            net_in0 = int(network_g_opt0.get('in_nc', 1))
            net_out0 = int(network_g_opt0.get('out_nc', 1))
            if infer_mode in ['', 'auto']:
                if proj_u16.ndim == 3 and proj_u16.shape[0] == 2:
                    infer_mode = 'paired2ch'
                elif proj_u16.ndim == 3 and proj_u16.shape[0] == 60 and net_in0 == 2 and net_out0 == 2:
                    infer_mode = 'pair60_stitch'
                else:
                    infer_mode = ''

            # Determine channel semantics.
            # IMPORTANT: We DO NOT support "noise-map as a second channel" anymore.
            # For this project, in_nc==2/out_nc==2 means paired A/P (anterior/posterior) views.
            network_g_opt = self.opt.get('network_g', {})
            network_in_nc = int(network_g_opt.get('in_nc', 1))
            network_out_nc = int(network_g_opt.get('out_nc', 1))

            if infer_mode in ['paired2ch', 'pair60_stitch']:
                if not (network_in_nc == 2 and network_out_nc == 2):
                    self.logger.warning(
                        f"infer_mode={infer_mode} requires network_g.in_nc==2 and out_nc==2, "
                        f"got in_nc={network_in_nc}, out_nc={network_out_nc}"
                    )
                    return None
            else:
                if not (network_in_nc == 1 and network_out_nc == 1):
                    self.logger.warning(
                        f"infer_mode={infer_mode or 'default'} expects a 1ch denoiser (in_nc==1,out_nc==1). "
                        f"If you are using a 2ch A/P model, set val.infer_mode to paired2ch or pair60_stitch. "
                        f"Got in_nc={network_in_nc}, out_nc={network_out_nc}"
                    )
                    return None

            # Detect if current network is a 3D conv model (expects volume input [1,C,60,128,128])
            # Heuristic: any Conv3d exists in net_g.
            def _is_3d_net(net: torch.nn.Module) -> bool:
                try:
                    for m in net.modules():
                        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
                            return True
                except Exception:
                    pass
                return False

            is_3d_net_g = _is_3d_net(self.net_g)

            # No noise-map related options are supported.

            # --- Inference normalization (fixed max_value, training-style) ---
            ds_train_opt = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
            max_value_train = float((val_opt or {}).get('max_value', ds_train_opt.get('max_value', 150.0)))
            if max_value_train < 1e-6:
                max_value_train = 1.0
            vmax_arr = np.full((int(proj_u16.shape[0]),), max_value_train, dtype=np.float32)

            # --- Display scaling for GIF (per-patient global max from ORIGINAL projection) ---
            # This is ONLY used for mapping frames to uint8 for visualization (normalize_to_u8),
            # not for network input.
            vmax_display = float(np.max(proj_u16))
            if vmax_display < 1e-6:
                vmax_display = 1.0

            # ------------------------------------------------------------------
            # infer_mode=paired2ch: (2,H,W) paired A/P input -> 1-frame GIF
            # ------------------------------------------------------------------
            if infer_mode == 'paired2ch':
                if not (proj_u16.ndim == 3 and proj_u16.shape[0] == 2):
                    self.logger.warning(f"paired2ch expects shape (2,H,W), got {proj_u16.shape}")
                    return None

                # flip posterior (LR) to align with anterior for inference/display
                posterior_flip = bool(val_opt.get('posterior_flip', True))
                pair = proj_u16.astype(np.float32, copy=False)
                if posterior_flip:
                    pair = pair.copy()
                    pair[1] = pair[1, :, ::-1]

                device = next(self.net_g.parameters()).device

                def _infer_2ch(net: torch.nn.Module) -> np.ndarray:
                    net.eval()
                    x = np.clip(pair, 0.0, None) / float(max_value_train)
                    xt = torch.from_numpy(x[None, ...]).to(device=device, dtype=torch.float32)  # (1,2,H,W)
                    with torch.no_grad():
                        yt = net(xt)
                        yt = torch.clamp(yt, min=0.0)
                    y = yt.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (2,H,W)
                    return y * float(max_value_train)

                den_g_2 = _infer_2ch(self.net_g)
                den_ema_2 = den_g_2
                if hasattr(self, 'net_g_ema'):
                    try:
                        den_ema_2 = _infer_2ch(self.net_g_ema)
                    except Exception:
                        den_ema_2 = den_g_2

                # Display mapping (use log1p for readability)
                use_log1p = bool(val_opt.get('log1p', True))
                vmax = float(vmax_display)
                if vmax < 1e-6:
                    vmax = 1.0

                def _to_u8(img: np.ndarray) -> np.ndarray:
                    x = np.clip(img.astype(np.float32, copy=False), 0.0, vmax)
                    if use_log1p:
                        x = np.log1p(x) / np.log1p(vmax)
                    else:
                        x = x / vmax
                    return (x * 255.0).round().clip(0, 255).astype(np.uint8)

                a_in = _to_u8(pair[0])
                p_in = _to_u8(pair[1])
                a_out = _to_u8(den_ema_2[0])
                p_out = _to_u8(den_ema_2[1])

                # 1-row 4-column composite
                H, W = a_in.shape
                canvas = Image.new("RGB", (W * 4, H), color=(0, 0, 0))
                for j, u8 in enumerate([a_in, p_in, a_out, p_out]):
                    im = Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")
                    canvas.paste(im, (j * W, 0))

                # minimal labels
                try:
                    draw = ImageDraw.Draw(canvas)
                    font = ImageFont.load_default()
                    labels = ["A_in", "P_in", "A_out(ema)", "P_out(ema)"]
                    for j, lab in enumerate(labels):
                        draw.text((j * W + 4, 4), lab, fill=(255, 255, 255), font=font)
                except Exception:
                    pass

                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(str(out_path), save_all=True, append_images=[], duration=200, loop=0, optimize=False)
                return str(out_path)

            # Load or compute BM3D denoising (with caching)
            # Use lq_path for cache key
            exp_name = self.opt.get('name', 'unknown')
            cache_dir = Path('experiments') / exp_name / 'cache' / 'bm3d'
            denoised_bm3d = self._load_or_compute_bm3d(proj_u16, cache_dir, lq_path)

            # Step number extraction removed - no longer used in GIF labels

            # Denoise all views (2D per-view inference) OR denoise full volume (3D inference)
            device = next(self.net_g.parameters()).device

            def _denoise_2d_per_view(net: torch.nn.Module) -> np.ndarray:
                net.eval()
                out = np.zeros_like(proj_u16, dtype=np.float32)
                with torch.no_grad():
                    for i in range(proj_u16.shape[0]):
                        proj_i = proj_u16[i].astype(np.float32, copy=False)
                        vmax_i = float(vmax_arr[i])
                        proj_normalized = proj_i / vmax_i
                        xt = torch.from_numpy(proj_normalized[None, None, ...]).to(device=device, dtype=torch.float32)
                        yt = net(xt)
                        yt = torch.clamp(yt, min=0.0)
                        y = yt.squeeze(0).squeeze(0).detach().cpu().numpy()
                        out[i] = y * vmax_i
                return out

            def _denoise_pair60_stitch(net: torch.nn.Module) -> np.ndarray:
                """Denoise a 60-view volume using a 2ch net by pairing (i, i+30), flip-aligning posterior, then stitching."""
                if not (proj_u16.ndim == 3 and proj_u16.shape[0] == 60):
                    raise ValueError(f"pair60_stitch expects proj_u16 shape (60,H,W), got {proj_u16.shape}")
                net.eval()
                mv = float(max_value_train)
                if mv < 1e-6:
                    mv = 1.0
                H, W = int(proj_u16.shape[1]), int(proj_u16.shape[2])
                # Pair: (0..29) anterior, (30..59) posterior
                a = proj_u16[:30].astype(np.float32, copy=False)  # (30,H,W)
                p = proj_u16[30:].astype(np.float32, copy=False)  # (30,H,W)
                # Flip posterior left-right to align with anterior
                posterior_flip = bool(val_opt.get('posterior_flip', True))
                if posterior_flip:
                    p = p[:, :, ::-1]
                x = np.stack([a / mv, p / mv], axis=1).astype(np.float32, copy=False)  # (30,2,H,W)
                xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    yt = net(xt)  # (30,2,H,W)
                    yt = torch.clamp(yt, min=0.0)
                y = yt.detach().cpu().numpy().astype(np.float32, copy=False)
                y = y * mv
                out = np.zeros((60, H, W), dtype=np.float32)
                out_a = y[:, 0]  # (30,H,W)
                out_p = y[:, 1]
                if posterior_flip:
                    out_p = out_p[:, :, ::-1]  # unflip to original posterior orientation
                out[:30] = out_a
                out[30:] = out_p
                return out

            def _denoise_3d_volume(net: torch.nn.Module) -> np.ndarray:
                """Denoise full (V,H,W) volume with one forward (3D net expects input [1,1,V,H,W])."""
                net.eval()
                # Normalize with the selected gif_norm_mode (vmax_arr is (60,))
                x_norm = (proj_u16 / vmax_arr[:, None, None]).astype(np.float32, copy=False)  # (60,H,W)
                xt = torch.from_numpy(x_norm[None, None, ...]).to(device=device, dtype=torch.float32)  # (1,1,60,H,W)
                with torch.no_grad():
                    yt = net(xt)
                    yt = torch.clamp(yt, min=0.0)
                y_norm = yt.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (60,H,W)
                out = y_norm * vmax_arr[:, None, None]
                return out.astype(np.float32, copy=False)

            # g / ema inference
            if infer_mode == 'pair60_stitch':
                denoised_g = _denoise_pair60_stitch(self.net_g)
            else:
                denoised_g = _denoise_3d_volume(self.net_g) if is_3d_net_g else _denoise_2d_per_view(self.net_g)

            # 推理 ema（如果存在）
            if hasattr(self, 'net_g_ema'):
                is_3d_net_ema = _is_3d_net(self.net_g_ema)
                if infer_mode == 'pair60_stitch':
                    denoised_ema = _denoise_pair60_stitch(self.net_g_ema)
                else:
                    denoised_ema = _denoise_3d_volume(self.net_g_ema) if is_3d_net_ema else _denoise_2d_per_view(self.net_g_ema)
            else:
                # 如果没有 EMA，使用 g 的结果
                denoised_ema = denoised_g.copy()

            # Generate Poisson sample from denoised_ema (for 5th column)
            def poisson_sample(img_count):
                """Generate Poisson sample from count image.

                Args:
                    img_count: Count domain image, np.float32

                Returns:
                    Poisson sampled image (noisier version)
                """
                # Clip negative values
                img_count = np.clip(img_count, 0, None)
                # Poisson sampling: y ~ Poisson(λ=img_count)
                img_noisy = np.random.poisson(lam=img_count).astype(np.float32)
                return img_noisy

            poisson_sampled = np.zeros_like(denoised_ema, dtype=np.float32)
            for i in range(denoised_ema.shape[0]):
                poisson_sampled[i] = poisson_sample(denoised_ema[i])

            # ===== Optional: Poisson calibration accumulation during validation =====
            # We accumulate stats here (cheap, since we already have proj_u16 + denoised_ema).
            # Final plot/csv/summary is saved once per validation in nondist_validation().
            if isinstance(poisson_calib_ctx, dict) and poisson_calib_ctx.get('enable', False):
                try:
                    bin_edges = poisson_calib_ctx['bin_edges']
                    num_bins = int(poisson_calib_ctx['num_bins'])
                    n = poisson_calib_ctx['n']
                    sum_lam = poisson_calib_ctx['sum_lambda']
                    sum_r = poisson_calib_ctx['sum_r']
                    sum_r2 = poisson_calib_ctx['sum_r2']

                    y_all = np.clip(proj_u16.astype(np.float32, copy=False), 0.0, None)
                    lam_all = np.clip(denoised_ema.astype(np.float32, copy=False), 0.0, None)

                    for vi in range(int(y_all.shape[0])):
                        y = y_all[vi].reshape(-1).astype(np.float64, copy=False)
                        lam = lam_all[vi].reshape(-1).astype(np.float64, copy=False)
                        r = y - lam

                        idx = np.digitize(lam, bin_edges, right=False) - 1
                        idx = np.clip(idx, 0, num_bins - 1).astype(np.int64, copy=False)

                        n += np.bincount(idx, minlength=num_bins).astype(np.float64, copy=False)
                        sum_lam += np.bincount(idx, weights=lam, minlength=num_bins).astype(np.float64, copy=False)
                        sum_r += np.bincount(idx, weights=r, minlength=num_bins).astype(np.float64, copy=False)
                        sum_r2 += np.bincount(idx, weights=r * r, minlength=num_bins).astype(np.float64, copy=False)
                        poisson_calib_ctx['total_pixels'] = int(poisson_calib_ctx.get('total_pixels', 0) + lam.size)

                    poisson_calib_ctx['num_files_used'] = int(poisson_calib_ctx.get('num_files_used', 0) + 1)
                except Exception as e:
                    self.logger.warning(f"Poisson calibration accumulation failed, skipping. Reason: {e}")

            # ===== Optional: LPIPS between poisson_sample and original, averaged over views =====
            # NOTE: LPIPS is trained on natural RGB images; interpret with caution for SPECT projections.
            lpips_poisson_means: dict[str, float] = {}  # net -> mean, for writing onto subplot
            lpips_poisson_repeats: dict[str, int] = {}  # net -> repeats used
            if bool(val_opt.get('compute_lpips_poisson_vs_original', False)):
                try:
                    from basicsr.metrics import calculate_lpips
                except Exception as e:
                    self.logger.warning(f"LPIPS metric import failed, skipping. Reason: {e}")
                else:
                    # Use a fixed linear max_value to map count-domain images into [0,1] for LPIPS.
                    # Default: training dataset max_value (often 150.0)
                    ds_train_opt = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
                    lpips_max_value = float(val_opt.get('lpips_max_value', ds_train_opt.get('max_value', 150.0)))
                    view_stride = int(val_opt.get('lpips_view_stride', 1))
                    if view_stride < 1:
                        view_stride = 1

                    # Repeat Poisson sampling K times, compute LPIPS for each, then average.
                    # Support multiple LPIPS backbones (alex/vgg) with different repeat counts.
                    # To keep it fast, compute LPIPS across all selected views in ONE batch call.
                    nets_opt = val_opt.get('lpips_nets', None)
                    if nets_opt is None:
                        # backward-compatible: lpips_net
                        nets = [str(val_opt.get('lpips_net', 'alex')).lower()]
                    elif isinstance(nets_opt, (list, tuple)):
                        nets = [str(x).lower() for x in nets_opt if str(x).strip()]
                    else:
                        nets = [str(nets_opt).lower()]

                    # Per-net repeats (fallback to legacy lpips_poisson_repeats)
                    legacy_repeats = int(val_opt.get('lpips_poisson_repeats', 10))
                    if legacy_repeats < 1:
                        legacy_repeats = 1
                    repeats_alex = int(val_opt.get('lpips_poisson_repeats_alex', legacy_repeats))
                    repeats_vgg = int(val_opt.get('lpips_poisson_repeats_vgg', legacy_repeats))
                    if repeats_alex < 1:
                        repeats_alex = 1
                    if repeats_vgg < 1:
                        repeats_vgg = 1

                    repeats_by_net = {
                        'alex': repeats_alex,
                        'vgg': repeats_vgg,
                    }
                    # Filter to supported nets we know how to configure repeats for
                    nets = [n for n in nets if n in repeats_by_net]
                    if not nets:
                        nets = ['alex']

                    device_lp = 'cuda' if torch.cuda.is_available() else 'cpu'
                    idx = np.arange(0, int(proj_u16.shape[0]), view_stride, dtype=np.int64)

                    ref_np = (np.clip(proj_u16[idx], 0.0, lpips_max_value) / lpips_max_value).astype(np.float32, copy=False)
                    ref_t = torch.from_numpy(ref_np[:, None, ...])  # (V,1,H,W) in [0,1]

                    # We share the same Poisson draws across nets for the overlap part to reduce work.
                    max_repeats = max(int(repeats_by_net.get(n, 1)) for n in nets)
                    vals_by_net: dict[str, list[float]] = {n: [] for n in nets}

                    for r in range(max_repeats):
                        ps = np.zeros_like(ref_np, dtype=np.float32)
                        for j, vi in enumerate(idx):
                            ps[j] = poisson_sample(denoised_ema[int(vi)])
                        pred_np = (np.clip(ps, 0.0, lpips_max_value) / lpips_max_value).astype(np.float32, copy=False)
                        pred_t = torch.from_numpy(pred_np[:, None, ...])  # (V,1,H,W) in [0,1]

                        for net_name in nets:
                            need = int(repeats_by_net.get(net_name, 1))
                            if r >= need:
                                continue
                            vals_by_net[net_name].append(
                                float(
                                    calculate_lpips(
                                        pred_t,
                                        ref_t,
                                        input_order='CHW',
                                        net=net_name,
                                        device=device_lp,
                                    )
                                )
                            )

                    for net_name in nets:
                        vv = vals_by_net.get(net_name, [])
                        lpips_poisson_means[net_name] = float(np.mean(vv)) if len(vv) > 0 else float('nan')
                        lpips_poisson_repeats[net_name] = int(repeats_by_net.get(net_name, 1))
                    sample_name = Path(lq_path).stem if lq_path else Path(output_path).parent.name
                    for net_name in nets:
                        m = lpips_poisson_means.get(net_name, float('nan'))
                        k = lpips_poisson_repeats.get(net_name, 0)
                        self.logger.info(
                            f"[val][{sample_name}] LPIPS({net_name}) poisson_vs_original: {m:.6f} "
                            f"(repeats={k}, views={len(idx)}, stride={view_stride}, max_value={lpips_max_value:g})"
                        )
                        if tb_logger is not None and getattr(self, 'opt', {}).get('rank', 0) == 0:
                            tb_logger.add_scalar(f"metrics/lpips_poisson_vs_original_{net_name}", m, current_iter)

            # Compute residuals
            proj_f32 = proj_u16.astype(np.float32)
            # Residual definition for visualization:
            #   residual = denoised - original
            # This matches the intuition: positive means we added counts / brightened.
            residual_ema = denoised_ema - proj_f32

            # Anscombe transform for residual
            def anscombe_forward(x):
                x = np.clip(x, 0.0, None)
                return (2.0 * np.sqrt(x + 3.0 / 8.0)).astype(np.float32)

            residual_ema_anscombe = anscombe_forward(denoised_ema) - anscombe_forward(proj_f32)

            # ========== Count Statistics ==========
            # Calculate total counts for all projections (sum over all 60 views)
            total_counts_orig = float(np.sum(proj_f32))
            total_counts_denoised_g = float(np.sum(denoised_g))
            total_counts_denoised_ema = float(np.sum(denoised_ema))
            total_counts_bm3d = float(np.sum(denoised_bm3d))
            total_counts_poisson = float(np.sum(poisson_sampled))

            # Calculate reduction percentages
            reduction_g = (total_counts_orig - total_counts_denoised_g) / max(total_counts_orig, 1.0) * 100.0
            reduction_ema = (total_counts_orig - total_counts_denoised_ema) / max(total_counts_orig, 1.0) * 100.0
            reduction_bm3d = (total_counts_orig - total_counts_bm3d) / max(total_counts_orig, 1.0) * 100.0
            reduction_poisson = (total_counts_orig - total_counts_poisson) / max(total_counts_orig, 1.0) * 100.0

            # Overlay text: total counts over 60 views + delta% vs original.
            # Use delta% consistent with counts.txt: (new - original)/original * 100.
            def _delta_pct(new: float, base: float) -> float:
                if abs(base) < 1e-12:
                    return float('nan')
                return (new - base) / base * 100.0

            def _fmt_counts(x: float) -> str:
                try:
                    return f"{x:,.0f}"
                except Exception:
                    return str(x)

            # Stats text rendering:
            # - per_panel: render one compact line BELOW each sub-panel (recommended)
            # - overlay: draw stats inside each sub-panel (old behavior)
            # - bottom: render ONE compact line in a bottom black bar (whole canvas)
            # - none: no stats on GIF (still logged)
            stats_mode = str((val_opt or {}).get('gif_stats_mode', 'per_panel')).strip().lower()
            if stats_mode not in ['per_panel', 'overlay', 'bottom', 'none']:
                stats_mode = 'per_panel'
            overlay_enabled = bool((val_opt or {}).get('gif_show_total_counts', True)) and (stats_mode == 'overlay')
            overlay_lines = {
                # Use ASCII-only labels to avoid missing-glyph "tofu" squares in PIL default fonts.
                'Original': f"Sum={_fmt_counts(total_counts_orig)}\nd%={_delta_pct(total_counts_orig, total_counts_orig):+.2f}%",
                'BM3D': f"Sum={_fmt_counts(total_counts_bm3d)}\nd%={_delta_pct(total_counts_bm3d, total_counts_orig):+.2f}%",
                'Denoised (g)': f"Sum={_fmt_counts(total_counts_denoised_g)}\nd%={_delta_pct(total_counts_denoised_g, total_counts_orig):+.2f}%",
                'Denoised (ema)': f"Sum={_fmt_counts(total_counts_denoised_ema)}\nd%={_delta_pct(total_counts_denoised_ema, total_counts_orig):+.2f}%",
                'Poisson Sample': f"Sum={_fmt_counts(total_counts_poisson)}\nd%={_delta_pct(total_counts_poisson, total_counts_orig):+.2f}%",
            }
            if lpips_poisson_means:
                # Keep deterministic order alex -> vgg if present
                for net_name in ['alex', 'vgg']:
                    if net_name not in lpips_poisson_means:
                        continue
                    m = float(lpips_poisson_means.get(net_name, float('nan')))
                    if not np.isfinite(m):
                        continue
                    k = int(lpips_poisson_repeats.get(net_name, 0))
                    overlay_lines['Poisson Sample'] += f"\nLPIPS({net_name})={m:.5f} (K={k})"

            # Per-view count statistics (to diagnose if reduction is uniform across views)
            per_view_counts_orig = np.sum(proj_f32, axis=(1, 2))  # (60,)
            per_view_counts_ema = np.sum(denoised_ema, axis=(1, 2))  # (60,)
            per_view_reduction_ema = (per_view_counts_orig - per_view_counts_ema) / np.maximum(per_view_counts_orig, 1.0) * 100.0

            # High-count vs Low-count region analysis
            # Define high-count as > median, low-count as <= median
            median_count = np.median(proj_f32[proj_f32 > 0]) if np.any(proj_f32 > 0) else 0.0
            high_count_mask = proj_f32 > median_count
            low_count_mask = (proj_f32 > 0) & (proj_f32 <= median_count)

            if np.any(high_count_mask):
                high_count_orig = float(np.sum(proj_f32[high_count_mask]))
                high_count_ema = float(np.sum(denoised_ema[high_count_mask]))
                high_reduction_ema = (high_count_orig - high_count_ema) / max(high_count_orig, 1.0) * 100.0
            else:
                high_count_orig = high_count_ema = high_reduction_ema = 0.0

            if np.any(low_count_mask):
                low_count_orig = float(np.sum(proj_f32[low_count_mask]))
                low_count_ema = float(np.sum(denoised_ema[low_count_mask]))
                low_reduction_ema = (low_count_orig - low_count_ema) / max(low_count_orig, 1.0) * 100.0
            else:
                low_count_orig = low_count_ema = low_reduction_ema = 0.0

            # Log detailed count statistics
            self.logger.info(
                f"[Count Stats] {osp.basename(lq_path)} | "
                f"Original: {total_counts_orig:,.0f} | "
                f"Denoised(g): {total_counts_denoised_g:,.0f} ({reduction_g:+.2f}%) | "
                f"Denoised(ema): {total_counts_denoised_ema:,.0f} ({reduction_ema:+.2f}%) | "
                f"BM3D: {total_counts_bm3d:,.0f} ({reduction_bm3d:+.2f}%)"
            )
            self.logger.info(
                f"[Count Stats Detail] {osp.basename(lq_path)} | "
                f"Per-view reduction (ema): min={per_view_reduction_ema.min():.2f}%, "
                f"max={per_view_reduction_ema.max():.2f}%, mean={per_view_reduction_ema.mean():.2f}%, "
                f"std={per_view_reduction_ema.std():.2f}%"
            )
            self.logger.info(
                f"[Count Stats Region] {osp.basename(lq_path)} | "
                f"High-count (>median={median_count:.1f}): {high_count_orig:,.0f} → {high_count_ema:,.0f} ({high_reduction_ema:+.2f}%) | "
                f"Low-count (≤median): {low_count_orig:,.0f} → {low_count_ema:,.0f} ({low_reduction_ema:+.2f}%)"
            )

            # Compute vmax for visualization
            def compute_vmax(x, mode='p99.9'):
                if mode == 'max':
                    return float(np.max(x))
                if mode.startswith('p'):
                    p = float(mode[1:])
                    return float(np.percentile(x, p))
                return float(np.max(x))

            # For GIF display we use per-patient global max from ORIGINAL projection (stable across views),
            # and keep log1p in normalize_to_u8 for dynamic range.
            vmax_orig = float(vmax_display)
            vmax_denoised = float(vmax_display)

            # For residual: use absolute value to compute symmetric range, but don't clip
            # This ensures both positive and negative residuals are visible
            residual_abs = np.abs(residual_ema)
            residual_anscombe_abs = np.abs(residual_ema_anscombe)
            vmax_residual = compute_vmax(residual_abs, 'p99.9')
            vmax_residual_anscombe = compute_vmax(residual_anscombe_abs, 'p99.9')
            # Use symmetric range: [-vmax, +vmax] to show both positive and negative
            vmin_residual = -vmax_residual
            vmin_residual_anscombe = -vmax_residual_anscombe

            # Normalization functions
            def normalize_to_u8(frame, vmin, vmax, gamma=0.8, log1p=True):
                f = frame.astype(np.float32)
                f = np.clip(f, vmin, vmax)
                f = (f - vmin) / max(1e-8, (vmax - vmin))
                if log1p:
                    f = np.log1p(9.0 * f) / math.log1p(9.0)
                if gamma != 1.0:
                    f = np.power(np.clip(f, 0.0, 1.0), gamma)
                return (f * 255.0 + 0.5).astype(np.uint8)

            def apply_colormap(data, vmin, vmax, cmap_name='RdBu_r'):
                data_norm = (data - vmin) / max(1e-8, (vmax - vmin))
                data_norm = np.clip(data_norm, 0.0, 1.0)
                try:
                    cmap = matplotlib.colormaps[cmap_name]
                except (KeyError, AttributeError):
                    cmap = cm.get_cmap(cmap_name)
                colored = cmap(data_norm)
                colored = (colored[:, :, :3] * 255).astype(np.uint8)
                return Image.fromarray(colored, mode="RGB")

            def draw_label(img, text):
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                pad = 4
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                x0, y0 = pad, pad
                draw.rectangle([x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2], fill=(0, 0, 0))
                draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
                return img

            def draw_label_bottom(img, text):
                """Draw label on bottom-left. Supports multiline via '\\n'."""
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                pad = 4
                lines = str(text).split('\n')
                widths = []
                heights = []
                for ln in lines:
                    bb = draw.textbbox((0, 0), ln, font=font)
                    widths.append(bb[2] - bb[0])
                    heights.append(bb[3] - bb[1])
                tw = int(max(widths) if widths else 0)
                th = int(sum(heights) + max(0, (len(lines) - 1) * 2))
                x0 = pad
                y0 = img.height - pad - th
                draw.rectangle([x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2], fill=(0, 0, 0))
                yy = y0
                for ln, hh in zip(lines, heights):
                    draw.text((x0, yy), ln, fill=(255, 255, 255), font=font)
                    yy += int(hh) + 2
                return img

            # Render stats in a bottom black bar BELOW an image (for each panel or whole canvas).
            def _with_bottom_bar(img: Image.Image, text: str, bar_h: int, pad_x: int, pad_y: int) -> Image.Image:
                out = Image.new('RGB', (img.width, img.height + bar_h), color=(0, 0, 0))
                out.paste(img, (0, 0))
                if text:
                    d = ImageDraw.Draw(out)
                    try:
                        font_bar = ImageFont.load_default()
                    except Exception:
                        font_bar = None
                    try:
                        d.multiline_text(
                            (pad_x, img.height + pad_y),
                            text,
                            fill=(255, 255, 255),
                            font=font_bar,
                            spacing=2,
                        )
                    except Exception:
                        pass
                return out

            def _wrap_text_to_width(text: str, max_w: int, draw: "ImageDraw.ImageDraw", font) -> str:
                """Greedy wrap by spaces to fit max_w. Keeps existing newlines as hard breaks."""
                if max_w <= 8:
                    return text
                lines_out: list[str] = []
                for para in str(text).split('\n'):
                    words = [w for w in para.split(' ') if w != '']
                    if not words:
                        lines_out.append('')
                        continue
                    cur = words[0]
                    for w in words[1:]:
                        cand = cur + ' ' + w
                        try:
                            bb = draw.textbbox((0, 0), cand, font=font)
                            tw = int(bb[2] - bb[0])
                        except Exception:
                            tw = len(cand) * 6
                        if tw <= max_w:
                            cur = cand
                        else:
                            lines_out.append(cur)
                            cur = w
                    lines_out.append(cur)
                return '\n'.join(lines_out)

            def _pct_of_orig(x: float) -> float:
                if abs(total_counts_orig) < 1e-12:
                    return float('nan')
                return x / total_counts_orig * 100.0

            show_counts = bool((val_opt or {}).get('gif_show_total_counts', True))

            # Per-panel compact multi-lines (default).
            panel_line = {}
            if show_counts:
                # Show 2 lines for readability (avoid overly long single lines).
                panel_line['Original'] = (
                    f"Sum={_fmt_counts(total_counts_orig)}\n"
                    f"d%={_delta_pct(total_counts_orig, total_counts_orig):+.2f}%"
                )
                panel_line['BM3D'] = (
                    f"Sum={_fmt_counts(total_counts_bm3d)}\n"
                    f"d%={_delta_pct(total_counts_bm3d, total_counts_orig):+.2f}%"
                )
                panel_line['Denoised (g)'] = (
                    f"Sum={_fmt_counts(total_counts_denoised_g)}\n"
                    f"d%={_delta_pct(total_counts_denoised_g, total_counts_orig):+.2f}%"
                )
                panel_line['Denoised (ema)'] = (
                    f"Sum={_fmt_counts(total_counts_denoised_ema)}\n"
                    f"d%={_delta_pct(total_counts_denoised_ema, total_counts_orig):+.2f}%"
                )
                # Poisson gets LPIPS summary if available
                lp_short = []
                if lpips_poisson_means:
                    for net_name in ['alex', 'vgg']:
                        m = float(lpips_poisson_means.get(net_name, float('nan')))
                        if not np.isfinite(m):
                            continue
                        lp_short.append(f"{net_name}={m:.5f}")
                lp_s = ("LPIPS: " + "  ".join(lp_short)) if lp_short else ""
                panel_line['Poisson Sample'] = (
                    f"Sum={_fmt_counts(total_counts_poisson)}\n"
                    f"d%={_delta_pct(total_counts_poisson, total_counts_orig):+.2f}%"
                    + (f"\n{lp_s}" if lp_s else "")
                )

            # Bar sizing (shared for alignment)
            pad_x = int((val_opt or {}).get('gif_stats_pad_x', 6))
            pad_y = int((val_opt or {}).get('gif_stats_pad_y', 4))
            # Default a bit larger; user can override in YAML.
            min_h = int((val_opt or {}).get('gif_stats_min_h', 36))
            bar_h_panel = min_h
            wrapped_panel_line: dict[str, str] = dict(panel_line)
            if stats_mode in ['per_panel', 'bottom'] and show_counts:
                # IMPORTANT: bar height must be computed AFTER wrapping, otherwise the bottom text gets clipped
                # (e.g., LPIPS alex/vgg line may wrap to 2 lines).
                try:
                    font_bar0 = ImageFont.load_default()
                except Exception:
                    font_bar0 = None
                # We will wrap based on actual panel width once we know it (in frame loop, first frame).
                # Until then, keep defaults; we'll finalize wrapped text + bar height lazily.

            # Whole-canvas bottom line (optional)
            stats_line = ""
            if stats_mode == 'bottom' and show_counts:
                parts = [
                    f"Orig Sum={_fmt_counts(total_counts_orig)}",
                    f"BM3D d%={_delta_pct(total_counts_bm3d, total_counts_orig):+.2f}%",
                    f"g d%={_delta_pct(total_counts_denoised_g, total_counts_orig):+.2f}%",
                    f"EMA d%={_delta_pct(total_counts_denoised_ema, total_counts_orig):+.2f}%",
                    f"Pois d%={_delta_pct(total_counts_poisson, total_counts_orig):+.2f}%",
                ]
                if lpips_poisson_means:
                    lp_parts = []
                    for net_name in ['alex', 'vgg']:
                        m = float(lpips_poisson_means.get(net_name, float('nan')))
                        if not np.isfinite(m):
                            continue
                        lp_parts.append(f"{net_name}={m:.5f}")
                    if lp_parts:
                        parts.append("LPIPS(pois~orig): " + " ".join(lp_parts))
                stats_line = " | ".join(parts)

            # Generate frames
            frames = []

            for i in range(proj_u16.shape[0]):
                # 1. Original
                u8_orig = normalize_to_u8(proj_f32[i], 0.0, vmax_orig, gamma=0.8, log1p=True)
                img_orig = Image.fromarray(u8_orig, mode="L").convert("RGB")
                img_orig = draw_label(img_orig, "Original")
                if overlay_enabled:
                    img_orig = draw_label_bottom(img_orig, overlay_lines['Original'])
                elif stats_mode == 'per_panel' and show_counts:
                    # finalize wrapping + bar height once, using actual panel width
                    if i == 0:
                        try:
                            tmp = Image.new('RGB', (img_orig.width, img_orig.height), color=(0, 0, 0))
                            dtmp = ImageDraw.Draw(tmp)
                            f0 = ImageFont.load_default()
                        except Exception:
                            dtmp, f0 = None, None
                        if dtmp is not None:
                            max_w = img_orig.width - 2 * pad_x
                            max_th = 0
                            for k, t0 in panel_line.items():
                                wt = _wrap_text_to_width(t0, max_w, dtmp, f0) if t0 else ""
                                wrapped_panel_line[k] = wt
                                if wt:
                                    try:
                                        bb = dtmp.multiline_textbbox((0, 0), wt, font=f0, spacing=2)
                                        th = int(bb[3] - bb[1])
                                    except Exception:
                                        th = 28
                                    max_th = max(max_th, th)
                            bar_h_panel = max(min_h, max_th + 2 * pad_y)

                    img_orig = _with_bottom_bar(img_orig, wrapped_panel_line.get('Original', ''), bar_h_panel, pad_x, pad_y)

                # 2. BM3D Denoised
                u8_bm3d = normalize_to_u8(denoised_bm3d[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_bm3d = Image.fromarray(u8_bm3d, mode="L").convert("RGB")
                img_bm3d = draw_label(img_bm3d, "BM3D")
                if overlay_enabled:
                    img_bm3d = draw_label_bottom(img_bm3d, overlay_lines['BM3D'])
                elif stats_mode == 'per_panel' and show_counts:
                    img_bm3d = _with_bottom_bar(img_bm3d, wrapped_panel_line.get('BM3D', ''), bar_h_panel, pad_x, pad_y)

                # 3. Denoised (g)
                u8_g = normalize_to_u8(denoised_g[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_g = Image.fromarray(u8_g, mode="L").convert("RGB")
                img_g = draw_label(img_g, "Denoised (g)")
                if overlay_enabled:
                    img_g = draw_label_bottom(img_g, overlay_lines['Denoised (g)'])
                elif stats_mode == 'per_panel' and show_counts:
                    img_g = _with_bottom_bar(img_g, wrapped_panel_line.get('Denoised (g)', ''), bar_h_panel, pad_x, pad_y)

                # 4. Denoised (ema)
                u8_ema = normalize_to_u8(denoised_ema[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_ema = Image.fromarray(u8_ema, mode="L").convert("RGB")
                img_ema = draw_label(img_ema, "Denoised (ema)")
                if overlay_enabled:
                    img_ema = draw_label_bottom(img_ema, overlay_lines['Denoised (ema)'])
                elif stats_mode == 'per_panel' and show_counts:
                    img_ema = _with_bottom_bar(img_ema, wrapped_panel_line.get('Denoised (ema)', ''), bar_h_panel, pad_x, pad_y)

                # 5. Poisson Sample (from denoised_ema)
                u8_poisson = normalize_to_u8(poisson_sampled[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_poisson = Image.fromarray(u8_poisson, mode="L").convert("RGB")
                img_poisson = draw_label(img_poisson, "Poisson Sample")
                if overlay_enabled:
                    img_poisson = draw_label_bottom(img_poisson, overlay_lines['Poisson Sample'])
                elif stats_mode == 'per_panel' and show_counts:
                    img_poisson = _with_bottom_bar(img_poisson, wrapped_panel_line.get('Poisson Sample', ''), bar_h_panel, pad_x, pad_y)

                # 6. Residual
                img_residual = apply_colormap(residual_ema[i], vmin_residual, vmax_residual, 'RdBu_r')
                img_residual = draw_label(img_residual, "Residual (red=+, blue=-)")
                if stats_mode == 'per_panel' and show_counts:
                    img_residual = _with_bottom_bar(img_residual, "", bar_h_panel, pad_x, pad_y)

                # 7. Residual Anscombe
                img_residual_anscombe = apply_colormap(residual_ema_anscombe[i], vmin_residual_anscombe, vmax_residual_anscombe, 'RdBu_r')
                img_residual_anscombe = draw_label(img_residual_anscombe, "Residual Anscombe")
                if stats_mode == 'per_panel' and show_counts:
                    img_residual_anscombe = _with_bottom_bar(img_residual_anscombe, "", bar_h_panel, pad_x, pad_y)

                # Combine into canvas (7 columns)
                canvas = Image.new('RGB', (img_orig.width * 7, img_orig.height))
                canvas.paste(img_orig, (0, 0))
                canvas.paste(img_bm3d, (img_orig.width, 0))
                canvas.paste(img_g, (img_orig.width * 2, 0))
                canvas.paste(img_ema, (img_orig.width * 3, 0))
                canvas.paste(img_poisson, (img_orig.width * 4, 0))
                canvas.paste(img_residual, (img_orig.width * 5, 0))
                canvas.paste(img_residual_anscombe, (img_orig.width * 6, 0))

                if stats_mode == 'bottom' and show_counts:
                    canvas = _with_bottom_bar(canvas, stats_line, bar_h_panel, pad_x, pad_y)
                frames.append(canvas)

            # Save GIF
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000.0 / 8.0),  # 8 fps
                loop=0,
                optimize=False
            )

            return str(output_path)

        except Exception as e:
            self.logger.warning(f"Error generating GIF: {e}", exc_info=True)
            return None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """
        Override validation to generate GIFs for 3D data.

        For 3D data (projection sequences), generates 6-view GIFs.
        For 2D data, uses standard SRModel validation.
        """
        # Try to extract actual iteration number from checkpoint path if current_iter is not a number
        if not isinstance(current_iter, (int, float)):
            # Try to get iteration from pretrain_network_g path
            pretrain_path = self.opt.get('path', {}).get('pretrain_network_g')
            if pretrain_path:
                import re
                match = re.search(r'net_g[_-](\d+)\.pth', pretrain_path)
                if match:
                    current_iter = int(match.group(1))
                else:
                    # Try to find latest checkpoint
                    models_dir = self.opt.get('path', {}).get('models', f"experiments/{self.opt.get('name', 'unknown')}/models")
                    if osp.exists(models_dir):
                        import glob
                        checkpoints = glob.glob(osp.join(models_dir, 'net_g_*.pth'))
                        if checkpoints:
                            # Extract iteration from latest checkpoint
                            latest = max(checkpoints, key=lambda x: int(re.search(r'net_g_(\d+)\.pth', x).group(1)) if re.search(r'net_g_(\d+)\.pth', x) else 0)
                            match = re.search(r'net_g_(\d+)\.pth', latest)
                            if match:
                                current_iter = int(match.group(1))

        # Get validation config from datasets.val (where dataset config is) or top-level val
        val_opt = self.opt.get('val', {})
        datasets_val_opt = self.opt.get('datasets', {}).get('val', {})

        # Merge: datasets.val takes priority for dataset-specific settings
        merged_val_opt = {**val_opt, **datasets_val_opt}

        generate_gif = merged_val_opt.get('generate_gif', False)

        # If GIF generation is disabled, use standard validation
        if not generate_gif:
            return super().nondist_validation(dataloader, current_iter, tb_logger, save_img)

        # Check if this is 3D data validation
        dataset_name = dataloader.dataset.opt['name']
        gif_output_dir = merged_val_opt.get('gif_output_dir', None)
        if gif_output_dir is None:
            exp_name = self.opt.get('name', 'unknown')
            gif_output_dir = f'experiments/{exp_name}/visualization/gifs'
        else:
            # Replace {name} placeholder with actual experiment name
            exp_name = self.opt.get('name', 'unknown')
            gif_output_dir = gif_output_dir.format(name=exp_name)

        # For 3D projection data, skip standard validation (which expects 2D images)
        # and directly generate GIFs

        eval_networks = merged_val_opt.get('eval_networks', None)
        if eval_networks is None:
            eval_networks = ['ema'] if hasattr(self, 'net_g_ema') else ['g']
        eval_networks = [str(x).lower() for x in eval_networks]

        # Only generate GIF for the first network to save time
        net_tag = eval_networks[0]

        # Optional: Poisson calibration stats during validation (aggregate over the samples we process for GIFs)
        poisson_calib_opt = merged_val_opt.get('poisson_calibration_opt', {}) or {}
        poisson_calib_ctx = None
        if bool(poisson_calib_opt.get('enable', False)):
            try:
                bin_width = float(poisson_calib_opt.get('bin_width', 1.0))
                max_bin = float(poisson_calib_opt.get('max_bin', 150.0))
                if bin_width <= 0:
                    raise ValueError('bin_width must be > 0')
                bin_edges = np.arange(0.0, max_bin + bin_width, bin_width, dtype=np.float32)
                num_bins = int(len(bin_edges))
                poisson_calib_ctx = {
                    'enable': True,
                    'bin_width': bin_width,
                    'max_bin': max_bin,
                    'bin_edges': bin_edges,
                    'num_bins': num_bins,
                    'min_count': int(poisson_calib_opt.get('min_count', 20000)),
                    'n': np.zeros(num_bins, dtype=np.float64),
                    'sum_lambda': np.zeros(num_bins, dtype=np.float64),
                    'sum_r': np.zeros(num_bins, dtype=np.float64),
                    'sum_r2': np.zeros(num_bins, dtype=np.float64),
                    'num_files_used': 0,
                    'total_pixels': 0,
                }
            except Exception as e:
                self.logger.warning(f"Invalid poisson_calibration_opt, disabled. Reason: {e}")
                poisson_calib_ctx = None

        num_gifs = 0
        sample_idx = 0  # Global sample index across all batches
        for idx, val_data in enumerate(dataloader):
            # Handle different data formats
            # SPECTProjectionDataset returns numpy arrays, standard datasets return tensors
            if isinstance(val_data.get('lq'), np.ndarray):
                # SPECTProjectionDataset: lq is (60, 128, 128) numpy array
                batch_size = 1
            elif isinstance(val_data.get('lq'), torch.Tensor):
                batch_size = int(val_data['lq'].shape[0]) if val_data['lq'].dim() > 0 else 1
            else:
                batch_size = 1

            for b in range(batch_size):
                # Get sample
                one = {}
                for k, v in val_data.items():
                    if isinstance(v, torch.Tensor):
                        one[k] = v[b:b+1] if v.dim() > 0 else v
                    elif isinstance(v, np.ndarray):
                        # For SPECTProjectionDataset, keep numpy array as-is
                        one[k] = v
                    elif isinstance(v, list):
                        one[k] = [v[b]] if b < len(v) else v[0]
                    else:
                        one[k] = v

                lq_path = one.get('lq_path', [''])[0] if isinstance(one.get('lq_path'), list) else one.get('lq_path', '')

                # Check if this is 3D data
                is_3d = self._is_3d_data(lq_path)
                if not is_3d:
                    continue  # Skip 2D data

                # Generate GIF with experiment name to avoid conflicts
                # Use per-sample subdirectory to avoid overwriting
                # ⭐ FIX: Use sample_idx to ensure unique directory names for each sample
                # Extract base filename and combine with index to avoid overwriting
                base_name = osp.splitext(osp.basename(lq_path))[0] if lq_path else 'sample'
                img_name = f"{base_name}_{sample_idx:03d}"  # e.g., "ProjectionImage1_000", "ProjectionImage1_001"
                exp_name = self.opt.get('name', 'unknown')

                # Create subdirectory for each sample to avoid overwriting
                sample_dir = osp.join(gif_output_dir, img_name)
                os.makedirs(sample_dir, exist_ok=True)

                gif_name = f"{exp_name}_iter{current_iter}_{net_tag}.gif"
                gif_path = osp.join(sample_dir, gif_name)

                # Increment sample index for next sample
                sample_idx += 1

                # Check if GIF already exists (skip if exists to avoid re-generation)
                if osp.exists(gif_path):
                    self.logger.info(f'GIF already exists, skipping: {osp.basename(gif_path)}')
                    num_gifs += 1
                    if num_gifs >= merged_val_opt.get('max_gifs_per_val', 3):
                        break
                    continue

                result = self._generate_validation_gif(
                    val_data=one,
                    output_path=gif_path,
                    current_iter=current_iter,
                    net_tag=net_tag,
                    tb_logger=tb_logger,
                    val_opt=merged_val_opt,
                    poisson_calib_ctx=poisson_calib_ctx,
                )

                if result:
                    num_gifs += 1
                    self.logger.info(f'Generated GIF {num_gifs}/{merged_val_opt.get("max_gifs_per_val", 3)}: {osp.basename(gif_path)}')
                else:
                    self.logger.warning(f'Failed to generate GIF: {osp.basename(gif_path)}')

                # Limit number of GIFs per validation to avoid excessive I/O
                if num_gifs >= merged_val_opt.get('max_gifs_per_val', 3):
                    break

            if num_gifs >= merged_val_opt.get('max_gifs_per_val', 3):
                break

        if num_gifs > 0:
            self.logger.info(f"✅ Generated {num_gifs} GIF(s) at iter {current_iter}")

        # Save poisson calibration plot/csv once per validation
        if isinstance(poisson_calib_ctx, dict) and poisson_calib_ctx.get('enable', False) and self.opt.get('rank', 0) == 0:
            try:
                n = poisson_calib_ctx['n']
                sum_lam = poisson_calib_ctx['sum_lambda']
                sum_r = poisson_calib_ctx['sum_r']
                sum_r2 = poisson_calib_ctx['sum_r2']
                min_count = int(poisson_calib_ctx.get('min_count', 20000))
                valid = n >= float(min_count)

                mean_lam = np.zeros_like(n, dtype=np.float64)
                mean_r = np.zeros_like(n, dtype=np.float64)
                var_r = np.zeros_like(n, dtype=np.float64)
                mean_lam[valid] = sum_lam[valid] / n[valid]
                mean_r[valid] = sum_r[valid] / n[valid]
                var_r[valid] = sum_r2[valid] / n[valid] - mean_r[valid] ** 2

                ratio = np.zeros_like(mean_lam, dtype=np.float64)
                ratio[valid] = var_r[valid] / np.maximum(mean_lam[valid], 1e-8)

                # weighted fit (force through origin): var ≈ a * mean
                x = mean_lam[valid]
                y = var_r[valid]
                w = n[valid]
                if x.size >= 1:
                    ww = w / np.maximum(float(w.max()), 1.0)
                    denom = float(np.sum(ww * x * x))
                    if denom > 0:
                        a = float(np.sum(ww * x * y) / denom)
                    else:
                        a = float('nan')
                    b = 0.0
                else:
                    a, b = float('nan'), float('nan')

                exp_name = self.opt.get('name', 'unknown')
                out_dir = str(poisson_calib_opt.get('out_dir', f"experiments/{exp_name}/analysis/poisson_calibration_iter{int(current_iter):d}"))
                out_dir = out_dir.format(name=exp_name, iter=int(current_iter))
                os.makedirs(out_dir, exist_ok=True)

                bin_edges = poisson_calib_ctx['bin_edges']
                num_bins = int(poisson_calib_ctx['num_bins'])
                bin_width = float(poisson_calib_ctx['bin_width'])
                max_bin = float(poisson_calib_ctx['max_bin'])

                centers = np.zeros(num_bins, dtype=np.float64)
                if num_bins >= 2:
                    centers[:-1] = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                centers[-1] = max_bin + bin_width / 2.0

                csv_path = osp.join(out_dir, "poisson_calibration_bins.csv")
                header = "bin_center,n,mean_lambda,var_residual,ratio_var_over_mean,valid"
                rows = np.stack([centers, n, mean_lam, var_r, ratio, valid.astype(np.float64)], axis=1)
                np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=(7.2, 5.4), dpi=150)
                ax = fig.add_subplot(111)
                ax.scatter(mean_lam[valid], var_r[valid], s=12, c='tab:blue', alpha=0.75, label='bins (valid)')
                xx = np.linspace(0, max_bin, 200, dtype=np.float64)
                ax.plot(xx, xx, 'k--', linewidth=1.0, label='Poisson ideal: var=mean')
                if np.isfinite(a) and np.isfinite(b):
                    ax.plot(xx, a * xx, color='tab:red', linewidth=1.5, label=f'fit (0-intercept): var={a:.3f}*mean')
                ax.set_xlabel("Mean(denoised) in count domain")
                ax.set_ylabel("Var(residual = denoised - y)")
                ax.set_title(f"Poisson calibration @ iter {int(current_iter)} (files={int(poisson_calib_ctx.get('num_files_used', 0))})")
                ax.grid(True, alpha=0.25)
                ax.legend(loc='upper left', fontsize=8)
                fig.tight_layout()

                plot_path = osp.join(out_dir, "poisson_calibration_plot.png")
                fig.savefig(plot_path)
                plt.close(fig)

                summary_path = osp.join(out_dir, "poisson_calibration_summary.txt")
                med_ratio = float(np.median(ratio[valid])) if np.any(valid) else float('nan')
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write("Poisson calibration during validation\n")
                    f.write(f"iter: {int(current_iter)}\n")
                    f.write(f"files_used: {int(poisson_calib_ctx.get('num_files_used', 0))}\n")
                    f.write(f"total_pixels: {int(poisson_calib_ctx.get('total_pixels', 0))}\n")
                    f.write(f"valid_bins (n>={min_count}): {int(np.sum(valid))}/{int(len(valid))}\n")
                    f.write(f"weighted fit (0-intercept): var ≈ {a:.6f} * mean\n")
                    f.write(f"median(var/mean): {med_ratio:.6f}\n")
                    f.write(f"saved: {plot_path}\n")
                    f.write(f"saved: {csv_path}\n")

                self.logger.info(
                    f"[val] Poisson calibration saved: {out_dir} | "
                    f"fit(0-intercept) var≈{a:.3f}*mean | median(var/mean)={med_ratio:.3f} | "
                    f"files={int(poisson_calib_ctx.get('num_files_used', 0))}"
                )
                if tb_logger is not None:
                    tb_logger.add_scalar("metrics/poisson_calib_slope", a, current_iter)
                    tb_logger.add_scalar("metrics/poisson_calib_intercept", b, current_iter)
                    tb_logger.add_scalar("metrics/poisson_calib_median_var_over_mean", med_ratio, current_iter)
            except Exception as e:
                self.logger.warning(f"Failed to save poisson calibration during validation: {e}", exc_info=True)

