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

            # Merge validation options (top-level val + datasets.val) if not provided by caller
            if val_opt is None:
                _val_opt = self.opt.get('val', {}) or {}
                _datasets_val_opt = self.opt.get('datasets', {}).get('val', {}) or {}
                val_opt = {**_val_opt, **_datasets_val_opt}

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
                # Direct access: SPECTProjectionDataset returns (60, 128, 128) array
                proj_u16 = proj_data.astype(np.float32)
                if proj_u16.shape != (60, 128, 128):
                    self.logger.warning(f"Invalid projection shape: {proj_u16.shape}, expected (60, 128, 128)")
                    return None
            elif lq_path and Path(lq_path).exists():
                # Fallback: load from file
                proj_u16 = np.fromfile(lq_path, dtype=np.uint16)
                if proj_u16.size == 60 * 128 * 128:
                    proj_u16 = proj_u16.reshape(60, 128, 128).astype(np.float32)
                else:
                    self.logger.warning(f"Invalid projection size: {lq_path} (expected 60×128×128, got {proj_u16.size})")
                    return None
            else:
                self.logger.warning(f"Cannot load projection data from: {lq_path}")
                return None

            # Check network input channels to determine if we need noise map
            network_g_opt = self.opt.get('network_g', {})
            network_in_nc = network_g_opt.get('in_nc', 1)
            use_noise_map = (network_in_nc == 2)  # Use noise map if network expects 2 channels
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

            # Get noise map settings from config (if using noise map)
            train_opt = self.opt.get('datasets', {}).get('train', {})
            if use_noise_map:
                use_global_noise_map = train_opt.get('use_global_noise_map', True)
                noise_map_eps = train_opt.get('noise_map_eps', 1e-6)
            else:
                use_global_noise_map = False
                noise_map_eps = 1e-6

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

            # Load or compute BM3D denoising (with caching)
            # Use lq_path for cache key
            exp_name = self.opt.get('name', 'unknown')
            cache_dir = Path('experiments') / exp_name / 'cache' / 'bm3d'
            denoised_bm3d = self._load_or_compute_bm3d(proj_u16, cache_dir, lq_path)

            # Step number extraction removed - no longer used in GIF labels

            # Helper function to compute noise map (same as dataset)
            # Only used when use_noise_map=True (network expects 2 channels)
            def compute_noise_map(img: np.ndarray, vmax: float) -> np.ndarray:
                """Compute noise map for a single image."""
                if not use_noise_map:
                    return None  # Not needed for single-channel networks
                if use_global_noise_map:
                    # Use vmax/100 as global noise map (constant)
                    noise_level = vmax / 100.0 if vmax > 1e-6 else 0.01
                    return np.full_like(img, noise_level, dtype=np.float32)
                else:
                    # Per-pixel noise map: 1/√pixel_value
                    return 1.0 / np.sqrt(np.maximum(img, noise_map_eps))

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
                        if use_noise_map:
                            noise_map = compute_noise_map(proj_i, vmax_i)
                            input_data = np.stack([proj_normalized, noise_map], axis=-1)  # (H, W, 2)
                            xt = torch.from_numpy(input_data.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
                        else:
                            xt = torch.from_numpy(proj_normalized[None, None, ...]).to(device=device, dtype=torch.float32)
                        yt = net(xt)
                        yt = torch.clamp(yt, min=0.0)
                        y = yt.squeeze(0).squeeze(0).detach().cpu().numpy()
                        out[i] = y * vmax_i
                return out

            def _denoise_3d_volume(net: torch.nn.Module) -> np.ndarray:
                """Denoise full (60,128,128) volume with one forward."""
                if use_noise_map:
                    raise NotImplementedError("3D net + noise_map (in_nc==2) is not supported in GIF inference.")
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
            denoised_g = _denoise_3d_volume(self.net_g) if is_3d_net_g else _denoise_2d_per_view(self.net_g)

            # 推理 ema（如果存在）
            if hasattr(self, 'net_g_ema'):
                is_3d_net_ema = _is_3d_net(self.net_g_ema)
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

            # ===== Optional: LPIPS (alex) between poisson_sample and original, averaged over views =====
            # NOTE: LPIPS is trained on natural RGB images; interpret with caution for SPECT projections.
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
                    lpips_net = str(val_opt.get('lpips_net', 'alex')).lower()
                    view_stride = int(val_opt.get('lpips_view_stride', 1))
                    if view_stride < 1:
                        view_stride = 1

                    vals = []
                    for i in range(0, int(proj_u16.shape[0]), view_stride):
                        a = np.clip(proj_u16[i], 0.0, lpips_max_value) / lpips_max_value
                        b = np.clip(poisson_sampled[i], 0.0, lpips_max_value) / lpips_max_value
                        # calculate_lpips expects [0,1] with HWC/CHW; for 2D arrays HWC is fine.
                        vals.append(
                            calculate_lpips(
                                b,
                                a,
                                input_order='HWC',
                                net=lpips_net,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                            )
                        )

                    lpips_mean = float(np.mean(vals)) if len(vals) > 0 else float('nan')
                    sample_name = Path(lq_path).stem if lq_path else Path(output_path).parent.name
                    self.logger.info(
                        f"[val][{sample_name}] LPIPS({lpips_net}) poisson_vs_original: {lpips_mean:.6f} "
                        f"(views={len(vals)}, stride={view_stride}, max_value={lpips_max_value:g})"
                    )
                    if tb_logger is not None and getattr(self, 'opt', {}).get('rank', 0) == 0:
                        tb_logger.add_scalar('metrics/lpips_poisson_vs_original', lpips_mean, current_iter)

            # Compute residuals
            proj_f32 = proj_u16.astype(np.float32)
            residual_ema = proj_f32 - denoised_ema

            # Anscombe transform for residual
            def anscombe_forward(x):
                x = np.clip(x, 0.0, None)
                return (2.0 * np.sqrt(x + 3.0 / 8.0)).astype(np.float32)

            residual_ema_anscombe = anscombe_forward(proj_f32) - anscombe_forward(denoised_ema)

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

            overlay_enabled = bool((val_opt or {}).get('gif_show_total_counts', True))
            overlay_lines = {
                'Original': f"Σ={_fmt_counts(total_counts_orig)}\nΔ={_delta_pct(total_counts_orig, total_counts_orig):+.2f}%",
                'BM3D': f"Σ={_fmt_counts(total_counts_bm3d)}\nΔ={_delta_pct(total_counts_bm3d, total_counts_orig):+.2f}%",
                'Denoised (g)': f"Σ={_fmt_counts(total_counts_denoised_g)}\nΔ={_delta_pct(total_counts_denoised_g, total_counts_orig):+.2f}%",
                'Denoised (ema)': f"Σ={_fmt_counts(total_counts_denoised_ema)}\nΔ={_delta_pct(total_counts_denoised_ema, total_counts_orig):+.2f}%",
                'Poisson Sample': f"Σ={_fmt_counts(total_counts_poisson)}\nΔ={_delta_pct(total_counts_poisson, total_counts_orig):+.2f}%",
            }

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

            # Generate frames
            frames = []

            for i in range(proj_u16.shape[0]):
                # 1. Original
                u8_orig = normalize_to_u8(proj_f32[i], 0.0, vmax_orig, gamma=0.8, log1p=True)
                img_orig = Image.fromarray(u8_orig, mode="L").convert("RGB")
                img_orig = draw_label(img_orig, "Original")
                if overlay_enabled:
                    img_orig = draw_label_bottom(img_orig, overlay_lines['Original'])

                # 2. BM3D Denoised
                u8_bm3d = normalize_to_u8(denoised_bm3d[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_bm3d = Image.fromarray(u8_bm3d, mode="L").convert("RGB")
                img_bm3d = draw_label(img_bm3d, "BM3D")
                if overlay_enabled:
                    img_bm3d = draw_label_bottom(img_bm3d, overlay_lines['BM3D'])

                # 3. Denoised (g)
                u8_g = normalize_to_u8(denoised_g[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_g = Image.fromarray(u8_g, mode="L").convert("RGB")
                img_g = draw_label(img_g, "Denoised (g)")
                if overlay_enabled:
                    img_g = draw_label_bottom(img_g, overlay_lines['Denoised (g)'])

                # 4. Denoised (ema)
                u8_ema = normalize_to_u8(denoised_ema[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_ema = Image.fromarray(u8_ema, mode="L").convert("RGB")
                img_ema = draw_label(img_ema, "Denoised (ema)")
                if overlay_enabled:
                    img_ema = draw_label_bottom(img_ema, overlay_lines['Denoised (ema)'])

                # 5. Poisson Sample (from denoised_ema)
                u8_poisson = normalize_to_u8(poisson_sampled[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_poisson = Image.fromarray(u8_poisson, mode="L").convert("RGB")
                img_poisson = draw_label(img_poisson, "Poisson Sample")
                if overlay_enabled:
                    img_poisson = draw_label_bottom(img_poisson, overlay_lines['Poisson Sample'])

                # 6. Residual
                img_residual = apply_colormap(residual_ema[i], vmin_residual, vmax_residual, 'RdBu_r')
                img_residual = draw_label(img_residual, "Residual (red=+, blue=-)")

                # 7. Residual Anscombe
                img_residual_anscombe = apply_colormap(residual_ema_anscombe[i], vmin_residual_anscombe, vmax_residual_anscombe, 'RdBu_r')
                img_residual_anscombe = draw_label(img_residual_anscombe, "Residual Anscombe")

                # Combine into canvas (7 columns)
                canvas = Image.new('RGB', (img_orig.width * 7, img_orig.height))
                canvas.paste(img_orig, (0, 0))
                canvas.paste(img_bm3d, (img_orig.width, 0))
                canvas.paste(img_g, (img_orig.width * 2, 0))
                canvas.paste(img_ema, (img_orig.width * 3, 0))
                canvas.paste(img_poisson, (img_orig.width * 4, 0))
                canvas.paste(img_residual, (img_orig.width * 5, 0))
                canvas.paste(img_residual_anscombe, (img_orig.width * 6, 0))

                # No additional text overlay - only column labels are shown
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

                # weighted fit var ≈ a*mean + b
                x = mean_lam[valid]
                y = var_r[valid]
                w = n[valid]
                if x.size >= 2:
                    A = np.vstack([x, np.ones_like(x)]).T
                    W = np.diag(w / np.maximum(w.max(), 1.0))
                    coef = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
                    a, b = float(coef[0]), float(coef[1])
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
                    ax.plot(xx, a * xx + b, color='tab:red', linewidth=1.5, label=f'fit: var={a:.3f}*mean+{b:.3f}')
                ax.set_xlabel("Mean(denoised) in count domain")
                ax.set_ylabel("Var(residual = y - denoised)")
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
                    f.write(f"weighted fit: var ≈ {a:.6f} * mean + {b:.6f}\n")
                    f.write(f"median(var/mean): {med_ratio:.6f}\n")
                    f.write(f"saved: {plot_path}\n")
                    f.write(f"saved: {csv_path}\n")

                self.logger.info(
                    f"[val] Poisson calibration saved: {out_dir} | "
                    f"fit var≈{a:.3f}*mean+{b:.3f} | median(var/mean)={med_ratio:.3f} | "
                    f"files={int(poisson_calib_ctx.get('num_files_used', 0))}"
                )
                if tb_logger is not None:
                    tb_logger.add_scalar("metrics/poisson_calib_slope", a, current_iter)
                    tb_logger.add_scalar("metrics/poisson_calib_intercept", b, current_iter)
                    tb_logger.add_scalar("metrics/poisson_calib_median_var_over_mean", med_ratio, current_iter)
            except Exception as e:
                self.logger.warning(f"Failed to save poisson calibration during validation: {e}", exc_info=True)

