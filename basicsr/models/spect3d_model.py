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
        - Filename contains "ProjectionImage" (raw projection file)
        - Dataset type is SPECTProjectionDataset (designed for 3D data)
        """
        path_lower = lq_path.lower()
        # Check filename patterns (raw projection files)
        if 'projectionimage' in path_lower:
            return True
        # Check dataset type
        val_opt = self.opt.get('val', {})
        if val_opt.get('type') == 'SPECTProjectionDataset':
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
    ) -> Optional[str]:
        """
        Generate a 6-view GIF for 3D projection data validation.

        Views: Original | BM3D | Denoised(g) | Denoised(ema) | Residual | Residual Anscombe

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

            # Get noise map settings from config (if using noise map)
            train_opt = self.opt.get('datasets', {}).get('train', {})
            if use_noise_map:
                use_global_noise_map = train_opt.get('use_global_noise_map', True)
                noise_map_eps = train_opt.get('noise_map_eps', 1e-6)
            else:
                use_global_noise_map = False
                noise_map_eps = 1e-6

            # Load or compute BM3D denoising (with caching)
            # Use lq_path for cache key
            exp_name = self.opt.get('name', 'unknown')
            cache_dir = Path('experiments') / exp_name / 'cache' / 'bm3d'
            denoised_bm3d = self._load_or_compute_bm3d(proj_u16, cache_dir, lq_path)

            # Extract step number from current_iter
            # Handle both int and string (from test.py, current_iter might be experiment name)
            if isinstance(current_iter, (int, float)):
                step_str = f"{int(current_iter):,}"
            else:
                # If current_iter is a string (e.g., experiment name), try to extract number or use as-is
                try:
                    # Try to extract number from string (e.g., "1000" or "iter1000")
                    import re
                    match = re.search(r'\d+', str(current_iter))
                    if match:
                        step_str = f"{int(match.group()):,}"
                    else:
                        step_str = str(current_iter)
                except:
                    step_str = str(current_iter)

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

            # Denoise all views (single-channel or 2-channel input based on network config)
            device = next(self.net_g.parameters()).device

            # 分别推理 g 和 ema
            self.net_g.eval()
            denoised_g = np.zeros_like(proj_u16, dtype=np.float32)

            with torch.no_grad():
                for i in range(proj_u16.shape[0]):
                    # Per-angle normalization
                    proj_i = proj_u16[i].astype(np.float32, copy=False)
                    vmax_i = float(np.max(proj_i))
                    if vmax_i < 1e-6:
                        vmax_i = 1.0

                    # Normalize projection
                    proj_normalized = proj_i / vmax_i

                    # Prepare input based on network configuration
                    if use_noise_map:
                        # Compute noise map
                        noise_map = compute_noise_map(proj_i, vmax_i)
                        # Concatenate: [normalized_projection, noise_map] -> (H, W, 2)
                        input_data = np.stack([proj_normalized, noise_map], axis=-1)  # (H, W, 2)
                        # Convert to tensor: HWC -> CHW -> (1, 2, H, W)
                        xt = torch.from_numpy(input_data.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
                    else:
                        # Single-channel input: (H, W) -> (1, 1, H, W)
                        xt = torch.from_numpy(proj_normalized[None, None, ...]).to(device=device, dtype=torch.float32)

                    # Forward pass
                    yt = self.net_g(xt)
                    yt = torch.clamp(yt, min=0.0)
                    y = yt.squeeze(0).squeeze(0).detach().cpu().numpy()  # (H, W)
                    denoised_g[i] = y * vmax_i  # Denormalize

            # 推理 ema（如果存在）
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                denoised_ema = np.zeros_like(proj_u16, dtype=np.float32)
                with torch.no_grad():
                    for i in range(proj_u16.shape[0]):
                        # Per-angle normalization
                        proj_i = proj_u16[i].astype(np.float32, copy=False)
                        vmax_i = float(np.max(proj_i))
                        if vmax_i < 1e-6:
                            vmax_i = 1.0

                        # Normalize projection
                        proj_normalized = proj_i / vmax_i

                        # Prepare input based on network configuration
                        if use_noise_map:
                            # Compute noise map
                            noise_map = compute_noise_map(proj_i, vmax_i)
                            # Concatenate: [normalized_projection, noise_map] -> (H, W, 2)
                            input_data = np.stack([proj_normalized, noise_map], axis=-1)  # (H, W, 2)
                            # Convert to tensor: HWC -> CHW -> (1, 2, H, W)
                            xt = torch.from_numpy(input_data.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
                        else:
                            # Single-channel input: (H, W) -> (1, 1, H, W)
                            xt = torch.from_numpy(proj_normalized[None, None, ...]).to(device=device, dtype=torch.float32)

                        # Forward pass
                        yt = self.net_g_ema(xt)
                        yt = torch.clamp(yt, min=0.0)
                        y = yt.squeeze(0).squeeze(0).detach().cpu().numpy()  # (H, W)
                        denoised_ema[i] = y * vmax_i  # Denormalize
            else:
                # 如果没有 EMA，使用 g 的结果
                denoised_ema = denoised_g.copy()

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

            # Calculate reduction percentages
            reduction_g = (total_counts_orig - total_counts_denoised_g) / max(total_counts_orig, 1.0) * 100.0
            reduction_ema = (total_counts_orig - total_counts_denoised_ema) / max(total_counts_orig, 1.0) * 100.0
            reduction_bm3d = (total_counts_orig - total_counts_bm3d) / max(total_counts_orig, 1.0) * 100.0

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

            vmax_orig = compute_vmax(proj_f32, 'p99.9')
            vmax_denoised = compute_vmax(np.concatenate([denoised_g, denoised_ema, denoised_bm3d]), 'p99.9')

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

            # Generate frames
            frames = []
            angle_start = -180.0
            angle_step = 6.0

            for i in range(proj_u16.shape[0]):
                # 1. Original
                u8_orig = normalize_to_u8(proj_f32[i], 0.0, vmax_orig, gamma=0.8, log1p=True)
                img_orig = Image.fromarray(u8_orig, mode="L").convert("RGB")
                img_orig = draw_label(img_orig, "Original")

                # 2. BM3D Denoised
                u8_bm3d = normalize_to_u8(denoised_bm3d[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_bm3d = Image.fromarray(u8_bm3d, mode="L").convert("RGB")
                img_bm3d = draw_label(img_bm3d, "BM3D")

                # 3. Denoised (g)
                u8_g = normalize_to_u8(denoised_g[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_g = Image.fromarray(u8_g, mode="L").convert("RGB")
                img_g = draw_label(img_g, "Denoised (g)")

                # 4. Denoised (ema)
                u8_ema = normalize_to_u8(denoised_ema[i], 0.0, vmax_denoised, gamma=0.8, log1p=True)
                img_ema = Image.fromarray(u8_ema, mode="L").convert("RGB")
                img_ema = draw_label(img_ema, "Denoised (ema)")

                # 5. Residual
                img_residual = apply_colormap(residual_ema[i], vmin_residual, vmax_residual, 'RdBu_r')
                img_residual = draw_label(img_residual, "Residual (red=+, blue=-)")

                # 6. Residual Anscombe
                img_residual_anscombe = apply_colormap(residual_ema_anscombe[i], vmin_residual_anscombe, vmax_residual_anscombe, 'RdBu_r')
                img_residual_anscombe = draw_label(img_residual_anscombe, "Residual Anscombe")

                # Combine into canvas (6 columns)
                canvas = Image.new('RGB', (img_orig.width * 6, img_orig.height))
                canvas.paste(img_orig, (0, 0))
                canvas.paste(img_bm3d, (img_orig.width, 0))
                canvas.paste(img_g, (img_orig.width * 2, 0))
                canvas.paste(img_ema, (img_orig.width * 3, 0))
                canvas.paste(img_residual, (img_orig.width * 4, 0))
                canvas.paste(img_residual_anscombe, (img_orig.width * 5, 0))

                # Add step and angle info
                angle = angle_start + i * angle_step
                # Add count info only on first frame to avoid clutter
                if i == 0:
                    count_info = (
                        f"Counts: Orig={total_counts_orig:,.0f} | "
                        f"EMA={total_counts_denoised_ema:,.0f} ({reduction_ema:+.1f}%) | "
                        f"BM3D={total_counts_bm3d:,.0f} ({reduction_bm3d:+.1f}%)"
                    )
                    # Add region analysis on first frame
                    region_info = (
                        f"High: {high_reduction_ema:+.1f}% | Low: {low_reduction_ema:+.1f}%"
                    )
                    info_text = f"Step: {step_str}  |  View {i:02d}/59  |  Angle {angle:.0f}°  |  {count_info}  |  {region_info}"
                else:
                    info_text = f"Step: {step_str}  |  View {i:02d}/59  |  Angle {angle:.0f}°"
                canvas = draw_label(canvas, info_text)

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

