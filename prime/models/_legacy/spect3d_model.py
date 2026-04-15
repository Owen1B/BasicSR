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

from prime.models._legacy.sr_model_ext import SRModel
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
        # Cache LPIPS networks across validations to avoid re-loading weights every time.
        self._lpips_cache: dict[str, torch.nn.Module] = {}

    @torch.no_grad()
    def _nondist_validation_projection_light(self, dataloader, current_iter, merged_val_opt: dict, eval_networks: list[str]):
        """Lightweight validation/test path for SPECTProjectionEvalDataset.

        This avoids SRModel's 2D validation pipeline (which assumes 4D NCHW tensors),
        and provides a fast way to run 3D inference (optionally with view subsampling).
        """
        ds_train_opt = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
        max_value = float(merged_val_opt.get('max_value', ds_train_opt.get('max_value', 150.0)))
        if max_value < 1e-6:
            max_value = 1.0

        # Optional: subsample views to speed up inference (e.g. 2 -> 30 views from a 60-view input)
        try:
            view_stride = int(merged_val_opt.get('view_stride', 1))
        except Exception:
            view_stride = 1
        if view_stride < 1:
            view_stride = 1

        # Optional PLL metric (self-supervised Poisson NLL): lower is better.
        metrics = merged_val_opt.get('metrics', None) or {}
        want_pll = False
        pll_key = None
        pll_eps = 1.0e-8
        pll_full = True
        for m, opt_ in metrics.items():
            opt_type = str((opt_ or {}).get('type', '')).strip().lower()
            if opt_type in ['pll', 'pllloss', 'poisson_nll', 'poisson_nll_mean', 'poisson_nll_loss']:
                want_pll = True
                pll_key = str(m)
                pll_eps = float((opt_ or {}).get('eps', 1.0e-8))
                pll_full = bool((opt_ or {}).get('full', True))
                break

        pll_sum = {tag: 0.0 for tag in eval_networks}
        n_used = 0

        for val_data in dataloader:
            # Extract projection: (V,H,W) in count domain (uint16 in file; here float32)
            proj_data = None
            for key in ['proj_sequence', 'lq']:
                if key not in val_data:
                    continue
                data = val_data[key]
                if isinstance(data, torch.Tensor):
                    if data.dim() == 4:  # [B,V,H,W]
                        proj_data = data[0].detach().cpu().numpy()
                    elif data.dim() == 3:  # [V,H,W]
                        proj_data = data.detach().cpu().numpy()
                    break
                if isinstance(data, np.ndarray):
                    proj_data = data
                    break
            if proj_data is None:
                continue
            proj_u16 = np.asarray(proj_data, dtype=np.float32)
            if not (proj_u16.ndim == 3 and int(proj_u16.shape[0]) >= 2):
                continue
            v_total = int(proj_u16.shape[0])
            if view_stride > 1:
                proj_u16 = proj_u16[::view_stride]
            v_used = int(proj_u16.shape[0])

            # One-line confirmation in logs: how many views are actually used.
            try:
                lq_path = val_data.get('lq_path', '')
                if isinstance(lq_path, list):
                    lq_path = lq_path[0] if lq_path else ''
                sample_name = str(lq_path)
            except Exception:
                sample_name = ''
            self.logger.info(f"[val][light] sample={sample_name} views_used={v_used}/{v_total} view_stride={view_stride}")

            # Input: (1,1,V,H,W) normalized to [0,1]
            x = (proj_u16 / max_value).astype(np.float32, copy=False)
            xt = torch.from_numpy(x[None, None, ...]).to(device=self.device, dtype=torch.float32)

            for tag in eval_networks:
                net = self.net_g_ema if (tag == 'ema' and hasattr(self, 'net_g_ema')) else self.net_g
                net.eval()
                yt = net(xt)
                yt = torch.clamp(yt, min=0.0)
                den = yt[0, 0] * float(max_value)  # (V,H,W) in count domain

                if want_pll:
                    import torch.nn.functional as F
                    y = torch.from_numpy(np.clip(np.rint(proj_u16), 0.0, None).astype(np.float32, copy=False)).to(self.device)
                    lam = torch.clamp(den, min=float(pll_eps))
                    # Use log_input=True with log(lam) for numerical stability and full Poisson likelihood if requested.
                    val_pll = F.poisson_nll_loss(torch.log(lam), y, log_input=True, full=pll_full, reduction='mean')
                    pll_sum[tag] += float(val_pll.detach().cpu().item())

            n_used += 1

        if want_pll and n_used > 0:
            for tag in eval_networks:
                v = pll_sum[tag] / float(n_used)
                self.logger.info(f"[val][light] {pll_key}_{tag}: {v:.6g} (n={n_used}, view_stride={view_stride})")
        else:
            self.logger.info(f"[val][light] done (n={n_used}, view_stride={view_stride})")

    def _get_lpips_model(self, net: str, device: torch.device) -> Optional[torch.nn.Module]:
        """Get (and cache) LPIPS model. Returns None if lpips dependency missing."""
        name = str(net).lower().strip()
        if not name:
            return None
        if name in self._lpips_cache:
            m = self._lpips_cache[name]
            try:
                m = m.to(device)
            except Exception:
                pass
            return m
        try:
            import lpips  # type: ignore
        except Exception as e:
            self.logger.warning(f"LPIPS dependency not available, skipping. Reason: {e}")
            return None
        m = lpips.LPIPS(net=name).to(device)
        m.eval()
        self._lpips_cache[name] = m
        return m

    @torch.no_grad()
    def _lpips_poisson_vs_original_mean_batch(
        self,
        *,
        orig_count: np.ndarray,
        denoised_lambda_count: np.ndarray,
        lpips_net: str = "alex",
        repeats: int = 100,
        max_value: float = 150.0,
        batch: int = 4,
        view_chunk: int = 10,
        view_stride: int = 1,
        device: torch.device,
    ) -> float:
        """Compute mean LPIPS(lpips_net) between A=original and B=Poisson(denoised_lambda), averaged over views + repeats.

        This is the same quantity you've been estimating with scripts, but accelerated with:
        - Poisson sampling in batches
        - LPIPS forward in view chunks to control VRAM
        """
        mv = float(max(max_value, 1e-6))
        reps = int(repeats)
        if reps <= 0:
            return float("nan")
        bs = max(1, int(batch))
        vc = max(1, int(view_chunk))
        stride = max(1, int(view_stride))

        # (V,H,W) float32
        orig = np.asarray(orig_count, dtype=np.float32)
        lam = np.asarray(denoised_lambda_count, dtype=np.float32)
        if orig.ndim != 3 or lam.ndim != 3:
            raise ValueError(f"expected orig/lam shape (V,H,W), got orig={orig.shape} lam={lam.shape}")
        if orig.shape != lam.shape:
            raise ValueError(f"orig/lam shape mismatch: orig={orig.shape} lam={lam.shape}")

        # Select views
        v = int(orig.shape[0])
        sel = np.arange(0, v, stride, dtype=np.int64)
        orig = orig[sel]
        lam = lam[sel]
        v_sel = int(orig.shape[0])

        # Tensors
        orig_t = torch.from_numpy(np.clip(orig, 0.0, mv) / mv)[None, ...].to(device=device, dtype=torch.float32)  # (1,V,H,W) in [0,1]
        lam_t = torch.from_numpy(np.clip(lam, 0.0, None)).to(device=device, dtype=torch.float32)  # (V,H,W) count domain

        loss_fn = self._get_lpips_model(lpips_net, device)
        if loss_fn is None:
            return float("nan")

        total = 0.0
        done = 0
        while done < reps:
            k = min(bs, reps - done)
            # Poisson samples: (k,V,H,W) in count domain
            rate = lam_t[None, ...].expand(k, -1, -1, -1)
            samp = torch.poisson(rate)
            samp01 = torch.clamp(samp, 0.0, mv) / mv  # (k,V,H,W) in [0,1]

            # Mean LPIPS over views (chunked)
            lp_sum = torch.zeros((k,), device=device, dtype=torch.float32)
            for vs in range(0, v_sel, vc):
                ve = min(v_sel, vs + vc)
                chunk = ve - vs
                xs = samp01[:, vs:ve, :, :].reshape(k * chunk, 1, orig.shape[1], orig.shape[2]).repeat(1, 3, 1, 1)
                ys = orig_t.expand(k, -1, -1, -1)[:, vs:ve, :, :].reshape(k * chunk, 1, orig.shape[1], orig.shape[2]).repeat(1, 3, 1, 1)
                # LPIPS expects [-1,1]
                xs = xs * 2.0 - 1.0
                ys = ys * 2.0 - 1.0
                d = loss_fn(xs, ys).reshape(k, chunk).mean(dim=1)  # (k,)
                lp_sum += d * float(chunk)
            lp_mean = lp_sum / float(v_sel)  # (k,)
            total += float(lp_mean.sum().detach().cpu().item())
            done += k

        return float(total / float(max(reps, 1)))

    def _is_3d_data(self, lq_path: str) -> bool:
        """
        Detect if the data is 3D (e.g., projection sequence).

        Heuristics:
        - Filename contains "ProjectionImage" or "Proj4Filter" (raw projection file)
        - Dataset type is SPECTProjectionEvalDataset (designed for 3D data)
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
        if val_opt.get('type') == 'SPECTProjectionEvalDataset':
            return True
        if datasets_val_opt.get('type') == 'SPECTProjectionEvalDataset':
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
            from prime.runtime.anscombe import anscombe_forward, anscombe_inverse_unbiased
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

    def _denoise_pair60_stitch(
        self,
        proj_u16: np.ndarray,
        net: torch.nn.Module,
        max_value: float,
        device: torch.device,
        pair_bs: int = 4,
        posterior_flip: bool = True,
    ) -> np.ndarray:
        """Denoise a 60-view volume using a 2ch net by pairing (i, i+30), flip-aligning posterior, then stitching.

        Args:
            proj_u16: (60, H, W) count-domain projection
            net: 2ch 2D network (in_nc=2, out_nc=2)
            max_value: normalization factor (e.g. 150.0)
            device: torch device
            pair_bs: batch size for processing pairs (OOM safety)
            posterior_flip: whether to flip posterior LR for alignment

        Returns:
            (60, H, W) denoised projection in count domain
        """
        if not (proj_u16.ndim == 3 and proj_u16.shape[0] == 60):
            raise ValueError(f"pair60_stitch expects proj_u16 shape (60,H,W), got {proj_u16.shape}")

        net.eval()
        mv = float(max(max_value, 1e-6))
        H, W = int(proj_u16.shape[1]), int(proj_u16.shape[2])

        # Pair: (0..29) anterior, (30..59) posterior
        a = proj_u16[:30].astype(np.float32, copy=False)  # (30,H,W)
        p = proj_u16[30:].astype(np.float32, copy=False)  # (30,H,W)

        # Flip posterior left-right to align with anterior
        if posterior_flip:
            p = p[:, :, ::-1]

        x = np.stack([a / mv, p / mv], axis=1).astype(np.float32, copy=False)  # (30,2,H,W)

        # Run in batches to avoid OOM
        pair_bs = max(1, min(int(pair_bs), 30))
        yt_list = []
        with torch.no_grad():
            for s in range(0, 30, pair_bs):
                xt = torch.from_numpy(x[s:s + pair_bs]).to(device=device, dtype=torch.float32)
                try:
                    yt = net(xt)
                except torch.OutOfMemoryError:
                    if device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # Retry with batch=1
                    if pair_bs > 1:
                        pair_bs = 1
                        yt_list = []
                        for s2 in range(0, 30, 1):
                            xt2 = torch.from_numpy(x[s2:s2 + 1]).to(device=device, dtype=torch.float32)
                            yt2 = net(xt2)
                            yt_list.append(torch.clamp(yt2, min=0.0))
                        break
                    raise
                yt_list.append(torch.clamp(yt, min=0.0))

        yt = torch.cat(yt_list, dim=0)  # (30,2,H,W)
        y = yt.detach().cpu().numpy().astype(np.float32, copy=False) * mv

        out = np.zeros((60, H, W), dtype=np.float32)
        out_a = y[:, 0]  # (30,H,W)
        out_p = y[:, 1]

        if posterior_flip:
            out_p = out_p[:, :, ::-1]  # unflip to original posterior orientation

        out[:30] = out_a
        out[30:] = out_p

        return out

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
            # For SPECTProjectionEvalDataset, lq_path points directly to ProjectionImage*.dat
            lq_path = val_data.get('lq_path', [''])[0] if isinstance(val_data.get('lq_path'), list) else val_data.get('lq_path', '')

            # For SPECTProjectionEvalDataset, lq/proj_sequence is already the full projection sequence
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
                # - SPECTProjectionEvalDataset returns (60, H, W)
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
                posterior_flip = bool(val_opt.get('posterior_flip', True))
                denoised_g = self._denoise_pair60_stitch(proj_u16, self.net_g, max_value_train, device, pair_bs=4, posterior_flip=posterior_flip)
            else:
                denoised_g = _denoise_3d_volume(self.net_g) if is_3d_net_g else _denoise_2d_per_view(self.net_g)

            # 推理 ema（如果存在）
            if hasattr(self, 'net_g_ema'):
                is_3d_net_ema = _is_3d_net(self.net_g_ema)
                if infer_mode == 'pair60_stitch':
                    posterior_flip = bool(val_opt.get('posterior_flip', True))
                    denoised_ema = self._denoise_pair60_stitch(proj_u16, self.net_g_ema, max_value_train, device, pair_bs=4, posterior_flip=posterior_flip)
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

    def _generate_validation_mp4_thin_factors(
        self,
        *,
        val_data: dict,
        sample_dir: str,
        exp_name: str,
        current_iter: int,
        net_tag: str,
        tb_logger,
        val_opt: dict,
        poisson_calib_ctx: Optional[dict] = None,
    ) -> Optional[str]:
        """MP4 version of validation visualization (replacing legacy GIF when enabled).

        Layout:
          Rows: 20s (top) + thinned x2/x3/x4/x5 (below), longer time on top.
          Cols: Original | BM3D | Denoised(g) | Denoised(ema) | Poisson(ema) | Res(ans) | EMA diff

        Key design:
          - Gray columns are displayed in 20s-equivalent domain: multiply each row by k and use the SAME vmax (20s max).
          - Poisson(ema) is sampled once per row for the whole 60-view volume (stable across frames).
          - Overlay shows 60-view total counts in 'w' (1e4) and % diff vs theory (C20/k, based on original 20s).
        """
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception as e:
            self.logger.warning(f"[val] imageio(ffmpeg) not available, skip mp4 generation. Reason: {e}")
            return None
        try:
            # PIL is required for rendering text/frames.
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception as e:
            self.logger.warning(f"[val] PIL not available, skip mp4 generation. Reason: {e}")
            return None

        try:
            # ---- resolve proj ----
            proj_data = None
            for key in ['proj_sequence', 'lq']:
                if key in val_data:
                    data = val_data[key]
                    if isinstance(data, torch.Tensor):
                        # expected (1,1,V,H,W) or (1,V,H,W) etc
                        if data.dim() >= 3:
                            proj_data = data[0].detach().cpu().numpy()
                            if proj_data.ndim == 4 and proj_data.shape[0] == 1:
                                proj_data = proj_data[0]
                        break
                    if isinstance(data, np.ndarray):
                        proj_data = data
                        break
            if proj_data is None:
                return None
            proj_u16 = np.asarray(proj_data, dtype=np.float32)
            if not (proj_u16.ndim == 3 and int(proj_u16.shape[0]) >= 2):
                return None
            v_total = int(proj_u16.shape[0])

            # Optional: subsample views for MP4 rendering too (e.g. 2 -> 30 views from a 60-view input).
            try:
                view_stride = int(val_opt.get("view_stride", 1))
            except Exception:
                view_stride = 1
            if view_stride < 1:
                view_stride = 1
            idx_views = np.arange(0, v_total, view_stride, dtype=np.int64)
            if idx_views.size < 1:
                idx_views = np.arange(0, v_total, 1, dtype=np.int64)
            # Keep only selected views for the whole MP4 pipeline.
            if view_stride > 1:
                proj_u16 = proj_u16[idx_views]
            lq_path = val_data.get("lq_path", "")
            if isinstance(lq_path, list):
                lq_path = lq_path[0] if lq_path else ""
            lq_path = str(lq_path)

            # ---- helpers ----
            def _counts_int(x: np.ndarray) -> np.ndarray:
                y = np.rint(x.astype(np.float64, copy=False))
                y = np.clip(y, 0.0, None)
                return y.astype(np.int64, copy=False)

            def _anscombe_forward(x: np.ndarray) -> np.ndarray:
                return 2.0 * np.sqrt(np.maximum(x + 3.0 / 8.0, 0.0))

            def _norm_to_u8(x: np.ndarray, *, vmax: float, use_log1p: bool) -> np.ndarray:
                vmax = float(vmax)
                if vmax < 1e-6:
                    vmax = 1.0
                y = np.clip(x.astype(np.float32, copy=False), 0.0, vmax)
                if bool(use_log1p):
                    y = np.log1p(y) / np.log1p(vmax)
                else:
                    y = y / vmax
                return (y * 255.0).round().clip(0, 255).astype(np.uint8)

            def _signed_to_bwr(x: np.ndarray, *, vmax: float) -> np.ndarray:
                vmax = float(vmax)
                if vmax < 1e-6:
                    vmax = 1.0
                t = np.clip(x.astype(np.float32, copy=False), -vmax, vmax) / vmax
                u = (t + 1.0) * 0.5
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    try:
                        cmap = matplotlib.colormaps.get_cmap("bwr")
                    except Exception:
                        import matplotlib.cm as cm
                        cmap = cm.get_cmap("bwr")
                    return (cmap(u)[..., :3] * 255.0).round().astype(np.uint8)
                except Exception:
                    rgb = np.zeros((*u.shape, 3), dtype=np.float32)
                    lo = u <= 0.5
                    hi = ~lo
                    a = (u[lo] / 0.5)[:, None]
                    rgb[lo] = (1 - a) * np.array([0, 0, 255], dtype=np.float32) + a * np.array([255, 255, 255], dtype=np.float32)
                    b = ((u[hi] - 0.5) / 0.5)[:, None]
                    rgb[hi] = (1 - b) * np.array([255, 255, 255], dtype=np.float32) + b * np.array([255, 0, 0], dtype=np.float32)
                    return rgb.round().clip(0, 255).astype(np.uint8)

            def _fmt_counts_w_pct(actual: float, theory: float) -> str:
                theory = float(theory)
                actual = float(actual)
                w = actual / 1.0e4
                if w >= 100:
                    cstr = f"C={w:.0f}w"
                elif w >= 10:
                    cstr = f"C={w:.1f}w"
                else:
                    cstr = f"C={w:.2f}w"
                if theory <= 0:
                    return cstr
                pct = (actual - theory) / theory * 100.0
                pstr = f"{pct:+.0f}%" if abs(pct) >= 10 else f"{pct:+.1f}%"
                return f"{cstr} ({pstr})"

            def _fmt_w_int(actual: float) -> str:
                """Format total counts in integer W (1e4). Example: 140W."""
                w = float(actual) / 1.0e4
                return f"{int(round(w))}W"

            def _load_font(size: int = 12):
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size=size)
                except Exception:
                    return ImageFont.load_default()

            # ---- options ----
            ds_train_opt = (self.opt.get("datasets", {}) or {}).get("train", {}) or {}
            mv = float((val_opt or {}).get("max_value", ds_train_opt.get("max_value", 150.0)))
            mv = float(max(mv, 1e-6))

            factors = val_opt.get("mp4_thin_factors", [2, 3, 4, 5])
            try:
                factors = [float(x) for x in list(factors)]
            except Exception:
                factors = [2.0, 3.0, 4.0, 5.0]
            factors = [k for k in factors if float(k) > 1.0]
            seed = int(val_opt.get("mp4_thin_seed", 123))
            fps = float(val_opt.get("mp4_fps", 10.0))
            crf = int(val_opt.get("mp4_crf", 18))
            use_log1p = bool(val_opt.get("mp4_log1p", True))

            # ---- prepare rows ----
            y20i = _counts_int(proj_u16)
            c20 = float(np.sum(y20i.astype(np.float64, copy=False)))
            if c20 <= 0:
                c20 = 1.0
            vmax20 = float(np.max(y20i.astype(np.float32, copy=False)))
            if vmax20 < 1e-6:
                vmax20 = 1.0

            # Use FIXED low-dose splits (preferred): load from dataset-level cache if available.
            # Cache file name matches _load_or_compute_bm3d(): md5(lq_path_with_label)[:16]_{thin|bm3d}.npy
            def _md5_16(s: str) -> str:
                import hashlib
                return hashlib.md5(str(s).encode()).hexdigest()[:16]

            base_rows = [("20s", 1.0, y20i.astype(np.float32, copy=False))]
            rng = np.random.default_rng(seed)

            # Patient-level cache dir (datasets/SPECT229/<patient>/bm3d_cache[/seed{seed}])
            ds_cache_dir = None
            try:
                lp0 = str(lq_path).split("__")[0]
                lp_obj = Path(lp0)
                patient_dir = lp_obj.parent
                cand_root = patient_dir / "bm3d_cache"
                if cand_root.exists():
                    # Prefer seed-scoped readable cache if present
                    seed_dir = cand_root / f"seed{int(seed)}"
                    ds_cache_dir = seed_dir if seed_dir.exists() else cand_root
            except Exception:
                ds_cache_dir = None

            for k in factors:
                lab = f"x{k:g}"
                xk = None
                if ds_cache_dir is not None:
                    try:
                        # Prefer readable name first, then fallback to md5 name.
                        thin_path = ds_cache_dir / f"{lab}_thin.npy"
                        if thin_path.exists():
                            xk = np.load(thin_path).astype(np.float32, copy=False)
                        else:
                            key = _md5_16(f"{lp0}__{lab}")
                            thin_path2 = ds_cache_dir / f"{key}_thin.npy"
                            if thin_path2.exists():
                                xk = np.load(thin_path2).astype(np.float32, copy=False)
                    except Exception:
                        xk = None
                if xk is None:
                    require_cached = bool(val_opt.get("mp4_require_cached_thins", True))
                    if require_cached:
                        raise FileNotFoundError(
                            f"[val][mp4] Missing cached low-dose split: {thin_path}. "
                            f"Run: python scripts/precompute_validation_bm3d.py --seed {seed} --overwrite"
                        )
                    # Optional fallback: generate on-the-fly (deterministic given seed)
                    xk_i = rng.binomial(y20i, 1.0 / float(k)).astype(np.int64, copy=False)
                    xk = xk_i.astype(np.float32, copy=False)
                # If cached thins were loaded for full 60 views, subsample to match selected views.
                if view_stride > 1 and xk is not None and xk.ndim == 3 and int(xk.shape[0]) == v_total:
                    xk = xk[idx_views]
                base_rows.append((lab, float(k), xk))

            def _sec(label: str, k: float) -> float:
                return 20.0 if label == "20s" else 20.0 / max(float(k), 1e-12)

            base_rows.sort(key=lambda t: _sec(t[0], t[1]), reverse=True)

            # infer
            net_ema = self.net_g_ema if hasattr(self, "net_g_ema") else self.net_g
            net_g = self.net_g
            device = next(net_ema.parameters()).device if hasattr(net_ema, "parameters") else next(net_g.parameters()).device

            def _is_3d_net(net0: torch.nn.Module) -> bool:
                try:
                    for m in net0.modules():
                        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
                            return True
                except Exception:
                    pass
                return False

            def _infer_with(netx: torch.nn.Module, proj: np.ndarray) -> np.ndarray:
                infer_mode = str(val_opt.get("infer_mode", "")).strip().lower()
                net_in0 = int((self.opt.get("network_g", {}) or {}).get("in_nc", 1))
                net_out0 = int((self.opt.get("network_g", {}) or {}).get("out_nc", 1))
                if infer_mode in ["", "auto"]:
                    if proj.ndim == 3 and proj.shape[0] == 60 and net_in0 == 2 and net_out0 == 2:
                        infer_mode = "pair60_stitch"
                    else:
                        infer_mode = ""
                if infer_mode == "pair60_stitch":
                    posterior_flip = bool(val_opt.get("posterior_flip", True))
                    return self._denoise_pair60_stitch(proj, netx, float(mv), device=device, pair_bs=int(val_opt.get("pair60_stitch_batch", 4)), posterior_flip=posterior_flip)
                if _is_3d_net(netx):
                    x = (proj / mv).astype(np.float32, copy=False)
                    xt = torch.from_numpy(x[None, None, ...]).to(device=device, dtype=torch.float32)
                    netx.eval()
                    with torch.no_grad():
                        yt = netx(xt)
                        yt = torch.clamp(yt, min=0.0)
                    return yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
                out = np.zeros_like(proj, dtype=np.float32)
                netx.eval()
                with torch.no_grad():
                    for i in range(int(proj.shape[0])):
                        xt = torch.from_numpy((proj[i] / mv)[None, None, ...]).to(device=device, dtype=torch.float32)
                        yt = netx(xt)
                        yt = torch.clamp(yt, min=0.0)
                        out[i] = yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
                return out

            # Optional tqdm inside MP4 generation (can be slow due to BM3D/LPIPS/60-frame rendering)
            mp4_tqdm = bool(val_opt.get("mp4_tqdm", ("debug" in str(exp_name).lower())))

            rows = []
            base_rows_iter = base_rows
            if mp4_tqdm:
                base_rows_iter = tqdm(
                    list(base_rows),
                    desc=f"mp4-rows[{Path(sample_dir).name}]",
                    leave=False,
                    dynamic_ncols=True,
                )
            for label, k, orig in base_rows_iter:
                # Denoise with both G and EMA
                den_g = np.clip(_infer_with(net_g, orig).astype(np.float32, copy=False), 0.0, None)
                den_ema = np.clip(_infer_with(net_ema, orig).astype(np.float32, copy=False), 0.0, None)
                # BM3D on original (cached)
                #
                # Priority:
                # 1) If lq_path is inside datasets/SPECT229/<patient>/..., prefer dataset-level cache:
                #       datasets/SPECT229/<patient>/bm3d_cache/<label>_<md5>_bm3d.npy
                #    This allows all experiments to reuse the same BM3D results without YAML changes.
                # 2) Else, fall back to experiment-local cache:
                #       experiments/<name>/cache/bm3d/<md5>_bm3d.npy
                #
                # NOTE: We include dose label in the key so different rows do not collide.
                bm3d_cache_dir = None
                try:
                    lp0 = str(lq_path).split("__")[0]
                    lp_obj = Path(lp0)
                    # If the projection file exists, use its parent as patient_dir.
                    # If not, still try to infer patient_dir from the path parts.
                    if lp_obj.exists():
                        patient_dir = lp_obj.parent
                    else:
                        patient_dir = lp_obj.parent
                    if "datasets" in patient_dir.parts and "SPECT229" in patient_dir.parts:
                        ds_cache_dir = patient_dir / "bm3d_cache"
                        if ds_cache_dir.exists():
                            bm3d_cache_dir = ds_cache_dir
                except Exception:
                    bm3d_cache_dir = None

                if bm3d_cache_dir is None:
                    bm3d_cache_root = val_opt.get("bm3d_cache_root", None)
                    if bm3d_cache_root is None or str(bm3d_cache_root).strip() in ["", "~", "null", "none"]:
                        bm3d_cache_dir = Path("experiments") / exp_name / "cache" / "bm3d"
                    else:
                        bm3d_cache_dir = Path(str(bm3d_cache_root).format(name=exp_name))

                # Prefer readable BM3D cache first (seed-scoped), then fallback to md5 cache via _load_or_compute_bm3d.
                den_bm3d = None
                try:
                    lp0 = str(lq_path).split("__")[0]
                    pdir = Path(lp0).parent
                    cand_root = pdir / "bm3d_cache"
                    if cand_root.exists():
                        seed_dir = cand_root / f"seed{int(seed)}"
                        ds_dir = seed_dir if seed_dir.exists() else cand_root
                        readable = ds_dir / f"{label}_bm3d.npy"
                        if readable.exists():
                            den_bm3d = np.load(readable).astype(np.float32, copy=False)
                except Exception:
                    den_bm3d = None
                if den_bm3d is None:
                    den_bm3d = self._load_or_compute_bm3d(orig, bm3d_cache_dir, lq_path=(lq_path + f"__{label}"))
                den_bm3d = np.clip(den_bm3d.astype(np.float32, copy=False), 0.0, None)
                # Poisson from EMA output (count domain)
                poi = np.random.default_rng(seed + int(round(float(k) * 1000.0))).poisson(lam=den_ema).astype(np.float32)
                res_ans = (_anscombe_forward(orig) - _anscombe_forward(den_ema)).astype(np.float32, copy=False)
                rows.append(
                    dict(
                        label=label,
                        k=float(k),
                        sec=_sec(label, k),
                        orig=orig,
                        bm3d=den_bm3d,
                        g=den_g,
                        ema=den_ema,
                        poi=poi,
                        res_ans=res_ans,
                        theory=(c20 / max(float(k), 1e-12)),
                    )
                )

            ema20 = None
            for r in rows:
                if r["label"] == "20s":
                    ema20 = r["ema"]
                    break
            if ema20 is None:
                ema20 = rows[0]["ema"]

            diffs_res = []
            for r in rows:
                r["diff_cnt"] = (r["ema"] * float(r["k"]) - ema20).astype(np.float32, copy=False)
                r["c_orig"] = float(np.sum(_counts_int(r["orig"]).astype(np.float64, copy=False)))
                r["c_bm3d"] = float(np.sum(_counts_int(r["bm3d"]).astype(np.float64, copy=False)))
                r["c_g"] = float(np.sum(_counts_int(r["g"]).astype(np.float64, copy=False)))
                r["c_ema"] = float(np.sum(r["ema"].astype(np.float64, copy=False)))
                r["c_poi"] = float(np.sum(_counts_int(r["poi"]).astype(np.float64, copy=False)))
                diffs_res.append(np.abs(r["res_ans"]).reshape(-1))
            flat_res = np.concatenate(diffs_res, axis=0) if diffs_res else np.array([1.0], dtype=np.float32)
            dv_res = float(np.percentile(flat_res, 99.5))
            if dv_res < 1e-6:
                dv_res = 1.0

            # ---- render frames ----
            W, H = 128, 128
            header_h = 22
            font = _load_font(12)
            col_labels = ["Original", "BM3D", "Denoised(g)", "Denoised(ema)", "Poisson(ema)", "Res(ans)", "EMA diff"]

            def _g2rgb(u8: np.ndarray) -> Image.Image:
                return Image.fromarray(np.repeat(u8[:, :, None], 3, axis=2), mode="RGB")

            frames = []
            vi_iter = range(int(proj_u16.shape[0]))
            if mp4_tqdm:
                vi_iter = tqdm(
                    vi_iter,
                    total=int(proj_u16.shape[0]),
                    desc=f"mp4-views[{Path(sample_dir).name}]",
                    leave=False,
                    dynamic_ncols=True,
                )
            for vi in vi_iter:
                row_imgs = []
                for r in rows:
                    k = float(r["k"])
                    a = _norm_to_u8(r["orig"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                    b = _norm_to_u8(r["bm3d"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                    c = _norm_to_u8(r["g"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                    d = _norm_to_u8(r["ema"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                    e = _norm_to_u8(r["poi"][vi] * k, vmax=vmax20, use_log1p=use_log1p)
                    # Res(ans) uses its OWN scale; EMA diff shares the same scale for direct visual comparison.
                    f = _signed_to_bwr(r["res_ans"][vi], vmax=dv_res)
                    g = _signed_to_bwr(r["diff_cnt"][vi], vmax=dv_res)

                    row_canvas = Image.new("RGB", (W * 7, H), color=(0, 0, 0))
                    row_canvas.paste(_g2rgb(a), (0 * W, 0))
                    row_canvas.paste(_g2rgb(b), (1 * W, 0))
                    row_canvas.paste(_g2rgb(c), (2 * W, 0))
                    row_canvas.paste(_g2rgb(d), (3 * W, 0))
                    row_canvas.paste(_g2rgb(e), (4 * W, 0))
                    row_canvas.paste(Image.fromarray(f, mode="RGB"), (5 * W, 0))
                    row_canvas.paste(Image.fromarray(g, mode="RGB"), (6 * W, 0))

                    draw = ImageDraw.Draw(row_canvas)
                    # Original column: put row label + ORIGINAL total counts on ONE line (avoid fullwidth '｜' which may not render).
                    # and only show integer W (no %).
                    if str(r["label"]).startswith("x"):
                        row_label = f"{r['sec']:.0f}s(x{k:g})"
                    else:
                        row_label = f"{r['sec']:.0f}s"
                    draw.text((4, 4), f"{row_label} / {_fmt_w_int(float(r['c_orig']))}", fill=255, font=font)
                    t = float(r["theory"])
                    draw.text((1 * W + 4, 4), _fmt_counts_w_pct(float(r["c_bm3d"]), t), fill=255, font=font)
                    draw.text((2 * W + 4, 4), _fmt_counts_w_pct(float(r["c_g"]), t), fill=255, font=font)
                    draw.text((3 * W + 4, 4), _fmt_counts_w_pct(float(r["c_ema"]), t), fill=255, font=font)
                    draw.text((4 * W + 4, 4), _fmt_counts_w_pct(float(r["c_poi"]), t), fill=255, font=font)
                    row_imgs.append(row_canvas)

                full = Image.new("RGB", (W * 7, header_h + H * len(row_imgs)), color=(0, 0, 0))
                draw = ImageDraw.Draw(full)
                for j, lab in enumerate(col_labels):
                    draw.text((j * W + 4, 3), lab, fill=255, font=font)
                draw.text((W * 7 - 240, 3), f"view={vi:02d}  ans_vmax={dv_res:.3g}", fill=255, font=font)
                for i, im in enumerate(row_imgs):
                    full.paste(im, (0, header_h + i * H))
                frames.append(full)

            # ---- write mp4 ----
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            mp4_name = f"{exp_name}_iter{int(current_iter)}_{net_tag}_thin2345.mp4"
            mp4_path = osp.join(sample_dir, mp4_name)
            with imageio.get_writer(
                str(mp4_path),
                format="FFMPEG",
                fps=float(fps) if float(fps) > 0 else 10.0,
                codec="libx264",
                # Avoid imageio auto-resize to macro_block_size=16.
                # We control compatibility ourselves and prefer keeping exact pixels for analysis.
                macro_block_size=1,
                # yuv420p is broadly compatible. Use writer arg instead of raw "-pix_fmt"
                # to avoid "Multiple -pix_fmt options specified" warnings from ffmpeg.
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", str(int(crf))],
            ) as w:
                for fr in frames:
                    w.append_data(np.asarray(fr.convert("RGB")))
            return str(mp4_path)
        except Exception as e:
            self.logger.warning(f"[val] Error generating MP4: {e}", exc_info=True)
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

        # Ensure current_iter is numeric (test.py passes opt['name'] which is a string).
        try:
            current_iter = int(current_iter)  # type: ignore[arg-type]
        except Exception:
            current_iter = 0

        # Get validation config from datasets.val (where dataset config is) or top-level val
        val_opt = self.opt.get('val', {})
        datasets_val_opt = self.opt.get('datasets', {}).get('val', {})

        # Merge: datasets.val takes priority for dataset-specific settings
        merged_val_opt = {**val_opt, **datasets_val_opt}

        generate_gif = bool(merged_val_opt.get('generate_gif', False))
        generate_mp4_thin = bool(merged_val_opt.get('generate_mp4_thin_factors', False))

        # Metrics / best checkpoint are handled here when generate_gif is enabled (because we bypass SRModel's 2D logic).
        with_metrics = merged_val_opt.get('metrics') is not None
        dataset_name = dataloader.dataset.opt['name']
        eval_networks = merged_val_opt.get('eval_networks', None)
        if eval_networks is None:
            eval_networks = ['ema'] if hasattr(self, 'net_g_ema') else ['g']
        eval_networks = [str(x).lower() for x in eval_networks]

        # If neither GIF nor MP4 visualization is enabled:
        # - For projection sequences, run a lightweight 3D inference loop (optionally subsampling views).
        # - Otherwise, fall back to standard SRModel validation.
        if (not generate_gif) and (not generate_mp4_thin):
            try:
                ds_opt = getattr(dataloader.dataset, 'opt', {}) or {}
                ds_type = str(ds_opt.get('type', '')).strip().lower()
            except Exception:
                ds_type = ''
            if ds_type == 'spectprojectionevaldataset':
                return self._nondist_validation_projection_light(dataloader, current_iter, merged_val_opt, eval_networks)
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

        mp4_output_dir = merged_val_opt.get('mp4_output_dir', None)
        if mp4_output_dir is None:
            mp4_output_dir = f'experiments/{exp_name}/visualization/mp4s'
        else:
            mp4_output_dir = str(mp4_output_dir).format(name=exp_name)

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

        # ---- init metrics containers (per SRModel style, with suffixed keys) ----
        if with_metrics:
            metric_names = list((merged_val_opt.get('metrics') or {}).keys())
            metric_keys = [f'{m}_{tag}' for tag in eval_networks for m in metric_names]
            if not hasattr(self, 'metric_results'):
                self.metric_results = {k: 0.0 for k in metric_keys}
            self._initialize_best_metric_results(dataset_name)
            record = self.best_metric_results[dataset_name]
            for tag in eval_networks:
                for m, content in (merged_val_opt.get('metrics') or {}).items():
                    key = f'{m}_{tag}'
                    if key in record:
                        continue
                    better = (content or {}).get('better', 'higher')
                    init_val = float('-inf') if better == 'higher' else float('inf')
                    record[key] = dict(better=better, val=init_val, iter=-1)
            self.metric_results = {k: 0.0 for k in self.metric_results}
            metric_count = 0

        # ---- realtime progress (useful to diagnose "not reached val" vs "val stuck") ----
        # Enable by default in debug runs; can be enabled explicitly via `val.val_tqdm: true`.
        use_val_tqdm = bool(merged_val_opt.get("val_tqdm", ("debug" in str(self.opt.get("name", "")).lower())))
        try:
            total_val = len(dataloader)
        except Exception:
            total_val = None

        # current_iter can be a string in test.py (it passes opt['name']). Make it robust for logging/paths.
        try:
            iter_i = int(current_iter)  # type: ignore[arg-type]
        except Exception:
            iter_i = 0
        self.logger.info(
            f"[val] Start validation: dataset={dataset_name} iter={iter_i} "
            f"mode={'mp4_thin' if generate_mp4_thin else 'gif'} "
            f"max_vis={merged_val_opt.get('max_gifs_per_val', 3)} total={total_val if total_val is not None else '?'}"
        )

        num_gifs = 0
        sample_idx = 0  # Global sample index across all batches
        base_iter = enumerate(dataloader)
        if use_val_tqdm:
            base_iter = tqdm(
                base_iter,
                total=total_val,
                desc=f"val[{dataset_name}] iter{iter_i}",
                leave=False,
                dynamic_ncols=True,
            )
        for idx, val_data in base_iter:
            # Handle different data formats
            # SPECTProjectionEvalDataset returns numpy arrays, standard datasets return tensors
            if isinstance(val_data.get('lq'), np.ndarray):
                # SPECTProjectionEvalDataset: lq is (60, 128, 128) numpy array
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
                        # For SPECTProjectionEvalDataset, keep numpy array as-is
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

                # ---- metrics: LPIPS mean (original vs Poisson(EMA output)) with GPU batching ----
                if with_metrics:
                    try:
                        # Only support custom metric type(s) we define for SPECT.
                        # Currently: lpips_poisson_mean
                        # NOTE: We compute using the SAME inference path as GIF generation (so A is original projection).
                        # To avoid duplicating inference logic, we reuse the denoised_ema computed in _generate_validation_gif
                        # by re-running the minimal forward here (still cheap relative to 100x LPIPS).
                        #
                        # We keep it robust: if anything fails for this sample, skip.
                        pass
                    except Exception:
                        pass

                # Generate GIF with experiment name to avoid conflicts
                # Use per-sample subdirectory to avoid overwriting
                # ⭐ FIX: Use sample_idx to ensure unique directory names for each sample
                # Extract base filename and combine with index to avoid overwriting
                base_name = osp.splitext(osp.basename(lq_path))[0] if lq_path else 'sample'
                img_name = f"{base_name}_{sample_idx:03d}"  # e.g., "ProjectionImage1_000", "ProjectionImage1_001"
                exp_name = self.opt.get('name', 'unknown')

                # Create subdirectory for each sample to avoid overwriting
                sample_dir = osp.join(mp4_output_dir if generate_mp4_thin else gif_output_dir, img_name)
                os.makedirs(sample_dir, exist_ok=True)

                gif_name = f"{exp_name}_iter{current_iter}_{net_tag}.gif"
                gif_path = osp.join(sample_dir, gif_name)

                # Increment sample index for next sample
                sample_idx += 1

                if generate_mp4_thin:
                    mp4_name = f"{exp_name}_iter{int(current_iter)}_{net_tag}_thin2345.mp4"
                    mp4_path = osp.join(sample_dir, mp4_name)
                    if osp.exists(mp4_path):
                        self.logger.info(f'MP4 already exists, skipping: {osp.basename(mp4_path)}')
                        num_gifs += 1
                        if num_gifs >= merged_val_opt.get('max_gifs_per_val', 3):
                            break
                        continue
                    self.logger.info(
                        f"[val][mp4] Start: sample={img_name} iter={int(current_iter)} "
                        f"factors={merged_val_opt.get('mp4_thin_factors', [2, 3, 4, 5])}"
                    )
                    result = self._generate_validation_mp4_thin_factors(
                        val_data=one,
                        sample_dir=sample_dir,
                        exp_name=exp_name,
                        current_iter=int(current_iter),
                        net_tag=net_tag,
                        tb_logger=tb_logger,
                        val_opt=merged_val_opt,
                        poisson_calib_ctx=poisson_calib_ctx,
                    )
                    if result:
                        num_gifs += 1
                        self.logger.info(f'Generated MP4 {num_gifs}/{merged_val_opt.get("max_gifs_per_val", 3)}: {osp.basename(result)}')
                    else:
                        self.logger.warning(f'Failed to generate MP4: {osp.basename(mp4_path)}')
                else:
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

        self.logger.info(f"[val] End validation: generated={num_gifs}")
        if num_gifs > 0:
            self.logger.info(f"✅ Generated {num_gifs} GIF(s) at iter {current_iter}")

        # ---- compute / log metrics & optionally save best ----
        if with_metrics:
            try:
                # We compute metric(s) by re-running a lightweight inference per sample
                # (no BM3D/no GIF rendering) and then running batch LPIPS sampling.
                def _is_3d_net(net: torch.nn.Module) -> bool:
                    try:
                        for m in net.modules():
                            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d):
                                return True
                    except Exception:
                        pass
                    return False

                def _infer_denoised(net: torch.nn.Module, proj_u16: np.ndarray, val_opt: dict) -> np.ndarray:
                    # Determine infer_mode auto/pair60_stitch.
                    infer_mode = str(val_opt.get('infer_mode', '')).strip().lower()
                    network_g_opt0 = self.opt.get('network_g', {}) or {}
                    net_in0 = int(network_g_opt0.get('in_nc', 1))
                    net_out0 = int(network_g_opt0.get('out_nc', 1))
                    if infer_mode in ['', 'auto']:
                        if proj_u16.ndim == 3 and proj_u16.shape[0] == 60 and net_in0 == 2 and net_out0 == 2:
                            infer_mode = 'pair60_stitch'
                        else:
                            infer_mode = ''

                    ds_train_opt = (self.opt.get('datasets', {}) or {}).get('train', {}) or {}
                    max_value_train = float((val_opt or {}).get('max_value', ds_train_opt.get('max_value', 150.0)))
                    if max_value_train < 1e-6:
                        max_value_train = 1.0
                    device = next(net.parameters()).device

                    def _denoise_2d_per_view() -> np.ndarray:
                        mv = float(max_value_train)
                        out = np.zeros_like(proj_u16, dtype=np.float32)
                        net.eval()
                        with torch.no_grad():
                            for i in range(int(proj_u16.shape[0])):
                                proj_i = proj_u16[i].astype(np.float32, copy=False)
                                xt = torch.from_numpy((proj_i / mv)[None, None, ...]).to(device=device, dtype=torch.float32)
                                yt = net(xt)
                                yt = torch.clamp(yt, min=0.0)
                                out[i] = yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
                        return out

                    def _denoise_3d_volume() -> np.ndarray:
                        mv = float(max_value_train)
                        x = (proj_u16 / mv).astype(np.float32, copy=False)
                        xt = torch.from_numpy(x[None, None, ...]).to(device=device, dtype=torch.float32)  # (1,1,V,H,W)
                        net.eval()
                        yt = net(xt)
                        yt = torch.clamp(yt, min=0.0)
                        y = yt[0, 0].detach().cpu().numpy().astype(np.float32, copy=False) * mv
                        return y

                    if infer_mode == 'pair60_stitch':
                        posterior_flip = bool(val_opt.get('posterior_flip', True))
                        pair_bs = int(val_opt.get('pair60_stitch_batch', val_opt.get('metric_infer_batch', 4)))
                        return self._denoise_pair60_stitch(proj_u16, net, max_value_train, device, pair_bs=pair_bs, posterior_flip=posterior_flip)
                    return _denoise_3d_volume() if _is_3d_net(net) else _denoise_2d_per_view()

                # Iterate again (cheap dataset) to compute metrics on up to val_metric_max_files (default: all)
                max_files = int(merged_val_opt.get('metric_max_files', -1))
                if max_files == 0:
                    max_files = -1

                metric_count = 0
                for val_data in dataloader:
                    # extract proj (same logic as GIF: look for proj_sequence/lq)
                    proj_data = None
                    for key in ['proj_sequence', 'lq']:
                        if key in val_data:
                            data = val_data[key]
                            if isinstance(data, torch.Tensor):
                                if data.dim() == 4:
                                    proj_data = data[0].cpu().numpy()
                                elif data.dim() == 3:
                                    proj_data = data.cpu().numpy()
                                break
                            if isinstance(data, np.ndarray):
                                proj_data = data
                                break
                    if proj_data is None:
                        continue
                    proj_u16 = proj_data.astype(np.float32, copy=False)
                    if not (proj_u16.ndim == 3 and int(proj_u16.shape[0]) >= 2):
                        continue

                    # Optional: subsample views for faster validation/test inference.
                    # Example: view_stride=2 keeps views [0,2,4,...] -> 30 views for a 60-view volume.
                    try:
                        view_stride = int(merged_val_opt.get('view_stride', 1))
                    except Exception:
                        view_stride = 1
                    if view_stride < 1:
                        view_stride = 1
                    if view_stride > 1:
                        proj_u16 = proj_u16[::view_stride]

                    # compute per network tag
                    for tag in eval_networks:
                        net = self.net_g_ema if (tag == 'ema' and hasattr(self, 'net_g_ema')) else self.net_g
                        den = _infer_denoised(net, proj_u16, merged_val_opt)
                        den = np.clip(den.astype(np.float32, copy=False), 0.0, None)

                        for m, opt_ in (merged_val_opt.get('metrics') or {}).items():
                            opt_type = str((opt_ or {}).get('type', '')).strip().lower()
                            if opt_type != 'lpips_poisson_mean':
                                # Support additional metrics below (e.g., pll)
                                if opt_type not in ['pll', 'pllloss', 'poisson_nll', 'poisson_nll_mean', 'poisson_nll_loss']:
                                    continue
                                # Poisson NLL (PLL) between EMA denoised lambda and observed counts.
                                # This is a self-supervised likelihood metric: lower is better.
                                try:
                                    import torch.nn.functional as F
                                    eps = float((opt_ or {}).get('eps', merged_val_opt.get('pll_eps', 1e-8)))
                                    full = bool((opt_ or {}).get('full', merged_val_opt.get('pll_full', True)))
                                    # target: integer-ish counts
                                    y = np.clip(np.rint(proj_u16.astype(np.float32, copy=False)), 0.0, None).astype(np.float32, copy=False)
                                    lam = np.clip(den.astype(np.float32, copy=False), eps, None).astype(np.float32, copy=False)
                                    dev = next(net.parameters()).device
                                    yt = torch.from_numpy(y).to(device=dev, dtype=torch.float32)
                                    lamt = torch.from_numpy(lam).to(device=dev, dtype=torch.float32)
                                    lamt = torch.clamp(lamt, min=eps)
                                    val_pll = F.poisson_nll_loss(torch.log(lamt), yt, log_input=True, full=full, reduction='mean')
                                    self.metric_results[f'{m}_{tag}'] += float(val_pll.detach().cpu().item())
                                except Exception:
                                    # Skip if anything fails for this sample/metric
                                    pass
                                continue
                            lpips_net = str((opt_ or {}).get('lpips_net', 'alex')).lower()
                            repeats = int((opt_ or {}).get('repeats', merged_val_opt.get(f'lpips_poisson_repeats_{lpips_net}', 100)))
                            mv = float((opt_ or {}).get('max_value', merged_val_opt.get('lpips_max_value', 150.0)))
                            batch = int((opt_ or {}).get('batch', merged_val_opt.get('lpips_poisson_batch', 4)))
                            view_chunk = int((opt_ or {}).get('view_chunk', merged_val_opt.get('lpips_poisson_view_chunk', 10)))
                            view_stride = int((opt_ or {}).get('view_stride', merged_val_opt.get('lpips_view_stride', 1)))
                            dev = next(net.parameters()).device
                            val = self._lpips_poisson_vs_original_mean_batch(
                                orig_count=proj_u16,
                                denoised_lambda_count=den,
                                lpips_net=lpips_net,
                                repeats=repeats,
                                max_value=mv,
                                batch=batch,
                                view_chunk=view_chunk,
                                view_stride=view_stride,
                                device=torch.device(dev),
                            )
                            self.metric_results[f'{m}_{tag}'] += float(val)

                    metric_count += 1
                    if max_files > 0 and metric_count >= max_files:
                        break

                # reduce mean
                for k in list(self.metric_results.keys()):
                    if k.split('_')[-1] in eval_networks:  # only the suffixed metrics
                        self.metric_results[k] /= float(max(metric_count, 1))
                        base_name = '_'.join(k.split('_')[:-1])
                        tag = k.split('_')[-1]
                        self._update_best_metric_result(dataset_name, k, self.metric_results[k], current_iter)

                # Optionally save best checkpoint (same behavior as SRModel)
                if bool(self.opt.get('val', {}).get('save_best_ckpt', False) or merged_val_opt.get('save_best_ckpt', False)):
                    best_metric = merged_val_opt.get('best_metric', None)
                    record = getattr(self, 'best_metric_results', {}).get(dataset_name, {})
                    if best_metric is None:
                        metric_names = list((merged_val_opt.get('metrics') or {}).keys())
                        best_metric = metric_names[0] if metric_names else None
                        if best_metric is not None and isinstance(eval_networks, list) and len(eval_networks) > 0:
                            best_metric = f'{best_metric}_{eval_networks[0]}'
                    else:
                        if best_metric not in record and isinstance(eval_networks, list) and len(eval_networks) == 1:
                            cand = f'{best_metric}_{eval_networks[0]}'
                            if cand in record:
                                best_metric = cand

                    if best_metric in record and int(record[best_metric].get('iter', -1)) == int(current_iter):
                        self.logger.info(f'[val] Saving best checkpoint: {best_metric} = {record[best_metric]["val"]:.6f} @ {current_iter}')
                        if hasattr(self, 'net_g_ema'):
                            self.save_network([self.net_g, self.net_g_ema], 'net_g', 'best', param_key=['params', 'params_ema'])
                        else:
                            self.save_network(self.net_g, 'net_g', 'best')

                # log
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            except Exception as e:
                self.logger.warning(f"[val] Failed to compute LPIPS-poisson metric: {e}", exc_info=True)

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
