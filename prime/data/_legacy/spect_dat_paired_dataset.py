from __future__ import annotations

from os import path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils import data as data
from scipy.ndimage import rotate as nd_rotate
import os

from basicsr.data.transforms import augment, paired_random_crop
from prime.runtime.anscombe import anscombe_forward, anscombe_inverse_algebraic, anscombe_inverse_unbiased
from prime.data.poisson_thinning import poisson_thinning_complement_split as n2n_binomial_split
from basicsr.utils.img_util import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import scandir


def _read_spect_dat(path: str, height: int, width: int) -> np.ndarray:
    """Read one SPECT .dat file as float32 and reshape to (2, H, W)."""
    arr = np.fromfile(path, dtype=np.float32)
    expected = 2 * height * width
    if arr.size != expected:
        raise ValueError(f'Invalid .dat size: {path}. Expected {expected} float32 values, got {arr.size}.')
    return arr.reshape(2, height, width)


def _to_views(dat_2hw: np.ndarray, posterior_flip: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Split (2,H,W) into (H,W) anterior, posterior_aligned."""
    anterior = dat_2hw[0]
    posterior = dat_2hw[1]
    if posterior_flip:
        posterior = np.fliplr(posterior)
    return anterior, posterior


def _ensure_hwc(img_hw: np.ndarray) -> np.ndarray:
    """Ensure image is HWC (H,W,1) float32."""
    if img_hw.ndim == 3:
        return img_hw.astype(np.float32, copy=False)
    return img_hw[..., None].astype(np.float32, copy=False)


def _parse_patient_from_basename(basename: str) -> str:
    """从诸如 `AnYufeng_pair00` 提取病人名 `AnYufeng`。"""
    if "_pair" in basename:
        return basename.split("_pair", 1)[0]
    return basename.split("_", 1)[0]


def _load_mu_map_zyx(mu_path: str) -> np.ndarray:
    """加载 μ-map，返回 (z,y,x) float32。当前仅支持 128^3。"""
    arr = np.fromfile(mu_path, dtype=np.float32)
    if arr.size == 128 * 128 * 128:
        return arr.reshape(128, 128, 128)
    raise ValueError(f"Unsupported μ-map size: {mu_path}, numel={arr.size}")


def _mu_project_two_views_zyx(
    mu_zyx: np.ndarray,
    angles_deg: Tuple[float, float],
    rotate_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 μ-map 做简化平行束投影，返回两张 view 的 (H,W)：
    - line integral: sum_y μ
    - mip: max_y μ

    输出二维图是 (z,x)，在本项目里等价于 (H,W)。
    """
    if mu_zyx.ndim != 3:
        raise ValueError(f"mu_zyx must be 3D, got {mu_zyx.shape}")

    outs_li = []
    outs_mip = []
    for a in angles_deg:
        rot = nd_rotate(
            mu_zyx,
            angle=float(a),
            axes=(1, 2),
            reshape=False,
            order=int(rotate_order),
            mode="constant",
            cval=0.0,
            prefilter=(rotate_order > 1),
        )
        outs_li.append(rot.sum(axis=1).astype(np.float32, copy=False))
        outs_mip.append(rot.max(axis=1).astype(np.float32, copy=False))

    li = np.stack(outs_li, axis=0)    # (2,H,W)
    mip = np.stack(outs_mip, axis=0)  # (2,H,W)
    return li, mip


def _norm_clip_div(x: np.ndarray, max_value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, float(max_value))
    return (x / float(max_value)).astype(np.float32, copy=False)


def _slice_range(items: List[str], start_idx: int, end_idx: Optional[int]) -> List[str]:
    if start_idx < 0:
        raise ValueError('start_idx must be >= 0')
    if end_idx is not None and end_idx < start_idx:
        raise ValueError('end_idx must be >= start_idx')
    return items[start_idx:end_idx]


def _apply_norm_linear(x: np.ndarray, max_value: float) -> np.ndarray:
    """Normalize to [0, 1]: x / max_value."""
    return (x / float(max_value)).astype(np.float32, copy=False)


def _apply_norm_anscombe(x: np.ndarray, max_value: float) -> np.ndarray:
    """Anscombe normalize to [0, 1]: A(x) / A(max_value)."""
    denom = anscombe_forward(np.array(max_value, dtype=np.float32))
    if float(denom) <= 0:
        raise ValueError(f'Invalid max_value for anscombe normalization: {max_value}')
    return (anscombe_forward(x) / float(denom)).astype(np.float32, copy=False)


@DATASET_REGISTRY.register()
class SPECTDatPairedDataset(data.Dataset):
    """SPECT .dat dataset for denoising/restoration.

    Each sample is a single `.dat` file stored as float32 and reshaped to (2, H, W):
      - index 0: anterior
      - index 1: posterior (optionally flipped left-right to align)

    Modes:
      - mode=paired: read (lq, gt) from two roots, same basename (e.g. 0001.dat).
      - mode=poisson: read gt only; generate lq by Poisson sampling per-pixel mean.
      - mode=n2n: read one count image only; perform Poisson/Binomial splitting per-pixel
        to get two complementary low-dose realizations for Noise2Noise training.
        Specifically, given pixel count n (integer), sample y ~ Binomial(n, p),
        then the other split is (n - y). Use one as input (lq) and the other as target (gt).

    View modes:
      - view_mode=stack2: output (H,W,2) stacked views (default).
      - view_mode=split1: treat each view as an independent (H,W,1) sample.
        view_select in {both, anterior_only, posterior_only}.

    Normalization (output range is [0, 1]):
      - norm.type=linear: x / max_value
      - norm.type=anscombe: A(x)/A(max_value) where A is Anscombe VST.
        norm.inverse in {algebraic, unbiased} is provided for downstream use.
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt

        self.phase = opt.get('phase', 'train')
        self.mode = opt.get('mode', 'paired')  # paired | poisson | n2n
        if self.mode not in ['paired', 'poisson', 'n2n']:
            raise ValueError(f'Unsupported mode: {self.mode}. Supported: paired | poisson | n2n')

        self.height = int(opt.get('height', 1024))
        self.width = int(opt.get('width', 256))
        self.posterior_flip = bool(opt.get('posterior_flip', True))

        # pairing
        self.gt_root = opt.get('dataroot_gt')
        self.lq_root = opt.get('dataroot_lq')
        if self.gt_root is None:
            raise ValueError('dataroot_gt is required')
        if self.mode == 'paired' and self.lq_root is None:
            raise ValueError('dataroot_lq is required for mode=paired')

        # n2n (binomial split) options
        # For p=0.5, the two splits correspond to two independent "2x fast scan" views whose sum is the original count.
        self.n2n_p = float(opt.get('n2n_p', 0.5))
        if not (0.0 < self.n2n_p < 1.0):
            raise ValueError(f'n2n_p must be in (0,1). Got: {self.n2n_p}')
        # Randomly swap which split is input vs target to make training symmetric.
        self.n2n_random_swap = bool(opt.get('n2n_random_swap', True))
        # How to convert float counts to integer trials for binomial splitting.
        # Most .dat files store integer counts as float32; this setting is a safeguard.
        self.n2n_round = str(opt.get('n2n_round', 'nearest'))  # nearest | floor
        if self.n2n_round not in ['nearest', 'floor']:
            raise ValueError(f'Unsupported n2n_round: {self.n2n_round}. Supported: nearest | floor')

        # subset slicing
        self.start_idx = int(opt.get('start_idx', 0))
        self.end_idx = opt.get('end_idx', None)
        self.end_idx = int(self.end_idx) if self.end_idx is not None else None

        # view handling
        self.view_mode = opt.get('view_mode', 'stack2')  # stack2 | split1
        self.view_select = opt.get('view_select', 'both')  # both | anterior_only | posterior_only
        # For split1: optionally force stack both views during val.
        self.force_stack_for_val = opt.get('force_stack_for_val', False)
        if self.view_mode not in ['stack2', 'split1']:
            raise ValueError(f'Unsupported view_mode: {self.view_mode}')
        if self.view_select not in ['both', 'anterior_only', 'posterior_only']:
            raise ValueError(f'Unsupported view_select: {self.view_select}')

        # force_stack_for_val only makes sense with split1 + both + val phase
        if self.force_stack_for_val:
            if self.view_mode != 'split1' or self.view_select != 'both':
                raise ValueError('force_stack_for_val requires view_mode=split1 and view_select=both')
            if self.phase != 'val':
                raise ValueError('force_stack_for_val should only be used in val phase')

        # scale alignment + normalization
        self.gt_scale_factor = float(opt.get('gt_scale_factor', 1.0))
        self.max_value = float(opt['max_value'])  # required (given constant)
        # Optional: clamp count domain values to [0, max_value] before normalization.
        # Useful when max_value is set by percentile (e.g., P99.9) and rare outliers exist.
        self.clip_max_value = bool(opt.get('clip_max_value', False))

        self.norm_opt = opt.get('norm', {}) or {}
        self.norm_type = self.norm_opt.get('type', 'linear')  # linear | anscombe
        if self.norm_type not in ['linear', 'anscombe']:
            raise ValueError(f'Unsupported norm.type: {self.norm_type}')
        self.anscombe_inverse = self.norm_opt.get('inverse', 'algebraic')  # algebraic | unbiased
        if self.anscombe_inverse not in ['algebraic', 'unbiased']:
            raise ValueError(f'Unsupported norm.inverse: {self.anscombe_inverse}')

        # augmentation / patch
        self.gt_size = opt.get('gt_size', None)
        self.use_hflip = bool(opt.get('use_hflip', False))
        self.use_rot = bool(opt.get('use_rot', False))

        # build file list
        self._basenames = self._scan_basenames()
        self._basenames = _slice_range(self._basenames, self.start_idx, self.end_idx)

        # Optional: μ-map auxiliary channels (condition input)
        # Goal: input = [proj, umap_integral, umap_mip], output = [proj]
        self.umap_aux_opt = opt.get("umap_aux", {}) or {}
        self.umap_aux_enable = bool(self.umap_aux_opt.get("enable", False))
        self.umap_root = self.umap_aux_opt.get("root", "datasets/SPECT229")
        self.umap_filename = self.umap_aux_opt.get("filename", "{patient}_PostAtten.dat")
        # For A/P paired projection, default view mapping matches convert_60views_to_ap.py: view0 & view30
        self.umap_anterior_view_index = int(self.umap_aux_opt.get("anterior_view_index", 0))
        self.umap_posterior_view_index = int(self.umap_aux_opt.get("posterior_view_index", 30))
        self.umap_start_angle_deg = float(self.umap_aux_opt.get("start_angle_deg", -180.0))
        self.umap_angle_step_deg = float(self.umap_aux_opt.get("angle_step_deg", 6.0))
        self.umap_rotate_order = int(self.umap_aux_opt.get("rotate_order", 1))
        # Which aux channels to include
        self.umap_use_line_integral = bool(self.umap_aux_opt.get("use_line_integral", True))
        self.umap_use_mip = bool(self.umap_aux_opt.get("use_mip", True))
        # Normalization for aux channels (to [0,1])
        self.umap_line_integral_max = float(self.umap_aux_opt.get("line_integral_max", 2.0))
        self.umap_mip_max = float(self.umap_aux_opt.get("mip_max", 0.05))
        self.umap_cache_root = self.umap_aux_opt.get("cache_root", None)
        self.umap_cache_root = str(self.umap_cache_root) if self.umap_cache_root not in [None, ""] else None
        # Cache per patient (to avoid repeated projections)
        self._umap_cache: dict[str, dict[str, np.ndarray]] = {}
        self._umap_warned_missing = False

        if self.umap_aux_enable:
            # For now we only support split1 training (single view) for channel concatenation.
            if not (self.view_mode == "split1" and not self.force_stack_for_val):
                raise ValueError("umap_aux currently requires view_mode=split1 (single-view).")

    def _scan_basenames(self) -> List[str]:
        # stable order: sorted by filename
        gt_files = [v for v in scandir(self.gt_root, full_path=False) if v.lower().endswith('.dat')]
        gt_basenames = sorted([osp.splitext(osp.basename(v))[0] for v in gt_files])

        if self.mode in ['poisson', 'n2n']:
            return gt_basenames

        # paired: keep intersection
        lq_files = [v for v in scandir(self.lq_root, full_path=False) if v.lower().endswith('.dat')]
        lq_set = set([osp.splitext(osp.basename(v))[0] for v in lq_files])
        names = [n for n in gt_basenames if n in lq_set]
        if len(names) == 0:
            raise FileNotFoundError(
                f'No paired .dat files found. gt_root={self.gt_root}, lq_root={self.lq_root}')
        return names

    def __len__(self) -> int:
        n = len(self._basenames)
        if self.view_mode == 'split1' and self.view_select == 'both' and not self.force_stack_for_val:
            return n * 2
        return n

    def _map_index(self, index: int) -> Tuple[int, int]:
        """Map dataset index to (base_index, view_id). view_id: 0=anterior, 1=posterior."""
        if self.view_mode == 'stack2':
            return index, -1
        # split1
        if self.force_stack_for_val:
            # In force_stack mode, treat as stack2 (return both views)
            return index, -1
        if self.view_select == 'both':
            return index // 2, index % 2
        if self.view_select == 'anterior_only':
            return index, 0
        return index, 1  # posterior_only

    def _load_pair(self, basename: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
        gt_path = osp.join(self.gt_root, f'{basename}.dat')
        gt_dat = _read_spect_dat(gt_path, self.height, self.width)
        gt_a, gt_p = _to_views(gt_dat, posterior_flip=self.posterior_flip)

        if self.mode == 'poisson':
            gt_scaled = np.clip(gt_a, 0.0, None) * self.gt_scale_factor
            gt_p_scaled = np.clip(gt_p, 0.0, None) * self.gt_scale_factor
            gt_img = np.stack([gt_scaled, gt_p_scaled], axis=2)  # HWC, C=2
            lq_img = np.stack(
                [
                    np.random.poisson(lam=gt_scaled).astype(np.float32),
                    np.random.poisson(lam=gt_p_scaled).astype(np.float32),
                ],
                axis=2,
            )
            lq_path = gt_path  # synthetic
            return lq_img, gt_img, lq_path, gt_path

        if self.mode == 'n2n':
            # For N2N we will split AFTER patch crop/augmentation for efficiency.
            # Here we return the base count image twice, and perform binomial split later in __getitem__.
            base_a = np.clip(gt_a, 0.0, None) * self.gt_scale_factor
            base_p = np.clip(gt_p, 0.0, None) * self.gt_scale_factor
            base_img = np.stack([base_a, base_p], axis=2).astype(np.float32, copy=False)
            lq_path = gt_path  # synthetic pair from one measurement
            return base_img, base_img.copy(), lq_path, gt_path

        # paired
        lq_path = osp.join(self.lq_root, f'{basename}.dat')
        lq_dat = _read_spect_dat(lq_path, self.height, self.width)
        lq_a, lq_p = _to_views(lq_dat, posterior_flip=self.posterior_flip)

        gt_scaled = gt_a * self.gt_scale_factor
        gt_p_scaled = gt_p * self.gt_scale_factor
        gt_img = np.stack([gt_scaled, gt_p_scaled], axis=2)
        lq_img = np.stack([lq_a, lq_p], axis=2)
        return lq_img, gt_img, lq_path, gt_path

    def _apply_norm(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.norm_type == 'linear':
            return _apply_norm_linear(lq, self.max_value), _apply_norm_linear(gt, self.max_value)
        return _apply_norm_anscombe(lq, self.max_value), _apply_norm_anscombe(gt, self.max_value)

    def __getitem__(self, index: int) -> Dict:
        base_index, view_id = self._map_index(index)
        basename = self._basenames[base_index]
        patient = _parse_patient_from_basename(basename)

        lq_img, gt_img, lq_path, gt_path = self._load_pair(basename)

        # choose view
        if self.view_mode == 'split1' and not self.force_stack_for_val:
            lq_img = _ensure_hwc(lq_img[:, :, view_id])
            gt_img = _ensure_hwc(gt_img[:, :, view_id])
        else:
            # stack2 or force_stack_for_val: keep both views as (H, W, 2)
            lq_img = lq_img.astype(np.float32, copy=False)
            gt_img = gt_img.astype(np.float32, copy=False)

        # Build μ-map aux channels (H,W,C_aux) BEFORE crop/augment, so we can crop consistently.
        aux_img = None
        if self.umap_aux_enable:
            if patient not in self._umap_cache:
                li_ap = None
                mip_ap = None

                # 1) Try load from disk cache if configured
                if self.umap_cache_root is not None:
                    cache_path = osp.join(self.umap_cache_root, f"{patient}_umap_aux_ap.npz")
                    if osp.exists(cache_path):
                        z = np.load(cache_path)
                        li_ap = z["li_ap"].astype(np.float32, copy=False)
                        mip_ap = z["mip_ap"].astype(np.float32, copy=False)

                # 2) Otherwise compute from μ-map
                if li_ap is None or mip_ap is None:
                    mu_path = osp.join(self.umap_root, patient, self.umap_filename.format(patient=patient))
                    if not osp.exists(mu_path):
                        if not self._umap_warned_missing:
                            self._umap_warned_missing = True
                            print(f"[WARN] μ-map file not found for aux channels: {mu_path}. Will use zeros.")
                        li_ap = np.zeros((2, self.height, self.width), dtype=np.float32)
                        mip_ap = np.zeros((2, self.height, self.width), dtype=np.float32)
                    else:
                        mu_zyx = _load_mu_map_zyx(mu_path)
                        a0 = self.umap_start_angle_deg + self.umap_anterior_view_index * self.umap_angle_step_deg
                        a1 = self.umap_start_angle_deg + self.umap_posterior_view_index * self.umap_angle_step_deg
                        li_ap, mip_ap = _mu_project_two_views_zyx(
                            mu_zyx=mu_zyx,
                            angles_deg=(a0, a1),
                            rotate_order=self.umap_rotate_order,
                        )
                        li_ap = _norm_clip_div(li_ap, self.umap_line_integral_max)
                        mip_ap = _norm_clip_div(mip_ap, self.umap_mip_max)

                        # Align posterior if dataset flips it
                        if self.posterior_flip:
                            li_ap[1] = np.fliplr(li_ap[1])
                            mip_ap[1] = np.fliplr(mip_ap[1])

                    # write cache (best-effort)
                    if self.umap_cache_root is not None:
                        try:
                            os.makedirs(self.umap_cache_root, exist_ok=True)
                            cache_path = osp.join(self.umap_cache_root, f"{patient}_umap_aux_ap.npz")
                            np.savez_compressed(cache_path, li_ap=li_ap, mip_ap=mip_ap)
                        except Exception:
                            pass

                self._umap_cache[patient] = {"li_ap": li_ap, "mip_ap": mip_ap}

            li_ap = self._umap_cache[patient]["li_ap"]
            mip_ap = self._umap_cache[patient]["mip_ap"]
            chans = []
            if self.umap_use_line_integral:
                chans.append(li_ap[view_id][..., None])
            if self.umap_use_mip:
                chans.append(mip_ap[view_id][..., None])
            if len(chans) == 0:
                raise ValueError("umap_aux.enable=True but no aux channels selected (use_line_integral/use_mip).")
            aux_img = np.concatenate(chans, axis=2).astype(np.float32, copy=False)  # (H,W,C_aux)

        # training augmentation / patch crop
        if self.phase == 'train':
            if self.gt_size is not None:
                gt_size = int(self.gt_size)
                # scale=1 for denoising by default
                scale = int(self.opt.get('scale', 1))
                if aux_img is None:
                    gt_img, lq_img = paired_random_crop(gt_img, lq_img, gt_size, scale, gt_path)
                else:
                    gt_img, (lq_img, aux_img) = paired_random_crop(gt_img, [lq_img, aux_img], gt_size, scale, gt_path)
            if self.use_hflip or self.use_rot:
                if aux_img is None:
                    gt_img, lq_img = augment([gt_img, lq_img], self.use_hflip, self.use_rot)
                else:
                    gt_img, lq_img, aux_img = augment([gt_img, lq_img, aux_img], self.use_hflip, self.use_rot)

        # n2n split (online): do it after crop/augment so we only sample on the patch.
        if self.mode == 'n2n':
            # At this point, gt_img contains the cropped base count image (scaled domain).
            lq_img, gt_img = n2n_binomial_split(
                gt_img,
                p=self.n2n_p,
                round_mode=self.n2n_round,
                random_swap=self.n2n_random_swap,
            )

        # Optional clamp before normalization so normalized values stay within [0, 1].
        if self.clip_max_value:
            lq_img = np.clip(lq_img, 0.0, self.max_value)
            gt_img = np.clip(gt_img, 0.0, self.max_value)

        # normalization
        lq_img, gt_img = self._apply_norm(lq_img, gt_img)

        # concat aux channels after norm (all in [0,1])
        if aux_img is not None:
            lq_img = np.concatenate([lq_img, aux_img], axis=2).astype(np.float32, copy=False)

        # to tensor: HWC -> CHW
        lq, gt = img2tensor([lq_img, gt_img], bgr2rgb=False, float32=True)

        out = {'lq': lq, 'gt': gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if self.view_mode == 'split1' and not self.force_stack_for_val:
            out['view_id'] = view_id
            out['patient'] = patient
        return out

    # Optional helpers for downstream inverse mapping (e.g., in validation/inference)
    def inverse_from_norm(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse normalized output back to scaled-count domain.

        Args:
            y_norm: normalized model output in [0, 1] domain.
        """
        # Before inverse transform: clamp negative values to 0.
        # Rationale: in count/Poisson setting, negative intensity is invalid and can
        # cause Anscombe inverse to behave poorly for negative inputs.
        y_norm = np.maximum(np.asarray(y_norm, dtype=np.float32), 0.0)
        if self.norm_type == 'linear':
            return (y_norm * self.max_value).astype(np.float32, copy=False)

        # anscombe: y_norm in [0,1] => y_scaled = y_norm * A(max_value)
        y_scaled = y_norm * float(anscombe_forward(np.array(self.max_value, dtype=np.float32)))
        if self.anscombe_inverse == 'unbiased':
            x_scaled = anscombe_inverse_unbiased(y_scaled)
        else:
            x_scaled = anscombe_inverse_algebraic(y_scaled)
        return x_scaled.astype(np.float32, copy=False)
