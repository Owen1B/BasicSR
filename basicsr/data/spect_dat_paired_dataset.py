from __future__ import annotations

from os import path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.anscombe import anscombe_forward, anscombe_inverse_algebraic, anscombe_inverse_unbiased
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


def _n2n_binomial_split(
    x_scaled: np.ndarray,
    p: float,
    round_mode: str,
    random_swap: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binomial split for Noise2Noise on count-domain image.

    Given non-negative count image x (float32, typically integer-valued), convert to integer trials n,
    then sample y ~ Binomial(n, p) and z = n - y. Optionally swap (y, z).

    Args:
        x_scaled: count-domain image, HWC float32, non-negative.
        p: probability in (0, 1).
        round_mode: 'nearest' or 'floor' for float->int conversion.
        random_swap: if True, randomly swap input/target with prob 0.5.
    """
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    x_scaled = np.clip(x_scaled, 0.0, None)
    if round_mode == 'nearest':
        n = np.rint(x_scaled)
    elif round_mode == 'floor':
        n = np.floor(x_scaled)
    else:
        raise ValueError(f'Unsupported round_mode: {round_mode}')
    n = np.clip(n, 0.0, None).astype(np.int64, copy=False)

    # vectorized binomial over HWC
    y = np.random.binomial(n=n, p=p).astype(np.int64, copy=False)
    z = (n - y).astype(np.int64, copy=False)

    y_f = y.astype(np.float32, copy=False)
    z_f = z.astype(np.float32, copy=False)

    if random_swap and (np.random.rand() < 0.5):
        return z_f, y_f
    return y_f, z_f


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
        self.force_stack_for_val = opt.get('force_stack_for_val', False)  # For split1: force stack both views during val
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

        lq_img, gt_img, lq_path, gt_path = self._load_pair(basename)

        # choose view
        if self.view_mode == 'split1' and not self.force_stack_for_val:
            lq_img = _ensure_hwc(lq_img[:, :, view_id])
            gt_img = _ensure_hwc(gt_img[:, :, view_id])
        else:
            # stack2 or force_stack_for_val: keep both views as (H, W, 2)
            lq_img = lq_img.astype(np.float32, copy=False)
            gt_img = gt_img.astype(np.float32, copy=False)

        # training augmentation / patch crop
        if self.phase == 'train':
            if self.gt_size is not None:
                gt_size = int(self.gt_size)
                # scale=1 for denoising by default
                scale = int(self.opt.get('scale', 1))
                gt_img, lq_img = paired_random_crop(gt_img, lq_img, gt_size, scale, gt_path)
            if self.use_hflip or self.use_rot:
                gt_img, lq_img = augment([gt_img, lq_img], self.use_hflip, self.use_rot)

        # n2n split (online): do it after crop/augment so we only sample on the patch.
        if self.mode == 'n2n':
            # At this point, gt_img contains the cropped base count image (scaled domain).
            lq_img, gt_img = _n2n_binomial_split(
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

        # to tensor: HWC -> CHW
        lq, gt = img2tensor([lq_img, gt_img], bgr2rgb=False, float32=True)

        out = {'lq': lq, 'gt': gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if self.view_mode == 'split1' and not self.force_stack_for_val:
            out['view_id'] = view_id
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


