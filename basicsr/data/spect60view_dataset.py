from __future__ import annotations

from os import path as osp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.anscombe import anscombe_forward, anscombe_inverse_algebraic, anscombe_inverse_unbiased
from basicsr.utils.img_util import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import scandir


def _ensure_hwc(img_hw: np.ndarray) -> np.ndarray:
    """Ensure image is HWC (H,W,1) float32."""
    if img_hw.ndim == 3:
        return img_hw.astype(np.float32, copy=False)
    return img_hw[..., None].astype(np.float32, copy=False)


def _parse_patient_from_basename(basename: str) -> str:
    """从文件名（不含扩展）提取病人名。

    兼容：
    - `AnYufeng` / `AnYufeng_proj` / `AnYufeng_ProjectionImage` 等：取第一个 `_` 前缀
    - `AnYufeng_pair00`：取 `_pair` 前缀
    """
    if "_pair" in basename:
        return basename.split("_pair", 1)[0]
    return basename.split("_", 1)[0]


def _slice_range(items: List[str], start_idx: int, end_idx: Optional[int]) -> List[str]:
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if end_idx is not None and end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")
    return items[start_idx:end_idx]


def _apply_norm_linear(x: np.ndarray, max_value: float) -> np.ndarray:
    return (x / float(max_value)).astype(np.float32, copy=False)


def _apply_norm_anscombe(x: np.ndarray, max_value: float) -> np.ndarray:
    denom = anscombe_forward(np.array(max_value, dtype=np.float32))
    if float(denom) <= 0:
        raise ValueError(f"Invalid max_value for anscombe normalization: {max_value}")
    return (anscombe_forward(x) / float(denom)).astype(np.float32, copy=False)


def _n2n_binomial_split(
    x_scaled: np.ndarray,
    p: float,
    round_mode: str,
    random_swap: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binomial split for Noise2Noise on count-domain image (HWC float32)."""
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    x_scaled = np.clip(x_scaled, 0.0, None)
    if round_mode == "nearest":
        n = np.rint(x_scaled)
    elif round_mode == "floor":
        n = np.floor(x_scaled)
    else:
        raise ValueError(f"Unsupported round_mode: {round_mode}")
    n = np.clip(n, 0.0, None).astype(np.int64, copy=False)

    y = np.random.binomial(n=n, p=p).astype(np.int64, copy=False)
    z = (n - y).astype(np.int64, copy=False)

    y_f = y.astype(np.float32, copy=False)
    z_f = z.astype(np.float32, copy=False)
    if random_swap and (np.random.rand() < 0.5):
        return z_f, y_f
    return y_f, z_f


def _load_view_from_dat(
    dat_path: str,
    view_idx: int,
    views: int,
    height: int,
    width: int,
    dtype: str,
) -> np.ndarray:
    """Load a single view (H,W) from a (V,H,W) .dat file using memmap."""
    if dtype == "float32":
        np_dtype = np.float32
        bytes_per = 4
    elif dtype == "uint16":
        np_dtype = np.uint16
        bytes_per = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use float32 or uint16.")

    expected_elems = views * height * width
    expected_bytes = expected_elems * bytes_per
    st = osp.getsize(dat_path)
    if st != expected_bytes:
        raise ValueError(f"Invalid .dat size: {dat_path}. Expected {expected_bytes} bytes, got {st}.")

    mm = np.memmap(dat_path, dtype=np_dtype, mode="r", shape=(views, height, width))
    v = np.asarray(mm[int(view_idx)], dtype=np.float32)  # convert to float32
    return v


@DATASET_REGISTRY.register()
class SPECT60ViewDatDataset(data.Dataset):
    """60-view SPECT projection dataset for training with per-view samples.

    Each file is a single patient projection cube stored as (V,H,W) in a `.dat` file.
    The dataset exposes each view as one sample, and can optionally add μ-map aux channels:
      input = [proj, umap_line_integral(view), umap_mip(view)]
      output = [proj]
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        self.phase = opt.get("phase", "train")

        self.mode = opt.get("mode", "n2n")  # n2n | poisson | paired
        if self.mode not in ["n2n", "poisson", "paired"]:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported: n2n | poisson | paired")

        self.views = int(opt.get("views", 60))
        self.height = int(opt.get("height", 128))
        self.width = int(opt.get("width", 128))
        self.dtype = str(opt.get("dtype", "float32"))  # float32 | uint16
        # Allow recursive scanning, e.g. datasets/SPECT229/<patient>/*_Proj4Filter.dat
        self.pattern_gt = str(opt.get("pattern_gt", "**/*.dat"))
        self.pattern_lq = str(opt.get("pattern_lq", self.pattern_gt))

        # roots
        self.gt_root = opt.get("dataroot_gt")
        self.lq_root = opt.get("dataroot_lq")
        if self.gt_root is None:
            raise ValueError("dataroot_gt is required")
        if self.mode == "paired" and self.lq_root is None:
            raise ValueError("dataroot_lq is required for mode=paired")

        # file list
        self.start_idx = int(opt.get("start_idx", 0))
        self.end_idx = opt.get("end_idx", None)
        self.end_idx = int(self.end_idx) if self.end_idx is not None else None
        self._basenames = self._scan_basenames()
        self._basenames = _slice_range(self._basenames, self.start_idx, self.end_idx)

        # normalization
        self.gt_scale_factor = float(opt.get("gt_scale_factor", 1.0))
        self.max_value = float(opt["max_value"])
        self.clip_max_value = bool(opt.get("clip_max_value", False))
        self.norm_opt = opt.get("norm", {}) or {}
        self.norm_type = self.norm_opt.get("type", "linear")  # linear | anscombe
        if self.norm_type not in ["linear", "anscombe"]:
            raise ValueError(f"Unsupported norm.type: {self.norm_type}")
        self.anscombe_inverse = self.norm_opt.get("inverse", "algebraic")
        if self.anscombe_inverse not in ["algebraic", "unbiased"]:
            raise ValueError(f"Unsupported norm.inverse: {self.anscombe_inverse}")

        # augmentation / patch
        self.gt_size = opt.get("gt_size", None)
        self.use_hflip = bool(opt.get("use_hflip", False))
        self.use_rot = bool(opt.get("use_rot", False))

        # n2n options
        self.n2n_p = float(opt.get("n2n_p", 0.5))
        if not (0.0 < self.n2n_p < 1.0):
            raise ValueError(f"n2n_p must be in (0,1). Got: {self.n2n_p}")
        self.n2n_random_swap = bool(opt.get("n2n_random_swap", True))
        self.n2n_round = str(opt.get("n2n_round", "nearest"))
        if self.n2n_round not in ["nearest", "floor"]:
            raise ValueError(f"Unsupported n2n_round: {self.n2n_round}. Supported: nearest | floor")

        # μ-map aux (expects precomputed per-view cache)
        self.umap_aux_opt = opt.get("umap_aux", {}) or {}
        self.umap_aux_enable = bool(self.umap_aux_opt.get("enable", False))
        self.umap_use_line_integral = bool(self.umap_aux_opt.get("use_line_integral", True))
        self.umap_use_mip = bool(self.umap_aux_opt.get("use_mip", True))
        self.umap_cache_root = self.umap_aux_opt.get("cache_root", None)
        self.umap_cache_root = str(self.umap_cache_root) if self.umap_cache_root not in [None, ""] else None
        self.umap_cache_suffix = str(self.umap_aux_opt.get("cache_suffix", f"views{self.views}"))  # views60 by default

        self._umap_cache: dict[str, dict[str, np.ndarray]] = {}
        self._umap_warned_missing = False

        if self.umap_aux_enable and self.umap_cache_root is None:
            raise ValueError("For 60-view training, umap_aux.enable=True requires umap_aux.cache_root (precomputed).")

    def _scan_basenames(self) -> List[str]:
        gt_root = Path(self.gt_root)
        if not gt_root.exists():
            raise FileNotFoundError(f"dataroot_gt not found: {self.gt_root}")

        gt_files = sorted([p for p in gt_root.glob(self.pattern_gt) if p.is_file()])
        if len(gt_files) == 0:
            raise FileNotFoundError(f"No .dat files found under dataroot_gt={self.gt_root} with pattern_gt={self.pattern_gt}")

        # Store relative paths (within gt_root) so __getitem__ can resolve to full path.
        gt_rel_paths = [str(p.relative_to(gt_root)) for p in gt_files]

        if self.mode in ["poisson", "n2n"]:
            return gt_rel_paths

        # paired: intersection
        lq_root = Path(self.lq_root)
        if not lq_root.exists():
            raise FileNotFoundError(f"dataroot_lq not found: {self.lq_root}")
        lq_files = sorted([p for p in lq_root.glob(self.pattern_lq) if p.is_file()])
        lq_rel = set([str(p.relative_to(lq_root)) for p in lq_files])
        names = [rp for rp in gt_rel_paths if rp in lq_rel]
        if len(names) == 0:
            raise FileNotFoundError(
                f"No paired .dat found. gt_root={self.gt_root} pattern_gt={self.pattern_gt}, "
                f"lq_root={self.lq_root} pattern_lq={self.pattern_lq}"
            )
        return names

    def __len__(self) -> int:
        return len(self._basenames) * self.views

    def _map_index(self, index: int) -> Tuple[int, int]:
        base_index = index // self.views
        view_idx = index % self.views
        return base_index, view_idx

    def _apply_norm(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.norm_type == "linear":
            return _apply_norm_linear(lq, self.max_value), _apply_norm_linear(gt, self.max_value)
        return _apply_norm_anscombe(lq, self.max_value), _apply_norm_anscombe(gt, self.max_value)

    def _load_umap_aux(self, patient: str) -> dict[str, np.ndarray]:
        if patient in self._umap_cache:
            return self._umap_cache[patient]

        cache_path = osp.join(self.umap_cache_root, f"{patient}_umap_aux_{self.umap_cache_suffix}.npz")
        if not osp.exists(cache_path):
            if not self._umap_warned_missing:
                self._umap_warned_missing = True
                print(f"[WARN] μ-map aux cache not found: {cache_path}. Will use zeros.")
            li = np.zeros((self.views, self.height, self.width), dtype=np.float32)
            mip = np.zeros((self.views, self.height, self.width), dtype=np.float32)
        else:
            z = np.load(cache_path)
            # expected: li/mip with shape (V,H,W)
            li = z["li"].astype(np.float32, copy=False)
            mip = z["mip"].astype(np.float32, copy=False)
            if li.shape[0] != self.views:
                raise ValueError(f"li views mismatch in {cache_path}: got {li.shape}, expected V={self.views}")
        self._umap_cache[patient] = {"li": li, "mip": mip}
        return self._umap_cache[patient]

    def __getitem__(self, index: int) -> Dict:
        base_index, view_idx = self._map_index(index)
        rel_path = self._basenames[base_index]
        basename = osp.splitext(osp.basename(rel_path))[0]
        patient = _parse_patient_from_basename(basename)

        gt_path = osp.join(self.gt_root, rel_path)
        if self.mode == "paired":
            lq_path = osp.join(self.lq_root, rel_path)
        else:
            lq_path = gt_path

        # load view (H,W)
        gt_hw = _load_view_from_dat(gt_path, view_idx, self.views, self.height, self.width, self.dtype)
        if self.mode == "paired":
            lq_hw = _load_view_from_dat(lq_path, view_idx, self.views, self.height, self.width, self.dtype)
        elif self.mode == "poisson":
            gt_scaled = np.clip(gt_hw, 0.0, None) * self.gt_scale_factor
            lq_hw = np.random.poisson(lam=gt_scaled).astype(np.float32)
            gt_hw = gt_scaled
        else:
            # n2n: use gt as base count, split after crop/augment
            gt_hw = np.clip(gt_hw, 0.0, None) * self.gt_scale_factor
            lq_hw = gt_hw.copy()

        lq_img = _ensure_hwc(lq_hw)
        gt_img = _ensure_hwc(gt_hw)

        # aux channels (H,W,C_aux) before crop
        aux_img = None
        if self.umap_aux_enable:
            aux = self._load_umap_aux(patient)
            chans = []
            if self.umap_use_line_integral:
                chans.append(aux["li"][view_idx][..., None])
            if self.umap_use_mip:
                chans.append(aux["mip"][view_idx][..., None])
            if len(chans) == 0:
                raise ValueError("umap_aux.enable=True but no aux channels selected (use_line_integral/use_mip).")
            aux_img = np.concatenate(chans, axis=2).astype(np.float32, copy=False)

        # crop/augment
        if self.phase == "train":
            if self.gt_size is not None:
                gt_size = int(self.gt_size)
                scale = int(self.opt.get("scale", 1))
                if aux_img is None:
                    gt_img, lq_img = paired_random_crop(gt_img, lq_img, gt_size, scale, gt_path)
                else:
                    gt_img, (lq_img, aux_img) = paired_random_crop(gt_img, [lq_img, aux_img], gt_size, scale, gt_path)
            if self.use_hflip or self.use_rot:
                if aux_img is None:
                    gt_img, lq_img = augment([gt_img, lq_img], self.use_hflip, self.use_rot)
                else:
                    gt_img, lq_img, aux_img = augment([gt_img, lq_img, aux_img], self.use_hflip, self.use_rot)

        # n2n split after crop/augment
        if self.mode == "n2n":
            lq_img, gt_img = _n2n_binomial_split(
                gt_img,
                p=self.n2n_p,
                round_mode=self.n2n_round,
                random_swap=self.n2n_random_swap,
            )

        if self.clip_max_value:
            lq_img = np.clip(lq_img, 0.0, self.max_value)
            gt_img = np.clip(gt_img, 0.0, self.max_value)

        # normalize proj
        lq_img, gt_img = self._apply_norm(lq_img, gt_img)

        # concat aux after norm (aux is expected already in [0,1])
        if aux_img is not None:
            lq_img = np.concatenate([lq_img, aux_img], axis=2).astype(np.float32, copy=False)

        lq, gt = img2tensor([lq_img, gt_img], bgr2rgb=False, float32=True)
        return {
            "lq": lq,
            "gt": gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "patient": patient,
            "view_id": int(view_idx),
        }

    def inverse_from_norm(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse normalized output back to scaled-count domain (for debugging/validation)."""
        y_norm = np.maximum(np.asarray(y_norm, dtype=np.float32), 0.0)
        if self.norm_type == "linear":
            return (y_norm * self.max_value).astype(np.float32, copy=False)
        y_scaled = y_norm * float(anscombe_forward(np.array(self.max_value, dtype=np.float32)))
        if self.anscombe_inverse == "unbiased":
            x_scaled = anscombe_inverse_unbiased(y_scaled)
        else:
            x_scaled = anscombe_inverse_algebraic(y_scaled)
        return x_scaled.astype(np.float32, copy=False)


