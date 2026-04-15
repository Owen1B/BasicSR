from __future__ import annotations

from os import path as osp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.img_util import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from prime.data.poisson_thinning import (
    poisson_thinning_complement_split,
    poisson_thinning_same_dose_split,
)
from prime.runtime.anscombe import anscombe_forward, anscombe_inverse_algebraic, anscombe_inverse_unbiased


def _slice_range(items: List[str], start_idx: int, end_idx: Optional[int]) -> List[str]:
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if end_idx is not None and end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")
    return items[start_idx:end_idx]


def _as_int_list(values: Sequence[int], name: str) -> List[int]:
    out = [int(v) for v in values]
    if len(out) == 0:
        raise ValueError(f"{name} cannot be empty")
    return out


def _parse_patient_from_relpath(rel_path: str) -> str:
    stem = Path(rel_path).stem
    if "_pair" in stem:
        return stem.split("_pair", 1)[0]
    return stem.split("_", 1)[0]


def _read_volume_dat(path: str, views: int, height: int, width: int, dtype: str) -> np.ndarray:
    dtype_map = {
        "float32": np.float32,
        "uint16": np.uint16,
        "int16": np.int16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype={dtype}. Use one of {sorted(dtype_map)}")

    arr = np.fromfile(path, dtype=dtype_map[dtype])
    expected = views * height * width
    if arr.size != expected:
        raise ValueError(
            f"Invalid .dat size: {path}. Expected {expected} values "
            f"(views={views},h={height},w={width}), got {arr.size}."
        )
    return arr.reshape(views, height, width).astype(np.float32, copy=False)


def _apply_norm_linear(x: np.ndarray, max_value: float) -> np.ndarray:
    return (x / float(max_value)).astype(np.float32, copy=False)


def _apply_norm_anscombe(x: np.ndarray, max_value: float) -> np.ndarray:
    denom = anscombe_forward(np.array(max_value, dtype=np.float32))
    if float(denom) <= 0:
        raise ValueError(f"Invalid max_value for anscombe normalization: {max_value}")
    return (anscombe_forward(x) / float(denom)).astype(np.float32, copy=False)


@DATASET_REGISTRY.register()
class SPECTTrainDataset(data.Dataset):
    """Unified projection dataset with pluggable view sampling and tensor packing.

    Core data convention:
    - Raw input is always interpreted as one projection volume with shape (V, H, W).
    - Three view-sampling modes:
      1) single_view: per-view samples
      2) opposite_pair: opposite-view pairs (e.g., AP in V=2)
      3) n_views: arbitrary n views
    - Two packing modes:
      - 2d_channel: output CHW where C = sampled views
      - 3d_depth: output C,D,H,W with C=1 and D = sampled views
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        self.phase = str(opt.get("phase", "train"))
        self.mode = str(opt.get("mode", "paired")).lower().strip()  # paired | poisson | poisson_thinning
        if self.mode not in {"paired", "poisson", "poisson_thinning"}:
            raise ValueError(f"Unsupported mode={self.mode}. Use paired|poisson|poisson_thinning.")

        # Input schema
        self.input_layout = str(opt.get("input_layout", "ap2")).lower().strip()  # ap2 | volume
        if self.input_layout not in {"ap2", "volume"}:
            raise ValueError(f"Unsupported input_layout={self.input_layout}. Use ap2|volume.")

        self.views = int(opt.get("views", 2 if self.input_layout == "ap2" else 60))
        self.height = int(opt["height"])
        self.width = int(opt["width"])
        self.dtype = str(opt.get("dtype", "float32" if self.input_layout == "ap2" else "uint16")).lower().strip()
        self.posterior_flip = bool(opt.get("posterior_flip", self.input_layout == "ap2"))

        self.dataroot_gt = str(opt["dataroot_gt"])
        self.dataroot_lq = opt.get("dataroot_lq")
        self.dataroot_lq = str(self.dataroot_lq) if self.dataroot_lq is not None else None
        self.pattern_gt = str(opt.get("pattern_gt", "*.dat"))
        self.pattern_lq = str(opt.get("pattern_lq", self.pattern_gt))
        if self.mode == "paired" and self.dataroot_lq is None:
            raise ValueError("dataroot_lq is required when mode=paired.")

        self.start_idx = int(opt.get("start_idx", 0))
        end_idx = opt.get("end_idx", None)
        self.end_idx = int(end_idx) if end_idx is not None else None

        # View sampler
        sampler_opt = opt.get("view_sampler", {}) or {}
        self.sample_mode = str(sampler_opt.get("mode", "single_view")).lower().strip()
        if self.sample_mode not in {"single_view", "opposite_pair", "n_views"}:
            raise ValueError(f"Unsupported view_sampler.mode={self.sample_mode}.")

        self.single_view_indices: List[int] = []
        self.opposite_pair_bases: List[int] = []
        self.fixed_n_views_indices: List[int] = []
        self.n_views_strategy = "fixed"
        self.n_views = 1

        self._init_sampler(sampler_opt)

        # Packing
        self.pack_mode = str(opt.get("pack_mode", "2d_channel")).lower().strip()
        if self.pack_mode not in {"2d_channel", "3d_depth"}:
            raise ValueError(f"Unsupported pack_mode={self.pack_mode}. Use 2d_channel|3d_depth.")

        # Normalization + scaling
        self.gt_scale_factor = float(opt.get("gt_scale_factor", 1.0))
        self.max_value = float(opt["max_value"])
        self.clip_max_value = bool(opt.get("clip_max_value", False))
        self.norm_opt = opt.get("norm", {}) or {}
        self.norm_type = str(self.norm_opt.get("type", "linear")).lower().strip()
        if self.norm_type not in {"linear", "anscombe"}:
            raise ValueError(f"Unsupported norm.type={self.norm_type}. Use linear|anscombe.")
        self.anscombe_inverse = str(self.norm_opt.get("inverse", "algebraic")).lower().strip()
        if self.anscombe_inverse not in {"algebraic", "unbiased"}:
            raise ValueError(f"Unsupported norm.inverse={self.anscombe_inverse}. Use algebraic|unbiased.")

        # Poisson thinning options
        self.poisson_thinning_prob = float(opt.get("poisson_thinning_prob", 0.5))
        if not (0.0 < self.poisson_thinning_prob < 1.0):
            raise ValueError(f"poisson_thinning_prob must be in (0,1). Got {self.poisson_thinning_prob}.")
        self.poisson_thinning_random_swap = bool(opt.get("poisson_thinning_random_swap", False))
        self.poisson_thinning_round_mode = str(opt.get("poisson_thinning_round_mode", "nearest")).lower().strip()
        if self.poisson_thinning_round_mode not in {"nearest", "floor"}:
            raise ValueError(
                f"Unsupported poisson_thinning_round_mode={self.poisson_thinning_round_mode}. "
                "Use nearest|floor."
            )
        self.poisson_thinning_mode = str(opt.get("poisson_thinning_mode", "complement")).lower().strip()
        if self.poisson_thinning_mode not in {"complement", "same_dose"}:
            raise ValueError(
                f"Unsupported poisson_thinning_mode={self.poisson_thinning_mode}. Use complement|same_dose."
            )
        self.poisson_thinning_k_choices = opt.get("poisson_thinning_k_choices", None)
        if self.poisson_thinning_k_choices is not None:
            ks = [float(k) for k in list(self.poisson_thinning_k_choices)]
            ks = [k for k in ks if k > 1.0]
            if len(ks) == 0:
                raise ValueError("poisson_thinning_k_choices provided but no valid value > 1.")
            self.poisson_thinning_k_choices = ks

        # 2D crop/augment
        self.gt_size = opt.get("gt_size", None)
        self.use_hflip = bool(opt.get("use_hflip", False))
        self.use_rot = bool(opt.get("use_rot", False))
        if self.pack_mode == "3d_depth" and self.gt_size is not None:
            raise ValueError("gt_size random crop is only supported with pack_mode=2d_channel.")

        self._rel_paths = self._scan_rel_paths()
        self._rel_paths = _slice_range(self._rel_paths, self.start_idx, self.end_idx)
        if len(self._rel_paths) == 0:
            raise FileNotFoundError("No .dat files found after start_idx/end_idx slicing.")

    def _init_sampler(self, sampler_opt: Dict) -> None:
        if self.sample_mode == "single_view":
            idxs = sampler_opt.get("view_indices", None)
            if idxs is None:
                idxs = list(range(self.views))
            self.single_view_indices = _as_int_list(idxs, "view_sampler.view_indices")
            for i in self.single_view_indices:
                if i < 0 or i >= self.views:
                    raise ValueError(f"single_view index out of range: {i} not in [0, {self.views})")
            return

        if self.sample_mode == "opposite_pair":
            offset = int(sampler_opt.get("offset", self.views // 2))
            if offset <= 0 or offset >= self.views:
                raise ValueError(f"Invalid opposite_pair offset={offset} for views={self.views}")
            bases = sampler_opt.get("base_indices", None)
            if bases is None:
                bases = list(range(offset))
            self.opposite_pair_bases = _as_int_list(bases, "view_sampler.base_indices")
            for b in self.opposite_pair_bases:
                if b < 0 or b >= self.views:
                    raise ValueError(f"opposite_pair base index out of range: {b} not in [0, {self.views})")
            self._opposite_offset = offset
            return

        # n_views
        self.n_views = int(sampler_opt.get("n_views", self.views))
        if self.n_views <= 0 or self.n_views > self.views:
            raise ValueError(f"Invalid n_views={self.n_views} for views={self.views}")
        self.n_views_strategy = str(sampler_opt.get("strategy", "fixed")).lower().strip()
        if self.n_views_strategy not in {"fixed", "random", "contiguous_random"}:
            raise ValueError(
                f"Unsupported view_sampler.strategy={self.n_views_strategy}. "
                "Use fixed|random|contiguous_random."
            )
        if self.n_views_strategy == "fixed":
            fixed = sampler_opt.get("view_indices", None)
            if fixed is None:
                start = int(sampler_opt.get("start", 0))
                if start < 0 or start + self.n_views > self.views:
                    raise ValueError(f"Invalid fixed start={start} for n_views={self.n_views}, views={self.views}")
                fixed = list(range(start, start + self.n_views))
            fixed = _as_int_list(fixed, "view_sampler.view_indices")
            if len(fixed) != self.n_views:
                raise ValueError(
                    f"fixed view_indices length mismatch: len={len(fixed)} while n_views={self.n_views}"
                )
            for i in fixed:
                if i < 0 or i >= self.views:
                    raise ValueError(f"fixed view index out of range: {i} not in [0, {self.views})")
            self.fixed_n_views_indices = fixed

    def _scan_rel_paths(self) -> List[str]:
        gt_root = Path(self.dataroot_gt)
        if not gt_root.exists():
            raise FileNotFoundError(f"dataroot_gt not found: {self.dataroot_gt}")
        gt_files = sorted([p for p in gt_root.glob(self.pattern_gt) if p.is_file()])
        if len(gt_files) == 0:
            raise FileNotFoundError(
                f"No files under dataroot_gt={self.dataroot_gt} with pattern_gt={self.pattern_gt}"
            )
        gt_rel = [str(p.relative_to(gt_root)) for p in gt_files]

        if self.mode != "paired":
            return gt_rel

        lq_root = Path(self.dataroot_lq)
        if not lq_root.exists():
            raise FileNotFoundError(f"dataroot_lq not found: {self.dataroot_lq}")
        lq_files = sorted([p for p in lq_root.glob(self.pattern_lq) if p.is_file()])
        lq_rel = {str(p.relative_to(lq_root)) for p in lq_files}

        inter = [rp for rp in gt_rel if rp in lq_rel]
        if len(inter) == 0:
            raise FileNotFoundError(
                f"No paired files found: gt_root={self.dataroot_gt}, lq_root={self.dataroot_lq}, "
                f"pattern_gt={self.pattern_gt}, pattern_lq={self.pattern_lq}"
            )
        return inter

    def __len__(self) -> int:
        n = len(self._rel_paths)
        if self.sample_mode == "single_view":
            return n * len(self.single_view_indices)
        if self.sample_mode == "opposite_pair":
            return n * len(self.opposite_pair_bases)
        return n

    def _map_index(self, index: int) -> Tuple[int, List[int]]:
        if self.sample_mode == "single_view":
            per = len(self.single_view_indices)
            base = index // per
            vid = self.single_view_indices[index % per]
            return base, [vid]

        if self.sample_mode == "opposite_pair":
            per = len(self.opposite_pair_bases)
            base = index // per
            b = self.opposite_pair_bases[index % per]
            return base, [b, (b + self._opposite_offset) % self.views]

        # n_views
        base = index
        if self.n_views_strategy == "fixed":
            return base, list(self.fixed_n_views_indices)
        if self.n_views_strategy == "random":
            idxs = np.random.choice(self.views, size=self.n_views, replace=False)
            return base, sorted([int(v) for v in idxs.tolist()])

        # contiguous_random
        start = int(np.random.randint(0, self.views - self.n_views + 1))
        return base, list(range(start, start + self.n_views))

    def _load_volume(self, root: str, rel_path: str) -> np.ndarray:
        full = osp.join(root, rel_path)
        vol = _read_volume_dat(full, views=self.views, height=self.height, width=self.width, dtype=self.dtype)
        if self.input_layout == "ap2" and self.posterior_flip and self.views >= 2:
            vol = vol.copy()
            vol[1] = np.fliplr(vol[1])
        return vol

    def _apply_norm(self, lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.norm_type == "linear":
            return _apply_norm_linear(lq, self.max_value), _apply_norm_linear(gt, self.max_value)
        return _apply_norm_anscombe(lq, self.max_value), _apply_norm_anscombe(gt, self.max_value)

    def _apply_poisson_thinning(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.poisson_thinning_k_choices is not None and self.poisson_thinning_mode == "same_dose":
            k = float(np.random.choice(self.poisson_thinning_k_choices))
            return poisson_thinning_same_dose_split(
                x_scaled=x,
                k_factor=k,
                round_mode=self.poisson_thinning_round_mode,
                random_swap=self.poisson_thinning_random_swap,
            )
        return poisson_thinning_complement_split(
            x_scaled=x,
            prob=self.poisson_thinning_prob,
            round_mode=self.poisson_thinning_round_mode,
            random_swap=self.poisson_thinning_random_swap,
        )

    def _pack_2d(
        self,
        lq_sel: np.ndarray,
        gt_sel: np.ndarray,
        gt_path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (S,H,W) -> (H,W,S)
        lq_img = np.transpose(lq_sel, (1, 2, 0)).astype(np.float32, copy=False)
        gt_img = np.transpose(gt_sel, (1, 2, 0)).astype(np.float32, copy=False)

        if self.phase == "train":
            if self.gt_size is not None:
                gt_size = int(self.gt_size)
                scale = int(self.opt.get("scale", 1))
                gt_img, lq_img = paired_random_crop(gt_img, lq_img, gt_size, scale, gt_path)
            if self.use_hflip or self.use_rot:
                gt_img, lq_img = augment([gt_img, lq_img], self.use_hflip, self.use_rot)

        # For Poisson thinning, split after crop/augment to avoid sampling unused pixels.
        if self.mode == "poisson_thinning":
            lq_img, gt_img = self._apply_poisson_thinning(gt_img)

        if self.clip_max_value:
            lq_img = np.clip(lq_img, 0.0, self.max_value)
            gt_img = np.clip(gt_img, 0.0, self.max_value)

        lq_img, gt_img = self._apply_norm(lq_img, gt_img)
        lq_t, gt_t = img2tensor([lq_img, gt_img], bgr2rgb=False, float32=True)
        return lq_t, gt_t

    def _pack_3d(self, lq_sel: np.ndarray, gt_sel: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # (D,H,W)
        if self.mode == "poisson_thinning":
            lq_sel, gt_sel = self._apply_poisson_thinning(gt_sel)

        if self.clip_max_value:
            lq_sel = np.clip(lq_sel, 0.0, self.max_value)
            gt_sel = np.clip(gt_sel, 0.0, self.max_value)

        lq_sel, gt_sel = self._apply_norm(lq_sel, gt_sel)
        lq_t = torch.from_numpy(lq_sel[None, ...].astype(np.float32, copy=False))  # (1,D,H,W)
        gt_t = torch.from_numpy(gt_sel[None, ...].astype(np.float32, copy=False))  # (1,D,H,W)
        return lq_t, gt_t

    def __getitem__(self, index: int) -> Dict:
        base_idx, view_indices = self._map_index(index)
        rel = self._rel_paths[base_idx]
        patient = _parse_patient_from_relpath(rel)
        gt_path = osp.join(self.dataroot_gt, rel)

        gt_vol = self._load_volume(self.dataroot_gt, rel)
        gt_vol = np.clip(gt_vol, 0.0, None) * self.gt_scale_factor

        if self.mode == "paired":
            lq_path = osp.join(self.dataroot_lq, rel)
            lq_vol = self._load_volume(self.dataroot_lq, rel)
        elif self.mode == "poisson":
            lq_path = gt_path
            lq_vol = np.random.poisson(lam=gt_vol).astype(np.float32, copy=False)
        else:
            lq_path = gt_path
            lq_vol = gt_vol.copy()

        idx = np.asarray(view_indices, dtype=np.int64)
        gt_sel = np.asarray(gt_vol[idx], dtype=np.float32)
        lq_sel = np.asarray(lq_vol[idx], dtype=np.float32)

        if self.pack_mode == "2d_channel":
            lq_t, gt_t = self._pack_2d(lq_sel=lq_sel, gt_sel=gt_sel, gt_path=gt_path)
        else:
            lq_t, gt_t = self._pack_3d(lq_sel=lq_sel, gt_sel=gt_sel)

        out = {
            "lq": lq_t,
            "gt": gt_t,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "view_indices": idx.tolist(),
            "patient": patient,
        }
        if len(view_indices) == 1:
            out["view_id"] = int(view_indices[0])
        return out

    def inverse_from_norm(self, y_norm: np.ndarray) -> np.ndarray:
        y_norm = np.maximum(np.asarray(y_norm, dtype=np.float32), 0.0)
        if self.norm_type == "linear":
            return (y_norm * self.max_value).astype(np.float32, copy=False)
        y_scaled = y_norm * float(anscombe_forward(np.array(self.max_value, dtype=np.float32)))
        if self.anscombe_inverse == "unbiased":
            x_scaled = anscombe_inverse_unbiased(y_scaled)
        else:
            x_scaled = anscombe_inverse_algebraic(y_scaled)
        return x_scaled.astype(np.float32, copy=False)
