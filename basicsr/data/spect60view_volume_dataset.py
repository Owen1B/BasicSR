from __future__ import annotations

from os import path as osp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


def _parse_patient_from_filename(filename: str) -> str:
    # e.g. AnYufeng_Proj4Filter.dat -> AnYufeng
    stem = osp.splitext(osp.basename(filename))[0]
    if "_pair" in stem:
        return stem.split("_pair", 1)[0]
    return stem.split("_", 1)[0]


def _slice_range(items: List[str], start_idx: int, end_idx: Optional[int]) -> List[str]:
    if start_idx < 0:
        raise ValueError("start_idx must be >= 0")
    if end_idx is not None and end_idx < start_idx:
        raise ValueError("end_idx must be >= start_idx")
    return items[start_idx:end_idx]


def _n2n_binomial_split(x_scaled: np.ndarray, p: float, round_mode: str, random_swap: bool) -> Tuple[np.ndarray, np.ndarray]:
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


def _load_volume_u16(path: str, views: int, h: int, w: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint16)
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(f"Invalid projection size: {path}. Expected {expected} uint16 values, got {arr.size}.")
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


@DATASET_REGISTRY.register()
class SPECT60ViewVolumeDataset(data.Dataset):
    """Per-patient full-volume dataset for 3D UNet training.

    Each sample is a full sinogram volume:
      - proj: (D,H,W) where D=60
      - optional μ-map aux: (D,H,W) line-integral + mip per view

    Output tensors:
      - lq: (C_in,D,H,W) where C_in=1 or 3
      - gt: (1,D,H,W)
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt
        self.phase = opt.get("phase", "train")
        self.mode = opt.get("mode", "n2n")  # n2n | poisson | paired
        if self.mode not in ["n2n", "poisson", "paired"]:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported: n2n | poisson | paired")

        self.dataroot_gt = str(opt["dataroot_gt"])
        self.pattern_gt = str(opt.get("pattern_gt", "**/*_Proj4Filter.dat"))
        self.views = int(opt.get("views", 60))
        self.height = int(opt.get("height", 128))
        self.width = int(opt.get("width", 128))
        self.dtype = str(opt.get("dtype", "uint16"))
        if self.dtype != "uint16":
            raise ValueError("SPECT60ViewVolumeDataset currently supports dtype=uint16 only (Proj4Filter.dat).")

        self.max_value = float(opt["max_value"])
        self.gt_scale_factor = float(opt.get("gt_scale_factor", 1.0))
        self.clip_max_value = bool(opt.get("clip_max_value", False))

        # n2n options
        self.n2n_p = float(opt.get("n2n_p", 0.5))
        if not (0.0 < self.n2n_p < 1.0):
            raise ValueError(f"n2n_p must be in (0,1). Got: {self.n2n_p}")
        self.n2n_random_swap = bool(opt.get("n2n_random_swap", False))
        self.n2n_round = str(opt.get("n2n_round", "nearest"))
        if self.n2n_round not in ["nearest", "floor"]:
            raise ValueError(f"Unsupported n2n_round: {self.n2n_round}")

        # μ-map aux cache (required for 3ch input)
        self.umap_aux_opt = opt.get("umap_aux", {}) or {}
        self.umap_aux_enable = bool(self.umap_aux_opt.get("enable", False))
        self.umap_cache_root = self.umap_aux_opt.get("cache_root", None)
        self.umap_cache_root = str(self.umap_cache_root) if self.umap_cache_root not in [None, ""] else None
        self.umap_cache_suffix = str(self.umap_aux_opt.get("cache_suffix", f"views{self.views}"))
        self.umap_use_line_integral = bool(self.umap_aux_opt.get("use_line_integral", True))
        self.umap_use_mip = bool(self.umap_aux_opt.get("use_mip", True))
        if self.umap_aux_enable and self.umap_cache_root is None:
            raise ValueError("umap_aux.enable=True requires umap_aux.cache_root (precomputed).")

        self._umap_cache: dict[str, dict[str, np.ndarray]] = {}
        self._umap_warned_missing = False

        # scan file list
        root = Path(self.dataroot_gt)
        if not root.exists():
            raise FileNotFoundError(f"dataroot_gt not found: {self.dataroot_gt}")
        files = sorted([p for p in root.glob(self.pattern_gt) if p.is_file()])
        if len(files) == 0:
            raise FileNotFoundError(f"No gt files found: root={self.dataroot_gt}, pattern_gt={self.pattern_gt}")
        rels = [str(p.relative_to(root)) for p in files]

        start_idx = int(opt.get("start_idx", 0))
        end_idx = opt.get("end_idx", None)
        end_idx = int(end_idx) if end_idx is not None else None
        self.paths = _slice_range(rels, start_idx, end_idx)

    def __len__(self) -> int:
        return len(self.paths)

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
            li = z["li"].astype(np.float32, copy=False)
            mip = z["mip"].astype(np.float32, copy=False)
            if li.shape != (self.views, self.height, self.width):
                raise ValueError(f"li shape mismatch in {cache_path}: got {li.shape}, expected {(self.views, self.height, self.width)}")
            if mip.shape != (self.views, self.height, self.width):
                raise ValueError(f"mip shape mismatch in {cache_path}: got {mip.shape}, expected {(self.views, self.height, self.width)}")
        self._umap_cache[patient] = {"li": li, "mip": mip}
        return self._umap_cache[patient]

    def __getitem__(self, index: int) -> Dict:
        rel = self.paths[index]
        gt_path = osp.join(self.dataroot_gt, rel)
        patient = _parse_patient_from_filename(gt_path)

        vol = _load_volume_u16(gt_path, self.views, self.height, self.width)  # (D,H,W) float32
        vol = np.clip(vol, 0.0, None) * self.gt_scale_factor

        if self.mode == "poisson":
            gt = vol
            lq = np.random.poisson(lam=gt).astype(np.float32)
        elif self.mode == "n2n":
            # split the full volume
            lq, gt = _n2n_binomial_split(vol, p=self.n2n_p, round_mode=self.n2n_round, random_swap=self.n2n_random_swap)
        else:
            raise NotImplementedError("paired mode not implemented for volume dataset yet.")

        if self.clip_max_value:
            lq = np.clip(lq, 0.0, self.max_value)
            gt = np.clip(gt, 0.0, self.max_value)

        # normalize to [0,1]
        lq = (lq / self.max_value).astype(np.float32, copy=False)
        gt = (gt / self.max_value).astype(np.float32, copy=False)

        # build aux channels (already expected in [0,1])
        if self.umap_aux_enable:
            aux = self._load_umap_aux(patient)
            chans = [lq[None, ...]]  # (1,D,H,W)
            if self.umap_use_line_integral:
                chans.append(aux["li"][None, ...])
            if self.umap_use_mip:
                chans.append(aux["mip"][None, ...])
            lq_t = np.concatenate(chans, axis=0).astype(np.float32, copy=False)  # (C,D,H,W)
        else:
            lq_t = lq[None, ...]

        gt_t = gt[None, ...]  # (1,D,H,W)

        # to torch
        lq_tensor = torch.from_numpy(lq_t)
        gt_tensor = torch.from_numpy(gt_t)

        return {
            "lq": lq_tensor,
            "gt": gt_tensor,
            "lq_path": gt_path,
            "gt_path": gt_path,
            "patient": patient,
        }


