from __future__ import annotations

from os import path as osp
from typing import Any, Dict

import numpy as np
import torch

from basicsr.utils.registry import DATASET_REGISTRY
from prime.data.spect_train_dataset import SPECTTrainDataset, _parse_patient_from_relpath
from prime.runtime.support_conditioning import build_support_condition_tensors


@DATASET_REGISTRY.register()
class SPECTTrainSupportConditionDataset(SPECTTrainDataset):
    """3D tomo training dataset with explicit acquisition-support condition channels."""

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)
        if self.pack_mode != "3d_depth":
            raise ValueError("SPECTTrainSupportConditionDataset requires pack_mode=3d_depth.")

        support_opt = dict(opt.get("support_condition", {}) or {})
        if not bool(support_opt):
            raise ValueError("support_condition config is required for SPECTTrainSupportConditionDataset.")
        self.support_condition_opt = support_opt

    def _build_support_tensors(self, source_sel: np.ndarray) -> dict[str, torch.Tensor]:
        cond = build_support_condition_tensors(source_sel, self.support_condition_opt)
        out: dict[str, torch.Tensor] = {
            "support_mask": torch.from_numpy(cond["mask"][None, ...].astype(np.float32, copy=False)),
            "support_boundary_weight": torch.from_numpy(
                cond["boundary_weight"][None, ...].astype(np.float32, copy=False)
            ),
        }
        if "distance" in cond:
            out["support_distance"] = torch.from_numpy(cond["distance"][None, ...].astype(np.float32, copy=False))
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
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

        support_tensors = self._build_support_tensors(lq_sel)
        lq_t, gt_t = self._pack_3d(lq_sel=lq_sel, gt_sel=gt_sel)

        if "support_distance" in support_tensors:
            lq_t = torch.cat([lq_t, support_tensors["support_mask"], support_tensors["support_distance"]], dim=0)
        else:
            lq_t = torch.cat([lq_t, support_tensors["support_mask"]], dim=0)

        out: Dict[str, Any] = {
            "lq": lq_t,
            "gt": gt_t,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "view_indices": idx.tolist(),
            "patient": patient,
            "support_mask": support_tensors["support_mask"],
            "support_boundary_weight": support_tensors["support_boundary_weight"],
        }
        if "support_distance" in support_tensors:
            out["support_distance"] = support_tensors["support_distance"]
        return out
