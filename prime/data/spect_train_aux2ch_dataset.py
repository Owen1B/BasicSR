from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from basicsr.utils.registry import DATASET_REGISTRY
from prime.data.spect_train_dataset import SPECTTrainDataset, _read_volume_dat


@DATASET_REGISTRY.register()
class SPECTTrainAux2ChDataset(SPECTTrainDataset):
    """3D dataset with two input channels and one output channel.

    Channel 0:
      - Same as SPECTTrainDataset lq branch (can use poisson_thinning / k-factor).
    Channel 1:
      - Auxiliary aligned projection volume loaded from dataroot_aux.
      - Never participates in poisson thinning (kept deterministic).
    """

    def __init__(self, opt: Dict):
        super().__init__(opt)
        if self.pack_mode != "3d_depth":
            raise ValueError("SPECTTrainAux2ChDataset requires pack_mode=3d_depth.")

        self.dataroot_aux = str(opt["dataroot_aux"])
        self.pattern_aux = str(
            opt.get("pattern_aux", "{patient}/{patient}_PostAtten_thr01622_ProjAligned.dat")
        )
        self.aux_dtype = str(opt.get("aux_dtype", "float32")).lower().strip()
        self.aux_scale_factor = float(opt.get("aux_scale_factor", 1.0))
        self.aux_max_value = opt.get("aux_max_value", None)
        self.aux_max_value = float(self.aux_max_value) if self.aux_max_value is not None else None
        self.aux_clip_max_value = bool(opt.get("aux_clip_max_value", False))
        aux_norm_opt = opt.get("aux_norm", {}) or {}
        self.aux_norm_type = str(aux_norm_opt.get("type", "none")).lower().strip()
        if self.aux_norm_type not in {"none", "linear"}:
            raise ValueError(f"Unsupported aux_norm.type={self.aux_norm_type}. Use none|linear.")
        if self.aux_norm_type == "linear" and (self.aux_max_value is None or self.aux_max_value <= 0):
            raise ValueError("aux_norm.type=linear requires aux_max_value > 0.")

        self._aux_root = Path(self.dataroot_aux)
        if not self._aux_root.exists():
            raise FileNotFoundError(f"dataroot_aux not found: {self.dataroot_aux}")

    def _resolve_aux_path(self, patient: str) -> str:
        rel = self.pattern_aux.format(patient=patient)
        full = self._aux_root / rel
        if not full.exists():
            raise FileNotFoundError(
                f"Aux file not found for patient={patient}: {full}. "
                f"Check dataroot_aux/pattern_aux in yaml."
            )
        return str(full)

    def _load_aux_volume(self, patient: str) -> Tuple[np.ndarray, str]:
        aux_path = self._resolve_aux_path(patient)
        aux = _read_volume_dat(
            aux_path,
            views=self.views,
            height=self.height,
            width=self.width,
            dtype=self.aux_dtype,
        )
        aux = np.clip(aux, 0.0, None) * self.aux_scale_factor
        if self.aux_clip_max_value and self.aux_max_value is not None:
            aux = np.clip(aux, 0.0, self.aux_max_value)
        if self.aux_norm_type == "linear":
            aux = (aux / self.aux_max_value).astype(np.float32, copy=False)
        else:
            aux = aux.astype(np.float32, copy=False)
        return aux, aux_path

    def __getitem__(self, index: int) -> Dict:
        out = super().__getitem__(index)
        patient = str(out["patient"])
        view_indices = np.asarray(out["view_indices"], dtype=np.int64)

        aux_vol, aux_path = self._load_aux_volume(patient)
        aux_sel = np.asarray(aux_vol[view_indices], dtype=np.float32)
        aux_t = torch.from_numpy(aux_sel[None, ...].astype(np.float32, copy=False))  # (1,D,H,W)

        lq_t = out["lq"]
        if not isinstance(lq_t, torch.Tensor) or lq_t.ndim != 4:
            raise ValueError(
                f"Unexpected lq tensor shape for SPECTTrainAux2ChDataset: "
                f"type={type(lq_t)}, shape={getattr(lq_t, 'shape', None)}"
            )

        out["lq"] = torch.cat([lq_t, aux_t], dim=0)  # (2,D,H,W)
        out["aux_path"] = aux_path
        return out

