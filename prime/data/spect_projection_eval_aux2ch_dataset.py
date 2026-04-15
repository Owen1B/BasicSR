from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from basicsr.utils.registry import DATASET_REGISTRY
from prime.data.spect_projection_eval_dataset import SPECTProjectionEvalDataset, load_projection_dat


def _parse_patient_from_relpath(rel_path: str) -> str:
    stem = Path(rel_path).stem
    if "_pair" in stem:
        return stem.split("_pair", 1)[0]
    return stem.split("_", 1)[0]


@DATASET_REGISTRY.register()
class SPECTProjectionEvalAux2ChDataset(SPECTProjectionEvalDataset):
    """Validation/inference dataset that appends one aligned auxiliary channel."""

    def __init__(self, opt: Dict):
        super().__init__(opt)
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

    def __getitem__(self, index: int) -> Dict:
        out = super().__getitem__(index)
        rel_path = self.paths[index]
        patient = _parse_patient_from_relpath(rel_path)
        aux_path = self._resolve_aux_path(patient)

        aux = load_projection_dat(aux_path, views=self.views, h=self.height, w=self.width, dtype=self.aux_dtype)
        aux = np.clip(aux, 0.0, None) * self.aux_scale_factor
        if self.aux_clip_max_value and self.aux_max_value is not None:
            aux = np.clip(aux, 0.0, self.aux_max_value)
        if self.aux_norm_type == "linear":
            aux = (aux / self.aux_max_value).astype(np.float32, copy=False)
        else:
            aux = aux.astype(np.float32, copy=False)

        obs = np.asarray(out["lq"], dtype=np.float32)
        out["lq"] = np.stack([obs, aux], axis=0).astype(np.float32, copy=False)  # (2,V,H,W)
        out["proj_sequence"] = obs  # Keep metrics/mp4 projection source as observed channel.
        out["patient"] = patient
        out["aux_path"] = aux_path
        return out

