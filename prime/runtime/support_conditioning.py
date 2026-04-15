from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportConditionSpec:
    mode: str = "from_data_nonzero"
    threshold: float = 0.0
    reduction: str = "per_view_rows"
    fixed_row_start: int | None = None
    fixed_row_end: int | None = None
    include_distance: bool = True
    signed_distance: bool = True
    distance_normalizer: str = "support_half_height"
    boundary_band: int = 1
    boundary_emphasis: float = 3.0


def _normalize_spec(spec: dict[str, Any] | None) -> SupportConditionSpec:
    spec = dict(spec or {})
    return SupportConditionSpec(
        mode=str(spec.get("mode", "from_data_nonzero")).lower().strip(),
        threshold=float(spec.get("threshold", 0.0)),
        reduction=str(spec.get("reduction", "per_view_rows")).lower().strip(),
        fixed_row_start=(
            int(spec["fixed_row_start"]) if spec.get("fixed_row_start", None) is not None else None
        ),
        fixed_row_end=int(spec["fixed_row_end"]) if spec.get("fixed_row_end", None) is not None else None,
        include_distance=bool(spec.get("include_distance", True)),
        signed_distance=bool(spec.get("signed_distance", True)),
        distance_normalizer=str(spec.get("distance_normalizer", "support_half_height")).lower().strip(),
        boundary_band=max(0, int(spec.get("boundary_band", 1))),
        boundary_emphasis=max(0.0, float(spec.get("boundary_emphasis", 3.0))),
    )


def _row_mask_from_projection(proj: np.ndarray, spec: SupportConditionSpec) -> np.ndarray:
    if proj.ndim != 3:
        raise ValueError(f"Projection must have shape (V,H,W), got {proj.shape}")
    views, height, _ = proj.shape

    if spec.mode == "fixed_rows":
        if spec.fixed_row_start is None or spec.fixed_row_end is None:
            raise ValueError("fixed_rows mode requires fixed_row_start and fixed_row_end.")
        start = int(spec.fixed_row_start)
        end = int(spec.fixed_row_end)
        if start < 0 or end < start or end >= height:
            raise ValueError(
                f"Invalid fixed support rows start={start}, end={end} for height={height}."
            )
        row_mask = np.zeros((views, height), dtype=bool)
        row_mask[:, start : end + 1] = True
        return row_mask

    if spec.mode != "from_data_nonzero":
        raise ValueError(f"Unsupported support mode: {spec.mode}")

    row_mask = np.any(proj > float(spec.threshold), axis=2)
    if spec.reduction == "global_rows":
        global_rows = np.any(row_mask, axis=0, keepdims=True)
        row_mask = np.repeat(global_rows, repeats=views, axis=0)
    elif spec.reduction != "per_view_rows":
        raise ValueError(f"Unsupported support reduction: {spec.reduction}")
    return row_mask


def build_support_condition_tensors(
    proj: np.ndarray,
    spec: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Build acquisition-support conditioning tensors from one projection volume.

    Returns numpy arrays with shape `(V,H,W)`:
    - `mask`: hard support mask
    - `distance`: signed or inside-only distance to nearest support boundary
    - `boundary_weight`: positive weight map emphasizing rows near the boundary
    """

    arr = np.asarray(proj, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Projection must have shape (V,H,W), got {arr.shape}")

    cfg = _normalize_spec(spec)
    views, height, width = arr.shape
    rows = np.arange(height, dtype=np.float32)
    row_mask = _row_mask_from_projection(arr, cfg)

    mask_rows = row_mask.astype(np.float32, copy=False)
    distance_rows = np.zeros((views, height), dtype=np.float32)
    boundary_rows = np.zeros((views, height), dtype=np.float32)

    for view_idx in range(views):
        active = np.flatnonzero(row_mask[view_idx])
        if active.size == 0:
            continue

        start = int(active[0])
        end = int(active[-1])
        inside = (rows >= float(start)) & (rows <= float(end))

        d_lower = rows - float(start)
        d_upper = float(end) - rows
        nearest = np.minimum(d_lower, d_upper)

        if cfg.distance_normalizer == "support_half_height":
            scale = max((end - start + 1) * 0.5, 1.0)
        elif cfg.distance_normalizer == "height":
            scale = max(float(height - 1), 1.0)
        else:
            raise ValueError(f"Unsupported distance_normalizer: {cfg.distance_normalizer}")

        if cfg.signed_distance:
            signed = np.where(inside, nearest, -np.minimum(np.abs(d_lower), np.abs(d_upper)))
            distance_rows[view_idx] = signed / float(scale)
        else:
            distance_rows[view_idx] = np.where(inside, nearest / float(scale), 0.0)

        if cfg.boundary_band > 0 and cfg.boundary_emphasis > 0.0:
            boundary_rows[view_idx] = (
                inside.astype(np.float32) * (nearest <= float(cfg.boundary_band)).astype(np.float32)
            ) * float(cfg.boundary_emphasis)

    mask = np.broadcast_to(mask_rows[:, :, None], (views, height, width)).astype(np.float32, copy=True)
    distance = np.broadcast_to(distance_rows[:, :, None], (views, height, width)).astype(np.float32, copy=True)
    boundary_weight = np.broadcast_to(boundary_rows[:, :, None], (views, height, width)).astype(
        np.float32, copy=True
    )

    out = {
        "mask": mask,
        "boundary_weight": boundary_weight,
    }
    if cfg.include_distance:
        out["distance"] = distance
    return out
