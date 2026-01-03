from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VolumeShape:
    """用 (z,y,x) 表示 3D 体数据尺寸。"""

    z: int
    y: int
    x: int

    @property
    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.z, self.y, self.x)

    @property
    def numel(self) -> int:
        return int(self.z * self.y * self.x)


def load_dat_file(file_path: Path, shape: Tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    data = np.fromfile(file_path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"数据大小不匹配: 期望 {expected}, 实际 {data.size}，文件: {file_path}")
    return data.reshape(shape)


def load_projection_i16(path: Path, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    """读取投影 (views,h,w)，int16 -> float32（用于后续推理/统计）。"""
    path = Path(path)
    x = np.fromfile(str(path), dtype=np.int16)
    expected = int(views * h * w)
    if x.size != expected:
        raise ValueError(f"投影大小不匹配: {path}，期望 {expected}，实际 {x.size}")
    return x.reshape(views, h, w).astype(np.float32, copy=False)


def load_projection_f32(path: Path, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    """读取投影 (views,h,w)，float32（用于保留 denoised 的小数部分）。"""
    path = Path(path)
    x = np.fromfile(str(path), dtype=np.float32)
    expected = int(views * h * w)
    if x.size != expected:
        raise ValueError(f"投影大小不匹配(float32): {path}，期望 {expected}，实际 {x.size}")
    return x.reshape(views, h, w).astype(np.float32, copy=False)


def save_projection_f32(path: Path, proj: np.ndarray) -> None:
    """保存投影为 float32 原始 dat（views,h,w）。

    约定：投影在 count domain，应满足非负。这里会：
    - 将 NaN/Inf 转为 0
    - clip 到 >= 0
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(proj, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    x.tofile(str(path))


def save_projection_i16_round_clip(path: Path, proj: np.ndarray) -> None:
    """保存投影为 int16，使用 round() 并 clip 到非负（与我们 counts 统计口径一致）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.round(proj.astype(np.float64, copy=False))
    x = np.clip(x, 0.0, None)
    x.astype(np.int16).tofile(str(path))


def load_recon_f32(path: Path, shape: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    return load_dat_file(path, shape=shape, dtype=np.float32)


def load_volume_f32_candidates(path: Path, candidates: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """按候选 shape 自动识别并 reshape（用于衰减图等多种尺寸）。"""
    path = Path(path)
    x = np.fromfile(str(path), dtype=np.float32)
    n = int(x.size)
    for shape in candidates:
        if int(np.prod(shape)) == n:
            return x.reshape(shape)
    raise ValueError(f"体数据大小不匹配: {path}，元素数={n}，无法匹配 candidates={list(candidates)}")


def load_atten_f32(path: Path) -> np.ndarray:
    """加载衰减体（float32，尝试常见尺寸），输出统一为 [z,y,x]。"""
    # 常见：128^3 或 256x256x32
    return load_volume_f32_candidates(path, candidates=[(128, 128, 128), (32, 256, 256)])


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_existing(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.exists():
            out.append(p)
    return out











