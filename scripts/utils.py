#!/usr/bin/env python3
"""
兼容层：历史脚本曾直接从 `scripts/utils.py` 引用工具函数。

为了消除重复代码，通用逻辑已抽到 `prime/pipeline/` 下；
这里保留同名导出，避免脚本间导入路径变动带来的维护成本。
"""

from __future__ import annotations

import numpy as np

from prime.pipeline.io import load_dat_file, load_projection_i16 as load_projection_dat, load_recon_f32 as load_reconstruction
from prime.pipeline.viz import (
    Log1pNormalize,
    calculate_vmax_percentile,
    compute_mip,
    create_subplot_with_colorbar,
    get_colormap,
    rotate_volume_around_x,
)

__all__ = [
    "get_colormap",
    "Log1pNormalize",
    "create_subplot_with_colorbar",
    "rotate_volume_around_x",
    "compute_mip",
    "load_dat_file",
    "load_projection_dat",
    "load_reconstruction",
    "calculate_vmax_percentile",
    "np",
]














