"""
Dataset for loading raw SPECT projection sequences (ProjectionImage*.dat files).

This dataset is designed for validation/inference with 3D projection data.
It loads complete projection sequences (60 views, 128x128) from ProjectionImage*.dat files.
"""

from __future__ import annotations

from os import path as osp
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils import data as data

from basicsr.utils import scandir
from basicsr.utils.registry import DATASET_REGISTRY


def load_projection_u16(path: str, views: int = 60, h: int = 128, w: int = 128) -> np.ndarray:
    """Load projection sequence from uint16 file."""
    arr = np.fromfile(path, dtype=np.uint16)
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(f'Invalid projection size: {path}. Expected {expected} uint16 values, got {arr.size}.')
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


@DATASET_REGISTRY.register()
class SPECTProjectionDataset(data.Dataset):
    """
    Dataset for loading raw SPECT projection sequences.

    Loads complete projection sequences (60 views, 128x128) from ProjectionImage*.dat files.
    Designed for validation/inference with 3D data.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot (str): Root directory containing projection files.
            pattern (str): Glob pattern to find projection files. Default: '**/ProjectionImage*.dat'
            height (int): Projection height. Default: 128
            width (int): Projection width. Default: 128
            views (int): Number of views. Default: 60
            start_idx (int): Start index for file list. Default: 0
            end_idx (int): End index for file list. Default: None (use all)
    """

    def __init__(self, opt: Dict):
        super().__init__()
        self.opt = opt

        self.dataroot = opt['dataroot']
        self.pattern = opt.get('pattern', '**/ProjectionImage*.dat')
        self.height = int(opt.get('height', 128))
        self.width = int(opt.get('width', 128))
        self.views = int(opt.get('views', 60))

        # Build file list
        self.paths = self._scan_projection_files()

        # Slice range
        start_idx = int(opt.get('start_idx', 0))
        end_idx = opt.get('end_idx', None)
        if end_idx is not None:
            end_idx = int(end_idx)
        self.paths = self.paths[start_idx:end_idx]

        if len(self.paths) == 0:
            raise FileNotFoundError(f'No projection files found in {self.dataroot} with pattern {self.pattern}')

    def _scan_projection_files(self) -> List[str]:
        """Scan for projection files matching the pattern."""
        import glob

        dataroot_path = Path(self.dataroot)
        if not dataroot_path.exists():
            raise FileNotFoundError(f'Data root not found: {self.dataroot}')

        # Use glob to find files
        pattern_path = dataroot_path / self.pattern
        files = sorted(glob.glob(str(pattern_path), recursive=True))

        # Convert to relative paths for consistency
        return [str(Path(f).relative_to(dataroot_path)) if Path(f).is_relative_to(dataroot_path) else f
                for f in files]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict:
        """Load a projection sequence.

        Returns:
            Dict with:
            - 'lq': numpy array (60, 128, 128) - full projection sequence for GIF generation
            - 'gt': numpy array (60, 128, 128) - same as lq (for self-supervised)
            - 'lq_path': str - full path to projection file
            - 'gt_path': str - same as lq_path
            - 'proj_sequence': numpy array (60, 128, 128) - alias for lq (for SPECT3DModel)
        """
        rel_path = self.paths[index]
        full_path = Path(self.dataroot) / rel_path

        # Load projection sequence (60, 128, 128)
        proj = load_projection_u16(str(full_path), views=self.views, h=self.height, w=self.width)

        # For validation with SPECT3DModel, we return the full sequence as numpy array
        # The model's _generate_validation_gif will use this directly
        return {
            'lq': proj,  # Full projection sequence (60, 128, 128) as numpy array
            'gt': proj.copy(),  # Target: same (for self-supervised)
            'lq_path': str(full_path),  # Full path to original projection file
            'gt_path': str(full_path),
            'proj_sequence': proj,  # Alias for easier access in SPECT3DModel
        }

