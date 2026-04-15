"""Dataset for loading raw SPECT projection sequences (flat `.dat`).

This dataset is designed for validation/inference with projection volumes.
The binary layout is configured by YAML (`views/height/width/dtype`) and is
interpreted as a contiguous array with shape `(views, height, width)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY


def load_projection_dat(path: str, views: int, h: int, w: int, dtype: str = 'uint16') -> np.ndarray:
    """Load one projection sequence from flat `.dat` file.

    Args:
        path: input file path.
        views/h/w: expected tensor shape `(views, h, w)`.
        dtype: one of `uint16 | int16 | float32`.
    """
    dtype_map = {
        'uint16': np.uint16,
        'int16': np.int16,
        'float32': np.float32,
    }
    key = str(dtype).lower().strip()
    if key not in dtype_map:
        raise ValueError(f'Unsupported dtype={dtype}. Use one of {sorted(dtype_map.keys())}.')

    arr = np.fromfile(path, dtype=dtype_map[key])
    expected = views * h * w
    if arr.size != expected:
        raise ValueError(
            f'Invalid projection size: {path}. '
            f'Expected {expected} values for shape ({views},{h},{w}), got {arr.size}.'
        )
    return arr.reshape(views, h, w).astype(np.float32, copy=False)


@DATASET_REGISTRY.register()
class SPECTProjectionEvalDataset(data.Dataset):
    """
    Dataset for loading raw SPECT projection sequences.

    Loads complete projection sequences from flat `.dat` files.
    Designed for validation/inference with projection data.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot (str): Root directory containing projection files.
            pattern (str): Glob pattern to find projection files. Default: '**/ProjectionImage*.dat'
            height (int): Projection height. Default: 128
            width (int): Projection width. Default: 128
            views (int): Number of views. Default: 60
            dtype (str): 'uint16' | 'int16' | 'float32'. Default: 'uint16'
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
        self.dtype = str(opt.get('dtype', 'uint16')).lower().strip()

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
            - 'proj_sequence': numpy array (60, 128, 128) - alias for lq (for SPECTModel validation)
        """
        rel_path = self.paths[index]
        full_path = Path(self.dataroot) / rel_path

        # Load projection sequence (60, 128, 128)
        proj = load_projection_dat(str(full_path), views=self.views, h=self.height, w=self.width, dtype=self.dtype)

        # For validation with SPECTModel, return the full sequence as numpy array.
        return {
            'lq': proj,  # Full projection sequence (60, 128, 128) as numpy array
            'gt': proj.copy(),  # Target: same (for self-supervised)
            'lq_path': str(full_path),  # Full path to original projection file
            'gt_path': str(full_path),
            'proj_sequence': proj,  # Alias for easier access in SPECTModel
        }
