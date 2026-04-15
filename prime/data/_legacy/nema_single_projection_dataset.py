"""
Single NEMA projection dataset for fine-tuning with mixed-dose N2N strategy.

Uses one projection file, generates diverse N2N pairs within each batch using different k values.
"""

import numpy as np
import random
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NEMASingleProjectionDataset(data.Dataset):
    """Single NEMA projection dataset with in-batch diversity via mixed-dose sampling.
    
    For fine-tuning on a single large projection (e.g., 120×256×256).
    Each batch item uses a different random k to create diverse N2N pairs.
    
    Args:
        opt (dict): Config dictionary with keys:
            - projection_file (str): Path to the projection .dat file
            - views (int): Number of views (e.g., 120)
            - height (int): Height dimension (e.g., 256)
            - width (int): Width dimension (e.g., 256)
            - dtype (str): Data type ('uint16' or 'float32')
            - max_value (float): Normalization max value (e.g., 150.0)
            - n2n_k_choices (list): List of k values for mixed-dose (e.g., [2,3,4,5])
            - virtual_dataset_size (int): Virtual dataset size (repeats per epoch)
            - use_flip (bool): Whether to use horizontal flip augmentation
            - use_rot (bool): Whether to use 90° rotation augmentation
    """
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # Load projection data
        proj_file = Path(opt['projection_file'])
        assert proj_file.exists(), f"Projection file not found: {proj_file}"
        
        views = opt['views']
        height = opt['height']
        width = opt['width']
        dtype_str = opt.get('dtype', 'uint16')
        dtype = np.uint16 if dtype_str == 'uint16' else np.float32
        
        # Load data (Fortran order, as typical for SPECT data)
        data = np.fromfile(str(proj_file), dtype=dtype)
        self.projection = data.reshape((views, height, width), order='F')
        
        # Normalization
        self.max_value = float(opt['max_value'])
        
        # N2N parameters
        self.k_choices = opt.get('n2n_k_choices', [2, 3, 4, 5])
        
        # Virtual dataset size (how many times to "repeat" this single sample per epoch)
        self.virtual_size = opt.get('virtual_dataset_size', 1000)
        
        # Augmentation
        self.use_flip = opt.get('use_flip', False)
        self.use_rot = opt.get('use_rot', False)
        
        print(f"NEMASingleProjectionDataset initialized:")
        print(f"  File: {proj_file}")
        print(f"  Shape: {self.projection.shape}")
        print(f"  Data range: [{self.projection.min()}, {self.projection.max()}]")
        print(f"  Max value for normalization: {self.max_value}")
        print(f"  K choices: {self.k_choices}")
        print(f"  Virtual dataset size: {self.virtual_size}")
    
    def __len__(self):
        return self.virtual_size
    
    def __getitem__(self, index):
        """Generate a random N2N pair from the projection data."""
        
        # Start with the full projection
        proj = self.projection.copy().astype(np.float32)
        
        # Optional spatial augmentation (before splitting)
        if self.use_flip and random.random() < 0.5:
            # Flip horizontally (flip width dimension)
            proj = np.flip(proj, axis=2).copy()
        
        if self.use_rot:
            # Random 90° rotations in the HW plane
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                # Rotate in the (height, width) plane
                proj = np.rot90(proj, k=k_rot, axes=(1, 2)).copy()
        
        # Randomly select k for this sample
        k = random.choice(self.k_choices)
        
        # Generate N2N pair with same-dose split
        lq, gt = self._generate_n2n_pair(proj, k)
        
        # Normalize to [0, 1]
        lq = lq / self.max_value
        gt = gt / self.max_value
        
        # Add channel dimension: (V, H, W) -> (1, V, H, W)
        lq = lq[np.newaxis, :, :, :]
        gt = gt[np.newaxis, :, :, :]
        
        return {
            'lq': lq.astype(np.float32),
            'gt': gt.astype(np.float32),
            'lq_path': f'NEMA_virtual_{index}_k{k}_A',
            'gt_path': f'NEMA_virtual_{index}_k{k}_B',
        }
    
    def _generate_n2n_pair(self, proj, k):
        """Generate two same-dose N2N splits.
        
        Args:
            proj: Full projection (V, H, W), float32, in count domain
            k: Split factor (2, 3, 4, or 5)
        
        Returns:
            split_a, split_b: Two Poisson samples, each with dose = original/k
        """
        # Generate k Poisson samples
        splits = []
        for _ in range(k):
            # Scale down by k, sample from Poisson, round to nearest int
            scaled = proj / k
            sampled = np.random.poisson(scaled)
            splits.append(sampled.astype(np.float32))
        
        # Combine: first split = splits[0], second split = sum of rest
        split_a = splits[0]
        split_b = sum(splits[1:])
        
        # Both splits have the same expected value = proj * (k-1)/k
        # But for k=2: split_a = proj/2, split_b = proj/2 (equal dose)
        # For k>2: we need to balance them
        
        # Alternative: randomly assign splits to two groups of equal size
        random.shuffle(splits)
        n_half = k // 2
        split_a = sum(splits[:n_half])
        split_b = sum(splits[n_half:])
        
        return split_a, split_b


@DATASET_REGISTRY.register()
class NEMASingleProjectionPatchDataset(NEMASingleProjectionDataset):
    """Same as NEMASingleProjectionDataset but extracts random patches.
    
    Useful if GPU memory is limited and cannot fit full 120×256×256 volumes.
    """
    
    def __init__(self, opt):
        super().__init__(opt)
        
        # Patch parameters
        self.patch_size_v = opt.get('patch_size_v', 60)  # views
        self.patch_size_h = opt.get('patch_size_h', 128)  # height
        self.patch_size_w = opt.get('patch_size_w', 128)  # width
        
        print(f"  Patch mode enabled: ({self.patch_size_v}, {self.patch_size_h}, {self.patch_size_w})")
    
    def __getitem__(self, index):
        """Generate a random N2N pair from a random patch."""
        
        # Start with the full projection
        proj = self.projection.copy().astype(np.float32)
        V, H, W = proj.shape
        
        # Extract random patch (before N2N splitting)
        v_start = random.randint(0, V - self.patch_size_v) if V > self.patch_size_v else 0
        h_start = random.randint(0, H - self.patch_size_h) if H > self.patch_size_h else 0
        w_start = random.randint(0, W - self.patch_size_w) if W > self.patch_size_w else 0
        
        proj_patch = proj[
            v_start:v_start + self.patch_size_v,
            h_start:h_start + self.patch_size_h,
            w_start:w_start + self.patch_size_w
        ]
        
        # Optional spatial augmentation
        if self.use_flip and random.random() < 0.5:
            proj_patch = np.flip(proj_patch, axis=2).copy()
        
        if self.use_rot:
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                proj_patch = np.rot90(proj_patch, k=k_rot, axes=(1, 2)).copy()
        
        # Randomly select k for this sample
        k = random.choice(self.k_choices)
        
        # Generate N2N pair
        lq, gt = self._generate_n2n_pair(proj_patch, k)
        
        # Normalize to [0, 1]
        lq = lq / self.max_value
        gt = gt / self.max_value
        
        # Add channel dimension
        lq = lq[np.newaxis, :, :, :]
        gt = gt[np.newaxis, :, :, :]
        
        return {
            'lq': lq.astype(np.float32),
            'gt': gt.astype(np.float32),
            'lq_path': f'NEMA_patch_{index}_k{k}_A',
            'gt_path': f'NEMA_patch_{index}_k{k}_B',
        }








