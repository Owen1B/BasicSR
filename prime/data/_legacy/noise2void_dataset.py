"""Noise2Void dataset for self-supervised denoising.

Based on: "Noise2Void - Learning Denoising from Single Noisy Images" (CVPR 2019)
Reference implementation: /home/owen/code/noise2void_ref/dataset.py

Key idea: Mask random pixels and predict them from surrounding neighbors (blind-spot network).
"""
from __future__ import annotations

import numpy as np
from typing import Dict

from basicsr.data.spect_dat_paired_dataset import SPECTDatPairedDataset, _ensure_hwc
from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils.img_util import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def _generate_n2v_mask(
    img: np.ndarray,
    ratio: float = 0.9,
    window_size: tuple = (5, 5)
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Noise2Void mask: randomly mask pixels and replace with random neighbor.

    Reference: noise2void_ref/dataset.py lines 89-118

    Args:
        img: input image, shape (H, W, C), float32
        ratio: fraction of pixels to keep unmask (default 0.9 = mask 10%)
        window_size: neighborhood window for replacement (height, width)

    Returns:
        masked_img: image with masked pixels replaced by neighbors
        mask: binary mask, 1=keep, 0=masked (for loss calculation)
    """
    if img.ndim != 3:
        raise ValueError(f'Expected 3D image (H,W,C), got shape {img.shape}')

    H, W, C = img.shape
    num_mask_per_channel = int(H * W * (1.0 - ratio))

    mask = np.ones((H, W, C), dtype=np.float32)
    output = img.copy()

    for ich in range(C):
        # Randomly select pixels to mask
        idy_msk = np.random.randint(0, H, num_mask_per_channel)
        idx_msk = np.random.randint(0, W, num_mask_per_channel)

        # For each masked pixel, select a random neighbor within window
        # Offset range: [-window//2 + window%2, window//2 + window%2)
        idy_neigh = np.random.randint(
            -window_size[0] // 2 + window_size[0] % 2,
            window_size[0] // 2 + window_size[0] % 2,
            num_mask_per_channel
        )
        idx_neigh = np.random.randint(
            -window_size[1] // 2 + window_size[1] % 2,
            window_size[1] // 2 + window_size[1] % 2,
            num_mask_per_channel
        )

        # Compute neighbor coordinates with wrapping (periodic boundary)
        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        # Wrap around boundaries
        idy_msk_neigh = (idy_msk_neigh + (idy_msk_neigh < 0) * H - (idy_msk_neigh >= H) * H)
        idx_msk_neigh = (idx_msk_neigh + (idx_msk_neigh < 0) * W - (idx_msk_neigh >= W) * W)

        # Create index tuples
        id_msk = (idy_msk, idx_msk, ich)
        id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

        # Replace masked pixels with neighbor values
        output[id_msk] = img[id_msk_neigh]
        mask[id_msk] = 0.0

    return output, mask


@DATASET_REGISTRY.register()
class Noise2VoidDataset(SPECTDatPairedDataset):
    """Noise2Void dataset for SPECT denoising.

    Inherits from SPECTDatPairedDataset and adds N2V mask generation.
    Uses single noisy images (mode='single') and trains network to predict
    masked pixels from their neighbors.

    Additional options in opt dict:
        n2v_ratio (float): fraction of pixels to keep unmask (default 0.9)
        n2v_window_size (tuple): neighborhood window size (default (5, 5))

    Returns dict with keys:
        lq: masked input image (CHW tensor, normalized)
        gt: original noisy image (CHW tensor, normalized)
        mask: binary mask (CHW tensor), 1=keep, 0=masked (for loss calculation)
        lq_path, gt_path: file paths
    """

    def __init__(self, opt: Dict):
        # Force mode to 'single' for N2V (only uses noisy images)
        # We'll read from dataroot_gt and use it as both input and target
        if 'mode' not in opt:
            opt['mode'] = 'paired'  # Use paired mode but will generate mask

        super().__init__(opt)

        # N2V specific parameters
        self.n2v_ratio = float(opt.get('n2v_ratio', 0.9))
        window_size = opt.get('n2v_window_size', [5, 5])
        self.n2v_window_size = tuple(window_size) if isinstance(window_size, list) else window_size

        if self.n2v_ratio <= 0 or self.n2v_ratio >= 1:
            raise ValueError(f'n2v_ratio must be in (0, 1), got {self.n2v_ratio}')

    def __getitem__(self, index: int) -> Dict:
        # We need to intercept the data flow before img2tensor
        # so we can apply N2V mask in numpy domain
        
        # Get index mapping
        if self.view_mode == 'split1':
            base_index = index // 2
            view_id = index % 2
        else:
            base_index = index
            view_id = None
        
        basename = self._basenames[base_index]
        
        # Load data
        lq_img, gt_img, lq_path, gt_path = self._load_pair(basename)
        
        # Choose view
        if self.view_mode == 'split1':
            lq_img = _ensure_hwc(lq_img[:, :, view_id])
            gt_img = _ensure_hwc(gt_img[:, :, view_id])
        else:
            lq_img = lq_img.astype(np.float32, copy=False)
            gt_img = gt_img.astype(np.float32, copy=False)
        
        # Training augmentation / patch crop
        if self.phase == 'train':
            if self.gt_size is not None:
                gt_size = int(self.gt_size)
                scale = int(self.opt.get('scale', 1))
                gt_img, lq_img = paired_random_crop(gt_img, lq_img, gt_size, scale, gt_path)
            if self.use_hflip or self.use_rot:
                gt_img, lq_img = augment([gt_img, lq_img], self.use_hflip, self.use_rot)
        
        # Normalization (we use gt as noisy image for N2V)
        lq_img, gt_img = self._apply_norm(lq_img, gt_img)
        
        # N2V: Generate mask on the normalized gt_img (noisy image)
        masked_img, mask = _generate_n2v_mask(
            gt_img,
            ratio=self.n2v_ratio,
            window_size=self.n2v_window_size
        )
        
        # Convert to tensors: HWC -> CHW
        lq, gt = img2tensor([masked_img, gt_img], bgr2rgb=False, float32=True)
        mask_tensor = img2tensor([mask], bgr2rgb=False, float32=True)[0]
        
        # Return masked image as input (lq), original as target (gt), plus mask
        out = {
            'lq': lq,  # Masked input for network
            'gt': gt,  # Original noisy image (target)
            'mask': mask_tensor,  # Binary mask (1=keep, 0=masked for loss)
            'lq_path': gt_path,  # Same source (we only use gt for N2V)
            'gt_path': gt_path
        }
        
        if self.view_mode == 'split1':
            out['view_id'] = view_id
        
        return out

    def __len__(self):
        return super().__len__()

