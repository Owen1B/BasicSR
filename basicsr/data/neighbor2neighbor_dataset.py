"""Neighbor2Neighbor dataset for self-supervised denoising.

Based on: "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images" (CVPR 2021)
Reference implementation: https://github.com/TaoHuang2018/Neighbor2Neighbor

Key idea: Spatially downsample image into sub-images, train on complementary pixel pairs from 2x2 blocks.
"""
from __future__ import annotations

from typing import Dict

from basicsr.data.spect_dat_paired_dataset import SPECTDatPairedDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Neighbor2NeighborDataset(SPECTDatPairedDataset):
    """Neighbor2Neighbor dataset for SPECT denoising.

    Inherits from SPECTDatPairedDataset. The main N2B logic (mask pair generation,
    space-to-depth, subimage extraction) is implemented in Neighbor2NeighborModel
    since it requires dynamic per-batch processing.

    This dataset simply provides single noisy images (similar to N2N/N2V).
    The model will handle the spatial downsampling and mask generation.

    Usage:
        mode: 'paired'  # Use single noisy images from dataroot_gt
        dataroot_gt: path to noisy images
        dataroot_lq: same as dataroot_gt (or will be ignored in model)
    """

    def __init__(self, opt: Dict):
        # N2B uses single noisy images
        # Ensure mode is set (will use paired mode but ignore LQ in model)
        if 'mode' not in opt:
            opt['mode'] = 'paired'

        super().__init__(opt)

    def __getitem__(self, index: int) -> Dict:
        """Returns single noisy image (gt) for N2B training.

        The model will perform spatial downsampling and create sub-images online.
        """
        base_sample = super().__getitem__(index)

        # For N2B, we use the GT (noisy) image as input
        # The model will generate mask pairs and sub-images dynamically
        return {
            'lq': base_sample['gt'],  # Input = noisy image
            'gt': base_sample['gt'],  # Target = same noisy image
            'lq_path': base_sample.get('gt_path', ''),
            'gt_path': base_sample.get('gt_path', '')
        }

    def __len__(self):
        return super().__len__()

