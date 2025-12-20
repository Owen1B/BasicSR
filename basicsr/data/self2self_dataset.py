"""Self2Self dataset for self-supervised denoising.

Based on: "Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image" (CVPR 2020)
Reference implementation (TensorFlow): https://github.com/scut-mingqinchen/Self2Self

Key idea: Use dropout as a stochastic masking mechanism, train network to predict
itself from randomly dropped pixels. Inference uses Monte Carlo dropout averaging.

Note: This is a PyTorch re-implementation of the original TensorFlow code.
"""
from __future__ import annotations

from typing import Dict

from basicsr.data.spect_dat_paired_dataset import SPECTDatPairedDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Self2SelfDataset(SPECTDatPairedDataset):
    """Self2Self dataset for SPECT denoising.

    Inherits from SPECTDatPairedDataset. S2S trains a network with dropout
    to predict the noisy image from itself (self-supervised).

    The dropout mechanism (implemented in UNetDropout architecture and
    Self2SelfModel) serves as the masking strategy. During training,
    dropout randomly masks pixels, and the network learns to predict
    the full image from the partial information.

    During inference, multiple forward passes with different dropout masks
    are averaged (Monte Carlo Dropout) to get the final denoised result.

    Additional options in opt dict:
        s2s_augment (bool): enable random flip augmentation (default True)
    """

    def __init__(self, opt: Dict):
        # S2S uses single noisy images
        if 'mode' not in opt:
            opt['mode'] = 'paired'

        super().__init__(opt)

        # S2S specific parameters
        self.s2s_augment = bool(opt.get('s2s_augment', True))

    def __getitem__(self, index: int) -> Dict:
        """Returns single noisy image for S2S training.

        The network will use dropout during training to create masked versions.
        """
        base_sample = super().__getitem__(index)

        # For S2S, both input and target are the same noisy image
        # The dropout in the network provides the masking
        return {
            'lq': base_sample['gt'],  # Input = noisy image
            'gt': base_sample['gt'],  # Target = same noisy image
            'lq_path': base_sample.get('gt_path', ''),
            'gt_path': base_sample.get('gt_path', '')
        }

    def __len__(self):
        return super().__len__()

