"""Noise2Void model for self-supervised denoising.

Based on: "Noise2Void - Learning Denoising from Single Noisy Images" (CVPR 2019)

This implementation supports two modes:

1. **Mask-based N2V** (original approach):
   - Use standard network (e.g., UNetRes)
   - Mask random pixels in input and replace with neighbors
   - Compute loss only at masked positions
   - The mask is provided by Noise2VoidDataset

2. **Blind-spot N2V** (recommended):
   - Use blind-spot network (e.g., DBSNl)
   - No data masking needed (network architecture ensures blind-spot property)
   - Compute loss on all pixels
   - Use `blind_spot_mode: true` in config

The blind-spot mode is preferred as it:
- Uses all pixels for training (more efficient)
- Has stronger theoretical guarantees
- Requires less data preprocessing
"""
from collections import OrderedDict
import torch

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger


@MODEL_REGISTRY.register()
class Noise2VoidModel(SRModel):
    """Noise2Void training model.

    Supports both mask-based and blind-spot network modes.

    Expected data dict keys (mask-based mode):
        lq: masked input image
        gt: original noisy image (target)
        mask: binary mask (1=keep unmask, 0=masked positions for loss)

    Expected data dict keys (blind-spot mode):
        lq: noisy input image (same as gt)
        gt: noisy image (target = input for self-supervised)

    Config options:
        train:
            blind_spot_mode: true  # Use blind-spot network, no mask needed
    """

    def __init__(self, opt):
        super(Noise2VoidModel, self).__init__(opt)

        # Check if using blind-spot mode
        train_opt = opt.get('train', {})
        self.blind_spot_mode = bool(train_opt.get('blind_spot_mode', False))

        logger = get_root_logger()
        if self.blind_spot_mode:
            logger.info('Noise2Void: Using blind-spot mode (DBSNl-style network)')
        else:
            logger.info('Noise2Void: Using mask-based mode (standard network + data masking)')

    def feed_data(self, data):
        """Feed data to the model."""
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if not self.blind_spot_mode:
            # Mask-based mode: need mask from dataset
            if 'mask' in data:
                self.mask = data['mask'].to(self.device)
            else:
                # If no mask provided, use all pixels (fallback)
                self.mask = torch.ones_like(self.gt)

    def optimize_parameters(self, current_iter):
        """Optimize parameters."""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if self.blind_spot_mode:
            # Blind-spot mode: use all pixels for loss
            # The network architecture ensures it can't see the center pixel
            self._compute_loss_all_pixels(loss_dict)
        else:
            # Mask-based mode: only compute loss at masked positions
            self._compute_loss_masked(loss_dict, current_iter)

        l_total = loss_dict.get('l_pix', 0)
        if self.cri_perceptual:
            l_total = l_total + loss_dict.get('l_percep', 0) + loss_dict.get('l_style', 0)

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _compute_loss_all_pixels(self, loss_dict):
        """Compute loss on all pixels (for blind-spot mode)."""
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                loss_dict['l_style'] = l_style

    def _compute_loss_masked(self, loss_dict, current_iter):
        """Compute loss only at masked positions (for mask-based mode)."""
        # The mask from dataset: 1=keep (unmask), 0=masked (for loss)
        # We want to compute loss where mask==0, so use (1-mask) as weight
        loss_mask = 1.0 - self.mask

        # Debug: Check mask statistics (only first few iterations)
        if current_iter <= 10:
            num_masked = (loss_mask > 0).sum().item()
            total_pixels = loss_mask.numel()
            logger = get_root_logger()
            logger.info(f'N2V Debug iter {current_iter}: masked pixels={num_masked}/{total_pixels} '
                       f'({100*num_masked/total_pixels:.1f}%), mask_sum={loss_mask.sum().item():.1f}')

        # Pixel loss with mask weighting
        if self.cri_pix:
            # Check if this is PoissonNLL loss (needs special handling)
            if 'PoissonNLL' in self.cri_pix.__class__.__name__:
                # For Poisson loss, call with weight parameter
                l_pix = self.cri_pix(self.output, self.gt, weight=loss_mask)
                loss_dict['l_pix'] = l_pix
            else:
                # For other losses, compute element-wise and apply mask
                diff = self.output - self.gt

                # Compute element-wise loss based on loss type
                if 'Charbonnier' in self.cri_pix.__class__.__name__:
                    eps = getattr(self.cri_pix, 'eps', 1e-12)
                    elementwise_loss = torch.sqrt(diff ** 2 + eps)
                elif 'L1' in self.cri_pix.__class__.__name__:
                    elementwise_loss = torch.abs(diff)
                elif 'MSE' in self.cri_pix.__class__.__name__:
                    elementwise_loss = diff ** 2
                else:
                    # Generic case: MSE
                    elementwise_loss = diff ** 2

                # Apply mask: only compute loss at masked positions
                masked_loss = elementwise_loss * loss_mask

                # Reduction
                reduction = getattr(self.cri_pix, 'reduction', 'mean')
                if reduction == 'mean':
                    # Average over masked pixels only
                    l_pix = masked_loss.sum() / (loss_mask.sum() + 1e-8)
                elif reduction == 'sum':
                    l_pix = masked_loss.sum()
                else:  # 'none'
                    l_pix = masked_loss.mean()  # Fallback to mean

                # Apply loss weight
                if hasattr(self.cri_pix, 'loss_weight'):
                    l_pix = l_pix * self.cri_pix.loss_weight

                loss_dict['l_pix'] = l_pix

        # Perceptual loss (if enabled) - typically not used for N2V
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                loss_dict['l_style'] = l_style


@MODEL_REGISTRY.register()
class Noise2VoidBlindSpotModel(SRModel):
    """Simplified Noise2Void model for blind-spot networks only.

    This is a cleaner implementation that assumes you're using a blind-spot
    network (like DBSNl). No data masking is needed.

    The key insight is that with a blind-spot network:
    - The network output at position (i,j) does NOT see input at (i,j)
    - Therefore, we can directly train: output = network(noisy), loss = L(output, noisy)
    - The network learns to denoise because it can't just copy the input

    This is simpler and more efficient than mask-based N2V.

    Example config:
        model_type: Noise2VoidBlindSpotModel
        network_g:
            type: DBSNl
            in_nc: 2
            out_nc: 2
            base_ch: 128
            num_module: 9
    """

    def feed_data(self, data):
        """Feed data to the model.

        For blind-spot N2V, lq and gt are the same noisy image.
        """
        self.lq = data['lq'].to(self.device)
        # For self-supervised: gt = lq (same noisy image)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        else:
            self.gt = self.lq

    def optimize_parameters(self, current_iter):
        """Standard training: output = network(noisy), loss = L(output, noisy)."""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Pixel loss on all pixels
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (optional)
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
