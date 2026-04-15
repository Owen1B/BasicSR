"""Self2Self model for self-supervised denoising.

Based on: "Self2Self with Dropout: Learning Self-Supervised Denoising from Single Image" (CVPR 2020)
Reference implementation: https://github.com/scut-mingqinchen/Self2Self (demo_denoising.py)

Key features:
- Training: Dropout enabled, network predicts noisy image from itself
- Inference: Monte Carlo Dropout - average over N_PREDICTION forward passes
"""
from collections import OrderedDict
import torch

from prime.models._legacy.sr_model_ext import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger


@MODEL_REGISTRY.register()
class Self2SelfModel(SRModel):
    """Self2Self training model with Monte Carlo Dropout inference.

    During training, dropout is enabled and the network learns to predict
    the noisy image from a dropout-masked version of itself.

    During inference, multiple forward passes (N_PREDICTION times) are
    performed with different dropout masks, and the results are averaged
    to get the final denoised image.

    Config options (in train section):
        n_prediction (int): number of Monte Carlo samples during inference (default 100)
    """

    def __init__(self, opt):
        super(Self2SelfModel, self).__init__(opt)

        # S2S specific parameters
        val_opt = opt.get('val', {})
        self.n_prediction = int(val_opt.get('n_prediction', 100))

        logger = get_root_logger()
        logger.info(f'Self2Self: n_prediction={self.n_prediction} (Monte Carlo Dropout)')

    def optimize_parameters(self, current_iter):
        """Training step with dropout enabled.

        The network (with dropout) predicts the noisy image from itself.
        This is the standard S2S training procedure.
        """
        self.optimizer_g.zero_grad()

        # Ensure dropout is enabled during training
        self.net_g.train()

        # Forward pass: noisy -> network with dropout -> noisy
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Pixel loss: predict noisy from noisy
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # Perceptual loss (if enabled)
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

    def test(self):
        """Test/inference with Monte Carlo Dropout averaging.

        Reference: self2self_ref/demo_denoising.py lines 50-55
        Performs N_PREDICTION forward passes with dropout enabled,
        then averages the results.
        """
        if hasattr(self, 'net_g_ema'):
            net = self.net_g_ema
        else:
            net = self.net_g

        # Enable dropout during inference (Monte Carlo Dropout)
        net.train()  # This keeps dropout active

        # If using UNetDropout, explicitly set dropout mode
        if hasattr(net, 'set_dropout_mode'):
            net.set_dropout_mode(training=True)

        # Accumulate predictions
        with torch.no_grad():
            outputs = []
            for _ in range(self.n_prediction):
                output = net(self.lq)
                outputs.append(output)

            # Average over all predictions
            self.output = torch.stack(outputs, dim=0).mean(dim=0)

        # Restore network to eval mode after inference
        if not self.training:
            net.eval()
            if hasattr(net, 'set_dropout_mode'):
                net.set_dropout_mode(training=False)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Override validation to use Monte Carlo Dropout."""
        # Temporarily store training state
        was_training = self.net_g.training

        # Run standard validation (which calls self.test())
        super().nondist_validation(dataloader, current_iter, tb_logger, save_img)

        # Restore training state
        if was_training:
            self.net_g.train()
