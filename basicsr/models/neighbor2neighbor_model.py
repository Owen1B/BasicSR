"""Neighbor2Neighbor model for self-supervised denoising.

Based on: "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images" (CVPR 2021)
Reference implementation: https://github.com/TaoHuang2018/Neighbor2Neighbor (train.py lines 133-190)

Key idea:
- Spatially downsample image using 2x2 blocks
- Generate complementary mask pairs from each 2x2 block
- Use regularization loss to encourage consistency
"""
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger


# Global counter for reproducible random generation (from N2B reference)
_operation_seed_counter = 0


def _get_generator(device='cuda'):
    """Get a CUDA generator with incrementing seed for reproducibility."""
    global _operation_seed_counter
    _operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(_operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    """Rearrange blocks of spatial data into depth (channel dimension).

    Reference: neighbor2neighbor_ref/train.py:133-137

    Args:
        x: input tensor (N, C, H, W)
        block_size: size of blocks to rearrange (typically 2)

    Returns:
        tensor of shape (N, C * block_size^2, H // block_size, W // block_size)
    """
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)


def generate_mask_pair(img):
    """Generate complementary mask pairs from 2x2 blocks.

    Reference: neighbor2neighbor_ref/train.py:140-171

    For each 2x2 block of 4 pixels, randomly select 2 pixels as a pair.
    8 possible pairs: [0,1],[0,2],[1,3],[2,3],[1,0],[2,0],[3,1],[3,2]

    Args:
        img: input tensor (N, C, H, W)

    Returns:
        mask1, mask2: binary masks for selecting sub-images
    """
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)

    # 8 possible pairs from 2x2 block [0,1,2,3]
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)

    # Randomly select one of 8 pairs for each 2x2 block
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=_get_generator(img.device.type),
                  out=rd_idx)

    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)

    # Set masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    """Extract sub-image using mask from space-to-depth transformed image.

    Reference: neighbor2neighbor_ref/train.py:174-189

    Args:
        img: input tensor (N, C, H, W)
        mask: binary mask for selecting pixels

    Returns:
        subimage of shape (N, C, H // 2, W // 2)
    """
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)

    # Process per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


@MODEL_REGISTRY.register()
class Neighbor2NeighborModel(SRModel):
    """Neighbor2Neighbor training model.

    Implements the dual-loss strategy from N2B paper:
    - loss1: reconstruction loss between sub-images
    - loss2: regularization loss for consistency
    - Lambda dynamically increases during training

    Reference: neighbor2neighbor_ref/train.py:358-400
    """

    def __init__(self, opt):
        super(Neighbor2NeighborModel, self).__init__(opt)

        # N2B specific parameters
        train_opt = opt.get('train', {})
        self.lambda1 = float(train_opt.get('lambda1', 1.0))
        self.lambda2 = float(train_opt.get('lambda2', 1.0))
        self.increase_ratio = float(train_opt.get('increase_ratio', 2.0))
        self.total_iter = int(train_opt.get('total_iter', 20000))

        logger = get_root_logger()
        logger.info(f'Neighbor2Neighbor: lambda1={self.lambda1}, lambda2={self.lambda2}, '
                    f'increase_ratio={self.increase_ratio}')

    def optimize_parameters(self, current_iter):
        """N2B training with dual loss strategy."""
        self.optimizer_g.zero_grad()

        # Input is noisy image (same as gt in N2B)
        noisy = self.lq

        # Generate complementary mask pairs
        mask1, mask2 = generate_mask_pair(noisy)

        # Generate sub-images
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        # Get denoised full image (no gradient)
        with torch.no_grad():
            noisy_denoised = self.net_g(noisy)

        # Get denoised sub-images
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        # Forward: denoise sub1
        noisy_output = self.net_g(noisy_sub1)
        noisy_target = noisy_sub2

        # Compute lambda (dynamically increasing)
        Lambda = current_iter / self.total_iter * self.increase_ratio

        # Compute losses
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        loss_dict = OrderedDict()

        # Loss1: reconstruction loss
        if self.cri_pix:
            l_pix = self.cri_pix(noisy_output, noisy_target)
            loss_dict['l_n2b_recon'] = l_pix
        else:
            # Fallback to MSE
            l_pix = torch.mean(diff ** 2)
            loss_dict['l_n2b_recon'] = l_pix

        # Loss2: regularization loss (consistency)
        l_reg = Lambda * torch.mean((diff - exp_diff) ** 2)
        loss_dict['l_n2b_reg'] = l_reg
        loss_dict['lambda_n2b'] = torch.tensor(Lambda, device=noisy.device)

        # Total loss
        l_total = self.lambda1 * l_pix + self.lambda2 * l_reg
        loss_dict['l_total'] = l_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        """Test/validation: denoise full image.

        During validation, we use the full noisy image (not sub-images)
        to evaluate the denoising performance on complete images.
        """
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

