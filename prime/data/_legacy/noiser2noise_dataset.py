import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Noiser2NoiseDataset(data.Dataset):
    """Noiser2Noise dataset for self-supervised denoising.

    Strategy:
    - Use 1x noisy image as "clean" reference (GT)
    - Generate noisier version by Poisson sampling from 1x
    - Train network to predict 1x from noisier version

    Theory:
    - If y ~ Poisson(λ), then E[λ|y] = λ (approximately)
    - Here λ = 1x image, y = Poisson sampled from 1x
    - Network learns: f(y) → λ

    Advantages:
    - Only needs single 1x image (no split required)
    - Can control noise level by sampling
    - Theoretically sound with PoissonNLL loss

    Args:
        opt (dict): Config for dataset. It contains:
            dataroot_gt (str): Data root path for gt (1x noisy images)
            height, width (int): Image dimensions
            mode (str): 'paired' or 'n2n'
            norm (dict): Normalization config (should use 'linear' for Poisson)
    """

    def __init__(self, opt):
        super(Noiser2NoiseDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt.get('io_backend', dict(type='disk'))

        self.gt_folder = opt['dataroot_gt']

        # SPECT-specific parameters
        self.height = opt.get('height', 512)
        self.width = opt.get('width', 128)
        self.posterior_flip = opt.get('posterior_flip', False)
        self.view_mode = opt.get('view_mode', 'single')  # single | stack2

        # Data range
        self.gt_scale_factor = opt.get('gt_scale_factor', 1.0)
        self.max_value = opt.get('max_value', 100.0)

        # Normalization
        norm_cfg = opt.get('norm', {'type': 'linear'})
        self.norm_type = norm_cfg.get('type', 'linear')
        if self.norm_type not in ['linear']:
            logger = get_root_logger()
            logger.warning(f'Noiser2Noise works best with linear normalization (got {self.norm_type})')

        # Load dataset info
        self._load_dataset_info()

        logger = get_root_logger()
        logger.info(f'Noiser2Noise Dataset: {len(self.data_list)} samples, '
                   f'norm={self.norm_type}, max_value={self.max_value}')

    def _load_dataset_info(self):
        """Load SPECT .dat file paths."""
        import os

        self.data_list = []

        # Scan for .dat files
        start_idx = self.opt.get('start_idx', 0)
        end_idx = self.opt.get('end_idx', None)

        dat_files = sorted([f for f in os.listdir(self.gt_folder) if f.endswith('.dat')])
        if end_idx is None:
            end_idx = len(dat_files)

        selected_files = dat_files[start_idx:end_idx]

        for dat_file in selected_files:
            self.data_list.append(os.path.join(self.gt_folder, dat_file))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load 1x noisy image as GT (in count domain)
        gt_path = self.data_list[index]
        gt_img_count = self._load_dat_file(gt_path)  # [H, W, C] in count domain

        # Random crop in COUNT domain first
        gt_size = self.opt.get('gt_size', 256)
        if gt_size > 0:
            h, w, c = gt_img_count.shape
            top = np.random.randint(0, h - gt_size + 1)
            left = np.random.randint(0, w - gt_size + 1)
            gt_img_count = gt_img_count[top:top+gt_size, left:left+gt_size, :]

        # ONLINE Poisson sampling in COUNT domain
        # Generate noisier version by Poisson sampling from GT
        lq_img_count = self._poisson_sample(gt_img_count.copy())

        # Now normalize both (after sampling)
        gt_img_norm = self.normalize(gt_img_count)
        lq_img_norm = self.normalize(lq_img_count)

        # Augmentation (on normalized images)
        if self.opt.get('use_hflip', False) or self.opt.get('use_rot', False):
            gt_img_norm, lq_img_norm = augment([gt_img_norm, lq_img_norm],
                                               self.opt['use_hflip'],
                                               self.opt.get('use_rot'))

        # Convert to tensor: HWC -> CHW
        gt_img_norm, lq_img_norm = img2tensor([gt_img_norm, lq_img_norm],
                                              bgr2rgb=False, float32=True)

        return {
            'lq': lq_img_norm,  # Poisson sampled (noisier)
            'gt': gt_img_norm,  # Original 1x (less noisy)
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.data_list)

    def _load_dat_file(self, path):
        """Load SPECT .dat file and return normalized image."""
        img_bytes = self.file_client.get(path, 'gt')

        # Parse .dat file (int32, row-major)
        img_data = np.frombuffer(img_bytes, dtype=np.int32)
        img_data = img_data.astype(np.float32)

        # Reshape
        total_pixels = self.height * self.width
        if self.view_mode == 'stack2':
            # Two views stacked: [H, W, 2]
            img = img_data[:2 * total_pixels].reshape(self.height, self.width, 2)
        else:
            # Single view: [H, W]
            img = img_data[:total_pixels].reshape(self.height, self.width)
            img = np.expand_dims(img, axis=2)  # [H, W, 1]

        # Posterior flip
        if self.posterior_flip:
            half_h = self.height // 2
            img[:half_h] = np.flip(img[:half_h], axis=1)

        # Scale
        img = img * self.gt_scale_factor

        return img

    def _poisson_sample(self, img_count):
        """Generate Poisson sample from count image.

        Args:
            img_count: Count domain image [H, W, C], np.float32

        Returns:
            Poisson sampled image (noisier version)
        """
        # Clip negative values (shouldn't happen, but be safe)
        img_count = np.clip(img_count, 0, None)

        # Poisson sampling: y ~ Poisson(λ=img_count)
        # np.random.poisson expects integer lambda, but works with float
        img_noisy = np.random.poisson(lam=img_count).astype(np.float32)

        return img_noisy

    def normalize(self, img):
        """Normalize image to [0, 1] range."""
        if self.norm_type == 'linear':
            # Linear: x / max_value
            img_norm = img / self.max_value
        else:
            raise NotImplementedError(f'Unknown norm type: {self.norm_type}')

        return img_norm

    def inverse_from_norm(self, img_norm):
        """Convert normalized image back to count domain."""
        if self.norm_type == 'linear':
            return img_norm * self.max_value
        else:
            raise NotImplementedError(f'Unknown norm type: {self.norm_type}')
