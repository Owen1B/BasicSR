from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_psnr_range, calculate_ssim, calculate_ssim_range
from .lpips_metric import calculate_lpips
from .poisson_metric import calculate_poisson_fit

__all__ = [
    'calculate_psnr',
    'calculate_psnr_range',
    'calculate_ssim',
    'calculate_ssim_range',
    'calculate_niqe',
    'calculate_lpips',
    'calculate_poisson_fit',
]


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
