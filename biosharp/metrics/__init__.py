from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim_mae_mse_rmse import calculate_psnr, calculate_ssim, calculate_mse, calculate_rmse, calculate_mae


__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_mse', 'calculate_rmse', 'calculate_mae']


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