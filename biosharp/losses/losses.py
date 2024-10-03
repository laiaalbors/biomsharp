import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from biosharp.archs import get_Fre, Get_gradient_nopadding_d


@LOSS_REGISTRY.register()
class SGNetLoss(nn.Module):
    """
    Custom loss for SGNet, which includes spatial loss (L1), frequency amplitude loss,
    frequency phase loss, and gradient loss.

    Args:
        loss_weight_spatial (float): Weight for the spatial (L1) loss. Default: 1.0.
        loss_weight_fre_amp (float): Weight for the frequency amplitude loss. Default: 0.002.
        loss_weight_fre_pha (float): Weight for the frequency phase loss. Default: 0.002.
        loss_weight_grad (float): Weight for the gradient loss. Default: 0.001.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, reduction='mean'):
        super(SGNetLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: none, mean, sum.')

        self.reduction = reduction

        self.net_getFre = get_Fre()
        self.net_grad = Get_gradient_nopadding_d()

        # You can use torch's built-in L1Loss for simplicity
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, out, out_grad, target, **kwargs):
        """
        Args:
            out (Tensor): Predicted tensor.
            out_grad (Tensor): Predicted gradient of the output tensor, 
                               used for comparing with the ground truth 
                               gradient to sharpen the depth structure.
            target (Tensor): Ground truth tensor.
        """

        # Compute frequency domain outputs (amplitude and phase)
        out_amp, out_pha = self.net_getFre(out)
        gt_amp, gt_pha = self.net_getFre(target)

        # Compute ground truth gradients
        gt_grad = self.net_grad(target)

        # Loss computations
        loss_grad1 = self.l1_loss(out_grad, gt_grad)
        loss_fre_amp = self.l1_loss(out_amp, gt_amp)
        loss_fre_pha = self.l1_loss(out_pha, gt_pha)
        loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
        loss_spa = self.l1_loss(out, target)

        # Total loss
        total_loss = loss_spa + 0.002 * loss_fre + 0.001 * loss_grad1

        return total_loss
