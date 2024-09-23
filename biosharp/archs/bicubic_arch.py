import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BicubicInterpolation(nn.Module):
    def __init__(self,
                 img_size=64,
                 upscale=2,
                 img_range=1.,
                 **kwargs):
        super(BicubicInterpolation, self).__init__()

        # Initialize parameters
        self.img_size = img_size
        self.upscale = upscale
        self.img_range = img_range
        
        # Define interpolation modes for different scenarios
        self.interpolation_mode = 'bicubic'

    def forward(self, x):
        # Get the current image size
        batch_size, channels, height, width = x.size()
        
        # Determine target size based on the mode
        target_size = (self.img_size * self.upscale, self.img_size * self.upscale)

        # Perform bicubic interpolation
        x = F.interpolate(x, size=target_size, mode=self.interpolation_mode, align_corners=False)
        x = x * self.img_range
        
        return x