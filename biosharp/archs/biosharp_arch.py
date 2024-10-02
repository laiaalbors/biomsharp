import torch
import torch.nn as nn
from torchvision.transforms import functional
import sys

sys.modules["torchvision.transforms.functional_tensor"] = functional

from hat.archs.hat_arch import HAT

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BIOSHARP(nn.Module):
    def __init__(self,
                 sentinel_channels=3,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(BIOSHARP, self).__init__()
        
        # initialize HAT
        self.hat = HAT(img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size, compress_ratio, squeeze_factor, conv_scale, overlap_ratio, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint, upscale, img_range, upsampler, resi_connection, **kwargs)

        # biomass first transpose convolutions
        self.conv_first_biomass = nn.Sequential(
            nn.ConvTranspose2d(in_chans, embed_dim, 4, stride=2, padding=1, output_padding=0),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1, output_padding=0) if upscale == 4 else nn.Identity()
        )

        # optical first convolutions
        self.conv_first_optical = nn.Conv2d(sentinel_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        # Define CNN layer to process the concatenated output
        self.conv_after_concat = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, stride=1, padding=1)

    
    def forward(self, biomass, optical):
        # # Normalize biomass
        # mean_biomass = self.mean.type_as(biomass)
        # biomass = (biomass - mean_biomass) * self.img_range
        
        # # Normalize optocal data
        # mean_optical = self.mean.type_as(optical)
        # optical = (optical - mean_optical) * self.img_range

        # Shallow Feature Extraction
        biomass_output = self.conv_first_biomass(biomass)
        optical_output = self.conv_first_optical(optical)
        
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat((biomass_output, optical_output), dim=1)
        x = self.conv_after_concat(x)

        # HAT's Deep Feature Extraction
        x = self.hat.conv_after_body(self.hat.forward_features(x)) + x
        x = self.hat.conv_before_upsample(x)
        x = self.hat.conv_last(x)

        # x = x / self.img_range + mean_biomass

        return x