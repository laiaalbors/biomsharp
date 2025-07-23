import torch
import torch.nn as nn
from torchvision.transforms import functional
import sys

sys.modules["torchvision.transforms.functional_tensor"] = functional

from hat.archs.hat_arch import HAT, Upsample

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class BIOMSHARP(nn.Module):
    def __init__(self,
                 scale_pos='beginning',
                 guide_channels=3,
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
                 upsampler='tCNN',
                 resi_connection='1conv',
                 **kwargs):
        super(BIOMSHARP, self).__init__()

        self.scale_pos = scale_pos
        
        # initialize HAT
        self.hat = HAT(img_size, patch_size, in_chans, embed_dim, depths, num_heads, window_size, compress_ratio, squeeze_factor, conv_scale, overlap_ratio, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint, upscale, img_range, 'pixelshuffle', resi_connection, **kwargs)

        if self.scale_pos == 'beginning':

            # biomass first transpose convolutions
            if upsampler == 'tCNN':
                self.conv_first_biomass = nn.Sequential(
                    nn.ConvTranspose2d(in_chans, embed_dim, 4, stride=2, padding=1, output_padding=0),
                    nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1, output_padding=0) if upscale == 4 else nn.Identity()
                )
            elif upsampler == 'pixelshuffle':
                self.conv_first_biomass = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim, 3, 1, 1),
                    nn.LeakyReLU(inplace=True),
                    Upsample(upscale, embed_dim)
                )

            # optical first convolutions
            self.conv_first_optical = nn.Conv2d(guide_channels, embed_dim, kernel_size=3, stride=1, padding=1)

        elif self.scale_pos == 'end':

            # upscaling end
            self.conv_first_optical = nn.Sequential(
                nn.Conv2d(guide_channels, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1) 
            )
            self.conv_first_biomass = nn.Conv2d(in_chans, embed_dim, 3, 1, 1) 
            
            if upsampler == 'tCNN':
                self.upscaler = nn.Sequential(
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0),
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, output_padding=0) if upscale == 4 else nn.Identity()
                )
            elif upsampler == 'pixelshuffle':
                self.upscaler = Upsample(upscale, 64)

        else:
            raise ValueError("scale_pos has to be one of these: beginning, end.")

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
        if self.scale_pos == 'end':
            x = self.upscaler(x)
        x = self.hat.conv_last(x)

        # x = x / self.img_range + mean_biomass

        return x