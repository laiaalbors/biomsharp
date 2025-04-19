import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import os

from basicsr.utils.registry import ARCH_REGISTRY


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super(Conv2dBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        ])
        if batchnorm:
            layers.insert(4, nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


@ARCH_REGISTRY.register()
class ReUse(nn.Module):
    def __init__(self, in_chans, n_classes=1, n_filters=16, dropout=0.1, batchnorm=True):
        super(ReUse, self).__init__()
        
        self.enc1 = Conv2dBlock(in_chans, n_filters * 1, batchnorm=batchnorm)
        self.enc2 = Conv2dBlock(n_filters * 1, n_filters * 2, batchnorm=batchnorm)
        self.enc3 = Conv2dBlock(n_filters * 2, n_filters * 4, batchnorm=batchnorm)
        self.enc4 = Conv2dBlock(n_filters * 4, n_filters * 8, batchnorm=batchnorm)
        self.center = Conv2dBlock(n_filters * 8, n_filters * 16, batchnorm=batchnorm)
        
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = Conv2dBlock(n_filters * 16, n_filters * 8, batchnorm=batchnorm)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = Conv2dBlock(n_filters * 8, n_filters * 4, batchnorm=batchnorm)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = Conv2dBlock(n_filters * 4, n_filters * 2, batchnorm=batchnorm)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters * 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = Conv2dBlock(n_filters * 2, n_filters * 1, batchnorm=batchnorm)
        
        self.final = nn.Conv2d(n_filters, n_classes, kernel_size=1)
        
    def forward(self, x):
        c1 = self.enc1(x)
        p1 = nn.MaxPool2d(2)(c1)
        
        c2 = self.enc2(p1)
        p2 = nn.MaxPool2d(2)(c2)
        
        c3 = self.enc3(p2)
        p3 = nn.MaxPool2d(2)(c3)
        
        c4 = self.enc4(p3)
        p4 = nn.MaxPool2d(2)(c4)
        
        c5 = self.center(p4)
        
        u4 = self.up4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.dec4(u4)
        
        u3 = self.up3(u4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.dec3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.dec2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.dec1(u1)
        
        return self.final(u1)
