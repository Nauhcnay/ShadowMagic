""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ChatGPT
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.relu(self.conv(x)))
        return x * attention + x

class LayerAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size = 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        x_ = self.conv(avg_pool)
        x_ = self.relu(x)
        attention = self.sigmoid(x_)

        return x * attention + x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleDilatedConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, attention = False):
        super().__init__() 
        if attention:   
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
                LayerAttention(out_channels),
                SpatialAttention(out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownDilated(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleDilatedConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


# add WGAN
# thanks for https://www.kaggle.com/code/salimhammadi07/pix2pix-image-colorization-with-conditional-wgan
# this is a conditional discriminator
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, drop_out = False, dilation = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = stride, bias = False, dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = 'same', stride = 1, bias = False, dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.indentity_map = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = 'same', dilation = dilation)
        self.relu = nn.ReLU(inplace = True)
        if drop_out > 0:
            self.dropout = nn.Dropout2d(0.2)
        else:
            self.dropout = None

    def forward(self, x):
        # why? this means cut off one path of gradient backpropagate
        # the original implementation of ResNet doesn't do so:
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
        x_ = x.clone().detach()
        out_ = self.layers(x_)
        residual = self.indentity_map(x)
        out = out_ + residual
        if self.dropout is not None:
            return self.dropout(self.relu(out))
        else:
            return self.relu(out)


class DownResNet(nn.Module):
    def __init__(self, in_channels, out_channels, attn = False, drop_out = -1):
        super().__init__
        if attn:
            self.layers = nn.Sequential(
                nn.MaxPool2d(2),
                ResBlock(in_channels, out_channels, drop_out = drop_out),
                LayerAttention(out_channels),
                SpatialAttention(out_channels)
            )
        else:
            self.layers = nn.Sequential(
                nn.MaxPool2d(2),
                ResBlock(in_channels, out_channels, drop_out = drop_out)
            )
    def forward(self, x):
        return self.layers(x)

class UpResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.res_block = ResBlock(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim = 1)
        x = self.res_block(x)
        return x

class DilatedConvResNet(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ResBlock(in_channels, mid_channels, kernel_size=3, padding="same", dilation=2),
            ResBlock(mid_channels, out_channels, kernel_size=3, padding="same", dilation=2)
        )

    def forward(self, x):
        return self.double_conv(x)

