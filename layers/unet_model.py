""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from torch import nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, l1 = False, drop_out = False, attention = False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        if drop_out:
            self.drop_out = nn.Dropout(0.2)
        else:
            self.drop_out = None
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128, attention)
        self.down2 = Down(128, 256, attention)
        self.down3 = Down(256, 512, attention)
        factor = 2 if bilinear else 1
        self.down4 = DownDilated(513, 1024 // factor)
        self.bottle1 = DoubleDilatedConv(512, 512)
        self.bottle2 = DoubleDilatedConv(512, 512)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.out1 = OutConv(512 //factor, out_channels)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.out2 = OutConv(256 //factor, out_channels)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.out3 = OutConv(128 //factor, out_channels)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    # we need to add a label to the featrue map so that could cat to the  
    def forward(self, x, label):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # we will concat at this layers
        b, c, h, w = x4.shape
        # cat label to feature map
        x4_cat = torch.cat((x4, label.unsqueeze(-1).unsqueeze(-1).expand(b, 1, h, w)), dim = 1)
        x5 = self.down4(x4_cat)
        x6 = self.bottle1(x5)
        if self.drop_out is not None:
            x6 = self.drop_out(x6)
        x7 = self.bottle2(x6)
        if self.drop_out is not None:
            x7 = self.drop_out(x7)
        x = self.up1(x7, x4) # 64
        x_down_8x = self.out1(x)
        x = self.up2(x, x3) # 128
        x_down_4x = self.out2(x)
        x = self.up3(x, x2) # 256
        x_down_2x = self.out3(x)
        x = self.up4(x, x1) # 512
        logits = self.outc(x)
        return logits, x_down_2x, x_down_4x, x_down_8x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out = -1, attention = False):
        super().__init__()
        self.inc = ResBlock(in_channels, 64, drop_out = drop_out, wgan = True)
        self.down1 = DownResNet(64, 128, attention, drop_out = drop_out, wgan = True)
        self.down2 = DownResNet(128, 256, attention, drop_out = drop_out, wgan = True)
        self.down3 = DownResNet(256, 512, attention, drop_out = drop_out, wgan = True)
        self.down4 = DownResNet(513, 512, attention, drop_out = drop_out, wgan = True)
        self.bottle1 = DilatedConvResNet(512, 512, wgan = True)
        self.bottle2 = DilatedConvResNet(512, 512, wgan = True)
        self.up1 = UpResNet(1024, 256, wgan = True)
        self.up2 = UpResNet(512, 128, wgan = True)
        self.up3 = UpResNet(256, 64, wgan = True)
        self.up4 = UpResNet(128, 64, wgan = True)
        self.outc = ResBlock(64, out_channels, kernel_size = 1, wgan = True)
    
    def forward(self, x, label):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b, c, h, w = x4.shape
        x4_cat = torch.cat((x4, label.unsqueeze(-1).unsqueeze(-1).expand(b, 1, h, w)), dim = 1)
        x5 = self.down4(x4_cat)
        x6 = self.bottle1(x5)
        x7 = self.bottle2(x6)
        x = self.up1(x7, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 2):
        super().__init__()

        def critic_block(in_filters, out_filters, normalization = True, stride = 2, kernel_size = 4):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size = kernel_size, stride = stride, padding = 1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine = True))
            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers
        self.ds0 = nn.Sequential(*critic_block(in_channels, 16, False, 1, 3))
        self.ds2 = nn.Sequential(
            *critic_block(16, 32, True, 2, 4),
            *critic_block(32, 32, True, 1, 3),)
        self.ds4 = nn.Sequential(
            *critic_block(32, 64, True, 2, 4),
            *critic_block(64, 64, True, 1, 3),)
        self.ds8 = nn.Sequential(
            *critic_block(64, 128, True, 2, 4),
            *critic_block(128, 128, True, 1, 3),)
        self.ds16 = nn.Sequential(
            *critic_block(128, 256, True, 2, 4),
            *critic_block(256, 256, True, 1, 3),)
        self.ds32 = nn.Sequential(
            *critic_block(256, 512, True, 2, 4),
            *critic_block(512, 512, True, 1, 3),)
        self.outc = nn.Sequential(*critic_block(512, 1, False, 1, 3))

        # self.model = nn.Sequential(
        #     *critic_block(in_channels, 16, False, 1, 3),
        #     *critic_block(16, 32, True, 2, 4),
        #     *critic_block(32, 32, True, 1, 3),
        #     *critic_block(32, 64, True, 2, 4),
        #     *critic_block(64, 64, True, 1, 3),
        #     *critic_block(64, 128, True, 2, 4),
        #     *critic_block(128, 128, True, 1, 3),
        #     *critic_block(128, 256, True, 2, 4),
        #     *critic_block(256, 256, True, 1, 3),
        #     *critic_block(256, 512, True, 2, 4),
        #     *critic_block(512, 512, True, 1, 3),
        #     *critic_block(512, 1, False, 1, 3),
        #     # nn.AdaptiveAvgPool2d(1),
        #     # nn.Flatten(),
        #     # nn.Linear(512, 1)
        #     # nn.Sigmoid(),
        #     )
    
    def forward(self, img, label):
        # expand label to the same size as input image
        b, h, w = img.shape[0], img.shape[2], img.shape[3]
        label = label.unsqueeze(-1).unsqueeze(-1).expand(b, 1, h, w)
        # get input ready
        img = torch.cat((img, label), dim = 1)
        f_ds0 = self.ds0(img)
        f_ds2 = self.ds2(f_ds0)
        f_ds4 = self.ds4(f_ds2)
        f_ds8 = self.ds8(f_ds4)
        f_ds16 = self.ds16(f_ds8)
        f_ds32 = self.ds32(f_ds16)
        outc = self.outc(f_ds32)
        # output = self.model(img)
        return outc, (f_ds0, f_ds2, f_ds4, f_ds8, f_ds16, f_ds32)