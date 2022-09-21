""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, l1 = False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownDilated(513, 1024 // factor)
        self.bottle1 = DoubleDilatedConv(512, 512)
        self.bottle2 = DoubleDilatedConv(512, 512)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
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
        x7 = self.bottle2(x6)
        x = self.up1(x7, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
