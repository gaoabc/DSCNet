import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.IB import HSIC_IB_Loss


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class DSCNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.ib_loss_fn = HSIC_IB_Loss()

        # Encoder for RGB
        self.rgb_conv1 = ConvBlock(3, 32)
        self.rgb_conv2 = ConvBlock(32, 64)

        # Encoder for HSL
        self.hsl_conv1 = ConvBlock(3, 32)
        self.hsl_conv2 = ConvBlock(32, 64)

        # Fusion
        self.fuse = ConvBlock(128, 128)

        # Decoder
        self.dec1 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def encode_rgb(self, x):
        x = self.rgb_conv1(x)
        x = self.rgb_conv2(x)
        return x

    def encode_hsl(self, x):
        x = self.hsl_conv1(x)
        x = self.hsl_conv2(x)
        return x

    def forward(self, rgb, hsl, mask=None, collect_hsic=False):
        f_rgb = self.encode_rgb(rgb)
        f_hsl = self.encode_hsl(hsl)

        
        fused = torch.cat([f_rgb, f_hsl], dim=1)
        fused = self.fuse(fused)

        # Decoder
        x = self.dec1(fused)
        x = self.dec2(x)
        out = self.out_conv(x)

       
        if collect_hsic and mask is not None:
            ib_loss = self.ib_loss_fn(f_rgb, f_hsl, mask)
            return out, ib_loss
        else:
            return out, torch.tensor(0.0, device=rgb.device)

