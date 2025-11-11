# =============================================
# File: model.py
# Only model construction lives here
# =============================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SmallUNetSSL",
]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SmallUNetSSL(nn.Module):
    """Small UNet encoder decoder with a projection head for SSL.
    - forward returns reconstruction and bottleneck feature map
    - embed returns L2 normalized projection and bottleneck feature map
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16, bottleneck_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        # Decoder for reconstruction
        self.up1 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up3 = UpBlock(base_ch * 2, base_ch * 2, base_ch)
        self.up4 = UpBlock(base_ch, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Projection head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed_fc = nn.Linear(base_ch * 8, bottleneck_dim)
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proj_dim, bias=True),
        )

    def encode_feats(self, x: torch.Tensor):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)
        b = self.bottleneck(p4)
        return s1, s2, s3, s4, b

    def forward(self, x: torch.Tensor):
        s1, s2, s3, s4, b = self.encode_feats(x)
        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        recon = torch.sigmoid(self.out_conv(x))
        return recon, b

    def embed(self, x: torch.Tensor):
        s1, s2, s3, s4, b = self.encode_feats(x)
        pooled = self.gap(b).flatten(1)
        h = self.embed_fc(pooled)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z, b