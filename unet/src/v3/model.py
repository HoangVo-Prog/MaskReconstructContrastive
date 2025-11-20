# =============================================
# File: model.py
# Only model construction lives here
# =============================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = [
    "SmallUNetSSL",
]


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


def norm2d(num_channels: int, use_gn: bool = False, num_groups: int = 8) -> nn.Module:
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if use_gn else nn.BatchNorm2d(num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_gn: bool = False, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = norm2d(out_ch, use_gn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = norm2d(out_ch, use_gn)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x); out = self.n1(out); out = self.act(out)
        out = self.conv2(out); out = self.n2(out)
        out = self.se(out)
        out = self.act(out + identity)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_gn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ResBlock(out_ch + skip_ch, out_ch, use_gn=use_gn, se=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, skip_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, dw, 0, dh))
        if skip_mask is not None:
            skip = skip * (1.0 - skip_mask)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = False):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)) if learnable else torch.tensor(p)
        self.eps = eps
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p if self.learnable else self.p.detach()
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x


class SmallUNetSSL(nn.Module):
    """UNet encoder decoder with projection head for self supervised learning.
    forward returns (reconstruction, deep feature tuple)
    encoder_embed returns (normalized projection z, pre projection embedding h)
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        bottleneck_dim: int = 128,
        proj_dim: int = 128,
        use_gn: bool = False,
        use_se: bool = False,
        use_multiscale: bool = True,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, use_gn=use_gn, se=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock(base_ch, base_ch * 2, use_gn=use_gn, se=False)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, use_gn=use_gn, se=False)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResBlock(base_ch * 4, base_ch * 8, use_gn=use_gn, se=use_se)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResBlock(base_ch * 8, base_ch * 8, use_gn=use_gn, se=use_se)

        # Decoder for masked image modeling
        self.up1 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, use_gn=use_gn)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, use_gn=use_gn)
        self.up3 = UpBlock(base_ch * 2, base_ch * 2, base_ch, use_gn=use_gn)
        self.up4 = UpBlock(base_ch, base_ch, base_ch, use_gn=use_gn)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Projection heads per mode to avoid dim mismatch
        self.gem = GeM(p=3.0, learnable=False)
        ch_s3 = base_ch * 4
        ch_s4 = base_ch * 8
        ch_b  = base_ch * 8
        ch_ms = ch_s3 + ch_s4 + ch_b
        self.embed_fc = nn.ModuleDict({
            "s3":         nn.Linear(ch_s3, bottleneck_dim),
            "s4":         nn.Linear(ch_s4, bottleneck_dim),
            "bottleneck": nn.Linear(ch_b,  bottleneck_dim),
            "multiscale": nn.Linear(ch_ms,  bottleneck_dim),
        })

        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proj_dim, bias=True),
        )

    def encode_feats(self, x: torch.Tensor):
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        s3 = self.enc3(p2); p3 = self.pool3(s3)
        s4 = self.enc4(p3); p4 = self.pool4(s4)
        b = self.bottleneck(p4)
        return s1, s2, s3, s4, b

    def forward(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor] = None):
        s1, s2, s3, s4, b = self.encode_feats(x)
        m1 = m2 = m3 = m4 = None
        if pixel_mask is not None:
            m4 = F.interpolate(pixel_mask, size=s4.shape[-2:], mode="nearest")
            m3 = F.interpolate(pixel_mask, size=s3.shape[-2:], mode="nearest")
            m2 = F.interpolate(pixel_mask, size=s2.shape[-2:], mode="nearest")
            m1 = F.interpolate(pixel_mask, size=s1.shape[-2:], mode="nearest")
        x = self.up1(b, s4, skip_mask=m4)
        x = self.up2(x, s3, skip_mask=m3)
        x = self.up3(x, s2, skip_mask=m2)
        x = self.up4(x, s1, skip_mask=m1)
        recon = torch.sigmoid(self.out_conv(x))
        return recon, (s3, s4, b)

    def encoder_embed(self, x: torch.Tensor, mode: str = "multiscale"):
        s1, s2, s3, s4, b = self.encode_feats(x)

        if mode == "s3":
            pooled = self.gem(s3).flatten(1); head = "s3"
        elif mode == "s4":
            pooled = self.gem(s4).flatten(1); head = "s4"
        elif mode == "bottleneck":
            pooled = self.gem(b).flatten(1);  head = "bottleneck"
        else:
            if self.use_multiscale:
                pooled = torch.cat([self.gem(s3).flatten(1),
                                    self.gem(s4).flatten(1),
                                    self.gem(b).flatten(1)], dim=1)
                head = "multiscale"
            else:
                pooled = self.gem(b).flatten(1); head = "bottleneck"

        h = self.embed_fc[head](pooled)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z, h
