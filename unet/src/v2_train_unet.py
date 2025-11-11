
import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path



from alzheimer_unet_data import create_unet_dataloaders

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Utils
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image_grid(tensors: List[torch.Tensor], titles: List[str], out_path: str, nrow: int = 4):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        panels = []
        for t in tensors:
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            panels.append(t)
        b = min(p.size(0) for p in panels)
        fig, axes = plt.subplots(b, len(panels), figsize=(3 * len(panels), 3 * b))
        if b == 1:
            axes = np.expand_dims(axes, 0)
        for i in range(b):
            for j, p in enumerate(panels):
                ax = axes[i, j]
                img = p[i].detach().cpu().numpy()
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if i == 0 and j < len(titles):
                    ax.set_title(titles[j], fontsize=10)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


# =========================
# Preprocessing
# =========================

def otsu_threshold(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    # x: [B,1,H,W] in [0,1]
    B = x.size(0)
    thresholds = []
    for b in range(B):
        hist = torch.histc(x[b].flatten(), bins=bins, min=0.0, max=1.0)
        p = hist / hist.sum().clamp(min=1.0)
        omega = torch.cumsum(p, 0)
        mu = torch.cumsum(p * torch.arange(bins, device=x.device), 0)
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega)).clamp(min=1e-8)
        sigma_b2[torch.isnan(sigma_b2)] = -1
        t = torch.argmax(sigma_b2).item()
        thresholds.append((t + 0.5) / bins)
    return torch.tensor(thresholds, device=x.device, dtype=x.dtype).view(B, 1, 1, 1)


def brain_mask(x: torch.Tensor) -> torch.Tensor:
    # crude mask via Otsu + small closing by blur-threshold
    thr = otsu_threshold(x)
    m = (x > thr).float()
    # smooth-then-threshold to close holes
    m_blur = F.avg_pool2d(m, kernel_size=7, stride=1, padding=3)
    m = (m_blur > 0.2).float()
    return m


def bias_field_lite(x: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    # divide by heavy blur, then renormalize to [0,1]
    blur = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    blur = blur.clamp(min=1e-3)
    x_corr = x / blur
    x_corr = x_corr - x_corr.amin(dim=(2,3), keepdim=True)
    x_corr = x_corr / x_corr.amax(dim=(2,3), keepdim=True).clamp(min=1e-6)
    return x_corr


def tight_crop_and_resize(x: torch.Tensor, mask: torch.Tensor, out_hw: int) -> torch.Tensor:
    # crop to bounding box of mask, pad to square, resize back
    B, _, H, W = x.shape
    out = []
    for b in range(B):
        ys, xs = torch.where(mask[b, 0] > 0.0)
        if ys.numel() == 0:
            out.append(F.interpolate(x[b:b+1], size=(out_hw, out_hw), mode="bilinear", align_corners=False))
            continue
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        side = max(h, w)
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        y1s = max(0, cy - side // 2)
        x1s = max(0, cx - side // 2)
        y2s = min(H, y1s + side)
        x2s = min(W, x1s + side)
        crop = x[b:b+1, :, y1s:y2s, x1s:x2s]
        out.append(F.interpolate(crop, size=(out_hw, out_hw), mode="bilinear", align_corners=False))
    return torch.cat(out, dim=0)


def align_midline(x: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    # shift horizontally by up to +-max_shift to maximize L-R symmetry
    B, C, H, W = x.shape
    best = []
    for b in range(B):
        xb = x[b:b+1]
        best_score = -1e9
        best_img = xb
        for d in range(-max_shift, max_shift + 1):
            if d < 0:
                pad = (0, -d, 0, 0)  # pad right
                xs = F.pad(xb, pad, mode="replicate")[..., :W]
            elif d > 0:
                pad = (d, 0, 0, 0)   # pad left
                xs = F.pad(xb, pad, mode="replicate")[..., -W:]
            else:
                xs = xb
            left = xs[..., :W//2]
            right = torch.flip(xs[..., W//2:], dims=[-1])
            score = (left * right).mean()
            if score > best_score:
                best_score = score
                best_img = xs
        best.append(best_img)
    return torch.cat(best, dim=0)


def preprocess_batch(x: torch.Tensor, args) -> torch.Tensor:
    # x in [0,1], shape Bx1xHxW
    if args.pre_bias:
        x = bias_field_lite(x, kernel=31)
    if args.pre_norm or args.pre_crop:
        m = brain_mask(x)
    if args.pre_norm:
        # robust normalization within mask
        B = x.size(0)
        flat = x.view(B, -1)
        flat_m = m.view(B, -1)
        out = []
        for b in range(B):
            vals = flat[b][flat_m[b] > 0]
            if vals.numel() > 0:
                lo = torch.quantile(vals, 0.01)
                hi = torch.quantile(vals, 0.99)
                xb = x[b:b+1].clamp(min=lo.item(), max=hi.item())
                xb = (xb - xb.mean()) / (xb.std().clamp(min=1e-6))
                xb = (xb - xb.amin()) / (xb.amax().clamp(min=1e-6))
            else:
                xb = x[b:b+1]
            out.append(xb)
        x = torch.cat(out, dim=0)
    if args.pre_crop:
        x = tight_crop_and_resize(x, m, out_hw=args.image_size)
    if args.pre_align:
        x = align_midline(x, max_shift=4)
    return x.clamp(0.0, 1.0)


# =========================
# Model: Encoder with options
# =========================

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


def norm2d(num_channels, use_gn=False, num_groups=8):
    if use_gn:
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=False, se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = norm2d(out_ch, use_gn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = norm2d(out_ch, use_gn)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if se else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.conv1(x); out = self.n1(out); out = self.act(out)
        out = self.conv2(out); out = self.n2(out)
        out = self.se(out)
        out = self.act(out + identity)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_gn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ResBlock(out_ch + skip_ch, out_ch, use_gn=use_gn, se=False)

    def forward(self, x, skip, skip_mask=None):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, dw, 0, dh))
        if skip_mask is not None:
            skip = skip * (1.0 - skip_mask)  # zero masked areas in skip
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = False):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)) if learnable else torch.tensor(p)
        self.eps = eps
        self.learnable = learnable

    def forward(self, x):
        p = self.p if self.learnable else self.p.detach()
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x


class SmallUNetSSL(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, bottleneck_dim=128, proj_dim=128,
                 use_gn: bool = False, use_se: bool = False, use_multiscale: bool = True):
        super().__init__()
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

        # Decoder for reconstruction (still present for MIM)
        self.up1 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, use_gn=use_gn)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, use_gn=use_gn)
        self.up3 = UpBlock(base_ch * 2, base_ch * 2, base_ch, use_gn=use_gn)
        self.up4 = UpBlock(base_ch, base_ch, base_ch, use_gn=use_gn)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Projection head
        self.use_multiscale = use_multiscale
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem = GeM(p=3.0, learnable=False)
        if use_multiscale:
            in_embed = base_ch * 4 + base_ch * 8 + base_ch * 8  # s3 + s4 + b pooled
        else:
            in_embed = base_ch * 8  # bottleneck only
        self.embed_fc = nn.Linear(in_embed, bottleneck_dim)
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proj_dim, bias=True),
        )

    def encode_feats(self, x):
        s1 = self.enc1(x); p1 = self.pool1(s1)
        s2 = self.enc2(p1); p2 = self.pool2(s2)
        s3 = self.enc3(p2); p3 = self.pool3(s3)
        s4 = self.enc4(p3); p4 = self.pool4(s4)
        b = self.bottleneck(p4)
        return s1, s2, s3, s4, b

    def forward(self, x, pixel_mask: Optional[torch.Tensor] = None):
        # For reconstruction. Skip tensors are masked where the pixel mask is 1.
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

    def encoder_embed(self, x, mode: str = "multiscale"):
        # Returns L2-normalized projection z and raw embedding h
        s1, s2, s3, s4, b = self.encode_feats(x)
        if mode == "s4":
            pooled = self.gem(s4).flatten(1)
        elif mode == "bottleneck":
            pooled = self.gem(b).flatten(1)
        elif mode == "s3":
            pooled = self.gem(s3).flatten(1)
        else:  # multiscale default
            p3 = self.gem(s3).flatten(1)
            p4 = self.gem(s4).flatten(1)
            pb = self.gem(b).flatten(1)
            pooled = torch.cat([p3, p4, pb], dim=1)
        h = self.embed_fc(pooled)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z, h


# =========================
# Masking & Aug
# =========================

@dataclass
class MaskSpec:
    patch_size: int = 16
    mask_ratio_side: float = 0.35
    image_size: int = 192
    def grid_size(self) -> Tuple[int, int]:
        gh = self.image_size // self.patch_size
        gw = self.image_size // self.patch_size
        return gh, gw
    def half_grid_w(self) -> int:
        return (self.image_size // 2) // self.patch_size
    def num_patches_side(self) -> int:
        gh, hw = self.image_size // self.patch_size, self.half_grid_w()
        return gh * hw


def sample_masks_anti_mirror(batch_size: int, spec: MaskSpec, device: torch.device) -> torch.Tensor:
    H = W = spec.image_size
    P = spec.patch_size
    gh, gw = spec.grid_size()
    hw = spec.half_grid_w()
    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))

    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    for b in range(batch_size):
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(random.sample(all_left, per_side))
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [rc for rc in all_right if rc not in mirror_exclude]
        if per_side > len(right_candidates):
            right_sel = set(random.sample(all_right, per_side))
        else:
            right_sel = set(random.sample(right_candidates, per_side))
        for (r, c) in left_sel:
            hs = r * P; ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
        for (r, c) in right_sel:
            hs = r * P; ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
    return mask


class HalfAug(nn.Module):
    def __init__(self, p_noise=0.7, p_jitter=0.7, p_blur=0.2, noise_std=0.02, jitter_strength=0.1, blur_kernel=3):
        super().__init__()
        self.p_noise = p_noise
        self.p_jitter = p_jitter
        self.p_blur = p_blur
        self.noise_std = noise_std
        self.jitter_strength = jitter_strength
        self.blur_kernel = blur_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_noise > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_noise
            if mask.any():
                noise = torch.randn_like(x[mask]) * self.noise_std
                x[mask] = torch.clamp(x[mask] + noise, 0.0, 1.0)
        if self.p_jitter > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_jitter
            if mask.any():
                b_shift = (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                c_scale = 1.0 + (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                x[mask] = torch.clamp((x[mask] - 0.5) * c_scale + 0.5 + b_shift, 0.0, 1.0)
        if self.p_blur > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_blur
            if mask.any():
                x_blur = F.avg_pool2d(x[mask], kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
                x[mask] = x_blur
        return x


# =========================
# Losses
# =========================

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * pixel_mask
    denom = pixel_mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    B, _ = z1.size()
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim.to(torch.float32)
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -float('inf'))
    pos = torch.cat([torch.arange(B, 2 * B, device=sim.device),
                     torch.arange(0, B, device=sim.device)], dim=0)
    labels = pos
    loss = F.cross_entropy(sim, labels)
    return loss


def compute_embedding_variance(z_list: List[torch.Tensor]) -> Tuple[float, float]:
    if len(z_list) == 0:
        return 0.0, 0.0
    Z = torch.cat(z_list, dim=0)
    var = Z.var(dim=0, unbiased=False)
    return var.mean().item(), var.min().item()


# =========================
# t-SNE using encoder-only features
# =========================

@torch.no_grad()
def run_tsne_variants(model: SmallUNetSSL, loader: DataLoader, device: torch.device, out_prefix: str, max_items: int = 1000):
    model.eval()
    def collect(mode: str):
        embs, labels = [], []
        count = 0
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            # Use encoder-only features, no decoder involved
            _, h = model.encoder_embed(x, mode=mode)
            embs.append(F.normalize(h, dim=-1).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            count += x.size(0)
            if count >= max_items:
                break
        if not embs:
            return None, None
        return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)

    for mode in ["s4", "bottleneck", "multiscale"]:
        X, y = collect(mode)
        if X is None:
            continue
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
        X2 = tsne.fit_transform(X)
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=10, cmap="tab10", alpha=0.8)
        plt.colorbar(scatter, label="label")
        plt.tight_layout()
        path = f"{out_prefix}_enc_{mode}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()


# =========================
# Training
# =========================

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, val_loader, test_loader = create_unet_dataloaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        apply_unsharp=True,
        pin_memory=True,
    )

    model = SmallUNetSSL(
        in_ch=1,
        base_ch=args.base_ch,
        bottleneck_dim=args.bottleneck_dim,
        proj_dim=args.proj_dim,
        use_gn=args.use_gn,
        use_se=args.use_se,
        use_multiscale=args.use_multiscale
    ).to(device)
    print(f"Model params: {count_params(model) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    halfer = HalfAug(p_noise=0.7, p_jitter=0.7, p_blur=0.2, noise_std=0.02, jitter_strength=0.1, blur_kernel=3)
    spec = MaskSpec(patch_size=args.patch_size, mask_ratio_side=args.mask_ratio, image_size=args.image_size)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,step,loss_total,loss_recon,loss_contrast,emb_var_mean,emb_var_min\n")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses_recon, losses_con, emb_vars = [], [], []

        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            # Encoder-suitable preprocessing
            x = preprocess_batch(x, args)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)

                # Reconstruction with skip-masked decoder
                recon, _ = model.forward(x, pixel_mask=pixel_mask)
                loss_recon = masked_l1_loss(recon, x, pixel_mask)

                # Contrastive on halves
                B, C, H, W = x.size()
                mid = W // 2
                left = x[..., :mid]
                right = x[..., mid:]
                left_aug = halfer(left.clone())
                right_aug = halfer(right.clone())

                zL, _ = model.encoder_embed(left_aug, mode="multiscale" if args.use_multiscale else "bottleneck")
                zR, _ = model.encoder_embed(right_aug, mode="multiscale" if args.use_multiscale else "bottleneck")
                loss_con = nt_xent_loss(zL, zR, temperature=args.temperature)

                loss = args.lambda_recon * loss_recon + args.lambda_contrast * loss_con

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                mean_var, min_var = compute_embedding_variance([zL.detach(), zR.detach()])

            losses_recon.append(loss_recon.item())
            losses_con.append(loss_con.item())
            emb_vars.append(mean_var)

            if step % args.vis_every == 0:
                masked_input = x * (1.0 - pixel_mask)
                out_path = os.path.join(args.out_dir, f"train_epoch{epoch:03d}_step{step:05d}.png")
                save_image_grid([x, pixel_mask, masked_input, recon.clamp(0, 1), torch.abs(x - recon).clamp(0, 1)],
                                ["orig", "mask", "masked input", "recon", "residual"],
                                out_path)

            with open(csv_path, "a") as f:
                f.write(f"{epoch},{step},{loss.item():.6f},{loss_recon.item():.6f},{loss_con.item():.6f},"
                        f"{mean_var:.6f},{min_var:.6f}\n")

        print(f"Epoch {epoch:03d} | train recon {np.mean(losses_recon):.4f} | train con {np.mean(losses_con):.4f} "
              f"| emb var {np.mean(emb_vars):.5f} | lr {opt.param_groups[0]['lr']:.2e} | time {time.time() - t0:.1f}s")

        val_recon = evaluate_recon(model, val_loader, device, spec, args)
        print(f"Epoch {epoch:03d} | val recon {val_recon:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "val_recon": val_recon,
        }
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch{epoch:03d}.pt"))
        if val_recon < best_val:
            best_val = val_recon
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best.pt"))

        if epoch % args.tsne_every == 0:
            tsne_prefix = os.path.join(args.out_dir, f"tsne_epoch{epoch:03d}")
            run_tsne_variants(model, val_loader, device, tsne_prefix, max_items=args.tsne_max_items)

    test_recon = evaluate_recon(model, test_loader, device, spec, args)
    print(f"Final test recon {test_recon:.4f}")


@torch.no_grad()
def evaluate_recon(model: SmallUNetSSL, loader: DataLoader, device: torch.device, spec: MaskSpec, args) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        x = preprocess_batch(x, args)
        pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
        recon, _ = model.forward(x, pixel_mask=pixel_mask)
        loss_recon = masked_l1_loss(recon, x, pixel_mask)
        losses.append(loss_recon.item())
    return float(np.mean(losses)) if losses else 0.0


def build_argparser():
    p = argparse.ArgumentParser("Self-supervised UNet (encoder-centric) on 2D slices with MIM and InfoNCE")
    # Size & data
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)

    # Model
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--bottleneck-dim", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--use-gn", action="store_true", help="use GroupNorm in encoder/decoder")
    p.add_argument("--use-se", action="store_true", help="add SE blocks at high stages")
    p.add_argument("--use-multiscale", action="store_true", help="use multi-scale embedding (s3+s4+b)")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs_ssl_unet")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vis-every", type=int, default=200)
    p.add_argument("--tsne-every", type=int, default=5)
    p.add_argument("--tsne-max-items", type=int, default=1000)

    # Preprocessing flags
    p.add_argument("--pre-norm", action="store_true", help="robust intensity normalization inside brain mask")
    p.add_argument("--pre-crop", action="store_true", help="tight crop around brain and resize back")
    p.add_argument("--pre-bias", action="store_true", help="bias field lite correction")
    p.add_argument("--pre-align", action="store_true", help="midline alignment before splitting")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
