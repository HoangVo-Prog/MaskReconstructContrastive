# =============================================
# File: train_ssl_unet.py
# Training logic only
# =============================================
from __future__ import annotations

import os
import math
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SmallUNetSSL
from eval import evaluate_recon, run_tsne_variants

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from alzheimer_unet_data import create_unet_dataloaders


from sklearn.manifold import TSNE  # not directly used here but kept for parity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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


def save_image_grid(tensors: List[torch.Tensor], titles: List[str], out_path: str):
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
    thr = otsu_threshold(x)
    m = (x > thr).float()
    m_blur = F.avg_pool2d(m, kernel_size=7, stride=1, padding=3)
    m = (m_blur > 0.2).float()
    return m


def bias_field_lite(x: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    blur = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    blur = blur.clamp(min=1e-3)
    x_corr = x / blur
    x_corr = x_corr - x_corr.amin(dim=(2,3), keepdim=True)
    x_corr = x_corr / x_corr.amax(dim=(2,3), keepdim=True).clamp(min=1e-6)
    return x_corr


def tight_crop_and_resize(x: torch.Tensor, mask: torch.Tensor, out_hw: int) -> torch.Tensor:
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
    B, C, H, W = x.shape
    best = []
    for b in range(B):
        xb = x[b:b+1]
        best_score = -1e9
        best_img = xb
        for d in range(-max_shift, max_shift + 1):
            if d < 0:
                pad = (0, -d, 0, 0)
                xs = F.pad(xb, pad, mode="replicate")[..., :W]
            elif d > 0:
                pad = (d, 0, 0, 0)
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
    if args.pre_bias:
        x = bias_field_lite(x, kernel=31)
    if args.pre_norm or args.pre_crop:
        m = brain_mask(x)
    if args.pre_norm:
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
# SSIM
# =========================

def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    half = window_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    g = (gauss / gauss.sum()).unsqueeze(0)
    kernel2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return kernel2d


def ssim_index(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    assert x.shape == y.shape and x.dim() == 4 and x.size(1) == 1
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    kernel = _gaussian_window(window_size, sigma, device=x.device, dtype=x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=1)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=1) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp_min(1e-8)
    return ssim_map.mean(dim=(1, 2, 3))


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
        gh = self.image_size // self.patch_size
        hw = self.half_grid_w()
        return gh * hw


def sample_masks_anti_mirror(batch_size: int, spec: MaskSpec, device: torch.device) -> torch.Tensor:
    H = W = spec.image_size
    P = spec.patch_size
    gh, _gw = spec.grid_size()
    hw = spec.half_grid_w()
    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))
    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    for b in range(batch_size):
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(random.sample(all_left, per_side))
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [rc for rc in all_right if rc not in mirror_exclude]
        right_sel = set(random.sample(all_right if per_side > len(right_candidates) else right_candidates, per_side))
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

def mixed_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2
) -> torch.Tensor:
    """
    alpha_mask: trọng số cho vùng masked
    beta_unmask: trọng số cho vùng unmasked
    """
    diff = torch.abs(pred - target)

    m = pixel_mask
    um = 1.0 - m

    # L1 trung bình trên vùng masked
    masked_denom = m.sum().clamp(min=1.0)
    masked_l1 = (diff * m).sum() / masked_denom

    # L1 trung bình trên vùng unmasked
    unmasked_denom = um.sum().clamp(min=1.0)
    unmasked_l1 = (diff * um).sum() / unmasked_denom

    # Kết hợp
    return alpha_mask * masked_l1 + beta_unmask * unmasked_l1


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
# Training
# =========================

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ----- Output directories with clear run name -----
    ts = time.strftime('%Y%m%d-%H%M%S')
    enc_tag = 'ms' if args.use_multiscale else 'bn'
    norm_tag = 'GN' if args.use_gn else 'BN'
    se_tag = '_SE' if args.use_se else ''
    run_name = args.run_name if args.run_name else f"{ts}_img{args.image_size}_b{args.base_ch}_{enc_tag}_{norm_tag}{se_tag}"
    base_out = Path(args.out_dir) / run_name
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else base_out / 'checkpoints'
    vis_dir = base_out / 'vis'
    tsne_dir = base_out / 'tsne'
    plots_dir = base_out / 'plots'
    logs_dir = base_out / 'logs'
    for d in [base_out, ckpt_dir, vis_dir, tsne_dir, plots_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_unet_dataloaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        apply_unsharp=True,
        pin_memory=True,
        data_source="adni" if args.adni else "hf",
        adni_path=args.adni_path,
        adni_image_type=args.image_type
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

    best_val = float("inf")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        
        train_ssim_sum = 0.0
        train_img_count = 0
        
        # train accumulators
        train_mask_num = 0.0
        train_mask_den = 0.0
        train_unmask_num = 0.0
        train_unmask_den = 0.0
        train_total_num = 0.0
        train_total_den = 0.0

        
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            x = preprocess_batch(x, args)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
                x_masked = x * (1.0 - pixel_mask)
                recon, _ = model.forward(x_masked, pixel_mask=pixel_mask)
                if args.enable_masked_loss:
                    loss_recon = masked_l1_loss(recon, x, pixel_mask)
                else:
                    loss_recon = mixed_l1_loss(recon, x, pixel_mask)
                
            with torch.amp.autocast("cuda", enabled=False):
                ssim_sum = ssim_index(x.float(), recon.float()).sum()

            # L1 diffs
            diff = torch.abs(recon - x)

            # region masks
            m = pixel_mask
            um = 1.0 - m

            # accumulate numerators and denominators
            train_mask_num   += (diff * m).sum().item()
            train_mask_den   += m.sum().item()
            train_unmask_num += (diff * um).sum().item()
            train_unmask_den += um.sum().item()
            train_total_num  += diff.sum().item()
            train_total_den  += diff.numel()                            # total masked pixels
            train_ssim_sum += float(ssim_sum.item())                             # sum over images
            train_img_count += x.size(0)

            if args.enable_contrastive:
                B, C, H, W = x.size()
                mid = W // 2
                left = x[..., :mid]
                right = x[..., mid:]
                left_aug = halfer(left.clone())
                right_aug = halfer(right.clone())

                zL, _ = model.encoder_embed(left_aug, mode="multiscale" if args.use_multiscale else "bottleneck")
                zR, _ = model.encoder_embed(right_aug, mode="multiscale" if args.use_multiscale else "bottleneck")
                loss_con = nt_xent_loss(zL, zR, temperature=args.temperature)
                with torch.no_grad():
                    mean_var, min_var = compute_embedding_variance([zL.detach(), zR.detach()])

            else:
                loss_con = torch.tensor(0.0, device=x.device)
                mean_var = float("nan")
                min_var = float("nan")
                            
            loss = args.lambda_recon * loss_recon + args.lambda_contrast * loss_con

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        # Validation recon and SSIM
        val_recon = evaluate_recon(model, val_loader, device, spec, args)

        model.eval()
        val_mask_num = 0.0
        val_mask_den = 0.0
        val_unmask_num = 0.0
        val_unmask_den = 0.0
        val_total_num = 0.0
        val_total_den = 0.0
        val_ssim_sum = 0.0
        val_img_count = 0
        with torch.no_grad():
            for vb in val_loader:
                vx = vb['input'].to(device, non_blocking=True)
                vx = preprocess_batch(vx, args)
                vmask = sample_masks_anti_mirror(vx.size(0), spec, device)
                vx_masked = vx * (1.0 - vmask)
                vrecon, _ = model.forward(vx_masked, pixel_mask=vmask)

                vdiff = torch.abs(vrecon - vx)
                vm = vmask
                vum = 1.0 - vm

                val_mask_num   += (vdiff * vm).sum().item()
                val_mask_den   += vm.sum().item()
                val_unmask_num += (vdiff * vum).sum().item()
                val_unmask_den += vum.sum().item()
                val_total_num  += vdiff.sum().item()
                val_total_den  += vdiff.numel()
                with torch.amp.autocast("cuda", enabled=False):
                    val_ssim_sum += float(ssim_index(vx.float(), vrecon.float()).sum().item())
                val_img_count += vx.size(0)

        train_recon_masked   = train_mask_num   / max(train_mask_den,   1.0)
        train_recon_unmasked = train_unmask_num / max(train_unmask_den, 1.0)
        train_recon_total    = train_total_num  / max(train_total_den,  1.0)

        val_recon_masked     = val_mask_num     / max(val_mask_den,     1.0)
        val_recon_unmasked   = val_unmask_num   / max(val_unmask_den,   1.0)
        val_recon_total      = val_total_num    / max(val_total_den,    1.0)

        train_ssim_mean = (train_ssim_sum / max(train_img_count, 1))
        val_ssim_mean   = (val_ssim_sum   / max(val_img_count, 1))

        print(
            f"Epoch {epoch:03d} | "
            f"train L1 (M/U/T) {train_recon_masked:.4f}/{train_recon_unmasked:.4f}/{train_recon_total:.4f} | "
            f"val L1 (M/U/T) {val_recon_masked:.4f}/{val_recon_unmasked:.4f}/{val_recon_total:.4f} | "
            f"train SSIM {train_ssim_mean:.4f} | val SSIM {val_ssim_mean:.4f}"
        )
        
        if args.enable_contrastive:
            print(
                f"Epoch {epoch:03d} | "
                f"train con {loss_con} | mean_var {mean_var:.6f} | min_var {min_var:.6f}"
            )

        if epoch % args.vis_every == 0:
            vb = next(iter(val_loader))
            vx = vb['input'].to(device, non_blocking=True)
            vx = preprocess_batch(vx, args)
            vmask = sample_masks_anti_mirror(vx.size(0), spec, device)
            vrecon, _ = model.forward(vx, pixel_mask=vmask)
            vresid = torch.abs(vx - vrecon).clamp(0, 1)
            vmasked = vx * (1.0 - vmask)
            
            out_path = str(vis_dir / f'epoch_{epoch:03d}.png')

            if args.enable_masked_loss:
                recon_full = vmask * vrecon + (1.0 - vmask) * vx
                vresid = torch.abs(vx - recon_full).clamp(0, 1)
                save_image_grid(
                    [vx, vmask, vmasked, recon_full.clamp(0,1), vresid],
                    ['val: target', 'mask', 'masked', 'recon_full', 'residual'],
                    out_path
                )
            else:
                vresid = torch.abs(vx - vrecon).clamp(0, 1)
                vmasked = vx * (1.0 - vmask)

                save_image_grid(
                    [vx, vmask, vmasked, vrecon.clamp(0,1), vresid],
                    ['val: target', 'mask', 'masked', 'recon', 'residual'],
                    out_path
                )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "val_recon": val_recon,
        }
        torch.save(ckpt, str(ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"))
        if val_recon < best_val:
            best_val = val_recon
            torch.save(ckpt, str(ckpt_dir / "ckpt_best.pt"))

        # Append epoch metrics
        with open(str(logs_dir / 'epoch_log.csv'), 'a' if epoch > 1 else 'w') as ef:
            if epoch == 1:
                ef.write(
                    'epoch,'
                    'train_recon_masked,train_recon_unmasked,train_recon_total,'
                    'val_recon_masked,val_recon_unmasked,val_recon_total,'
                    'train_ssim,val_ssim\n'
                )
            ef.write(
                f"{epoch},"
                f"{train_recon_masked:.6f},{train_recon_unmasked:.6f},{train_recon_total:.6f},"
                f"{val_recon_masked:.6f},{val_recon_unmasked:.6f},{val_recon_total:.6f},"
                f"{train_ssim_mean:.6f},{val_ssim_mean:.6f}\n"
            )  
            
        # t SNE snapshots
        if epoch % args.tsne_every == 0 and not args.adni:
            tsne_prefix = str(tsne_dir / f"tsne_epoch{epoch:03d}")
            run_tsne_variants(model, val_loader, device, tsne_prefix, max_items=args.tsne_max_items)

    # Final test
    test_recon = evaluate_recon(model, test_loader, device, spec, args)
    print(f"Final test recon {test_recon:.4f}")

    # Plots at the end
    elog_path = logs_dir / 'epoch_log.csv'
    if elog_path.exists():
        df = pd.read_csv(elog_path)

        # whole image L1
        plt.figure(figsize=(6,4))
        plt.plot(df['epoch'], df['train_recon_total'], label='train_total')
        plt.plot(df['epoch'], df['val_recon_total'], label='val_total')
        plt.xlabel('epoch'); plt.ylabel('L1 whole image')
        plt.legend(); plt.tight_layout()
        plt.savefig(str(plots_dir / 'recon_total_curves.png'), dpi=150); plt.close()

        # masked region L1
        plt.figure(figsize=(6,4))
        plt.plot(df['epoch'], df['train_recon_masked'], label='train_masked')
        plt.plot(df['epoch'], df['val_recon_masked'], label='val_masked')
        plt.xlabel('epoch'); plt.ylabel('L1 masked region')
        plt.legend(); plt.tight_layout()
        plt.savefig(str(plots_dir / 'recon_masked_curves.png'), dpi=150); plt.close()

        # unmasked region L1
        plt.figure(figsize=(6,4))
        plt.plot(df['epoch'], df['train_recon_unmasked'], label='train_unmasked')
        plt.plot(df['epoch'], df['val_recon_unmasked'], label='val_unmasked')
        plt.xlabel('epoch'); plt.ylabel('L1 unmasked region')
        plt.legend(); plt.tight_layout()
        plt.savefig(str(plots_dir / 'recon_unmasked_curves.png'), dpi=150); plt.close()
        
        # SSIM curves
        plt.figure(figsize=(6,4))
        plt.plot(df['epoch'], df['train_ssim'], label='train_ssim')
        plt.plot(df['epoch'], df['val_ssim'], label='val_ssim')
        plt.xlabel('epoch'); plt.ylabel('SSIM')
        plt.legend(); plt.tight_layout()
        plt.savefig(str(plots_dir / 'ssim_curves.png'), dpi=150); plt.close()


def build_argparser():
    p = argparse.ArgumentParser("Self supervised UNet (encoder centric) on 2D slices with MIM and InfoNCE")
    
    # data source
    p.add_argument("--adni", default=False, help="whether to use ADNI dataset")
    p.add_argument("--adni-path", type=str, default="", help="path to ADNI data root if using ADNI")
    p.add_argument("--image-type", type=str, default="axial", choices=["axial", "coronal"], help="which slice orientation to use from 3D MRI")
     
    # Size and data
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
    p.add_argument("--use-gn", action="store_true")
    p.add_argument("--use-se", action="store_true")
    p.add_argument("--use-multiscale", action="store_true")

    # Training
    p.add_argument("--enable-contrastive", action="store_true", help="whether to enable contrastive loss")
    p.add_argument("--enable-masked-loss", action="store_true", help="whether to use masked loss on unmasked regions")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs_ssl_unet")
    p.add_argument("--run-name", type=str, default="", help="optional run name; if empty, auto named")
    p.add_argument("--ckpt-dir", type=str, default="", help="optional checkpoints dir; default under run dir")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vis-every", type=int, default=20)
    p.add_argument("--tsne-every", type=int, default=20)
    p.add_argument("--tsne-max-items", type=int, default=1000)

    # Preprocessing flags
    p.add_argument("--pre-norm", action="store_true")
    p.add_argument("--pre-crop", action="store_true")
    p.add_argument("--pre-bias", action="store_true")
    p.add_argument("--pre-align", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
