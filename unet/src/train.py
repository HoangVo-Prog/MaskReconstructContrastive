# =============================================
# File: train.py
# Training logic only
# =============================================
from __future__ import annotations

import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alzheimer_unet_data import create_unet_dataloaders
from model import SmallUNetSSL
from eval import evaluate_recon, run_tsne


# Utils

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image_grid(tensors: List[torch.Tensor], titles: List[str], out_path: str, nrow: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            import numpy as _np
            axes = _np.expand_dims(axes, 0)
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


# Masking utilities

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

        if per_side > len(right_candidates):
            right_sel = set(random.sample(all_right, per_side))
        else:
            right_sel = set(random.sample(right_candidates, per_side))

        for (r, c) in left_sel:
            hs = r * P
            ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

        for (r, c) in right_sel:
            hs = r * P
            ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

    return mask


# Augmentations for halves

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


# Losses

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target)
    masked = diff * pixel_mask
    denom = pixel_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    B, _ = z1.size()
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim.to(torch.float32)
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -float("inf"))
    pos = torch.cat([
        torch.arange(B, 2 * B, device=sim.device),
        torch.arange(0, B, device=sim.device),
    ], dim=0)
    labels = pos
    loss = F.cross_entropy(sim, labels)
    return loss


def compute_embedding_variance(z_list: List[torch.Tensor]) -> tuple[float, float]:
    if len(z_list) == 0:
        return 0.0, 0.0
    Z = torch.cat(z_list, dim=0)
    var = Z.var(dim=0, unbiased=False)
    return var.mean().item(), var.min().item()


# Training

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

    model = SmallUNetSSL(in_ch=1, base_ch=16, bottleneck_dim=128, proj_dim=args.proj_dim).to(device)
    print(f"Model params: {count_params(model) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

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
        losses_recon = []
        losses_con = []
        emb_vars = []

        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            with torch.amp.autocast(enabled=(device.type == "cuda" and args.amp), device_type=device.type):
                pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
                recon, _ = model.forward(x)
                loss_recon = masked_l1_loss(recon, x, pixel_mask)

                B, C, H, W = x.size()
                mid = W // 2
                left = x[..., :mid]
                right = x[..., mid:]

                left_aug = halfer(left.clone())
                right_aug = halfer(right.clone())

                zL, _ = model.embed(left_aug)
                zR, _ = model.embed(right_aug)
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

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train recon {np.mean(losses_recon):.4f} | train con {np.mean(losses_con):.4f} "
            f"| emb var {np.mean(emb_vars):.5f} | lr {lr_now:.2e} | time {time.time() - t0:.1f}s"
        )

        val_recon = evaluate_recon(model, val_loader, device, spec, args)
        print(f"Epoch {epoch:03d} | val recon {val_recon:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "val_recon": val_recon,
        }
        torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch{epoch:03d}.pt"))
        if val_recon < best_val:
            best_val = val_recon
            torch.save(ckpt, os.path.join(args.out_dir, "ckpt_best.pt"))

        if epoch % args.tsne_every == 0:
            tsne_path = os.path.join(args.out_dir, f"tsne_epoch{epoch:03d}.png")
            run_tsne(model, val_loader, device, tsne_path, max_items=args.tsne_max_items)

    test_recon = evaluate_recon(model, test_loader, device, spec, args)
    print(f"Final test recon {test_recon:.4f}")


def build_argparser():
    p = argparse.ArgumentParser("Self supervised UNet on 2D slices with MIM and InfoNCE")
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs_ssl_unet")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vis-every", type=int, default=200)
    p.add_argument("--tsne-every", type=int, default=5)
    p.add_argument("--tsne-max-items", type=int, default=1000)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)


