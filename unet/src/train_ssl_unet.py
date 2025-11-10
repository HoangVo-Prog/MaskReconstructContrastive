
import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Dataset module expected to be present in the same environment
# The user uploaded this earlier.
try:
    from alzheimer_unet_data import create_unet_dataloaders
except Exception as e:
    raise ImportError("Could not import create_unet_dataloaders from alzheimer_unet_data.py. "
                      "Please ensure the file is available and the function signature matches.") from e

# Optional: for t-SNE and plotting
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image_grid(tensors: List[torch.Tensor], titles: List[str], out_path: str, nrow: int = 4):
    # tensors: list of Bx1xHxW or BxHxW
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        panels = []
        for t in tensors:
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            panels.append(t)
        # take the first min batch across tensors
        b = min(p.size(0) for p in panels)
        # make grid per item in the batch
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


# -------------------------
# Model: Small UNet Encoder-Decoder + Projection Head
# -------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed for odd sizes
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SmallUNetSSL(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, bottleneck_dim=128, proj_dim=128):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)          # 1 -> 16
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)    # 16 -> 32
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # 32 -> 64
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)  # 64 -> 128
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)  # 128 -> 128

        # Decoder for reconstruction
        self.up1 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)   # 128 -> 64
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)   # 64 -> 32
        self.up3 = UpBlock(base_ch * 2, base_ch * 2, base_ch)       # 32 -> 16
        self.up4 = UpBlock(base_ch, base_ch, base_ch)               # 16 -> 16
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Projection head for contrastive branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed_fc = nn.Linear(base_ch * 8, bottleneck_dim)
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proj_dim, bias=True),
        )

    def encode_feats(self, x):
        # Return encoder feats and bottleneck feature map
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

    def forward(self, x):
        # Full forward with reconstruction output
        s1, s2, s3, s4, b = self.encode_feats(x)
        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        recon = torch.sigmoid(self.out_conv(x))
        return recon, b

    def embed(self, x):
        # Return normalized projection for contrastive (keeps grad; uses current train/eval mode)
        s1, s2, s3, s4, b = self.encode_feats(x)
        pooled = self.gap(b).flatten(1)
        h = self.embed_fc(pooled)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z, b  # return b if caller wants it

# -------------------------
# Masking utilities
# -------------------------

@dataclass
class MaskSpec:
    patch_size: int = 16
    mask_ratio_side: float = 0.35
    image_size: int = 192  # square
    # left half has width image_size // 2, half-grid width = (image_size // 2) // patch_size

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
    """
    Create a pixel-level mask tensor of shape [B, 1, H, W] with anti-mirror rule.
    Left and Right halves have separate random masks. No mirrored duplicates allowed.
    """
    H = W = spec.image_size
    P = spec.patch_size
    gh, gw = spec.grid_size()
    hw = spec.half_grid_w()
    per_side = int(math.floor(spec.mask_ratio_side * gh * hw))

    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Left side grid indices: rows 0..gh-1, cols 0..hw-1
        # Right side grid indices: rows 0..gh-1, cols 0..hw-1 (to be mapped to full cols 6..11)
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(random.sample(all_left, per_side))

        # Mirrored coordinates on the right side to exclude
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)
        # All right candidates
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [rc for rc in all_right if rc not in mirror_exclude]

        # If per_side is larger than available after exclusion, fallback to all_right without exclusion
        # This should almost never happen with ratio 0.35
        if per_side > len(right_candidates):
            right_sel = set(random.sample(all_right, per_side))
        else:
            right_sel = set(random.sample(right_candidates, per_side))

        # Paint the mask at pixel level for both halves
        # Left mapping: full-grid col = c
        for (r, c) in left_sel:
            hs = r * P
            ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

        # Right mapping: full-grid col = hw + c
        for (r, c) in right_sel:
            hs = r * P
            ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0

    return mask


# -------------------------
# Augmentations for halves
# -------------------------

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
        # x: Bx1xH xW_half in [0,1]
        if self.p_noise > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_noise
            if mask.any():
                noise = torch.randn_like(x[mask]) * self.noise_std
                x[mask] = torch.clamp(x[mask] + noise, 0.0, 1.0)

        if self.p_jitter > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_jitter
            if mask.any():
                # brightness and contrast jitter
                b_shift = (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                c_scale = 1.0 + (torch.rand(x[mask].size(0), 1, 1, 1, device=x.device) - 0.5) * 2 * self.jitter_strength
                x[mask] = torch.clamp((x[mask] - 0.5) * c_scale + 0.5 + b_shift, 0.0, 1.0)

        if self.p_blur > 0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_blur
            if mask.any():
                x_blur = F.avg_pool2d(x[mask], kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
                x[mask] = x_blur

        return x


# -------------------------
# Losses
# -------------------------

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    # All tensors Bx1xHxW
    diff = torch.abs(pred - target)
    masked = diff * pixel_mask
    denom = pixel_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    # z1, z2: [B, D], normalized
    B, _ = z1.size()
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temperature  # cosine sim since normalized

    # Do the softmax & CE in float32 for numerical stability even under AMP
    sim = sim.to(torch.float32)

    # Mask self-similarity with a dtype-safe fill value
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    fill_value = -float('inf')  # works for all IEEE dtypes
    sim = sim.masked_fill(diag, fill_value)

    # positives are (i, i+B) and (i+B, i)
    pos = torch.cat([torch.arange(B, 2 * B, device=sim.device),
                     torch.arange(0, B, device=sim.device)], dim=0)
    labels = pos  # [2B]
    loss = F.cross_entropy(sim, labels)
    return loss


# -------------------------
# Training and evaluation
# -------------------------

def compute_embedding_variance(z_list: List[torch.Tensor]) -> Tuple[float, float]:
    # z_list: list of [B, D] normalized embeddings
    if len(z_list) == 0:
        return 0.0, 0.0
    Z = torch.cat(z_list, dim=0)  # [N, D]
    var = Z.var(dim=0, unbiased=False)
    return var.mean().item(), var.min().item()


def run_tsne(model: SmallUNetSSL, loader: DataLoader, device: torch.device, out_path: str, max_items: int = 1000):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        count = 0
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)  # Bx1xHxW
            _, bmap = model.forward(x)
            pooled = F.adaptive_avg_pool2d(bmap, 1).flatten(1)
            h = model.embed_fc(pooled)
            z = F.normalize(h, dim=-1)
            embs.append(z.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            count += x.size(0)
            if count >= max_items:
                break
    if not embs:
        return
    X = np.concatenate(embs, axis=0)
    y = np.concatenate(labels, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    X2 = tsne.fit_transform(X)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=10, cmap="tab10", alpha=0.8)
    plt.colorbar(scatter, label="label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Dataloaders from provided module
    train_loader, val_loader, test_loader = create_unet_dataloaders(
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        apply_unsharp=True,       # required feature
        pin_memory=True,
    )

    # Model
    model = SmallUNetSSL(in_ch=1, base_ch=16, bottleneck_dim=128, proj_dim=args.proj_dim).to(device)
    print(f"Model params: {count_params(model) / 1e6:.2f}M")

    # Optimizer and scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    # Augmentations for halves
    halfer = HalfAug(p_noise=0.7, p_jitter=0.7, p_blur=0.2, noise_std=0.02, jitter_strength=0.1, blur_kernel=3)

    # Mask spec
    spec = MaskSpec(patch_size=args.patch_size, mask_ratio_side=args.mask_ratio, image_size=args.image_size)

    # Logging
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
            x = batch["input"].to(device, non_blocking=True)  # Bx1xH xW
            with torch.amp.autocast(enabled=(device.type == "cuda" and args.amp), device_type=device.type):
                # Masks per batch
                pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)  # Bx1xHxW

                # Reconstruction branch on full image
                recon, _ = model.forward(x)
                loss_recon = masked_l1_loss(recon, x, pixel_mask)

                # Contrastive branch: split halves then photometric aug
                B, C, H, W = x.size()
                mid = W // 2
                left = x[..., :mid]   # Bx1xH x(W/2)
                right = x[..., mid:]  # Bx1xH x(W/2)

                # Apply photometric aug independently
                left_aug = halfer(left.clone())
                right_aug = halfer(right.clone())

                # Encode halves to get normalized projections
                zL, _ = model.embed(left_aug)
                zR, _ = model.embed(right_aug)
                loss_con = nt_xent_loss(zL, zR, temperature=args.temperature)

                loss = args.lambda_recon * loss_recon + args.lambda_contrast * loss_con

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Embedding variance on current batch
            with torch.no_grad():
                mean_var, min_var = compute_embedding_variance([zL.detach(), zR.detach()])

            losses_recon.append(loss_recon.item())
            losses_con.append(loss_con.item())
            emb_vars.append(mean_var)

            # Save a small panel every N steps
            if step % args.vis_every == 0:
                masked_input = x * (1.0 - pixel_mask)  # visualize the masked areas
                out_path = os.path.join(args.out_dir, f"train_epoch{epoch:03d}_step{step:05d}.png")
                save_image_grid([x, pixel_mask, masked_input, recon.clamp(0, 1), torch.abs(x - recon).clamp(0, 1)],
                                ["orig", "mask", "masked input", "recon", "residual"],
                                out_path)

            # CSV log
            with open(csv_path, "a") as f:
                f.write(f"{epoch},{step},{loss.item():.6f},{loss_recon.item():.6f},{loss_con.item():.6f},"
                        f"{mean_var:.6f},{min_var:.6f}\n")

        # End epoch summary
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train recon {np.mean(losses_recon):.4f} | train con {np.mean(losses_con):.4f} "
              f"| emb var {np.mean(emb_vars):.5f} | lr {lr_now:.2e} | time {time.time() - t0:.1f}s")

        # Simple validation on reconstruction loss
        val_recon = evaluate_recon(model, val_loader, device, spec, args)
        print(f"Epoch {epoch:03d} | val recon {val_recon:.4f}")

        # Save checkpoint
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

        # t-SNE every few epochs
        if epoch % args.tsne_every == 0:
            tsne_path = os.path.join(args.out_dir, f"tsne_epoch{epoch:03d}.png")
            run_tsne(model, val_loader, device, tsne_path, max_items=args.tsne_max_items)

    # Final test recon
    test_recon = evaluate_recon(model, test_loader, device, spec, args)
    print(f"Final test recon {test_recon:.4f}")


@torch.no_grad()
def evaluate_recon(model: SmallUNetSSL, loader: DataLoader, device: torch.device, spec: MaskSpec, args) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
        recon, _ = model.forward(x)
        loss_recon = masked_l1_loss(recon, x, pixel_mask)
        losses.append(loss_recon.item())
    return float(np.mean(losses)) if losses else 0.0


def build_argparser():
    p = argparse.ArgumentParser("Self-supervised UNet on 2D slices with MIM and InfoNCE")
    p.add_argument("--image-size", type=int, default=192, help="input size H and W")
    p.add_argument("--patch-size", type=int, default=16, help="patch size for MIM")
    p.add_argument("--mask-ratio", type=float, default=0.35, help="mask ratio per side for MIM")
    p.add_argument("--batch-size", type=int, default=64, help="batch size")
    p.add_argument("--val-size", type=float, default=0.2, help="validation split of train set")
    p.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--epochs", type=int, default=100, help="number of epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    p.add_argument("--lambda-recon", type=float, default=1.0, help="weight of reconstruction loss")
    p.add_argument("--lambda-contrast", type=float, default=1.0, help="weight of contrastive loss")
    p.add_argument("--proj-dim", type=int, default=128, help="projection dim for contrastive head")
    p.add_argument("--temperature", type=float, default=0.2, help="InfoNCE temperature")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--out-dir", type=str, default="runs_ssl_unet", help="output directory")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    p.add_argument("--amp", action="store_true", help="use mixed precision amp")
    p.add_argument("--vis-every", type=int, default=200, help="save recon panels every N train steps")
    p.add_argument("--tsne-every", type=int, default=5, help="run t-SNE every N epochs")
    p.add_argument("--tsne-max-items", type=int, default=1000, help="max items for t-SNE")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
