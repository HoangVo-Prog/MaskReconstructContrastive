# =============================================
# File: eval.py
# t SNE and evaluation helpers and a small CLI
# =============================================
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SmallUNetSSL

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alzheimer_unet_data import (
    create_unet_dataloader_from_folder_csv,
    mindset_colors_1,
    mindset_colors_2,
    mindset_idx_map_label_1,
    mindset_idx_map_label_2,
    mindset_label_map_idx_1,
    mindset_label_map_idx_2
)



# Minimal preprocessing and masking so eval can run standalone

@dataclass
class MaskSpec:
    patch_size: int = 16
    mask_ratio_side: float = 0.35
    image_size: int = 192
    def half_grid_w(self) -> int:
        return (self.image_size // 2) // self.patch_size


def sample_masks_anti_mirror(batch_size: int, spec: MaskSpec, device: torch.device) -> torch.Tensor:
    H = W = spec.image_size
    P = spec.patch_size
    gh = spec.image_size // spec.patch_size
    hw = spec.half_grid_w()
    per_side = int(np.floor(spec.mask_ratio_side * gh * hw))
    mask = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    import random as _random
    for b in range(batch_size):
        all_left = [(r, c) for r in range(gh) for c in range(hw)]
        left_sel = set(_random.sample(all_left, per_side))
        mirror_exclude = set((r, hw - 1 - c) for (r, c) in left_sel)
        all_right = [(r, c) for r in range(gh) for c in range(hw)]
        right_candidates = [rc for rc in all_right if rc not in mirror_exclude]
        right_sel = set(_random.sample(all_right if per_side > len(right_candidates) else right_candidates, per_side))
        for (r, c) in left_sel:
            hs = r * P; ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
        for (r, c) in right_sel:
            hs = r * P; ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
    return mask


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
    if getattr(args, "pre_bias", False):
        x = bias_field_lite(x, kernel=31)
    if getattr(args, "pre_norm", False) or getattr(args, "pre_crop", False):
        m = brain_mask(x)
    if getattr(args, "pre_norm", False):
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
    if getattr(args, "pre_crop", False):
        x = tight_crop_and_resize(x, m, out_hw=getattr(args, "image_size", 192))
    if getattr(args, "pre_align", False):
        x = align_midline(x, max_shift=4)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def evaluate_recon(model: SmallUNetSSL, loader: DataLoader, device: torch.device, spec: MaskSpec, args) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        x = preprocess_batch(x, args)
        pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
        recon, _ = model.forward(x, pixel_mask=pixel_mask)
        diff = torch.abs(recon - x) * pixel_mask
        loss_recon = diff.sum() / pixel_mask.sum().clamp(min=1.0)
        losses.append(loss_recon.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def run_tsne_variants(model: SmallUNetSSL, loader: DataLoader, device: torch.device, out_prefix: str, max_items: int = 1000, label="label"):
    model.eval()

    modes = ["s4", "bottleneck"]
    if getattr(model, "use_multiscale", False):
        modes.append("multiscale")

    def collect(mode: str):
        embs, labels = [], []
        count = 0
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            _, h = model.encoder_embed(x, mode=mode)
            embs.append(F.normalize(h, dim=-1).cpu().numpy())
            labels.append(batch.get(label, torch.zeros(x.size(0), dtype=torch.long, device=device)).cpu().numpy())
            count += x.size(0)
            if count >= max_items:
                break
        if not embs:
            return None, None
        return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)

    for mode in modes:
        X, y = collect(mode)
        if X is None:
            continue
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
        X2 = tsne.fit_transform(X)

        plt.figure(figsize=(6, 6))
        uniq = sorted(set(list(y)))
        for lbl in uniq:
            key = str(int(lbl)) if str(lbl).isdigit() else str(lbl)
            if label == "label_1":
                name_id = mindset_idx_map_label_1.get(key, key)
                print(name_id)
                name = mindset_label_map_idx_1.get(name_id, name_id)
                print(name)
                color = mindset_colors_1.get(name, "#888888")
                print(color)
            elif label == "label_2":
                name_id = mindset_idx_map_label_2.get(key, key)
                name = mindset_label_map_idx_2.get(name_id, name_id)
                color = mindset_colors_2.get(name, "#888888")
            # name = hf_idx_map_label.get(key, key)
            # color = hf_demantia_colors.get(name, "#888888")
            mask = (y == lbl)
            plt.scatter(X2[mask, 0], X2[mask, 1], s=10, c=color, label=name, alpha=0.9)
        plt.legend(loc="best", fontsize=8, frameon=False)
        plt.tight_layout()
        path = f"{out_prefix}_enc_{mode}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        

# Optional CLI to eval a checkpoint or make t SNE from it

def _load_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[SmallUNetSSL, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    model = SmallUNetSSL(
        in_ch=1,
        base_ch=cfg.get("base_ch", 16),
        bottleneck_dim=cfg.get("bottleneck_dim", 128),
        proj_dim=cfg.get("proj_dim", 128),
        use_gn=cfg.get("use_gn", False),
        use_se=cfg.get("use_se", False),
        use_multiscale=cfg.get("use_multiscale", True),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def build_argparser():
    p = argparse.ArgumentParser("Eval helpers for SSL UNet")
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--csv-map", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tsne", action="store_true")
    p.add_argument("--tsne-max-items", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="runs_eval")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_from_ckpt(args.ckpt, device)

    image_size = cfg.get("image_size", 192)

    val_loader = create_unet_dataloader_from_folder_csv(
        image_dir=args.image_dir,
        csv_map=args.csv_map,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_unsharp=True,
        pin_memory=True,
    )

    loader = {"train": None, "val": val_loader, "test": None}[args.split]
    spec = MaskSpec(patch_size=cfg.get("patch_size", 16), mask_ratio_side=cfg.get("mask_ratio", 0.35), image_size=image_size)

    os.makedirs(args.out_dir, exist_ok=True)
    recon = evaluate_recon(model, loader, device, spec, cfg)
    print(f"{args.split} recon {recon:.4f}")

    if args.tsne:
        tsne_prefix_1 = os.path.join(args.out_dir, f"tsne_{args.split}_1")
        run_tsne_variants(model, loader, device, tsne_prefix_1, max_items=args.tsne_max_items, label="label_1")
        
        tsne_prefix_2 = os.path.join(args.out_dir, f"tsne_{args.split}_2")
        run_tsne_variants(model, loader, device, tsne_prefix_2, max_items=args.tsne_max_items, label="label_2")

        print(f"Saved t SNE to {args.out_dir}_enc_*.png")


if __name__ == "__main__":
    main()
