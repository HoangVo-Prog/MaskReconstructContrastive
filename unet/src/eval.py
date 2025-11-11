# =============================================
# File: eval.py
# t SNE and evaluation helpers and a small CLI
# =============================================
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SmallUNetSSL
from alzheimer_unet_data import create_unet_dataloaders


# Basic util for reproducibility

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# MaskSpec duplicated here so eval can run on its own

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
        if per_side > len(right_candidates):
            right_sel = set(_random.sample(all_right, per_side))
        else:
            right_sel = set(_random.sample(right_candidates, per_side))
        for (r, c) in left_sel:
            hs = r * P
            ws = c * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
        for (r, c) in right_sel:
            hs = r * P
            ws = (hw + c) * P
            mask[b, 0, hs:hs + P, ws:ws + P] = 1.0
    return mask


@torch.no_grad()
def evaluate_recon(model: SmallUNetSSL, loader: DataLoader, device: torch.device, spec: MaskSpec, args) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        pixel_mask = sample_masks_anti_mirror(x.size(0), spec, device)
        recon, _ = model.forward(x)
        diff = torch.abs(recon - x)
        masked = diff * pixel_mask
        denom = pixel_mask.sum().clamp(min=1.0)
        loss_recon = masked.sum() / denom
        losses.append(loss_recon.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def run_tsne(model: SmallUNetSSL, loader: DataLoader, device: torch.device, out_path: str, max_items: int = 1000):
    model.eval()
    embs = []
    labels = []
    count = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        z, _ = model.embed(x)
        embs.append(z.cpu().numpy())
        if "label" in batch:
            labels.append(batch["label"].cpu().numpy())
        else:
            labels.append(np.zeros((z.size(0),), dtype=np.int64))
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


# Small CLI for offline eval

def build_argparser():
    p = argparse.ArgumentParser("Eval helpers for SSL UNet")
    p.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pt")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tsne", action="store_true", help="run t SNE plot")
    p.add_argument("--tsne-max-items", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="runs_eval")
    return p


def _load_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[SmallUNetSSL, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    proj_dim = ckpt_args.get("proj_dim", 128)
    model = SmallUNetSSL(in_ch=1, base_ch=16, bottleneck_dim=128, proj_dim=proj_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt_args


def main_eval():
    args = build_argparser().parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_args = _load_from_ckpt(args.ckpt, device)

    image_size = ckpt_args.get("image_size", 192)
    val_size = ckpt_args.get("val_size", 0.2)

    train_loader, val_loader, test_loader = create_unet_dataloaders(
        image_size=image_size,
        batch_size=args.batch_size,
        val_size=val_size,
        num_workers=args.num_workers,
        apply_unsharp=True,
        pin_memory=True,
    )

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    spec = MaskSpec(patch_size=ckpt_args.get("patch_size", 16),
                    mask_ratio_side=ckpt_args.get("mask_ratio", 0.35),
                    image_size=image_size)

    os.makedirs(args.out_dir, exist_ok=True)

    recon = evaluate_recon(model, loader, device, spec, ckpt_args)
    print(f"{args.split} recon {recon:.4f}")

    if args.tsne:
        tsne_path = os.path.join(args.out_dir, f"tsne_{args.split}.png")
        run_tsne(model, loader, device, tsne_path, max_items=args.tsne_max_items)
        print(f"Saved t SNE to {tsne_path}")


if __name__ == "__main__":
    main_eval()
