import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError

# Mindset mapping
label_map_idx = {
    'mtl_atrophy': 0,              # Mild_Demented
    'mtl_atrophy,other_atrophy': 1,# Moderate_Demented
    'mtl_atrophy,wmh': 3,          # Very_Mild_Demented
    'normal': 2,                   # Non_Demented
    'other_atrophy': 0,            # Mild_Demented
    'wmh': 3,                      # Very_Mild_Demented
    'wmh,other_atrophy': 1         # Moderate_Demented
}


class UnsharpMask(nn.Module):
    """
    Unsharp masking cho ảnh đơn kênh [0, 1].

    Tham số:
        kernel_size: kích thước kernel Gaussian (số lẻ)
        sigma: độ lệch chuẩn của Gaussian
        amount: hệ số sharpen
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, amount: float = 0.7):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount

        k = self.kernel_size
        ax = torch.arange(-k // 2 + 1.0, k // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k, k)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] hoặc [1, H, W]
        Trả về cùng shape đã sharpen.
        """
        single = False
        if x.dim() == 3:  # [1, H, W]
            x = x.unsqueeze(0)
            single = True

        kernel = self.kernel.to(x.device)
        blur = F.conv2d(x, kernel, padding=self.kernel_size // 2)
        sharp = x + self.amount * (x - blur)
        sharp = torch.clamp(sharp, 0.0, 1.0)

        if single:
            sharp = sharp.squeeze(0)
        return sharp


class AlzheimerUNetDataset(Dataset):
    """
    Dataset cho UNet reconstruction trên MRI.

    input:  ảnh đã Unsharp
    target: ảnh đã Unsharp
    original: ảnh gốc chưa Unsharp
    """

    def __init__(
        self,
        hf_dataset,
        image_size: int = 128,
        apply_unsharp: bool = True,
    ):
        self.hf_dataset = hf_dataset
        self.apply_unsharp = apply_unsharp

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        if apply_unsharp:
            self.unsharp = UnsharpMask(
                kernel_size=5,
                sigma=1.0,
                amount=0.7,
            )
        else:
            self.unsharp = None

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        row = self.hf_dataset[idx]
        pil_img = row["image"]

        x_orig = self.base_transform(pil_img)

        if self.unsharp is not None:
            x_proc = self.unsharp(x_orig)
        else:
            x_proc = x_orig

        sample = {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label": torch.tensor(row["label"]).long(),
        }
        return sample


def create_unet_dataloaders(
    batch_size: int = 8,
    val_size: float = 0.2,
    num_workers: int = 2,
    image_size: int = 128,
    apply_unsharp: bool = True,
    pin_memory: bool = True,
):
    """
    Tạo DataLoader cho UNet với dataset Falah/Alzheimer_MRI.

    Trả về:
        train_loader, val_loader, test_loader
    """

    raw_train = load_dataset("Falah/Alzheimer_MRI", split="train")
    raw_test = load_dataset("Falah/Alzheimer_MRI", split="test")

    indices = np.arange(len(raw_train))
    labels = np.array(raw_train["label"])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=42,
        stratify=labels,
    )

    hf_train = raw_train.select(train_idx)
    hf_val = raw_train.select(val_idx)

    print(f"Train size: {len(hf_train)}")
    print(f"Val size:   {len(hf_val)}")
    print(f"Test size:  {len(raw_test)}")

    train_ds = AlzheimerUNetDataset(
        hf_dataset=hf_train,
        image_size=image_size,
        apply_unsharp=apply_unsharp,
    )
    val_ds = AlzheimerUNetDataset(
        hf_dataset=hf_val,
        image_size=image_size,
        apply_unsharp=apply_unsharp,
    )
    test_ds = AlzheimerUNetDataset(
        hf_dataset=raw_test,
        image_size=image_size,
        apply_unsharp=apply_unsharp,
    )

    def make_loader(ds, shuffle: bool, drop_last: bool = False, pin_memory=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    train_loader = make_loader(train_ds, shuffle=True, drop_last=True, pin_memory=pin_memory)
    val_loader = make_loader(val_ds, shuffle=False, drop_last=False, pin_memory=pin_memory)
    test_loader = make_loader(test_ds, shuffle=False, drop_last=False, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


class FolderUNetDataset(Dataset):
    """
    Loads images using a CSV with columns:
      - 'img_path': relative or absolute path to image file
      - 'abnormal_type': mapped to label via `label_map_idx`

    Returns:
      {
        "input":    tensor [1,H,W] (optionally unsharp),
        "target":   same as input,
        "original": tensor [1,H,W] before unsharp,
        "label":    int64 tensor,
        "path":     str
      }
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        image_dir: str,
        csv_map: str,
        image_size: int = 128,
        apply_unsharp: bool = True,
        validate_images: bool = True,
        warn_limit: int = 20,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.apply_unsharp = apply_unsharp

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if csv_map is None or not Path(csv_map).exists():
            raise FileNotFoundError(f"CSV mapping not found: {csv_map}")

        # Read CSV and normalize fields
        df = pd.read_csv(csv_map)
        if "img_path" not in df.columns or "abnormal_type" not in df.columns:
            raise ValueError("CSV must contain 'img_path' and 'abnormal_type' columns")

        df["img_path"] = df["img_path"].astype(str).str.strip()
        df["abnormal_type"] = df["abnormal_type"].astype(str).str.strip().str.lower()

        # Map abnormal_type -> label
        def map_label(a: str) -> Optional[int]:
            return label_map_idx.get(a, None)
        df["label"] = df["abnormal_type"].map(map_label)

        # Report and drop rows with unknown mapping
        unknown = df[df["label"].isna()]
        if len(unknown) > 0:
            print(f"[FolderUNetDataset] Warning: {len(unknown)} rows have unknown abnormal_type "
                  f"and will be skipped. Examples: {unknown['abnormal_type'].unique()[:10]}")
            df = df.dropna(subset=["label"])

        # Build absolute paths
        def make_full_path(p: str) -> Path:
            p = p.strip()
            # If already absolute, keep; else join with image_dir
            return Path(p) if os.path.isabs(p) else (self.image_dir / p)

        df["full_path"] = df["img_path"].apply(make_full_path)

        # Optional pre-validation: existence + PIL-openable
        samples: List[Tuple[Path, int]] = []
        bad_count = 0
        for _, row in df.iterrows():
            path: Path = row["full_path"]
            label = int(row["label"])
            if not path.exists():
                if bad_count < warn_limit:
                    print(f"[FolderUNetDataset] Missing file: {path}")
                bad_count += 1
                continue
            if validate_images:
                try:
                    with Image.open(path) as im:
                        im.verify()  # quick header check
                except Exception as e:
                    if bad_count < warn_limit:
                        print(f"[FolderUNetDataset] Unreadable image skipped: {path} ({type(e).__name__})")
                    bad_count += 1
                    continue
            samples.append((path, label))

        if bad_count > warn_limit:
            print(f"[FolderUNetDataset] ...and {bad_count - warn_limit} more invalid/missing files skipped.")
        if len(samples) == 0:
            raise RuntimeError("No valid images after CSV mapping and validation.")

        self.samples = samples

        # Transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.unsharp = UnsharpMask(kernel_size=5, sigma=1.0, amount=0.7) if apply_unsharp else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                pil_img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            # If validation was off, we might still hit a bad image at runtime
            raise RuntimeError(f"Failed to open image: {path}") from e

        x_orig = self.base_transform(pil_img)  # [1,H,W]
        x_proc = self.unsharp(x_orig) if self.unsharp is not None else x_orig

        return {
            "input": x_proc,
            "target": x_proc,
            "original": x_orig,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }

# -------------------------
# Single DataLoader
# -------------------------
def create_unet_dataloader_from_folder_csv(
    image_dir: str,
    csv_map: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 128,
    apply_unsharp: bool = True,
    pin_memory: bool = True,
    shuffle: bool = True,
    validate_images: bool = True,
):
    """
    Build a single DataLoader using a CSV mapping.

    Args:
        image_dir: root folder for images
        csv_map: path to CSV with columns ['img_path','abnormal_type']
        batch_size, num_workers, image_size, apply_unsharp, pin_memory, shuffle:
            same semantics as before
        validate_images: if True, verify files are readable at init and skip bad ones

    Returns:
        DataLoader over dict samples
    """
    dataset = FolderUNetDataset(
        image_dir=image_dir,
        csv_map=csv_map,
        image_size=image_size,
        apply_unsharp=apply_unsharp,
        validate_images=validate_images,
    )
    print(f"Loaded {len(dataset)} valid images from CSV '{csv_map}' under '{image_dir}'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader


__all__ = [
    "UnsharpMask",
    "AlzheimerUNetDataset",
    "create_unet_dataloaders",
    "create_unet_dataloader_from_folder",
]
