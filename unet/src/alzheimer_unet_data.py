import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np


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


__all__ = [
    "UnsharpMask",
    "AlzheimerUNetDataset",
    "create_unet_dataloaders",
]
