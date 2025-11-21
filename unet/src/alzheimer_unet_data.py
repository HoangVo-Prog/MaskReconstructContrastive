import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nibabel as nib

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError


# Mindset mapping and fixed colors for t SNE legends

mindset_idx_map_label_1 = {
    '0': "Normal",
    '1': "MTL",
    '2': "Other",
    '3': "WMH",
}

mindset_label_map_idx_1 = {
    'mtl_atrophy': 1,              # Mild_Demented MTL
    'mtl_atrophy,other_atrophy': 1,# Moderate_Demented MTL
    'mtl_atrophy,wmh': 1,          # Very_Mild_Demented MTL
    'normal': 0,                   # Non_Demented N (nhãn 0) 4 màu hoàn toàn khác nhau 
    'other_atrophy': 2,            # Mild_Demented O
    'wmh': 3,                      # Very_Mild_Demented WMH
    'wmh,other_atrophy': 3         # Moderate_Demented WMH 
    # alzemer | normal | other 
}

mindset_colors_1 = {
    "MTL": "#e18775",
    "Normal": "#fdfdfd",
    "Other": "#7bbfc8",
    "WMH": "#fde2b6",
}


mindset_label_map_idx_2 = {
    'mtl_atrophy': 1,              # Mild_Demented MTL
    'mtl_atrophy,other_atrophy': 1,# Moderate_Demented MTL
    'mtl_atrophy,wmh': 1,          # Very_Mild_Demented MTL
    'normal': 0,                   # Non_Demented N (nhãn 0) 4 màu hoàn toàn khác nhau 
    'other_atrophy': 2,            # Mild_Demented O
    'wmh': 1,                      # Very_Mild_Demented WMH
    'wmh,other_atrophy': 1 
}

mindset_idx_map_label_2 = {
    '0': "Normal",
    '1': "Alzheimer",
    '2': "Other",
}

mindset_colors_2 = {
    "MTL": "#e18775",
    "Normal": "#fdfdfd",
    "Other": "#7bbfc8",
}
    

# Huggingface mapping and fixed colors for t SNE legends
hf_idx_map_label = {
    '0': "Mild_Demented",
    '1': "Moderate_Demented",
    '2': "Non_Demented",
    '3': "Very_Mild_Demented",
}

hf_demantia_colors = {
    "Moderate_Demented": "#a5352b",
    "Non_Demented": "#457eb7",
    "Mild_Demented": "#e18775",
    "Very_Mild_Demented": "#ffe9c6",
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


class AdniNiftiSliceDataset(Dataset):
    """
    Expands a folder of .nii or .nii.gz into 2D central slices.

    Each item:
      input    [1,H,W] optionally unsharp
      target   [1,H,W] same as input
      original [1,H,W] before unsharp
      label    int64  (from CSV if provided, else -1)
      path     str    original nii path
      slice_idx int   slice index in the chosen axis
    """

    ORIENT_TO_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

    def __init__(
        self,
        root_dir: str,
        image_size: int = 128,
        apply_unsharp: bool = True,
        adni_image_type: str = "axial",         # orientation for slicing
        adni_series_filter: Optional[List[str]] = None,  # substrings to keep in filenames
        adni_label_csv: Optional[str] = None,   # optional mapping CSV with columns filename,label
        middle_frac: float = 0.4,               # keep central fraction of slices
        middle_subsample: int = 1,              # subsample stride within middle segment
        validate_read: bool = True,
        warn_limit: int = 20,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"ADNI root not found: {self.root_dir}")

        adni_image_type = str(adni_image_type).lower().strip()
        if adni_image_type not in self.ORIENT_TO_AXIS:
            raise ValueError(f"adni_image_type must be one of {list(self.ORIENT_TO_AXIS.keys())}")
        self.axis = self.ORIENT_TO_AXIS[adni_image_type]

        self.image_size = image_size
        self.apply_unsharp = apply_unsharp
        self.middle_frac = float(middle_frac)
        self.middle_subsample = max(1, int(middle_subsample))

        # Load optional labels
        self.label_map = {}
        if adni_label_csv is not None:
            df_lab = pd.read_csv(adni_label_csv)
            if "filename" not in df_lab.columns or "label" not in df_lab.columns:
                raise ValueError("adni_label_csv must contain columns 'filename' and 'label'")
            # Normalize key to base filename without compression extension
            def keyize(fn: str) -> str:
                base = os.path.basename(fn)
                # strip .nii or .nii.gz
                if base.endswith(".nii.gz"):
                    base = base[:-7]
                elif base.endswith(".nii"):
                    base = base[:-4]
                return base
            for _, r in df_lab.iterrows():
                self.label_map[keyize(str(r["filename"]))] = int(r["label"])

        # Discover nii files
        nii_paths = []
        for p in self.root_dir.rglob("*"):
            name = p.name.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                nii_paths.append(p)

        # Optional series substring filter on filenames
        if adni_series_filter:
            filt_lower = [s.lower() for s in adni_series_filter]
            def keep(path: Path) -> bool:
                nm = path.name.lower()
                return any(s in nm for s in filt_lower)
            nii_paths = [p for p in nii_paths if keep(p)]

        if len(nii_paths) == 0:
            raise RuntimeError(f"No .nii or .nii.gz found under {self.root_dir}")

        # Index into slices lazily, but precompute slice indices per file
        self.index: List[Tuple[Path, int]] = []  # (nii_path, slice_idx)
        bad = 0
        for p in nii_paths:
            try:
                img = nib.load(str(p))
                shape = img.shape
                if len(shape) < 3:
                    # skip non 3D
                    continue
                depth = shape[self.axis]
                if depth < 8:
                    # too shallow to find meaningful middle
                    continue
                # central band
                band = int(round(self.middle_frac * depth))
                band = max(1, min(depth, band))
                start = (depth - band) // 2
                stop = start + band
                slice_indices = list(range(start, stop, self.middle_subsample))
                for s in slice_indices:
                    self.index.append((p, s))
            except Exception:
                if bad < warn_limit:
                    print(f"[AdniNiftiSliceDataset] Skipped unreadable: {p}")
                bad += 1
                continue
        if bad > warn_limit:
            print(f"[AdniNiftiSliceDataset] ...and {bad - warn_limit} more unreadable NIfTI skipped.")

        if len(self.index) == 0:
            raise RuntimeError("No valid middle slices discovered from ADNI volumes.")

        # Transforms
        self.to_tensor_resize = transforms.Compose([
            transforms.ToTensor(),                          # HWC [0,1] -> CHW
            transforms.Resize((image_size, image_size), antialias=True),
        ])
        self.unsharp = UnsharpMask(kernel_size=5, sigma=1.0, amount=0.7) if apply_unsharp else None

        # Small LRU cache for last loaded volume to avoid reloading for consecutive slices
        self._cache_path: Optional[Path] = None
        self._cache_data: Optional[np.ndarray] = None      # float32 volume in [0,1] after percentile norm

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume_norm01(self, path: Path) -> np.ndarray:
        # Cache the most recent volume
        if self._cache_path == path and self._cache_data is not None:
            return self._cache_data
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)
        # robust percentile scaling per volume
        lo = np.percentile(vol, 1.0)
        hi = np.percentile(vol, 99.0)
        if hi <= lo:
            hi = lo + 1e-6
        vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
        self._cache_path = path
        self._cache_data = vol
        return vol

    def _extract_slice(self, vol01: np.ndarray, slice_idx: int) -> np.ndarray:
        # axis order is (X,Y,Z). We slice along self.axis then make HxW
        if self.axis == 2:      # axial, slice [X,Y]
            sl = vol01[:, :, slice_idx]
        elif self.axis == 1:    # coronal, slice [X,Z]
            sl = vol01[:, slice_idx, :]
        else:                   # sagittal, slice [Y,Z]
            sl = vol01[slice_idx, :, :]
        # normalize per slice again to enhance contrast
        lo = np.percentile(sl, 1.0)
        hi = np.percentile(sl, 99.0)
        if hi <= lo:
            hi = lo + 1e-6
        sl = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
        return sl

    def __getitem__(self, idx: int):
        path, sidx = self.index[idx]
        vol01 = self._load_volume_norm01(path)
        sl = self._extract_slice(vol01, sidx)  # HxW in [0,1]
        
        # auto orient to vertical
        sl = self.ensure_vertical_orientation(sl)

        # to tensor via PIL path to reuse your resize
        img_u8 = (sl * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        x_orig = self.to_tensor_resize(pil)  # [1,H,W] in [0,1]

        # to tensor through PIL-like path to reuse your transforms shape
        img_u8 = (sl * 255.0).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        x_orig = self.to_tensor_resize(pil)  # [1,H,W] in [0,1]

        x_proc = self.unsharp(x_orig) if self.unsharp is not None else x_orig

        # map label if provided
        base = path.name
        if base.endswith(".nii.gz"):
            base = base[:-7]
        elif base.endswith(".nii"):
            base = base[:-4]
        label = self.label_map.get(base, -1)

        return {
            "input":    x_proc,
            "target":   x_proc,
            "original": x_orig,
            "label":    torch.tensor(label, dtype=torch.long),
            "path":     str(path),
            "slice_idx": int(sidx),
        }

    def _masked_percentile(self, img: np.ndarray, q: float) -> float:
        flat = img.reshape(-1)
        return float(np.percentile(flat, q))

    def _brain_mask_otsu(self, img01: np.ndarray) -> np.ndarray:
        # very light mask to suppress background in symmetry scores
        lo = self._masked_percentile(img01, 1.0)
        hi = self._masked_percentile(img01, 99.0)
        if hi <= lo:
            hi = lo + 1e-6
        x = np.clip((img01 - lo) / (hi - lo), 0.0, 1.0)
        thr = 0.0 + 0.2  # cheap threshold after robust scaling
        m = (x > thr).astype(np.float32)
        # quick blur then threshold to close holes
        import cv2
        m = cv2.GaussianBlur(m, (7, 7), 0)
        m = (m > 0.2).astype(np.float32)
        return m

    def _vertical_sym_mse(self, img01: np.ndarray, mask: np.ndarray) -> float:
        H, W = img01.shape
        w2 = W // 2
        left  = img01[:, :w2]
        right = img01[:, W - w2:]
        mL = mask[:, :w2]
        mR = mask[:, W - w2:]
        right_flipped = np.fliplr(right)
        mR_flip = np.fliplr(mR)
        m = (mL * mR_flip)
        denom = m.sum()
        if denom < 1.0:
            denom = left.size
            m = np.ones_like(left, dtype=np.float32)
        return float(((left - right_flipped) ** 2 * m).sum() / denom)

    def _horizontal_sym_mse(self, img01: np.ndarray, mask: np.ndarray) -> float:
        H, W = img01.shape
        h2 = H // 2
        top    = img01[:h2, :]
        bottom = img01[H - h2:, :]
        mT = mask[:h2, :]
        mB = mask[H - h2:, :]
        bottom_flipped = np.flipud(bottom)
        mB_flip = np.flipud(mB)
        m = (mT * mB_flip)
        denom = m.sum()
        if denom < 1.0:
            denom = top.size
            m = np.ones_like(top, dtype=np.float32)
        return float(((top - bottom_flipped) ** 2 * m).sum() / denom)

    def ensure_vertical_orientation(self, sl01: np.ndarray) -> np.ndarray:
        """
        Input: 2D numpy array in [0,1]
        Output: rotated 2D array so that left–right midline is vertical.
        Chooses rotation k in {0, 90, 270} that minimizes vertical symmetry MSE.
        """
        assert sl01.ndim == 2, "expected 2D slice"
        # light brain mask to reduce background influence
        m = self._brain_mask_otsu(sl01)

        def vscore(x):
            return self._vertical_sym_mse(x, m)

        cands = [
            (vscore(sl01), 0),
            (vscore(np.rot90(sl01, 1)), 1),
            (vscore(np.rot90(sl01, 3)), 3),
        ]
        _, k = min(cands, key=lambda t: t[0])
        if k == 0:
            return sl01
        return np.rot90(sl01, k)

    def ensure_vertical_pil(self, pil_img):
        """Convenience wrapper for PIL grayscale images."""
        import numpy as np
        arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
        arr_v = self.ensure_vertical_orientation(arr)
        from PIL import Image
        return Image.fromarray((arr_v * 255.0).astype(np.uint8), mode="L")

def create_unet_dataloaders(
    batch_size: int = 8,
    val_size: float = 0.2,
    num_workers: int = 2,
    image_size: int = 128,
    apply_unsharp: bool = True,
    pin_memory: bool = True,
    data_source: str = "hf",
    adni_path: Optional[str] = None,
    adni_image_type: str = "axial",                 # axial, coronal, sagittal
    adni_series_filter: Optional[List[str]] = None, # e.g., ["MT1","MPRAGE"]
    adni_label_csv: Optional[str] = None,           # optional filename->label csv
    adni_middle_frac: float = 0.4,                  # central fraction of slices
    adni_middle_subsample: int = 1,                 # stride in that band
    seed: int = 42,
):
    """
    Tạo DataLoader cho UNet với dataset Falah/Alzheimer_MRI.

    Trả về:
        train_loader, val_loader, test_loader
    """
    if data_source == "hf":
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
        hf_val   = raw_train.select(val_idx)

        print(f"Train size: {len(hf_train)}")
        print(f"Val size:   {len(hf_val)}")
        print(f"Test size:  {len(raw_test)}")

        train_ds = AlzheimerUNetDataset(hf_dataset=hf_train, image_size=image_size, apply_unsharp=apply_unsharp)
        val_ds   = AlzheimerUNetDataset(hf_dataset=hf_val,   image_size=image_size, apply_unsharp=apply_unsharp)
        test_ds  = AlzheimerUNetDataset(hf_dataset=raw_test, image_size=image_size, apply_unsharp=apply_unsharp)

        
    elif data_source == "adni":
        if adni_path is None:
            raise ValueError("For data_source='adni', please provide adni_path pointing to .nii files")

        full_ds = AdniNiftiSliceDataset(
            root_dir=adni_path,
            image_size=image_size,
            apply_unsharp=apply_unsharp,
            adni_image_type=adni_image_type,
            adni_series_filter=adni_series_filter,
            adni_label_csv=adni_label_csv,
            middle_frac=adni_middle_frac,
            middle_subsample=adni_middle_subsample,
        )
        N = len(full_ds)
        indices = np.arange(N)

        # Stratify only if labels are available and contain 2+ unique values
        labels = None
        if adni_label_csv is not None:
            labels = np.array([full_ds.label_map.get(
                os.path.basename(path)[:-7] if str(path).endswith(".nii.gz") else os.path.basename(path)[:-4],
                -1
            ) for path, _ in full_ds.index])
            uniq = np.unique(labels)
            if len(uniq) < 2 or np.any(labels < 0):
                labels = None  # fallback to random split

        if labels is None:
            tr_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=seed, shuffle=True)
        else:
            tr_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=seed, stratify=labels)

        # A small held out test from val side to match your HF signature
        # Here we split val 50 50 for val and test unless your ADNI supply a distinct test
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=seed)

        # Subset wrappers
        class _Subset(Dataset):
            def __init__(self, base, idxs):
                self.base = base
                self.idxs = list(idxs)
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.base[self.idxs[i]]

        train_ds = _Subset(full_ds, tr_idx)
        val_ds   = _Subset(full_ds, val_idx)
        test_ds  = _Subset(full_ds, test_idx)

        print(f"ADNI total slices: {N}")
        print(f"Train size: {len(train_ds)}")
        print(f"Val size:   {len(val_ds)}")
        print(f"Test size:  {len(test_ds)}")

    else:
        raise ValueError("data_source must be 'hf' or 'adni'")

    def make_loader(ds, shuffle: bool, drop_last: bool = False, pin_memory=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    train_loader = make_loader(train_ds, shuffle=True,  drop_last=True,  pin_memory=pin_memory)
    val_loader   = make_loader(val_ds,   shuffle=False, drop_last=False, pin_memory=pin_memory)
    test_loader  = make_loader(test_ds,  shuffle=False, drop_last=False, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


class FolderUNetDataset(Dataset):
    """
    Loads images using a CSV with columns:
      - 'img_path': relative or absolute path to image file
      - 'abnormal_type': mapped to labels via `mindset_label_map_idx_1` and `mindset_label_map_idx_2`

    Returns:
      {
        "input":    tensor [1,H,W] (optionally unsharp),
        "target":   same as input,
        "original": tensor [1,H,W] before unsharp,
        "label_1":  int64 tensor (e.g. detailed class),
        "label_2":  int64 tensor (e.g. Normal vs Alzheimer vs Other),
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

        # Map abnormal_type -> label_1, label_2
        def map_label_1(key: str) -> Optional[int]:
            return mindset_label_map_idx_1.get(key, None)

        def map_label_2(key: str) -> Optional[int]:
            return mindset_label_map_idx_2.get(key, None)

        df["label_1"] = df["abnormal_type"].map(map_label_1)
        df["label_2"] = df["abnormal_type"].map(map_label_2)

        # Report and drop rows with unknown mapping in either label_1 or label_2
        unknown = df[df["label_1"].isna() | df["label_2"].isna()]
        if len(unknown) > 0:
            print(
                f"[FolderUNetDataset] Warning: {len(unknown)} rows have unknown abnormal_type "
                f"for label_1 or label_2 and will be skipped. "
                f"Examples: {unknown['abnormal_type'].unique()[:10]}"
            )
            df = df.dropna(subset=["label_1", "label_2"])

        # Build absolute paths
        def make_full_path(p: str) -> Path:
            p = p.strip()
            # If already absolute, keep; else join with image_dir
            return Path(p) if os.path.isabs(p) else (self.image_dir / p)

        df["full_path"] = df["img_path"].apply(make_full_path)

        # Optional pre validation: existence + PIL openable
        samples: List[Tuple[Path, int, int]] = []
        bad_count = 0
        for _, row in df.iterrows():
            path: Path = row["full_path"]
            label_1 = int(row["label_1"])
            label_2 = int(row["label_2"])

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
                        print(
                            f"[FolderUNetDataset] Unreadable image skipped: "
                            f"{path} ({type(e).__name__})"
                        )
                    bad_count += 1
                    continue

            samples.append((path, label_1, label_2))

        if bad_count > warn_limit:
            print(
                f"[FolderUNetDataset] ...and {bad_count - warn_limit} more invalid or missing files skipped."
            )
        if len(samples) == 0:
            raise RuntimeError(
                "No valid images after CSV mapping and validation."
            )

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
        path, label_1, label_2 = self.samples[idx]
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
            "label_1": torch.tensor(label_1, dtype=torch.long),
            "label_2": torch.tensor(label_2, dtype=torch.long),
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
