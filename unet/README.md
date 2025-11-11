# Self-Supervised UNet Design Plan

This document outlines the design for a **self-supervised learning framework** that combines **Masked Image Modeling (MIM)** and **Contrastive Learning** on 2D medical slices.  

---

## 1. Backbone and Embeddings

- **Backbone**: Small UNet encoder–decoder  
  - Encoder depth: 4  
  - Base channels: 16  
  - Bottleneck: 128  
- **Embedding head**:  
  - Global average pooling of the encoder bottleneck → 128-d vector  
  - Projection MLP for contrastive learning  
- **Decoder head**:  
  - Uses encoder skip connections for masked reconstruction

---

## 2. Image Size and Preprocessing

- **Image size**: `192 × 192`  
  - Memory-efficient for 2D slicing  
  - Divisible by common patch sizes  
  - Clean midline split and patch grid (`192 ÷ 16 = 12`)
- **Unsharp filter**: Enabled  
  - Applied to the **entire slice before** splitting or masking  
  - Uses existing module defaults

---

## 3. Data Flow per Batch

1. Load grayscale slice (1×H×W)  
2. Resize to `192×192`, apply Unsharp  
3. **Split vertically** into Left / Right halves (`1×192×96`)  
4. Use halves as two views for **InfoNCE**  
5. Create **masked patch sets** with the *anti-mirror rule* so mirrored patches are never both masked in the same sample

---

## 4. Masked Patch Reconstruction (MIM)

- **Patch size**: `16×16`, non-overlapping  
  - Full image: 12×12 grid  
  - Each half: 12×6 grid  
- **Mask ratio**: 0.35 per side  
- **Anti-mirror rule**:  
  - If a Left patch `(r, c)` is masked, the mirrored Right patch `(r, 6−1−c)` is excluded from the Right mask  
- **Per-view disjointness**: Masks unique within each side  
- **Network input**: Full (unmasked) image  
- **Loss**: L1 on masked patches only (optionally small L2 later)

---

## 5. Contrastive InfoNCE (Two-Halves Positives)

- **Positive pair**: Left vs Right halves from the same slice  
- **Projection head**: 2-layer MLP → 128-d output  
  - BatchNorm + L2 normalization  
- **Temperature**: 0.2  
- **Negatives**: All other halves in the batch (2N symmetric loss)

---

## 6. Augmentations

- **Order**:  
  1. Apply Unsharp to full image  
  2. Split midline  
  3. Apply light photometric transforms per half
- **Allowed**: Gaussian noise, mild intensity jitter, light blur  
- **Avoid**: Crops, flips, or rotations that break midline alignment  
- _(Optional)_ Global light rotation *before* split if needed later

---

## 7. Logging and Collapse Checks

- **Loss tracking**: Reconstruction + contrastive  
- **Embedding variance**:  
  - Compute per-dimension batch variance of L2-normalized embeddings  
  - Log mean and min variance (collapse detection)
- **Qualitative panels**: Per-epoch visualization  
  - Original, mask map, reconstruction, residual  
- **t-SNE visualization**: Every 5 epochs  
  - On a fixed subset, colored by dataset label

---

## 8. Training Defaults

| Setting | Value |
|----------|-------|
| Batch size | 64 (or 32 if limited) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | Cosine decay with warmup |
| Precision | Mixed precision (AMP) |
| Loss weights | λ_recon = 1.0, λ_contrast = 1.0 |
| Epochs | 100 (early stop on validation loss) |

---

## 9. Deliverables

- **Training script or notebook**  
  - Uses existing DataLoader  
  - Adds self-supervised collate for halves and masks  
  - Logs all metrics and visuals  
- **Outputs**  
  - Reconstruction panels  
  - t-SNE figures (per schedule)  
  - Checkpoints  
  - Mini README for usage

---

## 10. Confirmation Checklist

Please confirm or edit the following before implementation:

1. `image_size = 192`, `patch_size = 16`  
2. `mask_ratio = 0.35` per side, with anti-mirror rule  
3. Photometric-only augmentations after splitting  
4. Projection dimension = 128, temperature = 0.2  
5. Losses: L1 (MIM) + InfoNCE (equal weights)  
6. t-SNE every 5 epochs, colored by dataset labels

---

**If approved, the next step is full implementation following this exact design.**
