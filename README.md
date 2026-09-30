# Brain Tumor Segmentation with Swin-UNet

A from-scratch implementation of a Swin Transformer–based U-Net for binary brain tumor segmentation on the BraTS 2020 dataset. This project was built to demonstrate practical, applied understanding of the Swin Transformer architecture (window attention, shifted-window attention, hierarchical multi-scale representations) by using it as the backbone of a full encoder-decoder segmentation pipeline, combined with U-Net design principles.

## Motivation

Most tutorials that use Swin Transformer stop at image classification, where the model only needs its final-stage output. This project intentionally goes further: it uses Swin as a **general-purpose vision backbone** by extracting features from **all four stages** and feeding them into a **U-Net-style decoder**, proving that Swin's hierarchical, multi-scale design (unlike plain ViT, which keeps a single resolution throughout) makes it suitable for dense prediction tasks such as segmentation — not just classification.

The project mirrors the same validation strategy used in the original Swin Transformer paper, which benchmarks the architecture across classification (ImageNet), detection (COCO), and segmentation (ADE20K) to support its "general-purpose backbone" claim. Here, the same idea is demonstrated at a smaller scale on medical imaging data.

## Dataset

**BraTS 2020** (Brain Tumor Segmentation Challenge 2020), sourced from Kaggle (`awsaf49/brats20-dataset-training-validation`).

- 371 patient cases with ground-truth segmentation masks (`BraTS2020_TrainingData`)
- Each case includes 4 MRI modalities: **FLAIR, T1, T1CE, T2** (3D volumes, shape `240 × 240 × 155`)
- Each case includes a segmentation mask (`seg.nii`) with 4 original classes: `0` (background), `1` (necrotic/non-enhancing core), `2` (peritumoral edema), `4` (GD-enhancing tumor)
- `BraTS2020_ValidationData` was **not used** — it ships without ground-truth masks (withheld by the original competition organizers for official scoring), so it has no value for local training or evaluation.

### Why 2D instead of 3D

BraTS volumes are inherently 3D, but this project deliberately works in **2D**, slicing each volume along the axial plane. This decision was made because:

- The project's goal is to demonstrate understanding of **Swin Transformer V1**, which is a 2D architecture. Using a native 3D Swin variant (e.g. SwinUNETR) would require learning a different architecture entirely, diluting the project's core purpose.
- 2D training is dramatically lighter on GPU memory and time — a necessity given the free-tier compute budget (Kaggle's 30 GPU-hours/week).

## Data Pipeline

Implemented in `datasets.py` via a custom `BraTSDataset` (PyTorch `Dataset`).

**Key design decisions:**

| Decision | Choice | Reasoning |
|---|---|---|
| Dimensionality | 2D axial slices | Matches Swin V1's native 2D design; far lighter on compute |
| Modalities used | All 4 (FLAIR, T1, T1CE, T2) stacked as channels | Richer input than a single modality; more clinically realistic |
| Segmentation target | Binary (tumor vs. background) | Reduces class imbalance complexity and lets the project focus on proving the architecture works, rather than fighting multi-class imbalance |
| Slice filtering | Only slices with ≥1% tumor pixels are kept | Removes the large majority of near-empty background-only slices, which would otherwise dominate training and waste compute |
| Resize | 240×240 → 224×224 | Matches the standard input size Swin's ImageNet-pretrained weights expect |
| Normalization | Per-slice z-score (mean 0, std 1) | Standard practice in medical imaging, where intensity ranges vary slice to slice and scan to scan |
| Mask resizing | Nearest-neighbor interpolation | Preserves exact binary values (0/1); linear interpolation would introduce invalid fractional values |

**Indexing strategy:** rather than loading every modality volume up front, the dataset builds a lightweight index of `(patient_id, slice_index)` pairs at initialization (scanning only the segmentation volumes to determine tumor content), then loads the actual image data lazily inside `__getitem__`. This keeps memory usage low while still allowing fast, filtered access to only the relevant slices.

**Robustness:** the indexing step wraps file loading in a `try/except`, skipping any patient with missing or corrupted files (a known issue with a small number of cases in the public BraTS2020 Kaggle mirror) instead of crashing the whole pipeline.

**Result:** 371 patients → **17,510 valid 2D slices** after filtering.

### Train/Validation Split

The split (85% / 15%) is performed **at the patient level, not the slice level**. This is a deliberate and important choice: slices from the same patient are highly similar to their neighbors, so splitting at the slice level would leak information between training and validation, producing an artificially inflated (and misleading) validation score. Splitting by patient ID guarantees no patient's data appears in both sets.

## Model Architecture: Swin-UNet

Implemented across `models/swin_encoder.py`, `models/patch_expanding.py`, `models/decoder_block.py`, and `models/swin_unet.py`.

This is a full encoder-decoder U-Net where **both halves are built from Swin Transformer components** — not just the encoder — closely following the design proposed in the original Swin-UNet research, rather than a simplified CNN-decoder shortcut.

### Encoder: Swin-Small (pretrained)

- Backbone: `swin_small_patch4_window7_224` from `timm`, pretrained on ImageNet.
- Loaded with `features_only=True` to expose all 4 hierarchical stages as separate feature maps (not just the final classification output) — this is exactly what makes multi-scale skip connections possible.
- **Stage outputs:**

| Stage | Resolution | Channels |
|---|---|---|
| 0 | 56×56 | 96 |
| 1 | 28×28 | 192 |
| 2 | 14×14 | 384 |
| 3 | 7×7 | 768 |

**Adapting the input layer to 4 channels:** Swin-Small's pretrained patch embedding expects 3-channel RGB input. Since this project stacks 4 MRI modalities as channels, the patch embedding's `Conv2d` layer was replaced with a 4-channel version. Rather than randomly initializing this new layer (which would discard useful pretrained structure), its weights were initialized by **averaging the original 3-channel weights and repeating that average across all 4 new input channels**. This lets the network start from a sensible, pretrained-informed state rather than from scratch, while still allowing the 4 channels to differentiate from each other during fine-tuning.

### Patch Expanding (custom, built from scratch)

No standard library provides a "patch expanding" layer — it is specific to Swin-UNet-style architectures and has no equivalent in general-purpose vision libraries. This was implemented manually as the mirror image of Swin's own Patch Merging operation:

- Patch Merging (encoder direction): combines a 2×2 block of neighboring patches into one, **halving resolution** and **doubling channels**.
- Patch Expanding (decoder direction, built here): a `Linear` layer first **doubles** the channel dimension, then a reshape/permute operation redistributes those channels across space, **doubling resolution** and **quartering channels down to half of the original** — the exact inverse transformation.

This was verified independently: an input of `(7, 7, 768)` correctly expands to `(14, 14, 384)`.

### Decoder: Swin Transformer Blocks + Skip Connections

Each decoder stage performs, in order:
1. **Patch Expanding** — upsamples resolution, reduces channels.
2. **Skip connection fusion** — concatenates the upsampled features with the corresponding encoder stage's output (same resolution level), then a `Linear` layer projects the concatenated features back down to the target channel width.
3. **Two Swin Transformer Blocks** (reused directly from `timm`'s `SwinTransformerBlock`, not implemented from scratch) — one with `shift_size=0` (regular windowed attention) and one with `shift_size=window_size // 2` (shifted-window attention). This paired-block pattern is the same shifted-window mechanism used throughout the encoder, ensuring information can flow *between* windows, not just within them.

Using `timm`'s existing `SwinTransformerBlock` for the decoder (rather than reimplementing window attention from scratch) was a deliberate engineering choice: it reduces implementation risk and debugging time while still requiring a solid understanding of the block's expected input shapes and parameters (input resolution, number of heads, window size) — which differ at every decoder stage and had to be configured correctly by hand.

**Skip connections** link every encoder stage to its matching decoder stage:

```
Encoder Stage 3 (7×7,   768) ──────────────► Decoder input
Encoder Stage 2 (14×14, 384) ──skip──► Decoder Block 1 → (14×14, 384)
Encoder Stage 1 (28×28, 192) ──skip──► Decoder Block 2 → (28×28, 192)
Encoder Stage 0 (56×56, 96)  ──skip──► Decoder Block 3 → (56×56, 96)
                                        Final Expanding → (224×224, 1)
```

### Final Output Head

The last decoder stage (56×56) is still 4× smaller than the original input resolution (224×224), because the initial patch embedding downsamples by a factor of 4 before the encoder even begins. Two consecutive Patch Expanding operations (56→112→224) close this gap, followed by a 1×1 `Conv2d` that maps the final feature channels to a single-channel segmentation logit map.

**Verified end-to-end shape flow:**
```
Input:  (B, 4, 224, 224)
Output: (B, 1, 224, 224)   — matches target mask shape exactly
```

### Why this architecture choice matters

This design was chosen deliberately over a simpler alternative (a plain convolutional decoder) because it proves a stronger claim: not only can Swin serve as a feature extractor for segmentation, but its own attention mechanism and hierarchical logic can be **mirrored and reused symmetrically** on the decoding side. Building the Patch Expanding layer from scratch, in particular, required genuinely understanding — not just applying — the Patch Merging operation well enough to invert it correctly.

## Loss Function

Implemented in `losses.py` as `CombinedLoss` = `0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss`.

- **Dice Loss** directly optimizes for mask overlap and is far more robust to class imbalance (tumor pixels are a small minority even after slice filtering) than pixel-wise loss alone.
- **BCE** contributes training stability and well-behaved gradients, complementing Dice's sometimes unstable gradients early in training.
- Combining both is a standard, well-established practice in medical image segmentation for exactly this reason.

## Training Setup

| Component | Choice | Reasoning |
|---|---|---|
| Optimizer | `AdamW` | Standard choice for Vision Transformers; decouples weight decay from the gradient update, unlike vanilla Adam |
| Learning rate | Encoder: `1e-5`, Decoder: `1e-4` (discriminative LR) | The encoder is already pretrained and should be fine-tuned gently to avoid destroying learned features; the decoder is training from scratch and needs a higher rate to learn quickly |
| Mixed precision | `torch.cuda.amp` (autocast + `GradScaler`) | Roughly halves training time and memory usage on the T4 GPU with negligible accuracy impact |
| Batch size | 8 | Fits comfortably in T4 memory with Swin-Small at mixed precision |
| Early stopping | Custom `EarlyStopping` class, monitoring **validation Dice**, `patience=5`, `min_delta=0.001` | Prevents overfitting and wastes no compute once the model stops improving; Dice was chosen over validation loss as the stopping criterion because it directly reflects segmentation quality, which is the metric that actually matters here |
| Checkpointing | Best model (highest val Dice) saved to `best_model.pt` after every improving epoch | Ensures the final evaluated model is the best one seen during training, not simply the last one |

## Evaluation

Implemented in `metrics.py` and `evaluate.py`. Five complementary metrics were computed per-image over the full validation set and averaged, rather than relying on a single number:

| Metric | Purpose |
|---|---|
| **Dice Similarity Coefficient** | Standard overlap measure for segmentation; matches the training loss for consistency |
| **IoU (Jaccard Index)** | A stricter overlap measure than Dice; commonly reported alongside it |
| **Precision** | Measures how many predicted tumor pixels are actually correct — penalizes over-segmentation |
| **Recall (Sensitivity)** | Measures how much of the true tumor area was found — penalizes under-segmentation / missed tumor |
| **HD95 (95th-percentile Hausdorff Distance)** | Measures *boundary* accuracy in pixels, computed via Euclidean distance transforms; the 95th percentile (rather than the true maximum) is used to avoid a single noisy outlier pixel distorting the score — standard practice in medical segmentation literature |

Precision and Recall were both included deliberately (not just Dice/IoU) because they reveal *which kind* of error the model tends to make — over-segmenting vs. missing tumor — information that a single overlap score cannot provide on its own.

## Results

| Metric | Score |
|---|---|
| Dice Similarity Coefficient | **0.9099** |
| IoU | **0.8421** |
| Precision | **0.9174** |
| Recall | **0.9145** |
| HD95 | **2.06 px** |

**Interpretation:**
- A Dice score of ~0.91 sits within the range reported by specialized, purpose-built BraTS segmentation research (typically 0.85–0.92 for whole-tumor segmentation), which is a strong result for a project of this scope.
- Precision (0.9174) and Recall (0.9145) are closely balanced, indicating the model is not systematically biased toward over- or under-predicting tumor area.
- An HD95 of ~2 pixels indicates that predicted tumor boundaries are, on average, spatially very close to the true boundaries — the model is not just getting the right general area, but the right shape.
- Qualitative visual comparisons (input / ground truth / prediction, sampled from the validation set) confirm these numbers: predicted masks closely match ground-truth tumor location and shape, with predictions appearing slightly smoother/broader than the more jagged ground-truth annotations — consistent with recall being marginally higher than precision.

## Project Structure

```
Brain-Tumor-Segmentation/
├── README.md
├── notebooks/
│   └── training_notebook.ipynb    # Full Kaggle notebook: data exploration, training, evaluation
├── configs/
│   └── config.yaml                 # All hyperparameters (data, model, training, loss) in one place
├── datasets.py                      # BraTSDataset: loading, filtering, normalization
├── losses.py                         # DiceLoss, CombinedLoss
├── metrics.py                         # Dice, IoU, Precision, Recall, HD95
├── train.py                            # train_one_epoch, validate_one_epoch
├── evaluate.py                          # Full validation-set evaluation loop
├── early_stopping.py                     # EarlyStopping utility class
└── models/
    ├── __init__.py
    ├── swin_encoder.py                    # Pretrained Swin-Small, 4-channel patch embedding
    ├── patch_expanding.py                  # Custom inverse of Patch Merging
    ├── decoder_block.py                     # Patch Expand + skip fusion + Swin blocks
    └── swin_unet.py                          # Full model: encoder + decoder + segmentation head
```

**Note on model weights:** the trained checkpoint (`best_model.pt`) is not committed to this repository, since GitHub's per-file size limit (100 MB without Git LFS) is close to or below the checkpoint's size. The weights are instead hosted externally — see the link below.

**Model checkpoint:** [it will be available soon]

## Engineering Workflow

- **Code lives on GitHub; compute happens on Kaggle.** All architecture and training code was written as proper, version-controlled `.py` modules in this repository rather than as a single monolithic notebook. Kaggle notebooks `git clone`/`git pull` the repository and execute the code against a free T4 GPU (Kaggle offers 30 GPU-hours/week, with more predictable session stability than Colab's free tier).
- **GPU acceleration:** trained on a Kaggle T4 GPU using mixed-precision training.
- Total training consumed under 10 of the available 30 weekly GPU-hours, with early stopping preventing unnecessary further compute usage once validation Dice plateaued.

## Key Takeaways / What This Project Demonstrates

1. **Practical understanding of Swin Transformer V1** — window attention, shifted-window attention, hierarchical patch merging, and how to extract and use *all* stage outputs, not just the final classification embedding.
2. **Ability to invert an architecture's core mechanism** — designing Patch Expanding from first principles, based on understanding *why* Patch Merging works, not just how to call it.
3. **Sound handling of pretrained weights under a domain mismatch** — adapting a 3-channel ImageNet-pretrained input layer to a 4-channel medical imaging input without discarding pretrained knowledge.
4. **Correct handling of medical imaging pitfalls** — patient-level (not slice-level) train/validation splitting to avoid data leakage, class-imbalance-aware loss design, and boundary-sensitive evaluation (HD95) rather than relying on overlap metrics alone.
5. **End-to-end engineering discipline** — a modular, version-controlled codebase separating data, model, training, and evaluation concerns, rather than an unstructured single notebook.
