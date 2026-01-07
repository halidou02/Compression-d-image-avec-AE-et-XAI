# Assumptions and Design Decisions

This document describes the default assumptions and design choices made during implementation.

## Dataset

- **COCO Images**: Using pre-resized 256×256 COCO images from `../CocoData`
- **Pseudo-labels**: For classification mode, images are assigned pseudo-labels based on index mod 10
- **Train/Val Split**: 90%/10% random split with fixed seed

## Architecture

### Encoder
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Output**: 512 channels, 8×8 spatial resolution for 256×256 input
- **Downsampling**: 32× total (256 → 8)

### SR-SC (Semantic Rate Selection)
- **Default Mode**: L2 energy per channel for importance scoring
- **XAI Mode**: Use Grad-CAM attention map when available
- **Selection**: Top-k channels based on keep_ratio

### ACC (Adaptive Channel Condition)
- **Input**: Channel features + SNR (dB)
- **Architecture**: Global pooling → MLP → sigmoid scaling
- **Hidden Dim**: 256

### PSSG (Variable Rate)
- **Smax**: 256 (maximum transmitted symbols)
- **Rate Mask**: Binary mask, first S = ceil(r × Smax) symbols retained
- **Projection**: Linear layer from flattened features to Smax

### Channel
- **Type**: AWGN (Additive White Gaussian Noise)
- **SNR Range**: 0-20 dB during training
- **Power Normalization**: Signals normalized to unit power before channel

### Decoders
- **Classification**: MLP with 2 hidden layers (512, 256)
- **Reconstruction**: Transposed convolutions, 5 upsampling stages

## Training

### Multi-SNR Training
- SNR uniformly sampled from [0, 20] dB per batch
- Rate uniformly sampled from [0.1, 1.0]

### Loss Functions
- **Reconstruction**: MSE + 0.1 × (1 - SSIM)
- **Classification**: CrossEntropyLoss

### Optimization
- **Optimizer**: Adam with lr=0.0005 (recon), lr=0.001 (cls)
- **Scheduler**: Cosine annealing
- **Gradient Clipping**: max_norm=1.0

## Rate Selection (OCR)

### Data-Level
- Build utility table U(γ, r) via grid evaluation
- Choose minimum r such that U(γ, r) ≥ threshold

### Instance-Level
- Pre-Net predicts expected quality from latent statistics
- Grid search over r values, select minimum satisfying threshold

## XAI

### Grad-CAM
- Target layer: Last conv block of encoder
- Upsampling: Bilinear interpolation to feature size
- Channel importance: Weighted sum with attention map

### Importance Conversion
```
importance[c] = Σ_{h,w} (attention[h,w] × |features[c,h,w]|) / Σ attention
```

## Defaults Summary

| Parameter | Default Value |
|-----------|---------------|
| Image Size | 256×256 |
| Smax | 256 |
| Encoder | ResNet-18 |
| SNR Range | [0, 20] dB |
| Rate Range | [0.1, 1.0] |
| Batch Size | 16 (recon), 32 (cls) |
| Learning Rate | 0.0005 (recon), 0.001 (cls) |
| λ_MSE | 1.0 |
| λ_SSIM | 0.1 |
