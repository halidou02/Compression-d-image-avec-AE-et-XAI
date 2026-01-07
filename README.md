# 📡 XAI-Guided Semantic Communication (JSCC)

Variable-rate Joint Source-Channel Coding with explainable AI guidance.

## 🏗️ Architecture

- **Encoder**: ResNet-50 (frozen BN) → 256 channels latent
- **SR-SC**: Ordered channel selection via SE-Block
- **PSSG**: Per-sample power normalization
- **Channel**: AWGN with SNR conditioning
- **Decoder**: Progressive U-Net (Self-Contained, no skip leak)

## 📊 Features

- ✅ Variable rate (0.1 - 1.0)
- ✅ Variable SNR (0 - 20 dB)
- ✅ Monotonic (rate↑ = quality↑)
- ✅ Teacher CAM guidance
- ✅ Budget loss for channel ordering

## 🚀 Quick Start

### Local Training
```bash
python -m src.train.train_noskip --batch_size 24 --epochs 100 --lr 2e-4
```

### Resume Training
```bash
python -m src.train.train_noskip --batch_size 24 --epochs 100 --lr 2e-4 --resume
```

### Colab Training
See `train_colab.ipynb` for Google Colab setup with A100.

## 📁 Structure

```
semantic_comm/
├── src/
│   ├── models/
│   │   ├── jscc_noskip.py      # Main pipeline
│   │   ├── encoder.py          # ResNet-50 encoder
│   │   ├── decoder_noskip.py   # Progressive decoder
│   │   ├── sr_sc.py            # Rate selection
│   │   └── pssg.py             # Power normalization
│   ├── train/
│   │   ├── train_noskip.py     # Main training script
│   │   └── train_prenet.py     # PreNet for adaptive rate
│   ├── channel/
│   │   └── awgn.py             # AWGN channel
│   └── utils/
│       ├── metrics.py          # PSNR, SSIM
│       └── gradcam.py          # Grad-CAM hooks
├── train_colab.ipynb           # Colab notebook
└── requirements.txt
```

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| PSNR | 27-28 dB |
| SSIM | 0.85-0.87 |
| Mono Score | 100% |

## 📚 References

Based on XAI-guided semantic communication principles.
