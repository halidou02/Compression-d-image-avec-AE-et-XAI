# Custom JSCC Training Guide

## 📋 Contexte

Script d'entraînement pour un modèle JSCC (Joint Source-Channel Coding) personnalisé avec:
- **Encoder/Decoder custom** (~30M params total)
- **MS-SSIM + Perceptual Loss** pour qualité visuelle
- **Teacher Grad-CAM** (ResNet-152) pour guidance sémantique
- **Progressive rate control** avec remask après AWGN

## 🏗️ Architecture

| Composant | Détails |
|-----------|---------|
| Encoder | 64→128→256→512→256 latent |
| Decoder | 256→512→256→128→64→3 |
| Latent | 256 canaux × 16×16 |
| Params | ~30M (14M enc + 15M dec) |

## 🔧 Installation

```bash
# Cloner le repo
git clone https://github.com/halidou02/Compression-d-image-avec-AE-et-XAI.git
cd Compression-d-image-avec-AE-et-XAI

# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# ou: source venv/bin/activate  # Linux

# Installer dépendances
pip install torch torchvision numpy pillow tqdm
```

## ✅ Vérification (smoke test)

```bash
python smoke_test.py
```

Résultat attendu:
```
ALL TESTS PASSED! ✓
- CustomEncoder: ~14M params
- CustomDecoder: ~15M params
- CustomJSCC: ~30M params
```

## 🚀 Lancement de l'entraînement

### Windows (RTX A4000 16Go)

```bash
python train_custom_jscc.py --data_dir C:\chemin\vers\coco --batch_size 16 --epochs 100
```

### Linux

```bash
python train_custom_jscc.py --data_dir /chemin/vers/coco --batch_size 16 --epochs 100
```

### Arguments disponibles

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *requis* | Chemin vers les images COCO (256×256) |
| `--batch_size` | 32 | Réduire à 16 si OOM |
| `--epochs` | 100 | Nombre d'epochs |
| `--lr` | 2e-4 | Learning rate initial |
| `--resume` | False | Reprendre depuis checkpoint |
| `--output_dir` | ./output_custom | Dossier de sortie |

## 📊 Monitoring

Les métriques sont sauvegardées dans `output_custom/metrics.csv`:
- train_loss, train_psnr, train_ssim, train_budget
- grid_avg_psnr, grid_avg_ssim, mono_score

## 🎯 Objectifs de performance

| Métrique | Cible |
|----------|-------|
| Grid PSNR | 27-28 dB |
| Grid SSIM | 0.85-0.87 |
| Mono Score | 100% |

## ⏱️ Temps estimé

| GPU | Batch | Temps/epoch | Total 100 epochs |
|-----|-------|-------------|------------------|
| RTX A4000 (16Go) | 16 | ~10 min | ~17h |
| RTX 3080 (10Go) | 12 | ~15 min | ~25h |

## 📝 Corrections appliquées

- ✅ MS-SSIM: formule corrigée (w[-1] exponent)
- ✅ FiLM: dtype/device safe (x.new_tensor)
- ✅ Rate control: remask après AWGN
- ✅ SR-SC: progressive coding (ordered selection)
