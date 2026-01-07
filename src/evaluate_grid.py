"""
Evaluate model on rate × SNR grid.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from src.models.xai_pipeline import XAIGuidedSemanticComm
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim


def evaluate_grid(model, loader, device, max_batches=50):
    """Evaluate on rate × SNR grid."""
    model.eval()
    
    RATES = [0.1, 0.25, 0.5, 0.75, 1.0]
    SNRS = [0, 5, 10, 15, 20]
    
    results = {}
    
    print("\nEvaluating on rate × SNR grid...")
    print("=" * 60)
    
    for rate in RATES:
        for snr in SNRS:
            psnr_list, ssim_list = [], []
            
            with torch.no_grad():
                for i, (images, targets) in enumerate(loader):
                    if i >= max_batches:
                        break
                    images, targets = images.to(device), targets.to(device)
                    result = model(images, snr_db=float(snr), rate=rate)
                    x_hat = result['output']
                    psnr_list.append(compute_psnr(x_hat, targets).mean().item())
                    ssim_list.append(compute_ssim(x_hat, targets).mean().item())
            
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            results[(rate, snr)] = {'psnr': avg_psnr, 'ssim': avg_ssim}
            print(f"rate={rate:.2f}, SNR={snr:2d}dB -> PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
    
    print("=" * 60)
    
    # Summary table
    print("\n📊 PSNR Table (rate × SNR):")
    print("rate\\SNR |", " | ".join([f"{s:5d}" for s in SNRS]))
    print("-" * 50)
    for rate in RATES:
        row = [f"{results[(rate, s)]['psnr']:5.2f}" for s in SNRS]
        print(f"  {rate:.2f}  |", " | ".join(row))
    
    # Check monotonicity
    print("\n🔍 Monotonicity check (PSNR should increase with rate):")
    for snr in SNRS:
        psnrs = [results[(r, snr)]['psnr'] for r in RATES]
        monotonic = all(psnrs[i] <= psnrs[i+1] for i in range(len(psnrs)-1))
        print(f"  SNR={snr}dB: {psnrs} -> {'✅ Monotonic' if monotonic else '⚠️ NOT monotonic'}")
    
    # Global average
    all_psnr = np.mean([v['psnr'] for v in results.values()])
    all_ssim = np.mean([v['ssim'] for v in results.values()])
    print(f"\n📈 Global average: PSNR={all_psnr:.2f}dB, SSIM={all_ssim:.4f}")
    
    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=False).to(device)
    
    # Try different checkpoints
    ckpt_paths = [
        Path(__file__).parent.parent / 'checkpoints' / 'best_gradcam.pt',
        Path(__file__).parent.parent / 'checkpoints' / 'best_finetune.pt',
    ]
    
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            print(f"\n🔄 Loading: {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"   Epoch: {ckpt.get('epoch', '?')}, PSNR: {ckpt.get('psnr', '?')}")
            break
    else:
        print("⚠️ No checkpoint found!")
    
    # Load data
    _, val_loader = get_dataloaders(
        root_dir=str(Path(__file__).parent.parent.parent / 'CocoData'),
        batch_size=32,
        image_size=256,
        num_workers=8
    )
    
    # Evaluate
    results = evaluate_grid(model, val_loader, device)
