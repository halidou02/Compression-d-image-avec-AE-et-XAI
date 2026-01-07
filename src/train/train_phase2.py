"""
Phase 2 Training: Multi-Rate Pair + L_mono + L_use + SSIM Schedule.

Key improvements:
- 2 forward passes per batch (rate_low, rate_high)
- L_mono: relu(L_high - L_low) to force monotonicity
- L_use: force channels [k:] to carry energy at rate=1.0
- SSIM schedule: 0.05 → 0.15 over epochs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import argparse
import logging
from tqdm import tqdm

from src.models.xai_pipeline import XAIGuidedSemanticComm
from src.models.sr_sc import compute_k
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim, ReconstructionLoss
from src.utils.gradcam import GradCAMHook

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def get_rate_pair():
    """Sample rate_low, rate_high pair."""
    if np.random.rand() < 0.7:
        # Fixed pairs from grid
        pairs = [(0.25, 0.5), (0.5, 0.75), (0.5, 1.0), (0.6, 1.0), (0.25, 1.0)]
        return pairs[np.random.randint(len(pairs))]
    else:
        # Random pair
        r_low = np.random.uniform(0.1, 0.8)
        r_high = min(1.0, r_low + np.random.uniform(0.15, 0.4))
        return r_low, r_high


def train_epoch_phase2(
    model, loader, optimizer, criterion, device, epoch,
    snr_range=(0, 20),
    lambda_mono=0.2, lambda_use=0.05,
    lambda_ssim_base=0.05, lambda_ssim_max=0.15, ssim_ramp_epochs=40,
    scheduler=None
):
    """Phase 2 training with multi-rate pairs."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    mono_m = AverageMeter()
    use_m = AverageMeter()
    
    # SSIM schedule: linear ramp
    if epoch <= ssim_ramp_epochs:
        lambda_ssim = lambda_ssim_base + (lambda_ssim_max - lambda_ssim_base) * (epoch / ssim_ramp_epochs)
    else:
        lambda_ssim = lambda_ssim_max
    
    pbar = tqdm(loader, desc=f'Training (λ_ssim={lambda_ssim:.3f})')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # Sample SNR
        if np.random.rand() < 0.5:
            snr = np.random.choice([0, 5, 10, 15, 20])
        else:
            snr = np.random.uniform(*snr_range)
        
        # Get rate pair
        r_low, r_high = get_rate_pair()
        
        optimizer.zero_grad()
        
        # === FORWARD LOW RATE ===
        result_low = model(images, snr_db=snr, rate=r_low, return_intermediate=True)
        out_low = result_low['output']
        
        # === FORWARD HIGH RATE ===
        result_high = model(images, snr_db=snr, rate=r_high, return_intermediate=True)
        out_high = result_high['output']
        xa_high = result_high.get('xa', result_high.get('features'))  # masked features
        
        # === LOSSES ===
        # Reconstruction loss for both
        L_low = criterion(out_low, targets)['total']
        L_high = criterion(out_high, targets)['total']
        
        # MSE for mono comparison (simpler than full loss)
        mse_low = ((out_low - targets).pow(2)).mean()
        mse_high = ((out_high - targets).pow(2)).mean()
        
        # L_mono: force L_high <= L_low (or mse_high <= mse_low)
        L_mono = torch.relu(mse_high - mse_low + 1e-4)
        
        # L_use: force channel [k:] utilization at high rate
        L_use = torch.tensor(0.0, device=device)
        if r_high >= 0.9:  # Only for near-full rate
            k = compute_k(r_high, 256)
            xa = xa_high
            e_all = xa.abs().mean(dim=(0, 2, 3))  # [256]
            e_tail = e_all[128:].mean()  # energy in top 128 channels
            e_head = e_all[:128].mean()  # energy in bottom 128
            ratio = e_tail / (e_head + 1e-8)
            L_use = torch.relu(0.15 - ratio)  # want at least 15% relative
        
        # SSIM loss (only on high rate output for simplicity)
        ssim_val = compute_ssim(out_high, targets)
        ssim_loss = (1 - ssim_val).mean()
        
        # Total loss
        L = L_low + L_high + lambda_ssim * ssim_loss + lambda_mono * L_mono + lambda_use * L_use
        
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Metrics (on high rate)
        with torch.no_grad():
            psnr = compute_psnr(out_high, targets).mean().item()
        
        loss_m.update(L.item(), B)
        psnr_m.update(psnr, B)
        mono_m.update(L_mono.item(), B)
        use_m.update(L_use.item(), B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'mono': f'{mono_m.avg:.4f}',
            'use': f'{use_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'mono': mono_m.avg, 
            'use': use_m.avg, 'lambda_ssim': lambda_ssim}


@torch.no_grad()
def validate_grid(model, loader, device, max_batches=20):
    """Grid validation with mono_score."""
    model.eval()
    
    GRID = [
        (0.25, 5.0), (0.25, 10.0), (0.25, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
        (0.6, 5.0), (0.6, 10.0), (0.6, 20.0),
        (1.0, 5.0), (1.0, 10.0), (1.0, 20.0),
    ]
    
    results = {}
    for rate, snr in GRID:
        psnr_m, ssim_m = AverageMeter(), AverageMeter()
        for i, (images, targets) in enumerate(loader):
            if i >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            result = model(images, snr_db=snr, rate=rate)
            psnr_m.update(compute_psnr(result['output'], targets).mean().item(), images.size(0))
            ssim_m.update(compute_ssim(result['output'], targets).mean().item(), images.size(0))
        results[(rate, snr)] = {'psnr': psnr_m.avg, 'ssim': ssim_m.avg}
    
    avg_psnr = np.mean([v['psnr'] for v in results.values()])
    avg_ssim = np.mean([v['ssim'] for v in results.values()])
    
    # Mono check: at each SNR, PSNR should increase with rate
    mono_pairs = 0
    total_pairs = 0
    tolerance = 0.1
    
    for snr in [5.0, 10.0, 20.0]:
        rates = [0.25, 0.5, 0.6, 1.0]
        psnrs = [results[(r, snr)]['psnr'] for r in rates]
        for i in range(len(rates) - 1):
            total_pairs += 1
            if psnrs[i] <= psnrs[i+1] + tolerance:
                mono_pairs += 1
    
    mono_score = mono_pairs / total_pairs if total_pairs > 0 else 0.0
    
    return {'grid_avg_psnr': avg_psnr, 'grid_avg_ssim': avg_ssim, 
            'grid': results, 'mono_score': mono_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lambda_mono', type=float, default=0.2)
    parser.add_argument('--lambda_use', type=float, default=0.05)
    parser.add_argument('--from_best', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent.parent
    
    logger.info("=" * 60)
    logger.info("Phase 2: Multi-Rate Pair + L_mono + L_use + SSIM Schedule")
    logger.info("=" * 60)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"λ_mono: {args.lambda_mono}, λ_use: {args.lambda_use}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root_dir=str(project_root.parent / 'CocoData'),
        batch_size=args.batch_size,
        image_size=256,
        num_workers=16
    )
    
    # Model
    model = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=True).to(device)
    
    # Load from best checkpoint
    if args.from_best:
        ckpt_path = project_root / 'checkpoints' / 'best_mono.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Loaded best_mono.pt (epoch {ckpt.get('epoch', '?')}, grid_psnr={ckpt.get('grid_psnr', 0):.2f})")
    
    # Criterion
    criterion = ReconstructionLoss(lambda_mse=1.0, lambda_ssim=0.0)  # SSIM handled separately
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    log_dir = project_root / 'results' / 'phase2'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    best_score = 0.0
    
    metrics_file = log_dir / 'metrics.csv'
    with open(metrics_file, 'w') as f:
        f.write('epoch,train_loss,train_psnr,train_mono,train_use,lambda_ssim,grid_avg_psnr,grid_avg_ssim,mono_score\n')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch_phase2(
            model, train_loader, optimizer, criterion, device, epoch,
            lambda_mono=args.lambda_mono, lambda_use=args.lambda_use,
            scheduler=scheduler
        )
        
        val_m = validate_grid(model, val_loader, device)
        
        logger.info(f"Train - PSNR: {train_m['psnr']:.2f}, Mono: {train_m['mono']:.4f}, Use: {train_m['use']:.4f}")
        logger.info(f"Grid  - PSNR: {val_m['grid_avg_psnr']:.2f}, SSIM: {val_m['grid_avg_ssim']:.4f}, Mono: {val_m['mono_score']:.2%}")
        
        # Log key grid points
        if epoch % 5 == 0:
            for r in [0.25, 0.6, 1.0]:
                p = val_m['grid'][(r, 10.0)]['psnr']
                s = val_m['grid'][(r, 10.0)]['ssim']
                logger.info(f"  rate={r:.2f}, SNR=10 -> PSNR={p:.2f}, SSIM={s:.4f}")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['mono']:.6f},{train_m['use']:.6f},{train_m['lambda_ssim']:.4f},{val_m['grid_avg_psnr']:.4f},{val_m['grid_avg_ssim']:.4f},{val_m['mono_score']:.4f}\n")
        
        # Best = grid_psnr + 5*mono_score
        score = val_m['grid_avg_psnr'] + 5 * val_m['mono_score']
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grid_psnr': val_m['grid_avg_psnr'],
                'grid_ssim': val_m['grid_avg_ssim'],
                'mono_score': val_m['mono_score']
            }, project_root / 'checkpoints' / 'best_phase2.pt')
            logger.info(f"[BEST] Score: {score:.2f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, project_root / 'checkpoints' / 'latest_phase2.pt')
    
    logger.info(f"\nComplete! Best Score: {best_score:.2f}")


if __name__ == '__main__':
    main()
