"""
Training for Progressive U-Net JSCC Architecture.

Key features:
- Multi-rate pair training
- Skip connections utilized via U-Net
- Progressive refinement (rate-dependent channel groups)
- Grid-based validation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from tqdm import tqdm

from src.models.xai_pipeline_unet import XAISemanticCommUNet
from src.models.sr_sc import compute_k
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


def train_epoch(
    model, loader, optimizer, device, epoch,
    snr_range=(0, 20), rate_range=(0.1, 1.0),
    lambda_ssim=0.1, scheduler=None
):
    """Training epoch with multi-rate sampling."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f'Training [Epoch {epoch}]')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # Sample rate (discrete + uniform)
        if np.random.rand() < 0.5:
            rate = np.random.choice([0.25, 0.5, 0.75, 1.0])
        else:
            rate = np.random.uniform(*rate_range)
        
        # Sample SNR
        if np.random.rand() < 0.5:
            snr = np.random.choice([0, 5, 10, 15, 20])
        else:
            snr = np.random.uniform(*snr_range)
        
        optimizer.zero_grad()
        
        result = model(images, snr_db=snr, rate=rate)
        x_hat = result['output']
        
        # MSE + SSIM loss
        mse_loss = F.mse_loss(x_hat, targets)
        ssim_val = compute_ssim(x_hat, targets)
        ssim_loss = (1 - ssim_val).mean()
        
        loss = mse_loss + lambda_ssim * ssim_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = ssim_val.mean().item()
        
        loss_m.update(loss.item(), B)
        psnr_m.update(psnr, B)
        ssim_m.update(ssim, B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'ssim': f'{ssim_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg}


@torch.no_grad()
def validate_grid(model, loader, device, max_batches=20):
    """Grid validation."""
    model.eval()
    
    GRID = [
        (0.25, 5.0), (0.25, 10.0), (0.25, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
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
    
    # Monotonicity check
    mono_ok = 0
    for snr in [5.0, 10.0, 20.0]:
        if results[(0.25, snr)]['psnr'] <= results[(0.5, snr)]['psnr'] <= results[(1.0, snr)]['psnr']:
            mono_ok += 1
    mono_score = mono_ok / 3.0
    
    return {'grid_avg_psnr': avg_psnr, 'grid_avg_ssim': avg_ssim, 
            'grid': results, 'mono_score': mono_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent.parent
    
    logger.info("=" * 60)
    logger.info("Training Progressive U-Net JSCC")
    logger.info("=" * 60)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root_dir=str(project_root.parent / 'CocoData'),
        batch_size=args.batch_size,
        image_size=256,
        num_workers=16
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Model
    model = XAISemanticCommUNet(num_channels=256, pretrained_encoder=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    log_dir = project_root / 'results' / 'unet'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 1
    best_psnr = 0.0
    
    # Resume
    ckpt_path = project_root / 'checkpoints' / 'best_unet.pt'
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('grid_psnr', 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
    
    # Metrics CSV
    metrics_file = log_dir / 'metrics.csv'
    if not args.resume or not metrics_file.exists():
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,train_psnr,train_ssim,grid_avg_psnr,grid_avg_ssim,mono_score\n')
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch(
            model, train_loader, optimizer, device, epoch,
            lambda_ssim=args.lambda_ssim, scheduler=scheduler
        )
        
        val_m = validate_grid(model, val_loader, device)
        
        logger.info(f"Train - Loss: {train_m['loss']:.4f}, PSNR: {train_m['psnr']:.2f}, SSIM: {train_m['ssim']:.4f}")
        logger.info(f"Grid  - PSNR: {val_m['grid_avg_psnr']:.2f}, SSIM: {val_m['grid_avg_ssim']:.4f}, Mono: {val_m['mono_score']:.2%}")
        
        # Log key points every 5 epochs
        if epoch % 5 == 0:
            for r in [0.25, 0.5, 1.0]:
                p = val_m['grid'][(r, 10.0)]['psnr']
                s = val_m['grid'][(r, 10.0)]['ssim']
                logger.info(f"  rate={r:.2f}, SNR=10 -> PSNR={p:.2f}, SSIM={s:.4f}")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},{val_m['grid_avg_psnr']:.4f},{val_m['grid_avg_ssim']:.4f},{val_m['mono_score']:.4f}\n")
        
        # Save best
        if val_m['grid_avg_psnr'] > best_psnr:
            best_psnr = val_m['grid_avg_psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grid_psnr': best_psnr,
                'grid_ssim': val_m['grid_avg_ssim'],
                'mono_score': val_m['mono_score']
            }, project_root / 'checkpoints' / 'best_unet.pt')
            logger.info(f"[BEST] Grid PSNR: {best_psnr:.2f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, project_root / 'checkpoints' / 'latest_unet.pt')
    
    logger.info(f"\nComplete! Best Grid PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
