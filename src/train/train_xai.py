"""
Training Script - FINAL CORRECTED VERSION.

Single backward, CAM monitoring uses existing gradients.
No retain_graph, no double backward.
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.xai_pipeline import XAIGuidedSemanticComm
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim, ReconstructionLoss
from src.utils.seed import set_seed, get_device
from src.utils.logging_ import setup_logger, AverageMeter
from src.utils.viz import plot_training_curves, visualize_reconstruction
from src.utils.gradcam import GradCAMHook, compute_budget_loss, visualize_gradcam, visualize_channel_energy


def train_epoch(
    model: XAIGuidedSemanticComm,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradcam_hook: GradCAMHook,
    snr_range: tuple = (0, 20),
    rate_range: tuple = (0.1, 1.0),
    scheduler = None
) -> dict:
    """
    Training with SINGLE backward.
    CAM monitoring uses gradients already computed.
    """
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    budget_m = AverageMeter()
    
    pbar = tqdm(loader, desc='Training')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # FIXED: Mix discrete + uniform to ensure edge cases like rate=1.0 are seen
        if np.random.rand() < 0.5:
            rate = np.random.choice([0.1, 0.25, 0.5, 0.75, 1.0])
        else:
            rate = np.random.uniform(*rate_range)
        
        if np.random.rand() < 0.5:
            snr = np.random.choice([0, 5, 10, 15, 20])
        else:
            snr = np.random.uniform(*snr_range)
        
        # === SINGLE BACKWARD PATTERN ===
        optimizer.zero_grad()
        
        # Forward
        result = model(images, snr_db=snr, rate=rate, return_intermediate=True)
        x_hat = result['output']
        features = result['features']  # [B, 256, 16, 16]
        
        # Loss
        loss_dict = criterion(x_hat, targets)
        L = loss_dict['total']
        
        # Single backward (populates gradients for CAM)
        L.backward()
        
        # === CAM MONITORING (no backward, uses existing grads) ===
        with torch.no_grad():
            try:
                cam = gradcam_hook.compute_cam()  # Uses grads from L.backward()
                energy = (cam * features.abs()).mean(dim=(2, 3))  # [B, 256]
                k = max(1, min(256, int(round(rate * 256))))
                budget = (energy[:, k:].sum() / (energy.sum() + 1e-8)).item()
            except:
                budget = 0.0
        
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Clear hook tensors
        gradcam_hook.clear()
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = compute_ssim(x_hat, targets).mean().item()
        
        loss_m.update(L.item(), B)
        psnr_m.update(psnr, B)
        ssim_m.update(ssim, B)
        budget_m.update(budget, B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'ssim': f'{ssim_m.avg:.4f}',
            'budget': f'{budget_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg, 'budget': budget_m.avg}


@torch.no_grad()
def validate(model, loader, criterion, device, snr=10.0, rate=0.5):
    """Single-point validation (for backward compat)."""
    model.eval()
    loss_m, psnr_m, ssim_m = AverageMeter(), AverageMeter(), AverageMeter()
    
    for images, targets in tqdm(loader, desc='Validating'):
        images, targets = images.to(device), targets.to(device)
        result = model(images, snr_db=snr, rate=rate)
        x_hat = result['output']
        
        loss_dict = criterion(x_hat, targets)
        loss_m.update(loss_dict['total'].item(), images.size(0))
        psnr_m.update(compute_psnr(x_hat, targets).mean().item(), images.size(0))
        ssim_m.update(compute_ssim(x_hat, targets).mean().item(), images.size(0))
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg}


@torch.no_grad()
def validate_grid(model, loader, criterion, device):
    """Grid-based validation for robust multi-rate/SNR evaluation."""
    model.eval()
    
    GRID = [
        (0.25, 5.0), (0.25, 10.0), (0.25, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
        (1.0, 5.0), (1.0, 10.0), (1.0, 20.0),
    ]
    
    results = {}
    total_psnr, total_ssim = 0.0, 0.0
    
    # Only use first N batches for speed
    max_batches = 20
    
    for rate, snr in GRID:
        psnr_m, ssim_m = AverageMeter(), AverageMeter()
        for i, (images, targets) in enumerate(loader):
            if i >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            result = model(images, snr_db=snr, rate=rate)
            x_hat = result['output']
            psnr_m.update(compute_psnr(x_hat, targets).mean().item(), images.size(0))
            ssim_m.update(compute_ssim(x_hat, targets).mean().item(), images.size(0))
        
        results[(rate, snr)] = {'psnr': psnr_m.avg, 'ssim': ssim_m.avg}
        total_psnr += psnr_m.avg
        total_ssim += ssim_m.avg
    
    avg_psnr = total_psnr / len(GRID)
    avg_ssim = total_ssim / len(GRID)
    
    return {'grid_avg_psnr': avg_psnr, 'grid_avg_ssim': avg_ssim, 'grid': results}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/recon.yaml')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_ssim', type=float, default=0.05)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(42)
    device = get_device()
    
    log_dir = project_root / 'results' / 'gradcam'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('train', log_dir / 'train.log')
    
    logger.info("=" * 50)
    logger.info("Training (Single Backward + CAM Monitoring)")
    logger.info("=" * 50)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    
    train_loader, val_loader = get_dataloaders(
        root_dir=project_root.parent / 'CocoData',
        batch_size=args.batch_size,
        image_size=config.get('model', {}).get('image_size', 256),
        num_workers=16
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    model = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=True).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Hook on layer3
    gradcam_hook = GradCAMHook(model.encoder.layer3)
    
    criterion = ReconstructionLoss(lambda_mse=1.0, lambda_ssim=args.lambda_ssim)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    (project_root / 'checkpoints').mkdir(exist_ok=True)
    start_epoch, best_psnr = 1, 0.0
    
    if args.resume:
        ckpt = project_root / 'checkpoints' / 'latest_gradcam.pt'
        if ckpt.exists():
            c = torch.load(ckpt, map_location=device)
            model.load_state_dict(c['model_state_dict'])
            start_epoch = c['epoch'] + 1
            best_psnr = c.get('best_psnr', 0.0)
            logger.info(f"Resumed from epoch {start_epoch-1}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch(
            model, train_loader, optimizer, criterion, device, gradcam_hook,
            snr_range=config['training']['snr_range'],
            rate_range=config['training']['rate_range'],
            scheduler=scheduler
        )
        val_m = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train - Loss: {train_m['loss']:.4f}, PSNR: {train_m['psnr']:.2f}, SSIM: {train_m['ssim']:.4f}, Budget: {train_m['budget']:.4f}")
        logger.info(f"Val   - Loss: {val_m['loss']:.4f}, PSNR: {val_m['psnr']:.2f}, SSIM: {val_m['ssim']:.4f}")
        
        # Viz every 10 epochs
        if epoch % 10 == 0:
            viz_dir = log_dir / 'gradcam_viz'
            viz_dir.mkdir(exist_ok=True)
            model.eval()
            model.zero_grad(set_to_none=True)
            imgs, tgts = next(iter(val_loader))
            imgs, tgts = imgs[:4].to(device), tgts[:4].to(device)
            
            # Hook on 256ch (channel_reduce) for better interpretability
            from src.utils.gradcam import GradCAMHook as GradCAMHook256, visualize_full
            hook_256 = GradCAMHook256(model.encoder.channel_reduce)
            
            with torch.enable_grad():
                res = model(imgs, snr_db=10.0, rate=0.5, return_intermediate=True)
                loss = ((res['output'] - tgts) ** 2).sum()
                loss.backward()
                cam = hook_256.compute_cam()
                energy = (cam * res['features'].abs()).mean(dim=(2, 3))
            
            # Full visualization: Input, Recon, Error, CAM, Overlays
            for i in range(min(4, imgs.shape[0])):
                visualize_full(
                    imgs[i], res['output'][i], cam[i],
                    str(viz_dir / f'full_e{epoch}_s{i}.png')
                )
            visualize_channel_energy(energy.mean(0), 0.5, str(viz_dir / f'energy_e{epoch}.png'))
            
            hook_256.remove()
            model.zero_grad(set_to_none=True)
            logger.info(f"Saved enhanced visualizations to {viz_dir}")
        
        if val_m['psnr'] > best_psnr:
            best_psnr = val_m['psnr']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'psnr': best_psnr}, project_root / 'checkpoints' / 'best_gradcam.pt')
            logger.info(f"[BEST] PSNR: {best_psnr:.2f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'best_psnr': best_psnr},
                  project_root / 'checkpoints' / 'latest_gradcam.pt')
        
        with open(log_dir / 'metrics.csv', 'a') as f:
            if epoch == 1:
                f.write('epoch,train_loss,train_psnr,train_ssim,train_budget,val_loss,val_psnr,val_ssim\n')
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},")
            f.write(f"{train_m['budget']:.6f},{val_m['loss']:.6f},{val_m['psnr']:.4f},{val_m['ssim']:.6f}\n")
    
    gradcam_hook.remove()
    logger.info(f"\nComplete! Best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
