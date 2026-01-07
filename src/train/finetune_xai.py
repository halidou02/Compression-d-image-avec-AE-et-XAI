"""
Fine-Tuning Script - LR/10 from Best Checkpoint.

Strategy:
- Load best_gradcam.pt (not latest)
- LR = 0.0001 (original 0.001 / 10)
- CosineAnnealingLR (gentle decay)
- Log LR per epoch for correlation with metrics
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
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.xai_pipeline import XAIGuidedSemanticComm
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim, ReconstructionLoss
from src.utils.seed import set_seed, get_device
from src.utils.logging_ import setup_logger, AverageMeter
from src.utils.viz import plot_training_curves, visualize_reconstruction
from src.utils.gradcam import GradCAMHook, visualize_gradcam, visualize_channel_energy


def train_epoch(
    model: XAIGuidedSemanticComm,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradcam_hook: GradCAMHook,
    snr_range: tuple = (0, 20),
    rate_range: tuple = (0.1, 1.0),
) -> dict:
    """Training with SINGLE backward. No per-step scheduler."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    budget_m = AverageMeter()
    
    pbar = tqdm(loader, desc='Fine-tuning')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        snr = np.random.uniform(*snr_range)
        rate = np.random.uniform(*rate_range)
        
        optimizer.zero_grad()
        
        # Forward
        result = model(images, snr_db=snr, rate=rate, return_intermediate=True)
        x_hat = result['output']
        features = result['features']
        
        # Loss
        loss_dict = criterion(x_hat, targets)
        L = loss_dict['total']
        
        # Single backward
        L.backward()
        
        # CAM monitoring (no backward, uses existing grads)
        with torch.no_grad():
            try:
                cam = gradcam_hook.compute_cam()
                energy = (cam * features.abs()).mean(dim=(2, 3))
                k = max(1, min(256, int(round(rate * 256))))
                budget = (energy[:, k:].sum() / (energy.sum() + 1e-8)).item()
            except:
                budget = 0.0
        
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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


def main():
    parser = argparse.ArgumentParser(description='Fine-tune from best checkpoint with LR/10')
    parser.add_argument('--config', default='configs/recon.yaml')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10, help='Fine-tuning epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='LR = original/10 (default: 0.0001)')
    parser.add_argument('--lambda_ssim', type=float, default=0.05)
    parser.add_argument('--checkpoint', default='best_gradcam.pt', help='Checkpoint to load')
    parser.add_argument('--no_scheduler', action='store_true', help='Disable scheduler (constant LR)')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(42)
    device = get_device()
    
    log_dir = project_root / 'results' / 'finetune'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('finetune', log_dir / 'finetune.log')
    
    logger.info("=" * 50)
    logger.info("FINE-TUNING (LR/10 + CosineAnnealing)")
    logger.info("=" * 50)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Checkpoint: {args.checkpoint}, Epochs: {args.epochs}")
    
    train_loader, val_loader = get_dataloaders(
        root_dir=project_root.parent / 'CocoData',
        batch_size=args.batch_size,
        image_size=config.get('model', {}).get('image_size', 256),
        num_workers=16
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Load model from best checkpoint
    model = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=True).to(device)
    
    ckpt_path = project_root / 'checkpoints' / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_psnr_loaded = ckpt.get('psnr', ckpt.get('best_psnr', 0.0))
    loaded_epoch = ckpt.get('epoch', 0)
    logger.info(f"Loaded checkpoint from epoch {loaded_epoch}, PSNR: {best_psnr_loaded:.2f}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Hook on layer3
    gradcam_hook = GradCAMHook(model.encoder.layer3)
    
    criterion = ReconstructionLoss(lambda_mse=1.0, lambda_ssim=args.lambda_ssim)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # CosineAnnealingLR - gentle decay per EPOCH (not per batch)
    if args.no_scheduler:
        scheduler = None
        logger.info("Scheduler: DISABLED (constant LR)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)
        logger.info(f"Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min={args.lr/10:.6f})")
    
    (project_root / 'checkpoints').mkdir(exist_ok=True)
    best_psnr = best_psnr_loaded
    
    for epoch in range(1, args.epochs + 1):
        # Log current LR
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} | LR: {current_lr:.6f} {'='*20}")
        
        train_m = train_epoch(
            model, train_loader, optimizer, criterion, device, gradcam_hook,
            snr_range=config['training']['snr_range'],
            rate_range=config['training']['rate_range'],
        )
        val_m = validate(model, val_loader, criterion, device)
        
        # Step scheduler AFTER epoch (not per batch)
        if scheduler:
            scheduler.step()
        
        logger.info(f"Train - Loss: {train_m['loss']:.4f}, PSNR: {train_m['psnr']:.2f}, SSIM: {train_m['ssim']:.4f}, Budget: {train_m['budget']:.4f}")
        logger.info(f"Val   - Loss: {val_m['loss']:.4f}, PSNR: {val_m['psnr']:.2f}, SSIM: {val_m['ssim']:.4f}")
        
        # Viz every 5 epochs
        if epoch % 5 == 0:
            viz_dir = log_dir / 'finetune_viz'
            viz_dir.mkdir(exist_ok=True)
            model.eval()
            model.zero_grad(set_to_none=True)
            imgs, tgts = next(iter(val_loader))
            imgs, tgts = imgs[:4].to(device), tgts[:4].to(device)
            
            from src.utils.gradcam import GradCAMHook as GradCAMHook256, visualize_full
            hook_256 = GradCAMHook256(model.encoder.channel_reduce)
            
            with torch.enable_grad():
                res = model(imgs, snr_db=10.0, rate=0.5, return_intermediate=True)
                loss = ((res['output'] - tgts) ** 2).sum()
                loss.backward()
                cam = hook_256.compute_cam()
                energy = (cam * res['features'].abs()).mean(dim=(2, 3))
            
            for i in range(min(4, imgs.shape[0])):
                visualize_full(
                    imgs[i], res['output'][i], cam[i],
                    str(viz_dir / f'full_ft_e{epoch}_s{i}.png')
                )
            visualize_channel_energy(energy.mean(0), 0.5, str(viz_dir / f'energy_ft_e{epoch}.png'))
            
            hook_256.remove()
            model.zero_grad(set_to_none=True)
            logger.info(f"Saved visualizations to {viz_dir}")
        
        if val_m['psnr'] > best_psnr:
            best_psnr = val_m['psnr']
            torch.save({
                'epoch': loaded_epoch + epoch,
                'model_state_dict': model.state_dict(),
                'psnr': best_psnr,
                'finetune_epoch': epoch
            }, project_root / 'checkpoints' / 'best_finetune.pt')
            logger.info(f"[NEW BEST] PSNR: {best_psnr:.2f} (+{best_psnr - best_psnr_loaded:.2f} vs loaded)")
        
        torch.save({
            'epoch': loaded_epoch + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'finetune_epoch': epoch
        }, project_root / 'checkpoints' / 'latest_finetune.pt')
        
        # CSV with LR column
        with open(log_dir / 'finetune_metrics.csv', 'a') as f:
            if epoch == 1:
                f.write('epoch,lr,train_loss,train_psnr,train_ssim,train_budget,val_loss,val_psnr,val_ssim\n')
            f.write(f"{epoch},{current_lr:.8f},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},")
            f.write(f"{train_m['budget']:.6f},{val_m['loss']:.6f},{val_m['psnr']:.4f},{val_m['ssim']:.6f}\n")
    
    gradcam_hook.remove()
    logger.info(f"\nFine-tuning complete!")
    logger.info(f"Best PSNR: {best_psnr:.2f} (improvement: +{best_psnr - best_psnr_loaded:.2f} dB)")


if __name__ == '__main__':
    main()
