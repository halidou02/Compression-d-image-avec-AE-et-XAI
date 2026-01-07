"""
Training Script for Reconstruction Task.

Trains the semantic communication system for image reconstruction.
Uses MSE + SSIM loss.
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.full_pipeline import SemanticCommSystem
from src.data.datasets import CocoDataset, get_dataloaders
from src.data.transforms import get_reconstruction_transforms
from src.utils.metrics import compute_psnr, compute_ssim, ReconstructionLoss
from src.utils.seed import set_seed, get_device
from src.utils.logging_ import setup_logger, AverageMeter
from src.utils.viz import plot_training_curves, visualize_reconstruction


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    snr_range: tuple = (0, 20),
    rate_range: tuple = (0.1, 1.0)
) -> dict:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter('loss')
    psnr_meter = AverageMeter('psnr')
    ssim_meter = AverageMeter('ssim')
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Sample random SNR and rate
        batch_size = images.size(0)
        snr = torch.FloatTensor(batch_size).uniform_(*snr_range).to(device)
        rate = np.random.uniform(*rate_range)
        
        # Forward pass
        optimizer.zero_grad()
        result = model(images, snr_db=snr, rate=rate)
        x_hat = result['output']
        
        # Compute loss
        loss_dict = criterion(x_hat, targets)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = compute_ssim(x_hat, targets).mean().item()
        
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(psnr, batch_size)
        ssim_meter.update(ssim, batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'psnr': f'{psnr_meter.avg:.2f}',
            'ssim': f'{ssim_meter.avg:.4f}'
        })
    
    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float = 10.0,
    rate: float = 1.0
) -> dict:
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter('loss')
    psnr_meter = AverageMeter('psnr')
    ssim_meter = AverageMeter('ssim')
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validating'):
            images = images.to(device)
            targets = targets.to(device)
            
            batch_size = images.size(0)
            snr = torch.full((batch_size,), snr_db, device=device)
            
            result = model(images, snr_db=snr, rate=rate)
            x_hat = result['output']
            
            loss_dict = criterion(x_hat, targets)
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = compute_ssim(x_hat, targets).mean().item()
            
            loss_meter.update(loss_dict['total'].item(), batch_size)
            psnr_meter.update(psnr, batch_size)
            ssim_meter.update(ssim, batch_size)
    
    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train Reconstruction Model')
    parser.add_argument('--config', type=str, default='configs/recon.yaml')
    parser.add_argument('--data_dir', type=str, default='../CocoData')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {'smax': 256, 'image_size': 256},
            'training': {
                'epochs': 50,
                'batch_size': 16,
                'lr': 0.0005,
                'snr_range': [0, 20],
                'rate_range': [0.1, 1.0],
                'lambda_mse': 1.0,
                'lambda_ssim': 0.1
            }
        }
    
    # Override with args
    epochs = args.epochs or config['training']['epochs']
    batch_size = args.batch_size or config['training']['batch_size']
    lr = args.lr or config['training']['lr']
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / 'results' / 'recon'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('train_recon', log_dir / 'train.log')
    logger.info(f"Training Reconstruction Model")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")
    
    # Data
    data_dir = project_root.parent / 'CocoData'
    train_loader, val_loader = get_dataloaders(
        root_dir=str(data_dir),
        batch_size=batch_size,
        image_size=config['model']['image_size'],
        task='reconstruction',
        num_workers=4,
        val_split=0.1
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = SemanticCommSystem(
        task='reconstruction',
        image_size=config['model']['image_size'],
        smax=config['model']['smax'],
        pretrained_encoder=True
    ).to(device)
    
    # Loss and optimizer
    criterion = ReconstructionLoss(
        lambda_mse=config['training']['lambda_mse'],
        lambda_ssim=config['training']['lambda_ssim']
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_psnr = 0
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            snr_range=config['training']['snr_range'],
            rate_range=config['training']['rate_range']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
        
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_psnrs.append(train_metrics['psnr'])
        val_psnrs.append(val_metrics['psnr'])
        
        # Save best
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'config': config
            }, project_root / 'checkpoints' / 'best_recon.pt')
            logger.info(f"Saved best model with PSNR: {best_psnr:.2f}")
    
    # Save final
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'config': config
    }, project_root / 'checkpoints' / 'final_recon.pt')
    
    # Plot curves
    plot_training_curves(
        train_losses, val_losses, train_psnrs, val_psnrs,
        metric_name='PSNR (dB)',
        save_path=log_dir / 'training_curves.png'
    )
    
    logger.info(f"\nTraining complete! Best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
