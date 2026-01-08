"""
Train PreNetAdaptive for Image + Network-aware Rate Prediction.

This script trains PreNetAdaptive to predict optimal rate given:
1. Image features from frozen JSCC encoder
2. Network conditions (SNR, bandwidth, latency, BER)

Training approach:
- Sample random network conditions
- For each (image, network) pair, find optimal rate via grid search
- Train PreNetAdaptive to predict this optimal rate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from tqdm import tqdm

from src.models.jscc_noskip import JSCCNoSkip
from src.models.prenet_adaptive import PreNetAdaptive
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def compute_channel_score(snr_db, bandwidth_mhz, latency_ms, ber_exp):
    """
    Compute channel quality score [0, 1] from network characteristics.
    Higher score = better channel.
    
    Args: All tensors [B, 1]
    Returns: score [B, 1]
    """
    # SNR score (0-1): 0 dB → 0, 30 dB → 1
    snr_score = torch.clamp(snr_db / 30.0, 0, 1)
    
    # Bandwidth score: 1 MHz → 0.1, 100 MHz → 1
    bw_score = torch.clamp(bandwidth_mhz / 100.0, 0.1, 1)
    
    # Latency score: 500 ms → 0.1, 5 ms → 1
    lat_score = torch.clamp(1.0 - latency_ms / 500.0, 0.1, 1)
    
    # BER score: 10^-2 → 0, 10^-6 → 1
    ber_score = torch.clamp((ber_exp - 2) / 4.0, 0, 1)
    
    # Weighted combination
    score = 0.50 * snr_score + 0.25 * bw_score + 0.10 * lat_score + 0.15 * ber_score
    
    return score


def compute_target_rate(snr_db, bandwidth_mhz, latency_ms, ber_exp, min_rate=0.1, max_rate=1.0):
    """
    Compute target rate from network characteristics.
    Good channel → high rate (can send more)
    Bad channel → low rate (must compress more)
    
    Returns: rate [B, 1] in [min_rate, max_rate]
    """
    score = compute_channel_score(snr_db, bandwidth_mhz, latency_ms, ber_exp)
    
    # Direct relationship: good channel → high rate
    rate = min_rate + (max_rate - min_rate) * score
    
    return rate


def sample_network_conditions(batch_size, device):
    """Sample random network conditions for training."""
    snr_db = torch.rand(batch_size, 1, device=device) * 28 + 2  # [2, 30] dB
    bandwidth_mhz = torch.rand(batch_size, 1, device=device) * 95 + 5  # [5, 100] MHz
    latency_ms = torch.rand(batch_size, 1, device=device) * 495 + 5  # [5, 500] ms
    ber_exp = torch.rand(batch_size, 1, device=device) * 4 + 2  # [2, 6] continuous
    
    return snr_db, bandwidth_mhz, latency_ms, ber_exp


def train_epoch(prenet, jscc, loader, optimizer, device):
    """Train one epoch with continuous rate targets."""
    prenet.train()
    jscc.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc='Training PreNetAdaptive')
    
    for images, _ in pbar:
        images = images.to(device)
        B = images.size(0)
        
        # Sample network conditions
        snr_db, bandwidth_mhz, latency_ms, ber_exp = sample_network_conditions(B, device)
        
        # Compute target rate from network characteristics (CONTINUOUS)
        target_rates = compute_target_rate(snr_db, bandwidth_mhz, latency_ms, ber_exp)
        
        # Get encoder features
        with torch.no_grad():
            enc_out = jscc.encoder(images)
            z0 = enc_out['z0']
            
            # Compute actual PSNR/SSIM at target rate for quality prediction training
            # Use mean rate for the batch for efficiency
            mean_rate = target_rates.mean().item()
            mean_snr = snr_db.mean().item()
            result_jscc = jscc(images, snr_db=mean_snr, rate=mean_rate)
            gt_psnr = compute_psnr(result_jscc['output'], images)  # [B]
            gt_ssim = compute_ssim(result_jscc['output'], images)  # [B]
            target_quality = torch.stack([gt_psnr / 50.0, gt_ssim], dim=1)  # [B, 2]
        
        # Forward through PreNet
        optimizer.zero_grad()
        
        result = prenet(
            z0,
            snr_db=snr_db,
            bandwidth_mhz=bandwidth_mhz,
            latency_ms=latency_ms,
            ber_exp=ber_exp
        )
        
        pred_rate = result['rate']
        
        # Loss: MSE on rate + quality prediction
        rate_loss = nn.MSELoss()(pred_rate, target_rates)
        
        if 'psnr' in result:
            quality_pred = torch.cat([result['psnr'] / 50, result['ssim']], dim=1)
            quality_loss = nn.MSELoss()(quality_pred, target_quality)
            loss = rate_loss + 0.3 * quality_loss
        else:
            loss = rate_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prenet.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        mae = torch.abs(pred_rate - target_rates).mean().item()
        total_loss += loss.item()
        total_mae += mae
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'rate_mae': f'{mae:.3f}'})
    
    return total_loss / num_batches, total_mae / num_batches


def validate(prenet, jscc, loader, device):
    """Validate on fixed network conditions."""
    prenet.eval()
    jscc.eval()
    
    test_conditions = [
        ("5G (SNR=20, BW=100)", 20.0, 100.0, 10.0, 6.0),
        ("WiFi (SNR=15, BW=20)", 15.0, 20.0, 30.0, 5.0),
        ("LTE (SNR=10, BW=10)", 10.0, 10.0, 50.0, 4.0),
        ("Satellite (SNR=5, BW=5)", 5.0, 5.0, 400.0, 3.0),
    ]
    
    results = {}
    
    with torch.no_grad():
        for name, snr, bw, lat, ber in test_conditions:
            pred_rates = []
            
            # Compute expected rate from formula (ground truth)
            expected_rate = compute_target_rate(
                torch.tensor([[snr]], device=device),
                torch.tensor([[bw]], device=device),
                torch.tensor([[lat]], device=device),
                torch.tensor([[ber]], device=device)
            ).item()
            
            for images, _ in loader:
                images = images.to(device)
                
                # Get features
                z0 = jscc.encoder(images)['z0']
                
                # Predict rate
                result = prenet(z0, snr, bw, lat, ber)
                pred_rate = result['rate'].squeeze().tolist()
                if isinstance(pred_rate, float):
                    pred_rate = [pred_rate]
                pred_rates.extend(pred_rate)
                
                if len(pred_rates) >= 50:
                    break
            
            avg_pred = np.mean(pred_rates)
            mae = abs(avg_pred - expected_rate)
            
            results[name] = {
                'pred_rate': avg_pred,
                'expected_rate': expected_rate,
                'mae': mae
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train PreNetAdaptive')
    parser.add_argument('--jscc_ckpt', type=str, required=True, help='Path to best_noskip.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).parent.parent.parent
    
    # Load frozen JSCC
    logger.info("Loading JSCC model...")
    jscc = JSCCNoSkip(num_channels=256, pretrained_encoder=False).to(device)
    ckpt = torch.load(args.jscc_ckpt, map_location=device, weights_only=False)
    jscc.load_state_dict(ckpt['model_state_dict'])
    jscc.eval()
    for p in jscc.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded JSCC from {args.jscc_ckpt}")
    
    # Initialize PreNetAdaptive
    logger.info("Initializing PreNetAdaptive...")
    prenet = PreNetAdaptive(num_channels=256, predict_quality=True).to(device)
    num_params = sum(p.numel() for p in prenet.parameters())
    logger.info(f"PreNetAdaptive parameters: {num_params:,}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root_dir=str(root.parent / 'CocoData'),
        batch_size=args.batch_size,
        image_size=256
    )
    
    # Optimizer
    optimizer = optim.AdamW(prenet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Output
    save_dir = root / 'results' / 'prenet_adaptive'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_mae = 999.0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_loss, train_mae = train_epoch(prenet, jscc, train_loader, optimizer, device)
        logger.info(f"Train - Loss: {train_loss:.4f}, Rate MAE: {train_mae:.3f}")
        
        # Validate
        val_results = validate(prenet, jscc, val_loader, device)
        
        total_mae = 0
        for name, res in val_results.items():
            logger.info(f"  {name}: pred={res['pred_rate']:.2f}, expected={res['expected_rate']:.2f}, MAE={res['mae']:.3f}")
            total_mae += res['mae']
        avg_mae = total_mae / len(val_results)
        
        scheduler.step()
        
        # Save best
        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': prenet.state_dict(),
                'mae': avg_mae,
            }, save_dir / 'best_prenet_adaptive.pt')
            logger.info(f"[BEST] Saved with MAE={avg_mae:.3f}")
    
    logger.info(f"\nTraining complete! Best MAE: {best_mae:.3f}")


if __name__ == '__main__':
    main()
