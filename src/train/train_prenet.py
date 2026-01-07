"""
Train PreNet for Adaptive Rate Selection.

This script trains a lightweight predictor (PreNet) to estimate 
the expected PSNR/SSIM for a given (Image, SNR, Rate) tuple.

Workflow:
1. Load frozen JSCC model (NoSkip).
2. For each batch:
   - Extract features z0.
   - For random rates/SNRs, run JSCC decoder.
   - Compute ground truth PSNR/SSIM.
3. Train PreNet to predict these metrics from feature stats (mu, sigma).
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
from src.models.prenet import PreNetWithRate
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def train_epoch(prenet, jscc_model, loader, optimizer, device):
    prenet.train()
    # JSCC is strictly evaluation
    jscc_model.eval()
    
    loss_meter = 0.0
    mae_psnr = 0.0
    
    pbar = tqdm(loader, desc='Training PreNet')
    
    for images, _ in pbar:
        images = images.to(device)
        B = images.size(0)
        
        # 1. Get stats from JSCC encoder (frozen)
        with torch.no_grad():
            enc_out = jscc_model.encoder(images)
            z0 = enc_out['z0']
            # PreNet inputs (mu, sigma) are computed inside PreNet from z0
            
        # 2. Sample random conditions
        rate = torch.rand(B, 1, device=device) * 0.9 + 0.1  # [0.1, 1.0]
        snr = torch.rand(B, 1, device=device) * 20.0        # [0, 20]
        
        # 3. Get Ground Truth Quality
        with torch.no_grad():
            # We need to run the rest of JSCC pipeline manually to inject specific rate/snr tensors
            # SR-SC
            # Note: SR-SC mask depends on scalar rate usually, but here we have batch of rates
            # Our current SR-SC implementation might expect scalar rate.
            # For simplicity in this v1 script, we pick ONE random rate per batch 
            # to accommodate SR-SC's likely scalar expectation, or we loop.
            # Let's verify SR-SC code... assumed scalar for now, so we sample 1 rate per batch.
            
            # Correction: To be efficient, let's sample 1 scalar rate and 1 scalar SNR per batch for now
            # Ideally PreNet should handle batch-wise mixed rates, but JSCC implementation limits us.
            
            r_val = float(np.random.uniform(0.1, 1.0))
            s_val = float(np.random.uniform(0.0, 20.0))
            
            res = jscc_model(images, snr_db=s_val, rate=r_val)
            x_hat = res['output']
            
            gt_psnr = compute_psnr(x_hat, images)  # [B]
            gt_ssim = compute_ssim(x_hat, images)  # [B]
            
            # Targets: normalize PSNR to [0,1] roughly (div by 50) for stability
            target = torch.stack([gt_psnr / 50.0, gt_ssim], dim=1) # [B, 2]
            
            # Inputs matching selection
            rate_in = torch.full((B, 1), r_val, device=device)
            snr_in = torch.full((B, 1), s_val, device=device)

        # 4. Predict
        # PreNet takes: z0, snr, rate
        preds = prenet(z0, snr_in, rate_in) # [B, 2]
        
        # 5. Loss
        loss = nn.MSELoss()(preds, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter += loss.item()
        mae_psnr += torch.abs(preds[:, 0]*50 - gt_psnr).mean().item()
        
        pbar.set_postfix({'loss': loss.item(), 'ma_psnr': mae_psnr / (pbar.n + 1)})
        
    return loss_meter / len(loader), mae_psnr / len(loader)


def validate(prenet, jscc_model, loader, device):
    prenet.eval()
    jscc_model.eval()
    
    mae_psnr = 0.0
    mae_ssim = 0.0
    
    # Test on fixed grid points like [0.5 rate, 10dB]
    target_r = 0.5
    target_s = 10.0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            B = images.size(0)
            
            # Ground Truth
            res = jscc_model(images, snr_db=target_s, rate=target_r)
            gt_psnr = compute_psnr(res['output'], images)
            gt_ssim = compute_ssim(res['output'], images)
            
            # Predict
            z0 = jscc_model.encoder(images)['z0']
            rate_in = torch.full((B, 1), target_r, device=device)
            snr_in = torch.full((B, 1), target_s, device=device)
            
            preds = prenet(z0, snr_in, rate_in)
            
            pred_psnr = preds[:, 0] * 50.0
            pred_ssim = preds[:, 1]
            
            mae_psnr += torch.abs(pred_psnr - gt_psnr).mean().item()
            mae_ssim += torch.abs(pred_ssim - gt_ssim).mean().item()
            
    return mae_psnr / len(loader), mae_ssim / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jscc_ckpt', type=str, required=True, help='Path to frozen best_noskip.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).parent.parent.parent
    
    # 1. Load Frozen JSCC
    logger.info("Loading JSCC model...")
    jscc = JSCCNoSkip(num_channels=256, pretrained_encoder=False).to(device)
    ckpt = torch.load(args.jscc_ckpt, map_location=device)
    jscc.load_state_dict(ckpt['model_state_dict'])
    jscc.eval()
    for p in jscc.parameters():
        p.requires_grad_(False)
        
    # 2. Init PreNet
    logger.info("Initializing PreNet...")
    # encoder outputs 256 channels
    prenet = PreNetWithRate(num_channels=256, task='regression').to(device)
    
    # 3. Data
    train_loader, val_loader = get_dataloaders(
        root_dir=str(root.parent / 'CocoData'),
        batch_size=args.batch_size,
        image_size=256
    )
    
    optimizer = optim.Adam(prenet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    
    save_dir = root / 'results' / 'prenet'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_mae = 999.0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        train_loss, train_mae = train_epoch(prenet, jscc, train_loader, optimizer, device)
        val_mae_p, val_mae_s = validate(prenet, jscc, val_loader, device)
        
        logger.info(f"Train MAE (PSNR): {train_mae:.2f} dB")
        logger.info(f"Val MAE (PSNR): {val_mae_p:.2f} dB, (SSIM): {val_mae_s:.4f}")
        
        scheduler.step(val_mae_p)
        
        if val_mae_p < best_mae:
            best_mae = val_mae_p
            torch.save(prenet.state_dict(), save_dir / 'best_prenet.pt')
            logger.info("Saved best PreNet.")

if __name__ == '__main__':
    main()
