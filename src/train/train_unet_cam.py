"""
U-Net Training with Teacher CAM + Budget Loss + High SSIM.

Features:
- Teacher CAM from ResNet-50 for CAM-weighted MSE
- Budget loss (energy in inactive channels)
- High SSIM weight (0.2) for maximizing SSIM
- MS-SSIM for better perceptual quality
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

from src.models.xai_pipeline_unet import XAISemanticCommUNet
from src.models.sr_sc import compute_k
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim
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


class TeacherCAM:
    """Frozen ResNet-50 for stable semantic CAM."""
    
    def __init__(self, device):
        self.teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.hook = GradCAMHook(self.teacher.layer3)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    def get_cam(self, images: torch.Tensor) -> torch.Tensor:
        x = (images - self.mean) / self.std
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.teacher(x)
            cls = logits.argmax(dim=1)
            score = logits.gather(1, cls.unsqueeze(1)).sum()
            self.teacher.zero_grad()
            score.backward()
        cam = self.hook.compute_cam()
        self.hook.clear()
        return cam.detach()


def train_epoch(
    model, teacher_cam, loader, optimizer, device, epoch,
    snr_range=(0, 20), rate_range=(0.1, 1.0),
    alpha=2.0, lambda_ssim=0.2, lambda_budget=0.02,
    scheduler=None
):
    """Training with Teacher CAM + Budget + High SSIM."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    budget_m = AverageMeter()
    
    pbar = tqdm(loader, desc=f'Training [Epoch {epoch}]')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # Sample rate and SNR
        if np.random.rand() < 0.5:
            rate = np.random.choice([0.25, 0.5, 0.75, 1.0])
        else:
            rate = np.random.uniform(*rate_range)
        
        if np.random.rand() < 0.5:
            snr = np.random.choice([0, 5, 10, 15, 20])
        else:
            snr = np.random.uniform(*snr_range)
        
        # Get Teacher CAM
        with torch.no_grad():
            cam = teacher_cam.get_cam(images)
            cam_full = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
            cam_full = cam_full / (cam_full.mean(dim=(2, 3), keepdim=True) + 1e-8)
            weight = 1.0 + alpha * cam_full
        
        optimizer.zero_grad()
        
        result = model(images, snr_db=snr, rate=rate, return_intermediate=True)
        x_hat = result['output']
        features = result['features']  # z0 [B, 256, 16, 16]
        
        # CAM-weighted MSE
        wmse = (weight * (x_hat - targets).pow(2)).mean()
        
        # SSIM loss (high weight for SSIM maximization)
        ssim_val = compute_ssim(x_hat, targets)
        ssim_loss = (1 - ssim_val).mean()
        
        # Budget loss
        cam_feat = F.interpolate(cam, size=(16, 16), mode='bilinear', align_corners=False)
        energy = (cam_feat * features.abs()).mean(dim=(2, 3))
        k = compute_k(rate, 256)
        energy_total = energy.sum(dim=1) + 1e-8
        energy_inactive = energy[:, k:].sum(dim=1)
        L_budget = (energy_inactive / energy_total).mean()
        
        # Total loss
        L = wmse + lambda_ssim * ssim_loss + lambda_budget * L_budget
        
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = ssim_val.mean().item()
        
        loss_m.update(L.item(), B)
        psnr_m.update(psnr, B)
        ssim_m.update(ssim, B)
        budget_m.update(L_budget.item(), B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'ssim': f'{ssim_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg, 'budget': budget_m.avg}


@torch.no_grad()
def validate_grid(model, loader, device, max_batches=20):
    """Grid validation with SSIM focus."""
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
    
    # Monotonicity
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
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--lambda_ssim', type=float, default=0.2)  # High for SSIM focus
    parser.add_argument('--lambda_budget', type=float, default=0.02)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent.parent
    
    logger.info("=" * 60)
    logger.info("U-Net + Teacher CAM + High SSIM Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"α: {args.alpha}, λ_ssim: {args.lambda_ssim}, λ_budget: {args.lambda_budget}")
    
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
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Teacher CAM
    teacher_cam = TeacherCAM(device)
    logger.info("Teacher CAM initialized")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    log_dir = project_root / 'results' / 'unet_cam'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 1
    best_ssim = 0.0
    
    # Resume
    ckpt_path = project_root / 'checkpoints' / 'best_unet_cam.pt'
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim = ckpt.get('grid_ssim', 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
    
    # Metrics CSV
    metrics_file = log_dir / 'metrics.csv'
    if not args.resume or not metrics_file.exists():
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,train_psnr,train_ssim,train_budget,grid_avg_psnr,grid_avg_ssim,mono_score\n')
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch(
            model, teacher_cam, train_loader, optimizer, device, epoch,
            alpha=args.alpha, lambda_ssim=args.lambda_ssim, lambda_budget=args.lambda_budget,
            scheduler=scheduler
        )
        
        val_m = validate_grid(model, val_loader, device)
        
        logger.info(f"Train - PSNR: {train_m['psnr']:.2f}, SSIM: {train_m['ssim']:.4f}, Budget: {train_m['budget']:.4f}")
        logger.info(f"Grid  - PSNR: {val_m['grid_avg_psnr']:.2f}, SSIM: {val_m['grid_avg_ssim']:.4f}, Mono: {val_m['mono_score']:.2%}")
        
        if epoch % 5 == 0:
            for r in [0.25, 0.5, 1.0]:
                p = val_m['grid'][(r, 10.0)]['psnr']
                s = val_m['grid'][(r, 10.0)]['ssim']
                logger.info(f"  rate={r:.2f}, SNR=10 -> PSNR={p:.2f}, SSIM={s:.4f}")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},{train_m['budget']:.6f},{val_m['grid_avg_psnr']:.4f},{val_m['grid_avg_ssim']:.4f},{val_m['mono_score']:.4f}\n")
        
        # Save best (by SSIM)
        if val_m['grid_avg_ssim'] > best_ssim:
            best_ssim = val_m['grid_avg_ssim']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grid_psnr': val_m['grid_avg_psnr'],
                'grid_ssim': best_ssim,
                'mono_score': val_m['mono_score']
            }, project_root / 'checkpoints' / 'best_unet_cam.pt')
            logger.info(f"[BEST] SSIM: {best_ssim:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, project_root / 'checkpoints' / 'latest_unet_cam.pt')
    
    logger.info(f"\nComplete! Best Grid SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    main()
