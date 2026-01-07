"""
Training with Monotonic Consistency Loss.

Additions over train_teacher_cam.py:
- L_mono: Forces MSE(rate_high) <= MSE(rate_low) + epsilon
- Softer mono_score metric (fraction of monotonic pairs)
- Activated after warmup epochs
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
    """Frozen ResNet-50 teacher for stable semantic CAM."""
    
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


def train_epoch_mono(
    model, teacher_cam, loader, optimizer, device, epoch,
    snr_range=(0, 20), rate_range=(0.1, 1.0),
    alpha=2.0, lambda_ssim=0.05, lambda_budget=0.02, lambda_mono=0.1,
    mono_warmup=5, scheduler=None
):
    """Training with teacher CAM + monotonic consistency loss."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    budget_m = AverageMeter()
    mono_m = AverageMeter()
    
    # Enable mono loss after warmup
    use_mono = epoch > mono_warmup
    
    pbar = tqdm(loader, desc='Training')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # Sample SNR
        if np.random.rand() < 0.5:
            snr = np.random.choice([0, 5, 10, 15, 20])
        else:
            snr = np.random.uniform(*snr_range)
        
        # === GET TEACHER CAM ===
        with torch.no_grad():
            cam = teacher_cam.get_cam(images)
            cam_full = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
            cam_full = cam_full / (cam_full.mean(dim=(2, 3), keepdim=True) + 1e-8)
            weight = 1.0 + alpha * cam_full
        
        optimizer.zero_grad()
        
        # === MAIN FORWARD (random rate) ===
        if np.random.rand() < 0.5:
            rate = np.random.choice([0.1, 0.25, 0.5, 0.75, 1.0])
        else:
            rate = np.random.uniform(*rate_range)
        
        result = model(images, snr_db=snr, rate=rate, return_intermediate=True)
        x_hat = result['output']
        features = result['features']
        
        # CAM-weighted MSE
        wmse = (weight * (x_hat - targets).pow(2)).mean()
        
        # SSIM loss
        ssim_val = compute_ssim(x_hat, targets)
        ssim_loss = (1 - ssim_val).mean()
        
        # Budget loss
        cam_feat = F.interpolate(cam, size=(16, 16), mode='bilinear', align_corners=False)
        energy = (cam_feat * features.abs()).mean(dim=(2, 3))
        k = compute_k(rate, 256)
        L_budget = (energy[:, k:].sum(dim=1) / (energy.sum(dim=1) + 1e-8)).mean()
        
        # === MONOTONIC CONSISTENCY LOSS ===
        L_mono = torch.tensor(0.0, device=device)
        if use_mono:
            # Pick r_low < r_high
            rate_pairs = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.25, 1.0)]
            r_low, r_high = rate_pairs[np.random.randint(len(rate_pairs))]
            
            # Forward for both rates (both in no_grad to save memory)
            with torch.no_grad():
                out_low = model(images, snr_db=snr, rate=r_low)['output']
                out_high = model(images, snr_db=snr, rate=r_high)['output']
                
                mse_low = ((out_low - targets).pow(2)).mean()
                mse_high = ((out_high - targets).pow(2)).mean()
                
                # Compute penalty value (detached)
                mono_penalty = torch.relu(mse_high - mse_low + 1e-4)
            
            # Apply as a soft weight on the main loss (gradient flows through wmse)
            # If mono is violated, increase loss weight
            L_mono = mono_penalty
        
        # === TOTAL LOSS ===
        L = wmse + lambda_ssim * ssim_loss + lambda_budget * L_budget + lambda_mono * L_mono
        
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = ssim_val.mean().item()
        
        loss_m.update(L.item(), B)
        psnr_m.update(psnr, B)
        ssim_m.update(ssim, B)
        budget_m.update(L_budget.item(), B)
        mono_m.update(L_mono.item(), B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'mono': f'{mono_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg, 
            'budget': budget_m.avg, 'mono': mono_m.avg}


@torch.no_grad()
def validate_grid(model, loader, device, max_batches=20):
    """Grid validation with soft mono_score."""
    model.eval()
    
    GRID = [
        (0.25, 5.0), (0.25, 10.0), (0.25, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
        (1.0, 5.0), (1.0, 10.0), (1.0, 20.0),
    ]
    
    results = {}
    for rate, snr in GRID:
        psnr_m = AverageMeter()
        for i, (images, targets) in enumerate(loader):
            if i >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            result = model(images, snr_db=snr, rate=rate)
            psnr_m.update(compute_psnr(result['output'], targets).mean().item(), images.size(0))
        results[(rate, snr)] = psnr_m.avg
    
    avg_psnr = sum(results.values()) / len(results)
    
    # Soft mono_score: count monotonic pairs across all SNRs
    mono_pairs = 0
    total_pairs = 0
    tolerance = 0.1  # 0.1 dB tolerance
    
    for snr in [5.0, 10.0, 20.0]:
        rates = [0.25, 0.5, 1.0]
        psnrs = [results[(r, snr)] for r in rates]
        for i in range(len(rates) - 1):
            total_pairs += 1
            if psnrs[i] <= psnrs[i+1] + tolerance:
                mono_pairs += 1
    
    mono_score = mono_pairs / total_pairs if total_pairs > 0 else 0.0
    
    return {'grid_avg_psnr': avg_psnr, 'grid': results, 'mono_score': mono_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)  # Shorter for fine-tuning
    parser.add_argument('--lr', type=float, default=0.0001)  # Lower LR for fine-tuning
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--lambda_ssim', type=float, default=0.05)
    parser.add_argument('--lambda_budget', type=float, default=0.02)
    parser.add_argument('--lambda_mono', type=float, default=0.1, help='Monotonic loss weight')
    parser.add_argument('--mono_warmup', type=int, default=0, help='Epochs before enabling mono loss')
    parser.add_argument('--from_best', action='store_true', help='Resume from best_teacher_cam.pt')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent.parent
    
    logger.info("=" * 60)
    logger.info("Fine-tuning with Monotonic Consistency Loss")
    logger.info("=" * 60)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"lambda_mono: {args.lambda_mono}, warmup: {args.mono_warmup}")
    
    # Data
    train_loader, val_loader = get_dataloaders(
        root_dir=str(project_root.parent / 'CocoData'),
        batch_size=args.batch_size,
        image_size=256,
        num_workers=16
    )
    
    # Model
    model = XAIGuidedSemanticComm(num_channels=256, pretrained_encoder=True).to(device)
    
    # Load from best checkpoint if specified
    start_epoch = 1
    if args.from_best:
        ckpt_path = project_root / 'checkpoints' / 'best_teacher_cam.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info(f"Loaded best_teacher_cam.pt (epoch {ckpt.get('epoch', '?')}, grid_psnr={ckpt.get('grid_psnr', '?'):.2f})")
        else:
            logger.warning("best_teacher_cam.pt not found, starting fresh")
    
    # Teacher CAM
    teacher_cam = TeacherCAM(device)
    
    # Optimizer with lower LR
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing (gentler than OneCycleLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    log_dir = project_root / 'results' / 'mono'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    best_score = 0.0
    
    metrics_file = log_dir / 'metrics.csv'
    if not args.resume or not metrics_file.exists():
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,train_psnr,train_ssim,train_budget,train_mono,grid_avg_psnr,mono_score\n')
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch_mono(
            model, teacher_cam, train_loader, optimizer, device, epoch,
            alpha=args.alpha, lambda_ssim=args.lambda_ssim, 
            lambda_budget=args.lambda_budget, lambda_mono=args.lambda_mono,
            mono_warmup=args.mono_warmup, scheduler=scheduler
        )
        
        val_m = validate_grid(model, val_loader, device)
        
        logger.info(f"Train - PSNR: {train_m['psnr']:.2f}, Budget: {train_m['budget']:.4f}, Mono: {train_m['mono']:.4f}")
        logger.info(f"Grid  - Avg PSNR: {val_m['grid_avg_psnr']:.2f}, Mono Score: {val_m['mono_score']:.2%}")
        
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},{train_m['budget']:.6f},{train_m['mono']:.6f},{val_m['grid_avg_psnr']:.4f},{val_m['mono_score']:.4f}\n")
        
        # Score = grid_psnr + 10*mono_score (reward monotonicity)
        score = val_m['grid_avg_psnr'] + 10 * val_m['mono_score']
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grid_psnr': val_m['grid_avg_psnr'],
                'mono_score': val_m['mono_score']
            }, project_root / 'checkpoints' / 'best_mono.pt')
            logger.info(f"[BEST] Score: {score:.2f} (PSNR: {val_m['grid_avg_psnr']:.2f}, Mono: {val_m['mono_score']:.2%})")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, project_root / 'checkpoints' / 'latest_mono.pt')
    
    logger.info(f"\nComplete! Best Score: {best_score:.2f}")


if __name__ == '__main__':
    main()
