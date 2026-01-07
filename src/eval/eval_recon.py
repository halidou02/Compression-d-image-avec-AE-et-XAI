"""
Reconstruction Evaluation Script.

Evaluates the trained reconstruction model across different SNR and rate values.
Builds utility tables Q(γ, r) for PSNR/SSIM.
"""
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.full_pipeline import SemanticCommSystem
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.seed import set_seed, get_device
from src.utils.logging_ import setup_logger
from src.utils.viz import visualize_reconstruction, plot_utility_table, plot_snr_vs_metric


def evaluate_single(
    model,
    loader,
    device,
    snr_db: float,
    rate: float
) -> dict:
    """Evaluate at single SNR and rate."""
    model.eval()
    
    psnrs, ssims = [], []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            
            batch_size = images.size(0)
            snr = torch.full((batch_size,), snr_db, device=device)
            
            result = model(images, snr_db=snr, rate=rate)
            x_hat = result['output']
            
            psnr = compute_psnr(x_hat, targets)
            ssim = compute_ssim(x_hat, targets)
            
            psnrs.extend(psnr.cpu().numpy())
            ssims.extend(ssim.cpu().numpy())
    
    return {
        'psnr': np.mean(psnrs),
        'psnr_std': np.std(psnrs),
        'ssim': np.mean(ssims),
        'ssim_std': np.std(ssims)
    }


def build_utility_table(
    model,
    loader,
    device,
    snr_values: list,
    rate_values: list
) -> dict:
    """Build utility tables Q(γ, r)."""
    psnr_table = np.zeros((len(snr_values), len(rate_values)))
    ssim_table = np.zeros((len(snr_values), len(rate_values)))
    
    for i, snr in enumerate(tqdm(snr_values, desc='SNR')):
        for j, rate in enumerate(rate_values):
            result = evaluate_single(model, loader, device, snr, rate)
            psnr_table[i, j] = result['psnr']
            ssim_table[i, j] = result['ssim']
    
    return {
        'psnr': psnr_table,
        'ssim': ssim_table,
        'snr_values': snr_values,
        'rate_values': rate_values
    }


def choose_rate(
    utility_table: np.ndarray,
    snr_idx: int,
    threshold: float,
    rate_values: list
) -> float:
    """
    Choose minimum rate satisfying threshold.
    
    Args:
        utility_table: [num_snr, num_rate] PSNR/SSIM values
        snr_idx: Index of current SNR
        threshold: Minimum acceptable quality
        rate_values: List of rate values
        
    Returns:
        Minimum rate satisfying threshold (or max rate if none)
    """
    row = utility_table[snr_idx]
    
    for j, (rate, quality) in enumerate(zip(rate_values, row)):
        if quality >= threshold:
            return rate
    
    return rate_values[-1]  # Return max rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_recon.pt')
    parser.add_argument('--data_dir', type=str, default='../CocoData')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / 'results' / 'recon'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('eval_recon', results_dir / 'eval.log')
    
    # Load model
    ckpt_path = project_root / args.checkpoint
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config', {'model': {'smax': 256, 'image_size': 256}})
    
    model = SemanticCommSystem(
        task='reconstruction',
        image_size=config['model']['image_size'],
        smax=config['model']['smax'],
        pretrained_encoder=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    
    # Data
    data_dir = project_root.parent / 'CocoData'
    _, val_loader = get_dataloaders(
        root_dir=str(data_dir),
        batch_size=args.batch_size,
        image_size=config['model']['image_size'],
        task='reconstruction',
        num_workers=4
    )
    
    # Evaluation grid
    snr_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    rate_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Build utility tables
    logger.info("Building utility tables...")
    tables = build_utility_table(model, val_loader, device, snr_values, rate_values)
    
    # Save tables
    np.savez(
        results_dir / 'utility_tables.npz',
        psnr=tables['psnr'],
        ssim=tables['ssim'],
        snr_values=snr_values,
        rate_values=rate_values
    )
    
    # Plot tables
    plot_utility_table(
        tables['psnr'], snr_values, rate_values,
        metric_name='PSNR (dB)',
        save_path=results_dir / 'psnr_table.png'
    )
    
    plot_utility_table(
        tables['ssim'], snr_values, rate_values,
        metric_name='SSIM',
        save_path=results_dir / 'ssim_table.png'
    )
    
    # Plot PSNR vs SNR at rate=1.0
    psnr_vs_snr = tables['psnr'][:, -1]
    plot_snr_vs_metric(
        snr_values, psnr_vs_snr,
        metric_name='PSNR (dB)',
        title='PSNR vs SNR (rate=1.0)',
        save_path=results_dir / 'psnr_vs_snr.png'
    )
    
    # Log summary
    logger.info("\n=== Results Summary ===")
    logger.info(f"PSNR range: {tables['psnr'].min():.2f} - {tables['psnr'].max():.2f} dB")
    logger.info(f"SSIM range: {tables['ssim'].min():.4f} - {tables['ssim'].max():.4f}")
    
    # Example rate selection
    psnr_threshold = 25.0
    logger.info(f"\nRate selection for PSNR >= {psnr_threshold} dB:")
    for i, snr in enumerate(snr_values):
        rate = choose_rate(tables['psnr'], i, psnr_threshold, rate_values)
        logger.info(f"  SNR={snr:2d}dB -> rate={rate:.1f}")
    
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
