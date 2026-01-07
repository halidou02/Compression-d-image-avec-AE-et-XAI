"""
Instance-Level Rate Selection using Pre-Net.

Uses trained Pre-Net to predict quality and select optimal rate for each image.
"""
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.full_pipeline import SemanticCommSystem
from src.models.prenet import PreNet
from src.data.datasets import get_dataloaders
from src.utils.metrics import compute_psnr, compute_ssim
from src.utils.seed import set_seed, get_device
from src.utils.logging_ import setup_logger


class InstanceLevelRateSelector:
    """
    Selects minimum rate satisfying quality threshold using Pre-Net predictions.
    """
    
    def __init__(
        self,
        model: SemanticCommSystem,
        prenet: PreNet,
        device: torch.device,
        rate_candidates: list = None,
        quality_threshold: float = 0.5,  # Normalized threshold
        task: str = 'reconstruction'
    ):
        self.model = model
        self.prenet = prenet
        self.device = device
        self.rate_candidates = rate_candidates or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.quality_threshold = quality_threshold
        self.task = task
        
        self.model.eval()
        self.prenet.eval()
    
    @torch.no_grad()
    def select_rate(
        self,
        image: torch.Tensor,
        snr_db: float
    ) -> dict:
        """
        Select minimum rate satisfying threshold for a single image.
        
        Args:
            image: [1, 3, H, W]
            snr_db: SNR value
            
        Returns:
            dict with selected_rate, predicted_quality, and all predictions
        """
        image = image.to(self.device)
        snr_tensor = torch.tensor([snr_db], device=self.device)
        
        # Get features through encoder
        features = self.model.encoder(image)
        
        predictions = {}
        
        # For each rate, use Pre-Net to predict quality
        for rate in self.rate_candidates:
            # Process through SR-SC and ACC
            xa, _ = self.model.sr_sc(features, keep_ratio=rate)
            x_prime = self.model.acc(xa, snr_tensor)
            
            # Pre-Net prediction
            pred = self.prenet(x_prime, snr_tensor)
            
            if self.task == 'reconstruction':
                quality = pred[0, 0].item()
            else:
                quality = pred[0, 0].item()
            
            predictions[rate] = quality
        
        # Select minimum rate meeting threshold
        selected_rate = self.rate_candidates[-1]  # Default to max
        for rate in self.rate_candidates:
            if predictions[rate] >= self.quality_threshold:
                selected_rate = rate
                break
        
        return {
            'selected_rate': selected_rate,
            'predicted_quality': predictions[selected_rate],
            'all_predictions': predictions
        }
    
    @torch.no_grad()
    def evaluate_rate_selection(
        self,
        loader: DataLoader,
        snr_db: float
    ) -> dict:
        """
        Evaluate rate selection on a dataset.
        """
        results = {
            'rates': [],
            'predicted_qualities': [],
            'actual_qualities': [],
            'bandwidth_savings': []
        }
        
        snr_tensor = torch.full((1,), snr_db, device=self.device)
        
        for images, targets in tqdm(loader, desc='Evaluating'):
            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)
                target = targets[i:i+1].to(self.device)
                
                # Select rate
                selection = self.select_rate(image, snr_db)
                selected_rate = selection['selected_rate']
                
                # Actually run the model at selected rate
                result = self.model(image, snr_db=snr_tensor, rate=selected_rate)
                output = result['output']
                
                # Compute actual quality
                if self.task == 'reconstruction':
                    actual_quality = compute_psnr(output, target).item() / 50.0
                else:
                    actual_quality = (output.argmax(1) == target).float().item()
                
                results['rates'].append(selected_rate)
                results['predicted_qualities'].append(selection['predicted_quality'])
                results['actual_qualities'].append(actual_quality)
                results['bandwidth_savings'].append(1.0 - selected_rate)
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints/best_recon.pt')
    parser.add_argument('--prenet_checkpoint', type=str, default='checkpoints/prenet.pt')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--threshold', type=float, default=0.5, help='Normalized quality threshold')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    project_root = Path(__file__).parent.parent.parent
    logger = setup_logger('rate_selection')
    
    # Load model
    model_ckpt = torch.load(project_root / args.model_checkpoint, map_location=device)
    config = model_ckpt.get('config', {'model': {'smax': 256, 'image_size': 256}})
    
    model = SemanticCommSystem(
        task='reconstruction',
        image_size=config['model']['image_size'],
        smax=config['model']['smax'],
        pretrained_encoder=False
    ).to(device)
    model.load_state_dict(model_ckpt['model_state_dict'])
    
    # Load Pre-Net
    prenet_ckpt = torch.load(project_root / args.prenet_checkpoint, map_location=device)
    prenet_config = prenet_ckpt['config']
    
    prenet = PreNet(
        num_channels=prenet_config['num_channels'],
        hidden_dim=prenet_config['hidden_dim'],
        num_rfcb=prenet_config['num_rfcb'],
        task=prenet_config['task']
    ).to(device)
    prenet.load_state_dict(prenet_ckpt['prenet_state_dict'])
    
    logger.info("Loaded model and Pre-Net")
    
    # Data
    data_dir = project_root.parent / 'CocoData'
    _, val_loader = get_dataloaders(
        root_dir=str(data_dir),
        batch_size=1,  # Process one at a time for rate selection
        image_size=config['model']['image_size'],
        task='reconstruction',
        num_workers=4
    )
    
    # Rate selector
    selector = InstanceLevelRateSelector(
        model=model,
        prenet=prenet,
        device=device,
        quality_threshold=args.threshold
    )
    
    # Evaluate
    results = selector.evaluate_rate_selection(val_loader, snr_db=args.snr)
    
    # Statistics
    avg_rate = np.mean(results['rates'])
    avg_predicted = np.mean(results['predicted_qualities'])
    avg_actual = np.mean(results['actual_qualities'])
    avg_savings = np.mean(results['bandwidth_savings'])
    
    logger.info(f"\n=== Rate Selection Results (SNR={args.snr}dB) ===")
    logger.info(f"Average selected rate: {avg_rate:.3f}")
    logger.info(f"Average predicted quality: {avg_predicted:.3f}")
    logger.info(f"Average actual quality: {avg_actual:.3f}")
    logger.info(f"Average bandwidth savings: {avg_savings:.1%}")
    logger.info(f"Prediction error: {abs(avg_predicted - avg_actual):.3f}")
    
    # Rate distribution
    from collections import Counter
    rate_dist = Counter(results['rates'])
    logger.info("\nRate distribution:")
    for rate in sorted(rate_dist.keys()):
        count = rate_dist[rate]
        pct = count / len(results['rates']) * 100
        logger.info(f"  Rate {rate:.1f}: {count} samples ({pct:.1f}%)")


if __name__ == '__main__':
    main()
