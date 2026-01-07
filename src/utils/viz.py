"""
Visualization utilities.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch  
        train_metrics: Training metric per epoch
        val_metrics: Validation metric per epoch
        metric_name: Name of the metric
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metric plot
    ax2.plot(epochs, train_metrics, 'b-', label='Train')
    ax2.plot(epochs, val_metrics, 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_snr_vs_metric(
    snr_values: List[float],
    metric_values: List[float],
    metric_name: str = 'Accuracy',
    title: str = 'Performance vs SNR',
    save_path: Optional[str] = None
) -> None:
    """
    Plot metric vs SNR curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, metric_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rate_vs_metric(
    rate_values: List[float],
    metric_values: List[float],
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
) -> None:
    """
    Plot metric vs transmission rate.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(rate_values, metric_values, 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Transmission Rate')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Transmission Rate')
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_utility_table(
    table: np.ndarray,
    snr_values: List[float],
    rate_values: List[float],
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
) -> None:
    """
    Plot utility table U(γ, r) as heatmap.
    
    Args:
        table: [num_snr, num_rate] utility values
        snr_values: SNR values
        rate_values: Rate values
        metric_name: Name of metric
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(table, aspect='auto', origin='lower', cmap='viridis')
    
    ax.set_xticks(range(len(rate_values)))
    ax.set_xticklabels([f'{r:.1f}' for r in rate_values])
    ax.set_yticks(range(len(snr_values)))
    ax.set_yticklabels([f'{s:.0f}' for s in snr_values])
    
    ax.set_xlabel('Rate')
    ax.set_ylabel('SNR (dB)')
    ax.set_title(f'Utility Table: {metric_name}')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize original vs reconstructed images.
    
    Args:
        original: Original images [B, 3, H, W]
        reconstructed: Reconstructed images [B, 3, H, W]
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    n = min(num_samples, original.shape[0])
    
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    
    for i in range(n):
        # Original
        img_orig = original[i].detach().cpu().permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        img_recon = reconstructed[i].detach().cpu().permute(1, 2, 0).numpy()
        img_recon = np.clip(img_recon, 0, 1)
        axes[1, i].imshow(img_recon)
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gradcam(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize Grad-CAM heatmap overlay.
    
    Args:
        image: Original image [3, H, W]
        heatmap: Grad-CAM heatmap [H, W]
        alpha: Heatmap opacity
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    img = image.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Heatmap
    hm = heatmap.cpu().numpy()
    axes[1].imshow(hm, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    
    # Overlay
    hm_resized = np.stack([hm] * 3, axis=-1)  # Make 3-channel
    hm_colored = plt.cm.jet(hm)[:, :, :3]  # Apply colormap
    overlay = (1 - alpha) * img + alpha * hm_colored
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
