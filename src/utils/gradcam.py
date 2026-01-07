"""
Enhanced Grad-CAM utilities with Error Map.

Adds:
- Error map visualization
- Hook on 256ch features (channel_reduce) instead of layer3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class GradCAMHook:
    """
    Hook-based Grad-CAM.
    Can hook on layer3 (1024ch) or channel_reduce (256ch).
    """
    
    def __init__(self, target_layer: nn.Module):
        self.activations = None
        self.gradients = None
        
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def compute_cam(self) -> torch.Tensor:
        """Compute Grad-CAM. Returns [B, 1, H, W] detached."""
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Forward and backward required")
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.detach()
    
    def compute_channel_energy(self, features: torch.Tensor) -> torch.Tensor:
        cam = self.compute_cam()
        energy = (cam * features.abs()).mean(dim=(2, 3))
        return energy
    
    def clear(self):
        self.activations = None
        self.gradients = None
    
    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def compute_budget_loss(cam_energy: torch.Tensor, rate: float, num_channels: int = 256) -> torch.Tensor:
    k = max(1, min(num_channels, int(round(rate * num_channels))))
    energy_total = cam_energy.sum(dim=1) + 1e-8
    energy_inactive = cam_energy[:, k:].sum(dim=1)
    return (energy_inactive / energy_total).mean()


def compute_error_map(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute per-pixel error map.
    Returns: [B, 1, H, W] normalized error
    """
    err = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    B = err.shape[0]
    err_flat = err.view(B, -1)
    err_min = err_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    err_max = err_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    err_norm = (err - err_min) / (err_max - err_min + 1e-8)
    
    return err_norm.detach()


def visualize_full(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    cam: torch.Tensor,
    save_path: str,
    alpha: float = 0.5
):
    """
    Full visualization: Input, Recon, Error, CAM, Overlays.
    """
    # Prepare data
    img = image.detach().cpu().numpy()
    if img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.transpose(img, (1, 2, 0))
    
    recon = reconstruction.detach().cpu().numpy()
    recon = np.clip(recon, 0, 1)
    recon = np.transpose(recon, (1, 2, 0))
    
    # Error map
    err = ((image - reconstruction) ** 2).mean(dim=0).detach().cpu().numpy()
    err = (err - err.min()) / (err.max() - err.min() + 1e-8)
    
    # CAM (resize to image size)
    cam_cpu = cam.cpu()
    if cam_cpu.shape[-1] != image.shape[-1]:
        cam_cpu = F.interpolate(
            cam_cpu.unsqueeze(0), size=image.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze(0)
    cam_np = cam_cpu.squeeze().numpy()
    
    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input, Recon, Error
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(recon)
    axes[0, 1].set_title('Reconstruction')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(err, cmap='hot')
    axes[0, 2].set_title('Error Map (MSE)')
    axes[0, 2].axis('off')
    
    # Row 2: CAM, Error Overlay, CAM Overlay
    axes[1, 0].imshow(cam_np, cmap='jet')
    axes[1, 0].set_title('Grad-CAM')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img)
    axes[1, 1].imshow(err, cmap='hot', alpha=alpha)
    axes[1, 1].set_title('Error Overlay')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img)
    axes[1, 2].imshow(cam_np, cmap='jet', alpha=alpha)
    axes[1, 2].set_title('CAM Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_gradcam(image, cam, save_path, alpha=0.5):
    """Simple CAM visualization (backward compatible)."""
    img = image.detach().cpu().numpy()
    if img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.transpose(img, (1, 2, 0))
    
    cam_cpu = cam.cpu()
    if cam_cpu.shape[-1] != image.shape[-1]:
        cam_cpu = F.interpolate(
            cam_cpu.unsqueeze(0), size=image.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze(0)
    cam_np = cam_cpu.squeeze().numpy()
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Input')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam_np, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cam_np, cmap='jet', alpha=alpha)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_channel_energy(cam_energy: torch.Tensor, rate: float, save_path: str):
    energy = cam_energy.detach().cpu().numpy()
    C = len(energy)
    k = max(1, int(round(rate * C)))
    
    plt.figure(figsize=(10, 4))
    colors = ['green' if i < k else 'red' for i in range(C)]
    plt.bar(range(C), energy, color=colors, alpha=0.7)
    plt.axvline(x=k-0.5, color='black', linestyle='--', linewidth=2, label=f'Budget k={k}')
    
    active = energy[:k].sum()
    inactive = energy[k:].sum()
    total = active + inactive + 1e-8
    
    plt.xlabel('Channel')
    plt.ylabel('CAM Energy')
    plt.title(f'In-budget: {active/total*100:.1f}%, Out: {inactive/total*100:.1f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
