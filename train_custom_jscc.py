"""
JSCC with Custom Autoencoder + Teacher Grad-CAM
================================================
Single-file training script for semantic communication.

Architecture:
- Custom Encoder: Conv-based (from scratch, no pretrained)
- Custom Decoder: Symmetric to encoder  
- SR-SC: Ordered channel selection
- PSSG: Power normalization
- AWGN: Channel simulation
- Teacher CAM: ResNet-152 (best) for semantic guidance

Usage:
    python train_custom_jscc.py --data_dir /path/to/coco --batch_size 32 --epochs 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================

class CocoDataset(Dataset):
    """COCO dataset for image reconstruction."""
    
    def __init__(self, root_dir: str, transform=None, max_images: int = None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        
        # Find all images
        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.images.extend(list(self.root_dir.glob(ext)))
        
        if max_images:
            self.images = self.images[:max_images]
        
        logger.info(f"Dataset: {len(self.images)} images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, img  # input, target
        except:
            # Return a random valid image on error
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """Create train and validation dataloaders."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    dataset = CocoDataset(data_dir, transform=transform)
    
    # 90/10 split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# CUSTOM ENCODER (From Scratch)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + residual)


class CustomEncoder(nn.Module):
    """
    Custom convolutional encoder - MEDIUM capacity (~20M params).
    Input:  [B, 3, 256, 256]
    Output: [B, 256, 16, 16]
    """
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
        # 256 -> 128 (MEDIUM: 128 channels instead of 64)
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=2, padding=3),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            ResBlock(128),  # Extra ResBlock for more capacity
        )
        
        # 128 -> 64 (MEDIUM: 256 channels instead of 128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            ResBlock(256),  # Extra ResBlock
        )
        
        # 64 -> 32 (MEDIUM: 512 channels instead of 256)
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            ResBlock(512),
            ResBlock(512),  # Extra ResBlock
        )
        
        # 32 -> 16 (MEDIUM: 1024 channels instead of 512)
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            ResBlock(1024),
            ResBlock(1024),  # Extra ResBlock
        )
        
        # Reduce to output channels (keep latent at 256)
        self.reduce = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.reduce(x)
        return x


# ============================================================================
# SR-SC (Semantic Rate Selection and Compression)
# ============================================================================

class SRSC(nn.Module):
    """Ordered channel selection based on SE-Block importance."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        self.channels = channels
    
    def forward(self, x, keep_ratio: float = 1.0):
        B, C, H, W = x.shape
        
        # SE weights
        se_weights = self.se(x)  # [B, C]
        
        # Apply SE weighting
        x = x * se_weights.view(B, C, 1, 1)
        
        # Compute k (number of active channels)
        k = max(1, min(C, round(keep_ratio * C)))
        
        # Create ordered mask (first k channels active)
        mask = torch.zeros(B, C, 1, 1, device=x.device)
        mask[:, :k] = 1.0
        
        return x * mask, mask, k


# ============================================================================
# PSSG (Power-Constrained Semantic Symbol Generation)
# ============================================================================

class PSSG(nn.Module):
    """Per-sample power normalization."""
    
    def forward(self, x, mask):
        B = x.size(0)
        
        # Compute power only on active channels
        active_elements = (x * mask).view(B, -1)
        mask_flat = mask.expand_as(x).reshape(B, -1)
        
        active_count = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        power = (active_elements ** 2).sum(dim=1, keepdim=True) / active_count
        power = power.clamp(min=1e-8)
        
        # Normalize
        scale = 1.0 / power.sqrt()
        x_normalized = x * scale.view(B, 1, 1, 1)
        
        return x_normalized, scale


# ============================================================================
# AWGN CHANNEL
# ============================================================================

class AWGNChannel(nn.Module):
    """Additive White Gaussian Noise channel."""
    
    def forward(self, x, snr_db: float):
        if self.training:
            # SNR to noise variance
            snr_linear = 10 ** (snr_db / 10)
            noise_var = 1.0 / snr_linear
            noise = torch.randn_like(x) * np.sqrt(noise_var)
            return x + noise
        else:
            # Less noise during eval for fair comparison
            snr_linear = 10 ** (snr_db / 10)
            noise_var = 1.0 / snr_linear
            noise = torch.randn_like(x) * np.sqrt(noise_var)
            return x + noise


# ============================================================================
# FiLM CONDITIONING
# ============================================================================

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation."""
    
    def __init__(self, channels: int, cond_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, channels * 2),
        )
        self.channels = channels
    
    def forward(self, x, rate: float, snr: float):
        B = x.size(0)
        
        # Normalize inputs
        rate_norm = (rate - 0.5) * 2  # [0.1, 1.0] -> [-0.8, 1.0]
        snr_norm = (snr - 10) / 10    # [0, 20] -> [-1, 1]
        
        cond = torch.tensor([[rate_norm, snr_norm]], device=x.device).expand(B, -1)
        params = self.mlp(cond)
        
        gamma = params[:, :self.channels].view(B, self.channels, 1, 1)
        beta = params[:, self.channels:].view(B, self.channels, 1, 1)
        
        return x * (1 + gamma) + beta


# ============================================================================
# CUSTOM DECODER
# ============================================================================

class CustomDecoder(nn.Module):
    """
    Custom decoder - MEDIUM capacity (~20M params).
    Input:  [B, 256, 16, 16] + rate/snr conditioning
    Output: [B, 3, 256, 256]
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # FiLM conditioning
        self.film = FiLMBlock(in_channels)
        
        # Progressive channel groups
        self.base_ch = in_channels // 2
        self.ref1_ch = in_channels // 4
        self.ref2_ch = in_channels - self.base_ch - self.ref1_ch
        
        # Initial processing (MEDIUM: 1024 channels)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 1024, 3, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            ResBlock(1024),
            ResBlock(1024),  # Extra ResBlock
        )
        
        # 16 -> 32 (MEDIUM: 512 channels)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            ResBlock(512),
            ResBlock(512),  # Extra ResBlock
        )
        
        # 32 -> 64 (MEDIUM: 256 channels)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            ResBlock(256),  # Extra ResBlock
        )
        
        # 64 -> 128 (MEDIUM: 128 channels)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            ResBlock(128),  # Extra ResBlock
        )
        
        # 128 -> 256 (MEDIUM: 64 channels)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        
        # Final output
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z, rate: float = 1.0, snr_db: float = 10.0):
        # FiLM conditioning
        z = self.film(z, rate, snr_db)
        
        # Decode
        x = self.initial(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.output(x)
        
        return x


# ============================================================================
# COMPLETE JSCC PIPELINE
# ============================================================================

class CustomJSCC(nn.Module):
    """Complete JSCC pipeline with custom encoder/decoder."""
    
    def __init__(self, num_channels: int = 256):
        super().__init__()
        
        self.encoder = CustomEncoder(out_channels=num_channels)
        self.sr_sc = SRSC(num_channels)
        self.pssg = PSSG()
        self.channel = AWGNChannel()
        self.decoder = CustomDecoder(in_channels=num_channels)
        
        self.num_channels = num_channels
    
    def forward(self, x, snr_db: float = 10.0, rate: float = 1.0, return_intermediate: bool = False):
        # Encode
        z = self.encoder(x)
        
        # Rate selection
        z_selected, mask, k = self.sr_sc(z, rate)
        
        # Power normalization
        z_norm, scale = self.pssg(z_selected, mask)
        
        # Channel
        z_noisy = self.channel(z_norm, snr_db)
        
        # Decode
        x_hat = self.decoder(z_noisy, rate, snr_db)
        
        if return_intermediate:
            return {
                'output': x_hat,
                'features': z_selected,
                'k': k,
                'mask': mask,
            }
        
        return x_hat


# ============================================================================
# TEACHER GRAD-CAM (ResNet-152 for best quality)
# ============================================================================

class TeacherGradCAM:
    """
    Teacher Grad-CAM using ResNet-152 (best pretrained model).
    Generates semantic importance maps for CAM-weighted loss and budget loss.
    """
    
    def __init__(self, device):
        self.device = device
        
        # Use ResNet-152 for best CAM quality
        self.model = models.resnet152(weights='IMAGENET1K_V2').to(device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Hook on layer4 for CAM
        self.activations = None
        self.gradients = None
        
        self.model.layer4.register_forward_hook(self._save_activation)
        self.model.layer4.register_full_backward_hook(self._save_gradient)
        
        logger.info("Teacher Grad-CAM initialized with ResNet-152")
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    @torch.no_grad()
    def get_cam(self, x):
        """Generate CAM heatmap for input images with proper ImageNet normalization."""
        # Resize to 224x224 and normalize for ImageNet
        x_in = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = x_in.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x_in.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_in = (x_in - mean) / std
        
        # Enable gradients temporarily for CAM
        with torch.enable_grad():
            x_grad = x_in.clone().requires_grad_(True)
            
            # Forward pass
            output = self.model(x_grad)
            
            # Get top class score
            scores = output.max(dim=1)[0]
            
            # Backward for gradients
            self.model.zero_grad()
            scores.sum().backward()
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam


# ============================================================================
# METRICS
# ============================================================================

def compute_psnr(x_hat, x, max_val=1.0):
    """Compute PSNR."""
    mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return psnr


def compute_ssim(x_hat, x, window_size=11):
    """Compute SSIM (fixed window construction)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    # Gaussian window - FIXED with torch.outer
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    window_2d = torch.outer(g, g)  # [ws, ws] - proper outer product
    window = window_2d.expand(3, 1, window_size, window_size).contiguous()
    
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=3)
    mu_y = F.conv2d(x_hat, window, padding=window_size // 2, groups=3)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2, groups=3) - mu_x_sq
    sigma_y_sq = F.conv2d(x_hat ** 2, window, padding=window_size // 2, groups=3) - mu_y_sq
    sigma_xy = F.conv2d(x * x_hat, window, padding=window_size // 2, groups=3) - mu_xy
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean(dim=(1, 2, 3))


def compute_msssim(x_hat, x, window_size=11, weights=None):
    """
    Compute MS-SSIM (Multi-Scale SSIM) for better perceptual quality.
    Uses 5 scales with standard weights from the original paper.
    """
    if weights is None:
        # Standard weights from Wang et al. 2003
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device)
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    window = window_2d.expand(3, 1, window_size, window_size).contiguous()
    
    msssim_vals = []
    mcs_vals = []
    
    for i in range(len(weights)):
        # Compute SSIM at this scale
        mu_x = F.conv2d(x, window, padding=window_size // 2, groups=3)
        mu_y = F.conv2d(x_hat, window, padding=window_size // 2, groups=3)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2, groups=3) - mu_x_sq
        sigma_y_sq = F.conv2d(x_hat ** 2, window, padding=window_size // 2, groups=3) - mu_y_sq
        sigma_xy = F.conv2d(x * x_hat, window, padding=window_size // 2, groups=3) - mu_xy
        
        # Contrast-structure (CS)
        cs = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)
        mcs_vals.append(cs.mean(dim=(1, 2, 3)))
        
        # Full SSIM (only needed for last scale)
        if i == len(weights) - 1:
            l = (2 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
            msssim_vals.append(l.mean(dim=(1, 2, 3)))
        
        # Downsample for next scale (except last)
        if i < len(weights) - 1:
            x = F.avg_pool2d(x, 2)
            x_hat = F.avg_pool2d(x_hat, 2)
    
    # Combine scales
    mcs_vals = torch.stack(mcs_vals[:-1], dim=1)  # [B, 4]
    msssim = msssim_vals[0] * torch.prod(mcs_vals ** weights[:-1].view(1, -1), dim=1)
    
    return msssim


class AverageMeter:
    """Track running average."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================================
# PERCEPTUAL LOSS (VGG19)
# ============================================================================

class PerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss for better visual quality.
    Compares features at multiple layers for texture and structure.
    """
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Load VGG19 pretrained
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        
        # Use features up to relu3_3 (layer 16) and relu4_3 (layer 25)
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Move entire module to device
        self.to(device)
        self.eval()
        logger.info("Perceptual Loss initialized with VGG19")
    
    def normalize(self, x):
        """Normalize input for VGG."""
        return (x - self.mean) / self.std
    
    def forward(self, x_hat, x):
        """Compute perceptual loss between reconstruction and original."""
        # Normalize
        x_hat_norm = self.normalize(x_hat)
        x_norm = self.normalize(x)
        
        # Extract features
        f1_hat = self.slice1(x_hat_norm)
        f1 = self.slice1(x_norm)
        
        f2_hat = self.slice2(f1_hat)
        f2 = self.slice2(f1)
        
        f3_hat = self.slice3(f2_hat)
        f3 = self.slice3(f2)
        
        # Compute L1 loss at each layer (L1 often better than L2 for perceptual)
        loss = F.l1_loss(f1_hat, f1) + F.l1_loss(f2_hat, f2) + F.l1_loss(f3_hat, f3)
        
        return loss


# ============================================================================
# LOSS SCHEDULING
# ============================================================================

def get_loss_weights(epoch, total_epochs):
    """Progressive loss scheduling with perceptual loss."""
    if epoch < 10:
        # Phase 1: MSE only
        lambda_ssim = 0.02 * epoch  # 0 -> 0.18
        lambda_perceptual = 0.0
        lambda_budget = 0.0
        alpha_cam = 0.0
    elif epoch < 20:
        # Phase 2: Add SSIM + Perceptual
        lambda_ssim = 0.15
        lambda_perceptual = 0.01  # Start perceptual loss
        lambda_budget = 0.0
        alpha_cam = 0.0
    elif epoch < 40:
        # Phase 3: Add budget
        lambda_ssim = 0.15
        lambda_perceptual = 0.02  # Increase perceptual
        progress = (epoch - 20) / 20
        lambda_budget = 0.005 * progress
        alpha_cam = 0.0
    else:
        # Phase 4: Full
        lambda_ssim = 0.15
        lambda_perceptual = 0.02
        lambda_budget = 0.005
        progress = min(1.0, (epoch - 40) / 20)
        alpha_cam = 0.5 + 1.5 * progress  # 0.5 -> 2.0
    
    return lambda_ssim, lambda_perceptual, lambda_budget, alpha_cam


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, teacher_cam, perceptual_loss, loader, optimizer, device, epoch, total_epochs, scheduler=None):
    """Train one epoch with perceptual loss for visual quality."""
    model.train()
    
    loss_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_m = AverageMeter()
    budget_m = AverageMeter()
    
    lambda_ssim, lambda_perceptual, lambda_budget, alpha_cam = get_loss_weights(epoch, total_epochs)
    
    # Eval grid for oversampling (70% from grid, 30% continuous)
    EVAL_RATES = [0.25, 0.5, 1.0]
    EVAL_SNRS = [5.0, 10.0, 20.0]
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} (λ_perc={lambda_perceptual:.2f}, α_cam={alpha_cam:.1f})')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        B = images.size(0)
        
        # Sample rate/SNR - 70% from eval grid, 30% continuous
        if np.random.rand() < 0.7:
            rate = float(np.random.choice(EVAL_RATES))
            snr = float(np.random.choice(EVAL_SNRS))
        else:
            rate = np.random.uniform(0.1, 1.0)
            snr = float(np.random.uniform(0, 20))
        
        optimizer.zero_grad()
        
        # Forward - get pre-mask z for budget loss
        z = model.encoder(images)
        z_selected, mask, k = model.sr_sc(z, rate)
        z_norm, scale = model.pssg(z_selected, mask)
        z_noisy = model.channel(z_norm, snr)
        x_hat = model.decoder(z_noisy, rate, snr)
        
        # MSE Loss
        mse_loss = F.mse_loss(x_hat, targets)
        
        # MS-SSIM Loss (better than SSIM for perceptual quality)
        msssim_val = compute_msssim(x_hat.clone(), targets.clone())
        ssim_val = compute_ssim(x_hat, targets)  # Keep for logging
        ssim_loss = (1 - msssim_val).mean()  # Use MS-SSIM for loss
        
        # Compute CAM once (if needed)
        cam = None
        if alpha_cam > 0 or lambda_budget > 0:
            cam = teacher_cam.get_cam(images)
        
        # CAM-Weighted MSE
        if alpha_cam > 0 and cam is not None:
            cam_full = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
            cam_full = cam_full / (cam_full.mean(dim=(2, 3), keepdim=True) + 1e-8)
            weight = 1.0 + alpha_cam * cam_full
            wmse = (weight * (x_hat - targets).pow(2)).mean()
        else:
            wmse = mse_loss
        
        # Budget Loss - using pre-mask features z (not z_selected which is already masked)
        if lambda_budget > 0 and cam is not None:
            cam_feat = F.interpolate(cam, size=(16, 16), mode='bilinear', align_corners=False)
            energy = (cam_feat * z.abs()).mean(dim=(2, 3))  # Use z, not z_selected
            energy_total = energy.sum(dim=1) + 1e-8
            energy_inactive = energy[:, k:].sum(dim=1)
            L_budget = (energy_inactive / energy_total).mean()
        else:
            L_budget = torch.tensor(0.0, device=device)
        
        # Perceptual Loss (VGG features)
        if lambda_perceptual > 0:
            L_perceptual = perceptual_loss(x_hat, targets)
        else:
            L_perceptual = torch.tensor(0.0, device=device)
        
        # Total Loss
        L = wmse + lambda_ssim * ssim_loss + lambda_perceptual * L_perceptual + lambda_budget * L_budget
        
        # Skip NaN
        if torch.isnan(L) or torch.isinf(L):
            continue
        
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Scheduler: fractional step for proper warm restarts
        if scheduler:
            scheduler.step((epoch - 1) + batch_idx / len(loader))
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = ssim_val.mean().item()
        
        loss_m.update(L.item(), B)
        psnr_m.update(psnr, B)
        ssim_m.update(ssim, B)
        budget_m.update(L_budget.item() if isinstance(L_budget, torch.Tensor) else L_budget, B)
        
        pbar.set_postfix({
            'loss': f'{loss_m.avg:.4f}',
            'psnr': f'{psnr_m.avg:.2f}',
            'ssim': f'{ssim_m.avg:.4f}'
        })
    
    return {'loss': loss_m.avg, 'psnr': psnr_m.avg, 'ssim': ssim_m.avg, 'budget': budget_m.avg}


@torch.no_grad()
def validate_grid(model, loader, device, max_batches=20):
    """Grid validation for rate × SNR."""
    model.eval()
    
    GRID = [
        (0.25, 5.0), (0.25, 10.0), (0.25, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
        (1.0, 5.0), (1.0, 10.0), (1.0, 20.0),
    ]
    
    results = {pt: {'psnr': [], 'ssim': []} for pt in GRID}
    
    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        images = images.to(device)
        targets = targets.to(device)
        
        for rate, snr in GRID:
            x_hat = model(images, snr_db=snr, rate=rate)
            psnr = compute_psnr(x_hat, targets).mean().item()
            ssim = compute_ssim(x_hat, targets).mean().item()
            results[(rate, snr)]['psnr'].append(psnr)
            results[(rate, snr)]['ssim'].append(ssim)
    
    # Aggregate
    for pt in GRID:
        results[pt]['psnr'] = np.mean(results[pt]['psnr'])
        results[pt]['ssim'] = np.mean(results[pt]['ssim'])
    
    # Average
    avg_psnr = np.mean([results[pt]['psnr'] for pt in GRID])
    avg_ssim = np.mean([results[pt]['ssim'] for pt in GRID])
    
    # Monotonicity check
    mono_ok = 0
    for snr in [5.0, 10.0, 20.0]:
        p25 = results[(0.25, snr)]['psnr']
        p50 = results[(0.5, snr)]['psnr']
        p100 = results[(1.0, snr)]['psnr']
        if p25 <= p50 <= p100:
            mono_ok += 1
    mono_score = mono_ok / 3
    
    return {
        'grid': results,
        'grid_avg_psnr': avg_psnr,
        'grid_avg_ssim': avg_ssim,
        'mono_score': mono_score,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Custom JSCC Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to COCO images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output_custom')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Custom JSCC Training (From Scratch Encoder/Decoder)")
    logger.info("=" * 60)
    logger.info(f"Device: {device}, Batch: {args.batch_size}, LR: {args.lr}")
    
    # Data
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)
    
    # Model
    model = CustomJSCC(num_channels=256).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Teacher CAM
    teacher_cam = TeacherGradCAM(device)
    
    # Perceptual Loss (VGG19)
    perceptual_loss = PerceptualLoss(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Resume
    start_epoch = 1
    best_score = 0
    ckpt_path = output_dir / 'best_custom_jscc.pt'
    
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_score = ckpt.get('score', 0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}")
    
    # Metrics file
    metrics_file = output_dir / 'metrics.csv'
    if not args.resume or not metrics_file.exists():
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,train_psnr,train_ssim,train_budget,grid_avg_psnr,grid_avg_ssim,mono_score\n')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")
        
        train_m = train_epoch(
            model, teacher_cam, perceptual_loss, train_loader, optimizer, device, epoch, args.epochs,
            scheduler=scheduler
        )
        
        val_m = validate_grid(model, val_loader, device)
        
        logger.info(f"Train - PSNR: {train_m['psnr']:.2f}, SSIM: {train_m['ssim']:.4f}, Budget: {train_m['budget']:.4f}")
        logger.info(f"Grid  - PSNR: {val_m['grid_avg_psnr']:.2f}, SSIM: {val_m['grid_avg_ssim']:.4f}, Mono: {val_m['mono_score']:.2%}")
        
        # Save metrics
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch},{train_m['loss']:.6f},{train_m['psnr']:.4f},{train_m['ssim']:.6f},{train_m['budget']:.6f},{val_m['grid_avg_psnr']:.4f},{val_m['grid_avg_ssim']:.4f},{val_m['mono_score']:.4f}\n")
        
        # Save best
        score = val_m['grid_avg_psnr'] + 5 * val_m['mono_score'] + 10 * val_m['grid_avg_ssim']
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'grid_psnr': val_m['grid_avg_psnr'],
                'grid_ssim': val_m['grid_avg_ssim'],
                'mono_score': val_m['mono_score'],
                'score': score,
            }, ckpt_path)
            logger.info(f"[BEST] Score: {score:.2f}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
