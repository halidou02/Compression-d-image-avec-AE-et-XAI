"""
Evaluation Metrics.

PSNR, SSIM, Accuracy, and other metrics for evaluation.
"""
import torch
import torch.nn.functional as F
import math
from typing import Optional


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model outputs [B, num_classes]
        labels: Ground truth labels [B,]
        top_k: Top-k accuracy (default 1 for standard accuracy)
        
    Returns:
        Accuracy as float in [0, 1]
    """
    _, pred = logits.topk(top_k, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        PSNR in dB [B,]
    """
    mse = F.mse_loss(img1, img2, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return psnr


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    size_average: bool = False
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of Gaussian window
        data_range: Range of pixel values
        size_average: If True, return scalar average
        
    Returns:
        SSIM values [B,] or scalar
    """
    C = img1.shape[1]
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    # 2D window
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(C, 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    # Constants for stability
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Mean over spatial dimensions
    ssim_val = ssim_map.mean(dim=(1, 2, 3))
    
    if size_average:
        return ssim_val.mean()
    return ssim_val


class SSIMLoss(torch.nn.Module):
    """
    SSIM Loss = 1 - SSIM for training.
    """
    
    def __init__(self, window_size: int = 11, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        ssim_val = compute_ssim(img1, img2, self.window_size, self.data_range, size_average=True)
        return 1 - ssim_val


class ReconstructionLoss(torch.nn.Module):
    """
    Combined reconstruction loss: MSE + (1 - SSIM).
    
    No explicit L2 - use AdamW weight_decay instead.
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_ssim: float = 0.1
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.mse = torch.nn.MSELoss()
        self.ssim_loss = SSIMLoss()
    
    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor
    ) -> dict:
        mse_loss = self.mse(x_hat, x)
        ssim_loss = self.ssim_loss(x_hat, x)
        
        total_loss = self.lambda_mse * mse_loss + self.lambda_ssim * ssim_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'ssim': ssim_loss
        }


def compute_bpp(
    rate: float,
    smax: int,
    image_size: int,
    bits_per_symbol: int = 8
) -> float:
    """
    Compute bits per pixel for transmission.
    
    Args:
        rate: Transmission rate (0, 1]
        smax: Maximum latent dimension
        image_size: Image size
        bits_per_symbol: Bits per transmitted symbol
        
    Returns:
        Bits per pixel
    """
    num_symbols = int(rate * smax)
    total_bits = num_symbols * bits_per_symbol
    num_pixels = image_size * image_size
    return total_bits / num_pixels


if __name__ == "__main__":
    # Test metrics
    img1 = torch.rand(2, 3, 64, 64)
    img2 = img1 + 0.1 * torch.randn_like(img1)
    img2 = img2.clamp(0, 1)
    
    psnr = compute_psnr(img1, img2)
    ssim = compute_ssim(img1, img2)
    
    print(f"PSNR: {psnr.mean():.2f} dB")
    print(f"SSIM: {ssim.mean():.4f}")
    
    # Test accuracy
    logits = torch.randn(8, 10)
    labels = torch.randint(0, 10, (8,))
    acc = compute_accuracy(logits, labels)
    print(f"Accuracy: {acc:.2%}")
