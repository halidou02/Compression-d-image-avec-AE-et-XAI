"""
SR-SC: Semantic Rate Selection and Compression (Option A - Fixed).

Fixes:
- Unified compute_k() with round + clamp
- Consistent k calculation across all modules
"""
import torch
import torch.nn as nn
from typing import Tuple


def compute_k(rate: float, C: int) -> int:
    """Unified k computation: round + clamp."""
    k = int(round(rate * C))
    return max(1, min(C, k))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for soft channel weighting."""
    
    def __init__(self, num_channels: int = 256, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, _, _ = x.shape
        y = self.squeeze(x).view(B, C)
        weights = self.excitation(y)
        weighted = x * weights.view(B, C, 1, 1)
        return weighted, weights


class SRSC(nn.Module):
    """
    Semantic Rate Selection with ORDERED CHANNELS (Fixed).
    
    Uses compute_k() for consistent k calculation.
    """
    
    def __init__(self, num_channels: int = 256, reduction: int = 16):
        super().__init__()
        
        self.num_channels = num_channels
        self.se_block = SEBlock(num_channels, reduction)
    
    def forward(
        self,
        features: torch.Tensor,
        keep_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with FIXED k computation (round + clamp).
        """
        B, C, H, W = features.shape
        device = features.device
        
        # SE-Block: soft reweighting
        weighted, weights = self.se_block(features)
        
        # FIXED: use compute_k (round + clamp)
        k = compute_k(keep_ratio, C)
        
        # Create ordered mask
        mask = torch.zeros(B, C, 1, 1, device=device, dtype=features.dtype)
        mask[:, :k, :, :] = 1.0
        
        # Apply mask
        xa = weighted * mask
        
        return xa, weights, mask


if __name__ == "__main__":
    sr_sc = SRSC(num_channels=256)
    
    # Test edge cases
    for rate in [0.1, 0.5, 0.999, 1.0]:
        k = compute_k(rate, 256)
        print(f"rate={rate:.3f} -> k={k}")
    
    features = torch.randn(2, 256, 16, 16)
    xa, weights, mask = sr_sc(features, keep_ratio=1.0)
    print(f"\nrate=1.0: mask active channels = {mask.sum().item() / 2}")
