"""
PSSG: Power-Splitting Symbol Generator - FIXED VERSION.

Fixes:
- Power normalization via MASK (not k)
- Per-sample normalization (not batch global)
"""
import torch
import torch.nn as nn


class PSSG(nn.Module):
    """
    Spatial-Preserving PSSG with MASK-BASED power normalization.
    
    FIXED: Uses actual mask for power calculation, per-sample normalization.
    """
    
    def __init__(self, num_channels: int = 256, normalize_power: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.normalize_power = normalize_power
    
    def forward(
        self,
        xa: torch.Tensor,
        mask: torch.Tensor,
        rate: float = 1.0
    ) -> torch.Tensor:
        """
        Power normalize xa using MASK (not k).
        
        Args:
            xa: [B, C, H, W] masked features from SR-SC
            mask: [B, C, 1, 1] binary mask
            rate: transmission rate (unused now, kept for API compat)
            
        Returns:
            z: [B, C, H, W] power-normalized symbols (per sample)
        """
        if not self.normalize_power:
            return xa
        
        B, C, H, W = xa.shape
        
        # Expand mask to full spatial dims
        mask_hw = mask.expand_as(xa)  # [B, C, H, W]
        
        # Count active symbols per sample: k * H * W
        active_count = mask.sum(dim=(1, 2, 3)) * (H * W)  # [B]
        active_count = active_count + 1e-8  # epsilon
        
        # Power per sample on active symbols only
        power = ((xa ** 2) * mask_hw).sum(dim=(1, 2, 3)) / active_count  # [B]
        
        # Normalize per sample
        z = xa / torch.sqrt(power.view(B, 1, 1, 1) + 1e-8)
        
        return z
    
    def get_num_symbols(self, spatial_size: int = 16) -> int:
        return self.num_channels * spatial_size * spatial_size


if __name__ == "__main__":
    pssg = PSSG()
    
    xa = torch.randn(2, 256, 16, 16)
    mask = torch.zeros(2, 256, 1, 1)
    mask[:, :128, :, :] = 1.0  # 50% active
    
    z = pssg(xa, mask, rate=0.5)
    
    # Verify power is ~1 per sample on active channels
    for b in range(2):
        active = z[b, :128]
        power = (active ** 2).mean()
        print(f"Sample {b}: active power = {power:.4f} (should be ~1.0)")
