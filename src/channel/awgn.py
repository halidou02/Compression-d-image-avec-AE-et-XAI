"""
AWGN Channel with Mask Cleaning (Option A).

After adding noise, re-apply the mask to clear noise from inactive channels.
This prevents the decoder from interpreting noise as features.
"""
import torch
import torch.nn as nn
import math
from typing import Union, Optional


class AWGNChannel(nn.Module):
    """
    AWGN Channel with MASK CLEANING.
    
    Key: z_rx = (z_tx + noise) * mask
    This clears noise from inactive channels.
    """
    
    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.default_snr_db = snr_db
    
    @staticmethod
    def snr_to_noise_std(snr_db: Union[float, torch.Tensor], signal_power: float = 1.0):
        """Convert SNR (dB) to noise standard deviation."""
        snr_linear = 10 ** (snr_db / 10)
        if isinstance(snr_linear, torch.Tensor):
            noise_std = torch.sqrt(torch.tensor(signal_power, device=snr_linear.device) / snr_linear)
        else:
            noise_std = math.sqrt(signal_power / snr_linear)
        return noise_std
    
    def forward(
        self,
        z: torch.Tensor,
        snr_db: Union[float, torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: add AWGN noise then CLEAN with mask.
        
        Args:
            z: Input signal [B, C, H, W] (power-normalized, masked)
            snr_db: SNR in dB
            mask: [B, C, 1, 1] binary mask for cleaning
            
        Returns:
            z_prime: Cleaned noisy signal [B, C, H, W]
        """
        if snr_db is None:
            snr_db = self.default_snr_db
        
        B = z.shape[0]
        device = z.device
        
        # Compute noise std (assuming unit power on active symbols)
        if isinstance(snr_db, torch.Tensor):
            noise_std = self.snr_to_noise_std(snr_db, 1.0)
            while noise_std.dim() < z.dim():
                noise_std = noise_std.unsqueeze(-1)
        else:
            noise_std = self.snr_to_noise_std(snr_db, 1.0)
        
        # Add noise
        noise = torch.randn_like(z) * noise_std
        z_noisy = z + noise
        
        # MASK CLEANING: clear noise from inactive channels
        if mask is not None:
            z_prime = z_noisy * mask
        else:
            z_prime = z_noisy
        
        return z_prime


if __name__ == "__main__":
    channel = AWGNChannel(snr_db=10.0)
    
    # Test with mask
    z = torch.randn(2, 256, 16, 16)
    mask = torch.zeros(2, 256, 1, 1)
    mask[:, :128, :, :] = 1.0
    z = z * mask  # Pre-masked
    
    z_noisy = channel(z, snr_db=10.0, mask=mask)
    
    # Check: inactive channels should be exactly zero
    inactive = z_noisy[:, 128:, :, :]
    print(f"Inactive channel max: {inactive.abs().max().item():.6f} (should be 0)")
    print("Test passed!" if inactive.abs().max() < 1e-6 else "Test FAILED!")
