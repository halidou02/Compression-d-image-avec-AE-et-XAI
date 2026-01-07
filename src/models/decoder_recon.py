"""
Spatial Decoder with FiLM Conditioning on RATE + SNR.

Better adaptation especially at low SNR.
"""
import torch
import torch.nn as nn
from typing import Union


class DecoderRecon(nn.Module):
    """
    Spatial decoder with FiLM conditioning on BOTH rate AND SNR.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        image_size: int = 256,
        base_channels: int = 64
    ):
        super().__init__()
        
        # Joint rate+SNR embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(2, 64),  # (rate, snr_normalized)
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        
        # FiLM parameters for each stage
        self.film1 = nn.Linear(64, base_channels * 4 * 2)  # gamma + beta
        self.film2 = nn.Linear(64, base_channels * 2 * 2)
        self.film3 = nn.Linear(64, base_channels)
        self.film4 = nn.Linear(64, base_channels)
        
        # Initial feature processing
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling: 16 -> 32 -> 64 -> 128 -> 256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.base_channels = base_channels
    
    def forward(
        self,
        z: torch.Tensor,
        rate: float = 1.0,
        snr_db: Union[float, torch.Tensor] = 10.0
    ) -> torch.Tensor:
        """
        Forward pass with rate+SNR conditioning.
        
        Args:
            z: [B, 256, 16, 16] noisy symbols
            rate: transmission rate
            snr_db: channel SNR in dB
        """
        B = z.shape[0]
        device = z.device
        
        # Prepare conditioning input: (rate, snr_normalized)
        # Normalize SNR to [0, 1] range (assuming 0-20 dB)
        if isinstance(snr_db, torch.Tensor):
            snr_norm = snr_db.float() / 20.0
            if snr_norm.dim() == 0:
                snr_norm = snr_norm.unsqueeze(0).expand(B)
        else:
            snr_norm = torch.full((B,), snr_db / 20.0, device=device)
        
        rate_tensor = torch.full((B,), rate, device=device)
        
        # Normalize rate from [0.1, 1.0] to [0, 1] (like SNR)
        rate_norm = (rate_tensor - 0.1) / (1.0 - 0.1)
        rate_norm = rate_norm.clamp(0, 1)
        
        condition = torch.stack([rate_norm, snr_norm], dim=1)  # [B, 2]
        cond_emb = self.condition_embed(condition)  # [B, 64]
        
        # Initial conv
        x = self.init_conv(z)
        
        # Up1 + FiLM
        x = self.up1(x)
        film1 = self.film1(cond_emb)
        gamma1, beta1 = film1.chunk(2, dim=1)
        x = x * (1 + gamma1.view(B, -1, 1, 1)) + beta1.view(B, -1, 1, 1)
        
        # Up2 + FiLM
        x = self.up2(x)
        film2 = self.film2(cond_emb)
        gamma2, beta2 = film2.chunk(2, dim=1)
        x = x * (1 + gamma2.view(B, -1, 1, 1)) + beta2.view(B, -1, 1, 1)
        
        # Up3 + FiLM
        x = self.up3(x)
        film3 = self.film3(cond_emb)
        x = x * (1 + film3.view(B, -1, 1, 1))
        
        # Up4 + FiLM
        x = self.up4(x)
        film4 = self.film4(cond_emb)
        x = x * (1 + film4.view(B, -1, 1, 1))
        
        # Final
        x_hat = self.final(x)
        
        return x_hat


if __name__ == "__main__":
    decoder = DecoderRecon(in_channels=256)
    z = torch.randn(2, 256, 16, 16)
    out = decoder(z, rate=0.5, snr_db=10.0)
    print(f"Input: {z.shape} -> Output: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
