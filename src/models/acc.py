"""
ACC: Adaptive Channel Condition module.

Re-weights feature channels based on channel SNR.
Allows the model to adapt its representation to channel quality.

Input: xa [B, C, Hf, Wf], gamma [B, 1] (SNR in dB)
Output: x_prime [B, C, Hf, Wf]
"""
import torch
import torch.nn as nn
from typing import Optional


class ACC(nn.Module):
    """
    Adaptive Channel Condition module.
    
    Scales feature channels based on SNR to adapt to channel conditions.
    Uses global pooling + MLP to generate per-channel scaling factors.
    """
    
    def __init__(
        self,
        num_channels: int = 2048,
        hidden_dim: int = 256,
        snr_embed_dim: int = 32
    ):
        """
        Args:
            num_channels: Number of feature channels
            hidden_dim: Hidden dimension of MLP
            snr_embed_dim: Dimension for SNR embedding
        """
        super().__init__()
        
        self.num_channels = num_channels
        
        # SNR embedding layer (scalar -> vector)
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(snr_embed_dim, snr_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # MLP: [channel stats + SNR embed] -> scale factors
        # Channel stats: global average pool gives [B, C]
        input_dim = num_channels + snr_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels),
            nn.Sigmoid()  # Scale factors in [0, 1]
        )
        
        # Optional: learnable base scale
        self.base_scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
    
    def forward(
        self,
        xa: torch.Tensor,
        gamma: torch.Tensor, 
        use_base_scale: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xa: Input features [B, C, H, W]
            gamma: SNR in dB [B, 1] or [B,]
            use_base_scale: Whether to multiply by learnable base scale
            
        Returns:
            x_prime: Scaled features [B, C, H, W]
        """
        B, C, H, W = xa.shape
        
        # Ensure gamma has shape [B, 1]
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        
        # Global average pooling to get channel statistics
        channel_stats = xa.mean(dim=(2, 3))  # [B, C]
        
        # Embed SNR
        snr_features = self.snr_embed(gamma)  # [B, snr_embed_dim]
        
        # Concatenate channel stats and SNR features
        combined = torch.cat([channel_stats, snr_features], dim=1)  # [B, C + snr_embed_dim]
        
        # Generate scale factors
        scale = self.mlp(combined)  # [B, C]
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # Apply scaling
        x_prime = xa * scale
        
        # Optional base scale
        if use_base_scale:
            x_prime = x_prime * self.base_scale
        
        return x_prime


class ACCSimple(nn.Module):
    """
    Simplified ACC using just channel attention mechanism.
    """
    
    def __init__(self, num_channels: int = 512, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_channels + 1, num_channels // reduction),  # +1 for SNR
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )
    
    def forward(self, xa: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xa.shape
        
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        
        # Channel statistics
        avg_out = self.avg_pool(xa).view(B, C)
        
        # Concatenate with SNR
        combined = torch.cat([avg_out, gamma], dim=1)
        
        # Generate scale
        scale = self.mlp(combined).unsqueeze(-1).unsqueeze(-1)
        
        return xa * scale


if __name__ == "__main__":
    # Test ACC
    acc = ACC(num_channels=512)
    
    xa = torch.randn(2, 512, 8, 8)
    gamma = torch.tensor([10.0, 5.0])  # SNR in dB
    
    x_prime = acc(xa, gamma)
    
    print(f"Input shape: {xa.shape}")
    print(f"SNR shape: {gamma.shape}")
    print(f"Output shape: {x_prime.shape}")
