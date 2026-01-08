"""
PreNetAdaptive: Image + Network-aware Rate Predictor.

Predicts optimal compression rate based on:
1. Image features (complexity, detail level)
2. Network conditions (SNR, bandwidth, latency, BER)

This allows instance-level rate adaptation that considers
both content complexity and channel quality.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NetworkConditions:
    """Network channel characteristics for rate selection."""
    snr_db: float  # Signal-to-Noise Ratio (dB), 0-30
    bandwidth_mhz: float = 20.0  # Bandwidth in MHz
    latency_ms: float = 50.0  # Round-trip latency in ms
    ber_exp: int = 4  # BER exponent (10^-x)


class ResidualFCBlock(nn.Module):
    """Residual Fully Connected Block."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class PreNetAdaptive(nn.Module):
    """
    Adaptive Rate Predictor combining image features and network conditions.
    
    Predicts optimal compression rate [0.1, 1.0] for a given image
    under specific network conditions.
    
    Input:
        - Image features z0: [B, C, H, W] from encoder
        - Network conditions: SNR, bandwidth, latency, BER
        
    Output:
        - Optimal rate: [B, 1] in range [0.1, 1.0]
        - Optional: Predicted quality (PSNR, SSIM)
    """
    
    def __init__(
        self,
        num_channels: int = 256,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        predict_quality: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            num_channels: Number of encoder output channels
            hidden_dim: Hidden dimension for MLP
            num_blocks: Number of residual FC blocks
            predict_quality: Also predict PSNR/SSIM
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.predict_quality = predict_quality
        
        # Image feature dimension: 2 * C (mean + std per channel)
        # Network condition dimension: 4 (SNR, BW, latency, BER)
        # Total input: 2*C + 4
        input_dim = 2 * num_channels + 4
        
        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Residual blocks for feature processing
        self.blocks = nn.ModuleList([
            ResidualFCBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Rate prediction head (outputs logit, then sigmoid to [0, 1])
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Outputs [0, 1]
        )
        
        # Quality prediction head (optional)
        if predict_quality:
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 2)  # [PSNR/50, SSIM]
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_image_stats(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute channel-wise statistics from encoder features.
        
        Args:
            z: Encoder output [B, C, H, W]
            
        Returns:
            mu: Channel means [B, C]
            sigma: Channel stds [B, C]
        """
        mu = z.mean(dim=(2, 3))  # [B, C]
        sigma = z.std(dim=(2, 3))  # [B, C]
        return mu, sigma
    
    def normalize_network_conditions(
        self,
        snr_db: torch.Tensor,
        bandwidth_mhz: torch.Tensor,
        latency_ms: torch.Tensor,
        ber_exp: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize network conditions to [-1, 1] range.
        
        Args:
            snr_db: [B, 1] SNR in dB (0-30)
            bandwidth_mhz: [B, 1] Bandwidth in MHz (1-100)
            latency_ms: [B, 1] Latency in ms (5-500)
            ber_exp: [B, 1] BER exponent (2-6)
        
        Returns:
            network_features: [B, 4] normalized features
        """
        # Normalize to roughly [-1, 1]
        snr_norm = (snr_db - 15) / 15  # 0→-1, 30→+1
        bw_norm = (bandwidth_mhz - 50) / 50  # 1→-0.98, 100→+1
        lat_norm = (latency_ms - 250) / 250  # 5→-0.98, 500→+1
        ber_norm = (ber_exp - 4) / 2  # 2→-1, 6→+1
        
        return torch.cat([snr_norm, bw_norm, lat_norm, ber_norm], dim=1)
    
    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        bandwidth_mhz: torch.Tensor,
        latency_ms: torch.Tensor,
        ber_exp: torch.Tensor,
        min_rate: float = 0.1,
        max_rate: float = 1.0
    ) -> dict:
        """
        Predict optimal rate for given image and network conditions.
        
        Args:
            z: Encoder features [B, C, H, W]
            snr_db: [B, 1] or scalar
            bandwidth_mhz: [B, 1] or scalar
            latency_ms: [B, 1] or scalar
            ber_exp: [B, 1] or scalar
            min_rate: Minimum rate
            max_rate: Maximum rate
            
        Returns:
            dict with:
                - 'rate': Optimal rate [B, 1]
                - 'quality': Predicted [PSNR, SSIM] if predict_quality=True
        """
        B = z.size(0)
        device = z.device
        
        # Compute image statistics
        mu, sigma = self.compute_image_stats(z)
        
        # Ensure network conditions are tensors with correct shape
        def to_tensor(x, name):
            if isinstance(x, (int, float)):
                return torch.full((B, 1), x, device=device, dtype=z.dtype)
            if x.dim() == 0:
                return x.view(1, 1).expand(B, 1)
            if x.dim() == 1:
                return x.view(B, 1)
            return x
        
        snr_db = to_tensor(snr_db, 'snr')
        bandwidth_mhz = to_tensor(bandwidth_mhz, 'bw')
        latency_ms = to_tensor(latency_ms, 'latency')
        ber_exp = to_tensor(ber_exp, 'ber')
        
        # Normalize network conditions
        net_features = self.normalize_network_conditions(
            snr_db, bandwidth_mhz, latency_ms, ber_exp
        )
        
        # Concatenate all features
        x = torch.cat([mu, sigma, net_features], dim=1)  # [B, 2C + 4]
        
        # Process through network
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        
        # Predict rate (scale from [0,1] sigmoid output to [min_rate, max_rate])
        rate_raw = self.rate_head(x)  # [B, 1] in [0, 1]
        rate = min_rate + (max_rate - min_rate) * rate_raw  # [B, 1] in [min_rate, max_rate]
        
        result = {'rate': rate}
        
        if self.predict_quality:
            quality = self.quality_head(x)  # [B, 2]
            result['psnr'] = quality[:, 0:1] * 50  # Scale back to dB
            result['ssim'] = torch.sigmoid(quality[:, 1:2])  # Ensure [0, 1]
        
        return result


class PreNetAdaptiveLite(nn.Module):
    """
    Lightweight version of PreNetAdaptive.
    
    Faster inference, less parameters, good for real-time applications.
    """
    
    def __init__(self, num_channels: int = 256, hidden_dim: int = 64):
        super().__init__()
        
        # Simplified: just image complexity score + network conditions
        input_dim = 4 + 4  # 4 image stats + 4 network params
        
        self.image_stats = nn.Sequential(
            nn.Linear(2 * num_channels, 4),  # Compress to 4 features
            nn.GELU()
        )
        
        self.rate_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        z: torch.Tensor,
        snr_db: float,
        bandwidth_mhz: float = 20.0,
        latency_ms: float = 50.0,
        ber_exp: int = 4
    ) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        
        # Image stats
        mu = z.mean(dim=(2, 3))
        sigma = z.std(dim=(2, 3))
        img_features = self.image_stats(torch.cat([mu, sigma], dim=1))
        
        # Network features (normalized)
        net_features = torch.tensor([
            [snr_db / 30, bandwidth_mhz / 100, 1 - latency_ms / 500, ber_exp / 6]
        ], device=device).expand(B, -1)
        
        # Combine and predict
        x = torch.cat([img_features, net_features], dim=1)
        rate = self.rate_predictor(x)
        
        # Scale to [0.1, 1.0]
        return 0.1 + 0.9 * rate


if __name__ == '__main__':
    # Test PreNetAdaptive
    print("=" * 60)
    print("PreNetAdaptive - Test")
    print("=" * 60)
    
    model = PreNetAdaptive(num_channels=256)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Fake encoder output
    z = torch.randn(4, 256, 16, 16)
    
    # Test with different network conditions
    results = model(
        z,
        snr_db=15.0,
        bandwidth_mhz=20.0,
        latency_ms=50.0,
        ber_exp=4
    )
    
    print(f"\nInput z shape: {z.shape}")
    print(f"Predicted rate: {results['rate'].squeeze().tolist()}")
    if 'psnr' in results:
        print(f"Predicted PSNR: {results['psnr'].squeeze().tolist()}")
        print(f"Predicted SSIM: {results['ssim'].squeeze().tolist()}")
    
    # Test with varying conditions
    print("\n" + "-" * 40)
    print("Rate prediction for different network conditions:")
    print("-" * 40)
    
    conditions = [
        ("Excellent (5G)", 25, 100, 5, 6),
        ("Good (WiFi)", 15, 20, 20, 5),
        ("Moderate (LTE)", 10, 10, 50, 4),
        ("Poor (Satellite)", 5, 5, 500, 3),
    ]
    
    for name, snr, bw, lat, ber in conditions:
        result = model(z[:1], snr, bw, lat, ber)
        print(f"{name}: rate={result['rate'].item():.2f}")
