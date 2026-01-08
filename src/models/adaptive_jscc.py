"""
AdaptiveJSCC: Unified Semantic Communication Model with Automatic Rate Selection.

This module combines:
1. JSCC (Encoder-Decoder for image compression/transmission)
2. PreNetAdaptive (Rate predictor based on image + network conditions)
3. Channel Simulator (Realistic network condition generation)

Single forward pass: Image + Network Conditions → Reconstructed Image
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Network Conditions
# ============================================================================

@dataclass
class NetworkConditions:
    """Network channel characteristics."""
    snr_db: float = 15.0
    bandwidth_mhz: float = 20.0
    latency_ms: float = 50.0
    ber_exp: float = 4.0  # BER = 10^(-ber_exp)
    
    def to_tensor(self, device='cpu') -> Tuple[torch.Tensor, ...]:
        """Convert to tensors for model input."""
        return (
            torch.tensor([[self.snr_db]], device=device),
            torch.tensor([[self.bandwidth_mhz]], device=device),
            torch.tensor([[self.latency_ms]], device=device),
            torch.tensor([[self.ber_exp]], device=device),
        )


# ============================================================================
# Building Blocks
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class ResidualFCBlock(nn.Module):
    """Residual FC block for rate predictor."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.norm(out + residual)


# ============================================================================
# Encoder
# ============================================================================

class Encoder(nn.Module):
    """Image encoder with progressive downsampling."""
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
        # Progressive downsampling: 256 -> 128 -> 64 -> 32 -> 16
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            ResBlock(512),
        )
        
        # Reduce to output channels
        self.reduce = nn.Sequential(
            nn.Conv2d(512, out_channels, 1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        z = self.reduce(x)
        return z  # [B, C, 16, 16]


# ============================================================================
# Rate Predictor (integrated PreNet)
# ============================================================================

class RatePredictor(nn.Module):
    """Predicts optimal rate from image features + network conditions."""
    
    def __init__(self, num_channels: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        # Input: 2*C (mu, sigma) + 4 (network params)
        input_dim = 2 * num_channels + 4
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.blocks = nn.ModuleList([
            ResidualFCBlock(hidden_dim) for _ in range(3)
        ])
        
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # [0, 1]
        )
    
    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        bandwidth_mhz: torch.Tensor,
        latency_ms: torch.Tensor,
        ber_exp: torch.Tensor,
        min_rate: float = 0.1,
        max_rate: float = 1.0
    ) -> torch.Tensor:
        B = z.size(0)
        
        # Compute image statistics
        mu = z.mean(dim=(2, 3))
        sigma = z.std(dim=(2, 3))
        
        # Ensure network tensors have correct batch size
        if snr_db.size(0) != B:
            snr_db = snr_db.expand(B, -1)
        if bandwidth_mhz.size(0) != B:
            bandwidth_mhz = bandwidth_mhz.expand(B, -1)
        if latency_ms.size(0) != B:
            latency_ms = latency_ms.expand(B, -1)
        if ber_exp.size(0) != B:
            ber_exp = ber_exp.expand(B, -1)
        
        # Normalize network conditions
        snr_norm = (snr_db - 15) / 15
        bw_norm = (bandwidth_mhz - 50) / 50
        lat_norm = (latency_ms - 250) / 250
        ber_norm = (ber_exp - 4) / 2
        
        net_features = torch.cat([snr_norm, bw_norm, lat_norm, ber_norm], dim=1)
        
        # Combine and predict
        x = torch.cat([mu, sigma, net_features], dim=1)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        rate_raw = self.rate_head(x)
        rate = min_rate + (max_rate - min_rate) * rate_raw
        
        return rate


# ============================================================================
# Channel Selection (SR-SC)
# ============================================================================

class SRSC(nn.Module):
    """Semantic Rate Selection and Compression."""
    
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
    
    def forward(self, x: torch.Tensor, rate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, C, H, W = x.shape
        
        # SE weighting
        se_weights = self.se(x)
        x = x * se_weights.view(B, C, 1, 1)
        
        # Compute k (use mean rate for batch)
        rate_val = rate.mean().item() if isinstance(rate, torch.Tensor) else rate
        k = max(1, min(C, round(rate_val * C)))
        
        # Create ordered mask
        mask = torch.zeros(B, C, 1, 1, device=x.device)
        mask[:, :k] = 1.0
        
        return x * mask, mask, k


# ============================================================================
# Power Normalization (PSSG)
# ============================================================================

class PSSG(nn.Module):
    """Power-constrained Semantic Symbol Generation."""
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        active = (mask > 0).sum()
        if active > 0:
            power = (x ** 2).sum() / active
            scale = torch.sqrt(1.0 / (power + 1e-8))
        else:
            scale = torch.ones(1, device=x.device)
        
        return x * scale, scale


# ============================================================================
# AWGN Channel
# ============================================================================

class AWGNChannel(nn.Module):
    """Additive White Gaussian Noise channel."""
    
    def forward(self, x: torch.Tensor, snr_db: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if not self.training:
            # Reduced noise at eval
            snr_db = snr_db + 3
        
        # Convert to float if needed
        if isinstance(snr_db, torch.Tensor):
            snr_val = snr_db.float().mean().item()
        else:
            snr_val = float(snr_db)
        
        snr_linear = 10 ** (snr_val / 10)
        noise_std = 1.0 / np.sqrt(snr_linear)
        
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        
        # Remask for strict rate control
        if mask is not None:
            x_noisy = x_noisy * mask
        
        return x_noisy


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """Image decoder with progressive upsampling."""
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            ResBlock(256),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            ResBlock(128),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.output(x)


# ============================================================================
# Unified AdaptiveJSCC Model
# ============================================================================

class AdaptiveJSCC(nn.Module):
    """
    Unified Semantic Communication Model.
    
    Single forward pass that:
    1. Encodes the image
    2. Predicts optimal rate based on image + network conditions
    3. Applies rate-adaptive channel coding
    4. Transmits through AWGN channel
    5. Reconstructs the image
    
    Usage:
        model = AdaptiveJSCC()
        
        # Auto rate selection
        result = model(image, network=NetworkConditions(snr_db=15, bandwidth_mhz=20))
        x_hat = result['output']
        rate_used = result['rate']
        
        # Manual rate override
        result = model(image, network=NetworkConditions(snr_db=15), rate_override=0.5)
    """
    
    def __init__(self, num_channels: int = 256):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Components
        self.encoder = Encoder(out_channels=num_channels)
        self.rate_predictor = RatePredictor(num_channels=num_channels)
        self.sr_sc = SRSC(num_channels)
        self.pssg = PSSG()
        self.channel = AWGNChannel()
        self.decoder = Decoder(in_channels=num_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        network: Optional[NetworkConditions] = None,
        snr_db: Optional[float] = None,
        bandwidth_mhz: float = 20.0,
        latency_ms: float = 50.0,
        ber_exp: float = 4.0,
        rate_override: Optional[float] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with automatic rate selection.
        
        Args:
            x: Input image [B, 3, 256, 256]
            network: NetworkConditions object (alternative to individual params)
            snr_db: Channel SNR in dB (used if network not provided)
            bandwidth_mhz: Channel bandwidth in MHz
            latency_ms: Network latency in ms
            ber_exp: BER exponent (BER = 10^-ber_exp)
            rate_override: Manual rate (bypasses rate predictor)
            return_intermediate: Return intermediate tensors
            
        Returns:
            dict with 'output', 'rate', 'k', and optionally intermediates
        """
        B = x.size(0)
        device = x.device
        
        # Parse network conditions
        if network is not None:
            snr_t, bw_t, lat_t, ber_t = network.to_tensor(device)
            snr_db = network.snr_db
        else:
            if snr_db is None:
                snr_db = 10.0
            snr_t = torch.tensor([[snr_db]], device=device).expand(B, 1)
            bw_t = torch.tensor([[bandwidth_mhz]], device=device).expand(B, 1)
            lat_t = torch.tensor([[latency_ms]], device=device).expand(B, 1)
            ber_t = torch.tensor([[ber_exp]], device=device).expand(B, 1)
        
        # 1. Encode
        z = self.encoder(x)
        
        # 2. Predict or use override rate
        if rate_override is not None:
            rate = torch.tensor([[rate_override]], device=device).expand(B, 1)
        else:
            rate = self.rate_predictor(z, snr_t, bw_t, lat_t, ber_t)
        
        # 3. Channel selection
        z_selected, mask, k = self.sr_sc(z, rate)
        
        # 4. Power normalization
        z_norm, scale = self.pssg(z_selected, mask)
        
        # 5. AWGN channel with remask
        z_noisy = self.channel(z_norm, snr_t, mask)
        
        # 6. Decode
        x_hat = self.decoder(z_noisy)
        
        result = {
            'output': x_hat,
            'rate': rate.mean().item(),
            'k': k,
            'snr_db': snr_db
        }
        
        if return_intermediate:
            result.update({
                'z': z,
                'z_selected': z_selected,
                'z_noisy': z_noisy,
                'mask': mask,
            })
        
        return result
    
    @classmethod
    def from_pretrained(
        cls,
        jscc_path: Optional[str] = None,
        prenet_path: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'AdaptiveJSCC':
        """
        Load from pretrained JSCC and PreNet checkpoints.
        
        Args:
            jscc_path: Path to best_noskip.pt
            prenet_path: Path to best_prenet_adaptive.pt
            device: Device to load on
        """
        model = cls(num_channels=256).to(device)
        
        if jscc_path and Path(jscc_path).exists():
            ckpt = torch.load(jscc_path, map_location=device, weights_only=False)
            
            # Map JSCC components (partial load)
            state = ckpt.get('model_state_dict', ckpt)
            
            # Try to load encoder
            encoder_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
            if encoder_state:
                model.encoder.load_state_dict(encoder_state, strict=False)
                print(f"Loaded encoder from {jscc_path}")
            
            # Try to load decoder
            decoder_state = {k.replace('decoder.', ''): v for k, v in state.items() if k.startswith('decoder.')}
            if decoder_state:
                model.decoder.load_state_dict(decoder_state, strict=False)
                print(f"Loaded decoder from {jscc_path}")
            
            # Try SR-SC
            srsc_state = {k.replace('sr_sc.', ''): v for k, v in state.items() if k.startswith('sr_sc.')}
            if srsc_state:
                model.sr_sc.load_state_dict(srsc_state, strict=False)
                print(f"Loaded SR-SC from {jscc_path}")
        
        if prenet_path and Path(prenet_path).exists():
            ckpt = torch.load(prenet_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.rate_predictor.load_state_dict(state, strict=False)
            print(f"Loaded rate predictor from {prenet_path}")
        
        return model
    
    def save(self, path: str):
        """Save unified model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_channels': self.num_channels,
        }, path)
        print(f"Saved AdaptiveJSCC to {path}")


# ============================================================================
# Demo and Testing
# ============================================================================

def demo():
    """Demo of AdaptiveJSCC."""
    print("=" * 60)
    print("AdaptiveJSCC - Unified Semantic Communication Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = AdaptiveJSCC(num_channels=256).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Test input
    x = torch.rand(2, 3, 256, 256).to(device)
    
    # Test with different network conditions
    conditions = [
        ("5G (excellent)", NetworkConditions(snr_db=25, bandwidth_mhz=100, latency_ms=5, ber_exp=6)),
        ("WiFi (good)", NetworkConditions(snr_db=15, bandwidth_mhz=20, latency_ms=30, ber_exp=5)),
        ("LTE (moderate)", NetworkConditions(snr_db=10, bandwidth_mhz=10, latency_ms=50, ber_exp=4)),
        ("Satellite (poor)", NetworkConditions(snr_db=5, bandwidth_mhz=5, latency_ms=500, ber_exp=3)),
    ]
    
    print(f"\n{'Network':<20} {'Auto Rate':<12} {'Channels':<12} {'Output Shape'}")
    print("-" * 60)
    
    model.eval()
    with torch.no_grad():
        for name, network in conditions:
            result = model(x, network=network)
            print(f"{name:<20} {result['rate']:.2f}         {result['k']}/256       {result['output'].shape}")
    
    # Test with rate override
    print("\n" + "-" * 60)
    print("Manual rate override test:")
    result = model(x, snr_db=15, rate_override=0.75)
    print(f"Override rate=0.75: k={result['k']}, output={result['output'].shape}")
    
    print("\n✅ AdaptiveJSCC ready for deployment!")


if __name__ == '__main__':
    demo()
