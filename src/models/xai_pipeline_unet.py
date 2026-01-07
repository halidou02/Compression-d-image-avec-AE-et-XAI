"""
XAI-Guided Semantic Communication Pipeline with U-Net Decoder.

Architecture:
- Encoder: ResNet-50 with multi-scale outputs (z0, s1, s2)
- SR-SC: Ordered channel selection
- PSSG: Per-sample power normalization
- AWGN: Channel with mask cleaning
- Decoder: Progressive U-Net with skips and refinement groups
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoder import SemanticEncoderUNet
from .decoder_unet import ProgressiveUNetDecoder
from .sr_sc import SRSC
from .pssg import PSSG
from ..channel.awgn import AWGNChannel


class XAISemanticCommUNet(nn.Module):
    """
    Complete JSCC pipeline with U-Net decoder.
    
    Key features:
    - Skip connections from encoder to decoder
    - Progressive refinement (rate=1.0 uses all channel groups)
    - GroupNorm for stability
    """
    
    def __init__(
        self,
        num_channels: int = 256,
        pretrained_encoder: bool = True
    ):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Encoder with multi-scale outputs
        self.encoder = SemanticEncoderUNet(
            pretrained=pretrained_encoder,
            out_channels=num_channels,
            freeze_bn=True
        )
        
        # Semantic Rate Selection and Compression
        self.sr_sc = SRSC(num_channels=num_channels)
        
        # Power normalization
        self.pssg = PSSG(normalize_power=True)
        
        # Channel
        self.channel = AWGNChannel()
        
        # Progressive U-Net decoder
        self.decoder = ProgressiveUNetDecoder(cond_dim=64)
    
    def forward(
        self,
        x: torch.Tensor,
        snr_db: float = 10.0,
        rate: float = 0.5,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, 256, 256]
            snr_db: Channel SNR in dB
            rate: Compression rate (0.1-1.0)
            return_intermediate: Return intermediate features
        
        Returns:
            Dict with 'output' and optionally intermediate features
        """
        B = x.size(0)
        device = x.device
        
        # === ENCODER (multi-scale) ===
        enc_out = self.encoder(x)
        z0 = enc_out['z0']  # [B, 256, 16, 16]
        s1 = enc_out['s1']  # [B, 256, 64, 64] - skip
        s2 = enc_out['s2']  # [B, 512, 32, 32] - skip
        
        # === SR-SC: Ordered channel mask ===
        xa, weights, mask = self.sr_sc(z0, keep_ratio=rate)
        
        # === PSSG: Power normalization ===
        z = self.pssg(xa, mask, rate=rate)
        
        # === AWGN Channel ===
        snr_tensor = torch.full((B,), snr_db, device=device)
        z_noisy = self.channel(z, snr_tensor, mask=mask)
        
        # === DECODER (Progressive U-Net) ===
        x_hat = self.decoder(z_noisy, s1, s2, rate=rate, snr_db=snr_tensor)
        
        result = {'output': x_hat}
        
        if return_intermediate:
            result.update({
                'features': z0,
                'xa': xa,
                'z': z,
                'z_noisy': z_noisy,
                'mask': mask,
                'weights': weights,
                's1': s1,
                's2': s2
            })
        
        return result


if __name__ == "__main__":
    # Test pipeline
    model = XAISemanticCommUNet(pretrained_encoder=False)
    
    x = torch.randn(2, 3, 256, 256)
    
    # Test different rates
    for rate in [0.25, 0.5, 1.0]:
        result = model(x, snr_db=10.0, rate=rate)
        print(f"rate={rate}: output {result['output'].shape}")
    
    # Test with intermediates
    result = model(x, snr_db=10.0, rate=0.5, return_intermediate=True)
    print(f"\nIntermediate keys: {list(result.keys())}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
