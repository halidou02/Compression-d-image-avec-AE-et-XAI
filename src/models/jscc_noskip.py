"""
JSCC Pipeline without Skip Leak.

All information must pass through the channel.
Decoder only receives z_noisy - no skip connections from encoder.
"""
import torch
import torch.nn as nn
from typing import Dict

from .encoder import SemanticEncoderUNet
from .decoder_noskip import SelfContainedDecoder
from .sr_sc import SRSC
from .pssg import PSSG
from ..channel.awgn import AWGNChannel


class JSCCNoSkip(nn.Module):
    """
    Physically correct JSCC: all info through channel.
    
    Key difference from XAISemanticCommUNet:
    - Decoder does NOT receive s1/s2 from encoder
    - All info must pass through SR-SC → PSSG → AWGN
    """
    
    def __init__(
        self,
        num_channels: int = 256,
        pretrained_encoder: bool = True
    ):
        super().__init__()
        
        self.num_channels = num_channels
        
        # Encoder (only z0 is transmitted, s1/s2 for Grad-CAM only)
        self.encoder = SemanticEncoderUNet(
            pretrained=pretrained_encoder,
            out_channels=num_channels,
            freeze_bn=True
        )
        
        self.sr_sc = SRSC(num_channels=num_channels)
        self.pssg = PSSG(normalize_power=True)
        self.channel = AWGNChannel()
        
        # Decoder without skip connections
        self.decoder = SelfContainedDecoder(in_channels=num_channels, cond_dim=64)
    
    def forward(
        self,
        x: torch.Tensor,
        snr_db: float = 10.0,
        rate: float = 0.5,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        device = x.device
        
        # Encoder
        enc_out = self.encoder(x)
        z0 = enc_out['z0']  # Only this goes through channel
        
        # SR-SC
        xa, weights, mask = self.sr_sc(z0, keep_ratio=rate)
        
        # PSSG
        z = self.pssg(xa, mask, rate=rate)
        
        # AWGN Channel
        snr_tensor = torch.full((B,), snr_db, device=device)
        z_noisy = self.channel(z, snr_tensor, mask=mask)
        
        # Decoder (ONLY z_noisy - no skips!)
        x_hat = self.decoder(z_noisy, rate=rate, snr_db=snr_tensor)
        
        result = {'output': x_hat}
        
        if return_intermediate:
            result.update({
                'features': z0,
                'xa': xa,
                'z': z,
                'z_noisy': z_noisy,
                'mask': mask,
                'weights': weights,
                's1': enc_out['s1'],  # For Grad-CAM only
                's2': enc_out['s2'],
            })
        
        return result


if __name__ == "__main__":
    model = JSCCNoSkip(pretrained_encoder=False)
    x = torch.randn(2, 3, 256, 256)
    
    for rate in [0.25, 0.5, 1.0]:
        result = model(x, snr_db=10.0, rate=rate)
        print(f"rate={rate}: output {result['output'].shape}")
    
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,}")
