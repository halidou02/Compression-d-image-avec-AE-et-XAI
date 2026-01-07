"""
XAI Semantic Communication Pipeline (Option A - Corrected).

Key fixes:
1. Ordered channels (deterministic mask)
2. Mask passed through pipeline for AWGN cleaning
3. Power norm on active channels only
4. Decoder conditioned on rate + SNR
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class XAIGuidedSemanticComm(nn.Module):
    """
    Corrected Semantic Communication with Option A fixes.
    """
    
    def __init__(
        self,
        task: str = 'reconstruction',
        image_size: int = 256,
        num_channels: int = 256,
        snr_db: float = 10.0,
        pretrained_encoder: bool = True
    ):
        super().__init__()
        
        from .encoder import SemanticEncoder
        from .sr_sc import SRSC
        from .pssg import PSSG
        from .decoder_recon import DecoderRecon
        from ..channel.awgn import AWGNChannel
        
        self.num_channels = num_channels
        self.default_snr = snr_db
        
        # Encoder: [B, 3, 256, 256] -> [B, 256, 16, 16]
        self.encoder = SemanticEncoder(
            pretrained=pretrained_encoder,
            out_channels=num_channels
        )
        
        # SR-SC: ordered channel masking + SE weighting
        self.sr_sc = SRSC(num_channels=num_channels)
        
        # PSSG: power norm on active channels
        self.pssg = PSSG(num_channels=num_channels, normalize_power=True)
        
        # AWGN with mask cleaning
        self.channel = AWGNChannel(snr_db=snr_db)
        
        # Decoder with rate+SNR conditioning
        self.decoder = DecoderRecon(in_channels=num_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        snr_db: Optional[Union[float, torch.Tensor]] = None,
        rate: float = 1.0,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with mask propagation.
        """
        B = x.shape[0]
        device = x.device
        
        # Handle SNR
        if snr_db is None:
            snr_db = self.default_snr
        if isinstance(snr_db, (int, float)):
            snr_tensor = torch.full((B,), snr_db, device=device)
        else:
            snr_tensor = snr_db
        
        # 1. Encode
        features = self.encoder(x)  # [B, 256, 16, 16]
        
        # 2. SR-SC: ordered masking + SE weighting, returns mask
        xa, weights, mask = self.sr_sc(features, keep_ratio=rate)
        
        # 3. PSSG: power norm on active channels only
        z = self.pssg(xa, mask=mask, rate=rate)
        
        # 4. Channel: AWGN + mask cleaning
        z_noisy = self.channel(z, snr_db=snr_tensor, mask=mask)
        
        # 5. Decode with rate+SNR conditioning
        x_hat = self.decoder(z_noisy, rate=rate, snr_db=snr_tensor)
        
        result = {
            'output': x_hat,
            'weights': weights,
            'mask': mask
        }
        
        if return_intermediate:
            result['features'] = features
            result['xa'] = xa
            result['z'] = z
            result['z_noisy'] = z_noisy
        
        return result
    
    def get_bandwidth_info(self, rate: float = 1.0) -> dict:
        """Get bandwidth info."""
        input_size = 3 * 256 * 256
        k = int(rate * self.num_channels)
        active_symbols = k * 16 * 16
        
        return {
            'input_values': input_size,
            'active_symbols': active_symbols,
            'total_symbols': self.num_channels * 16 * 16,
            'active_fraction': rate,
            'active_channels': k
        }


if __name__ == "__main__":
    print("Testing Corrected Pipeline...")
    
    model = XAIGuidedSemanticComm(pretrained_encoder=False)
    x = torch.randn(2, 3, 256, 256)
    
    result = model(x, snr_db=10.0, rate=0.5, return_intermediate=True)
    
    print(f"Input: {x.shape}")
    print(f"Output: {result['output'].shape}")
    print(f"Mask sum: {result['mask'].sum().item() / 2} active channels")
    
    # Verify mask cleaning worked
    z_noisy = result['z_noisy']
    inactive = z_noisy[:, 128:, :, :]
    print(f"Inactive channels max: {inactive.abs().max():.6f} (should be 0)")
    
    info = model.get_bandwidth_info(rate=0.5)
    print(f"Bandwidth: {info}")
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print("\nSUCCESS!")
