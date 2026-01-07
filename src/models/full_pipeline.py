"""
Full Semantic Communication Pipeline.

Combines all modules into a complete transmitter-channel-receiver system.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import SemanticEncoder
from .sr_sc import SRSC
from .acc import ACC
from .pssg import PSSG
from .decoder_cls import DecoderCls
from .decoder_recon import DecoderRecon
from ..channel.awgn import AWGNChannel


class SemanticCommSystem(nn.Module):
    """
    Complete Semantic Communication System.
    
    Pipeline: Encoder -> SR-SC -> ACC -> PSSG -> Channel -> Decoder
    """
    
    def __init__(
        self,
        task: str = 'classification',  # 'classification' or 'reconstruction'
        num_classes: int = 10,
        image_size: int = 256,
        smax: int = 256,
        snr_db: float = 10.0,
        pretrained_encoder: bool = True
    ):
        """
        Args:
            task: Task type
            num_classes: Number of classes (for classification)
            image_size: Input image size
            smax: Maximum latent dimension
            snr_db: Default channel SNR
            pretrained_encoder: Use pretrained encoder
        """
        super().__init__()
        
        self.task = task
        self.image_size = image_size
        self.smax = smax
        
        # Encoder
        self.encoder = SemanticEncoder(pretrained=pretrained_encoder)
        encoder_out_channels = 512  # ResNet-18
        encoder_spatial = image_size // 32  # 256 -> 8
        
        # SR-SC: Semantic importance selection
        self.sr_sc = SRSC(num_channels=encoder_out_channels)
        
        # ACC: Adaptive channel conditioning
        self.acc = ACC(num_channels=encoder_out_channels)
        
        # PSSG: Variable rate compression
        input_dim = encoder_out_channels * encoder_spatial * encoder_spatial
        self.pssg = PSSG(input_dim=input_dim, smax=smax)
        
        # Channel
        self.channel = AWGNChannel(snr_db=snr_db)
        
        # Decoder
        if task == 'classification':
            self.decoder = DecoderCls(smax=smax, num_classes=num_classes)
        else:
            self.decoder = DecoderRecon(smax=smax, image_size=image_size)
        
        # Store config
        self.encoder_out_channels = encoder_out_channels
        self.encoder_spatial = encoder_spatial
    
    def forward(
        self,
        x: torch.Tensor,
        snr_db: Optional[torch.Tensor] = None,
        rate: float = 1.0,
        attention_map: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete system.
        
        Args:
            x: Input images [B, 3, H, W]
            snr_db: Channel SNR in dB [B,] or scalar
            rate: Transmission rate in (0, 1]
            attention_map: Optional XAI attention map for SR-SC
            return_intermediates: If True, return intermediate tensors
            
        Returns:
            Dictionary with outputs:
                - 'output': Final output (logits or reconstructed image)
                - 'z': Transmitted latent
                - 'z_noisy': Received latent (after channel)
                - 'features': Encoder features (if return_intermediates)
        """
        B = x.shape[0]
        device = x.device
        
        # Default SNR if not provided
        if snr_db is None:
            snr_db = torch.full((B,), self.channel.default_snr_db, device=device)
        elif isinstance(snr_db, (int, float)):
            snr_db = torch.full((B,), snr_db, device=device)
        
        # Encode
        features = self.encoder(x)  # [B, C, Hf, Wf]
        
        # SR-SC: importance-based selection
        xa, importance = self.sr_sc(features, keep_ratio=rate, attention_map=attention_map)
        
        # ACC: adapt to channel conditions
        x_prime = self.acc(xa, snr_db)
        
        # PSSG: variable rate compression
        z, _ = self.pssg(x_prime, rate=rate)  # [B, Smax]
        
        # Channel: add noise
        z_noisy = self.channel(z, snr_db=snr_db)
        
        # Decode
        output = self.decoder(z_noisy)
        
        result = {
            'output': output,
            'z': z,
            'z_noisy': z_noisy,
            'importance': importance
        }
        
        if return_intermediates:
            result['features'] = features
            result['xa'] = xa
            result['x_prime'] = x_prime
        
        return result
    
    def encode(
        self,
        x: torch.Tensor,
        snr_db: Optional[torch.Tensor] = None,
        rate: float = 1.0,
        attention_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode input to latent (transmitter side).
        """
        B = x.shape[0]
        device = x.device
        
        if snr_db is None:
            snr_db = torch.full((B,), self.channel.default_snr_db, device=device)
        
        features = self.encoder(x)
        xa, _ = self.sr_sc(features, keep_ratio=rate, attention_map=attention_map)
        x_prime = self.acc(xa, snr_db)
        z, _ = self.pssg(x_prime, rate=rate)
        
        return z
    
    def transmit(
        self,
        z: torch.Tensor,
        snr_db: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transmit through channel.
        """
        return self.channel(z, snr_db=snr_db)
    
    def decode(self, z_noisy: torch.Tensor) -> torch.Tensor:
        """
        Decode received latent (receiver side).
        """
        return self.decoder(z_noisy)


class SemanticCommSystemDual(nn.Module):
    """
    Semantic Communication System supporting both tasks.
    Shares encoder but has separate decoders.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 256,
        smax: int = 256,
        snr_db: float = 10.0
    ):
        super().__init__()
        
        self.encoder = SemanticEncoder(pretrained=True)
        self.sr_sc = SRSC(num_channels=512)
        self.acc = ACC(num_channels=512)
        
        input_dim = 512 * (image_size // 32) ** 2
        self.pssg = PSSG(input_dim=input_dim, smax=smax)
        self.channel = AWGNChannel(snr_db=snr_db)
        
        self.decoder_cls = DecoderCls(smax=smax, num_classes=num_classes)
        self.decoder_recon = DecoderRecon(smax=smax, image_size=image_size)
    
    def forward(
        self,
        x: torch.Tensor,
        task: str,
        snr_db: Optional[torch.Tensor] = None,
        rate: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task selection.
        """
        B = x.shape[0]
        device = x.device
        
        if snr_db is None:
            snr_db = torch.full((B,), self.channel.default_snr_db, device=device)
        
        features = self.encoder(x)
        xa, importance = self.sr_sc(features, keep_ratio=rate)
        x_prime = self.acc(xa, snr_db)
        z, _ = self.pssg(x_prime, rate=rate)
        z_noisy = self.channel(z, snr_db=snr_db)
        
        if task == 'classification':
            output = self.decoder_cls(z_noisy)
        else:
            output = self.decoder_recon(z_noisy)
        
        return {'output': output, 'z': z, 'z_noisy': z_noisy}


if __name__ == "__main__":
    # Test full pipeline
    print("Testing Classification Pipeline...")
    model_cls = SemanticCommSystem(task='classification', num_classes=10, pretrained_encoder=False)
    
    x = torch.randn(2, 3, 256, 256)
    snr = torch.tensor([10.0, 5.0])
    
    result = model_cls(x, snr_db=snr, rate=0.5)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {result['output'].shape}")
    print(f"Latent shape: {result['z'].shape}")
    
    print("\nTesting Reconstruction Pipeline...")
    model_recon = SemanticCommSystem(task='reconstruction', pretrained_encoder=False)
    
    result = model_recon(x, snr_db=snr, rate=0.5)
    print(f"Reconstructed shape: {result['output'].shape}")
