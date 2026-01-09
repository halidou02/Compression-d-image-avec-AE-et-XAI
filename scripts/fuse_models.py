"""
Fusion Script: Merge JSCCNoSkip + PreNetAdaptive into a Unified Model.

This script:
1. Loads the trained JSCCNoSkip (encoder/decoder)
2. Loads the trained PreNetAdaptive (rate predictor)
3. Creates a unified model with both integrated
4. Saves the combined checkpoint
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

# Import the trained models
from src.models.jscc_noskip import JSCCNoSkip
from src.models.prenet_adaptive import PreNetAdaptive
from src.utils.metrics import compute_psnr, compute_ssim


@dataclass
class NetworkConditions:
    """Network channel characteristics."""
    snr_db: float = 15.0
    bandwidth_mhz: float = 20.0
    latency_ms: float = 50.0
    ber_exp: float = 4.0


class UnifiedJSCC(nn.Module):
    """
    Unified JSCC Model combining:
    - JSCCNoSkip (trained encoder/decoder)
    - PreNetAdaptive (trained rate predictor)
    
    Single forward pass: Image + Network Conditions → Reconstructed Image
    """
    
    def __init__(self, num_channels: int = 256):
        super().__init__()
        self.num_channels = num_channels
        
        # JSCC components (will load pretrained weights)
        self.jscc = JSCCNoSkip(num_channels=num_channels, pretrained_encoder=False)
        
        # Rate predictor (will load pretrained weights)
        self.rate_predictor = PreNetAdaptive(
            num_channels=num_channels,
            predict_quality=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        network: Optional[NetworkConditions] = None,
        snr_db: Optional[float] = None,
        bandwidth_mhz: float = 20.0,
        latency_ms: float = 50.0,
        ber_exp: float = 4.0,
        rate_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with automatic rate selection.
        
        Args:
            x: Input image [B, 3, 256, 256]
            network: NetworkConditions (alternative to individual params)
            snr_db: SNR in dB
            rate_override: Manual rate (bypasses PreNet)
        """
        B = x.size(0)
        device = x.device
        
        # Parse network conditions
        if network is not None:
            snr_db = network.snr_db
            bandwidth_mhz = network.bandwidth_mhz
            latency_ms = network.latency_ms
            ber_exp = network.ber_exp
        elif snr_db is None:
            snr_db = 10.0
        
        # Create tensors for PreNet
        snr_t = torch.tensor([[snr_db]], device=device).expand(B, 1)
        bw_t = torch.tensor([[bandwidth_mhz]], device=device).expand(B, 1)
        lat_t = torch.tensor([[latency_ms]], device=device).expand(B, 1)
        ber_t = torch.tensor([[ber_exp]], device=device).expand(B, 1)
        
        # 1. Get encoder features
        enc_out = self.jscc.encoder(x)
        z0 = enc_out['z0']
        
        # 2. Predict or override rate
        if rate_override is not None:
            rate = rate_override
        else:
            prenet_result = self.rate_predictor(z0, snr_t, bw_t, lat_t, ber_t)
            rate = prenet_result['rate'].mean().item()
        
        # 3. Run full JSCC with predicted rate
        result = self.jscc(x, snr_db=snr_db, rate=rate)
        
        return {
            'output': result['output'],
            'rate': rate,
            'k': result.get('k', int(rate * self.num_channels)),
            'snr_db': snr_db
        }
    
    @classmethod
    def from_pretrained(
        cls,
        jscc_path: str,
        prenet_path: str,
        device: str = 'cpu'
    ) -> 'UnifiedJSCC':
        """Load from pretrained checkpoints."""
        model = cls(num_channels=256).to(device)
        
        # Load JSCC
        if Path(jscc_path).exists():
            ckpt = torch.load(jscc_path, map_location=device, weights_only=False)
            model.jscc.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Loaded JSCC from {jscc_path}")
            psnr = ckpt.get('grid_psnr', ckpt.get('psnr', 0))
            print(f"   PSNR: {psnr:.2f} dB")
        else:
            print(f"⚠️ JSCC not found: {jscc_path}")
        
        # Load PreNet
        if Path(prenet_path).exists():
            ckpt = torch.load(prenet_path, map_location=device, weights_only=False)
            model.rate_predictor.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Loaded PreNet from {prenet_path}")
            mae = ckpt.get('mae', 0)
            print(f"   MAE: {mae:.4f}")
        else:
            print(f"⚠️ PreNet not found: {prenet_path}")
        
        model.eval()
        return model
    
    def save(self, path: str):
        """Save unified model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'jscc_state_dict': self.jscc.state_dict(),
            'prenet_state_dict': self.rate_predictor.state_dict(),
            'num_channels': self.num_channels,
        }, path)
        print(f"✅ Saved UnifiedJSCC to {path}")


def main():
    """Fuse JSCC + PreNet into unified model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuse JSCC and PreNet')
    parser.add_argument('--jscc', type=str, default='checkpoints/best_noskip.pt')
    parser.add_argument('--prenet', type=str, default='results/prenet_adaptive/best_prenet_adaptive.pt')
    parser.add_argument('--output', type=str, default='checkpoints/unified_jscc.pt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create and load model
    print("\n" + "=" * 60)
    print("Loading pretrained models...")
    print("=" * 60)
    
    model = UnifiedJSCC.from_pretrained(
        jscc_path=args.jscc,
        prenet_path=args.prenet,
        device=device
    )
    
    # Count parameters
    jscc_params = sum(p.numel() for p in model.jscc.parameters())
    prenet_params = sum(p.numel() for p in model.rate_predictor.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 Model Statistics:")
    print(f"   JSCC params:   {jscc_params:,}")
    print(f"   PreNet params: {prenet_params:,}")
    print(f"   Total params:  {total_params:,}")
    
    # Test with dummy input
    print("\n" + "=" * 60)
    print("Testing unified model...")
    print("=" * 60)
    
    x = torch.rand(1, 3, 256, 256).to(device)
    
    scenarios = [
        ("5G", NetworkConditions(snr_db=25, bandwidth_mhz=100, latency_ms=5, ber_exp=6)),
        ("WiFi", NetworkConditions(snr_db=15, bandwidth_mhz=20, latency_ms=30, ber_exp=5)),
        ("LTE", NetworkConditions(snr_db=10, bandwidth_mhz=10, latency_ms=50, ber_exp=4)),
        ("Satellite", NetworkConditions(snr_db=5, bandwidth_mhz=5, latency_ms=500, ber_exp=3)),
    ]
    
    print(f"\n{'Network':<12} {'Rate':<8} {'Channels':<12}")
    print("-" * 35)
    
    model.eval()
    with torch.no_grad():
        for name, network in scenarios:
            result = model(x, network=network)
            print(f"{name:<12} {result['rate']:.2f}     {result['k']}/256")
    
    # Save
    print("\n" + "=" * 60)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    
    print(f"\n🎉 Fusion complete!")
    print(f"   Unified model saved to: {args.output}")


if __name__ == '__main__':
    main()
