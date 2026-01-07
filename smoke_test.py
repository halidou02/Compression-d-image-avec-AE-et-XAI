"""Quick smoke test script."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
print("Testing Semantic Communication System...")

try:
    from src.models.encoder import SemanticEncoder
    print("  [OK] Encoder imported")
    
    from src.models.sr_sc import SRSC
    print("  [OK] SR-SC imported")
    
    from src.models.acc import ACC
    print("  [OK] ACC imported")
    
    from src.models.pssg import PSSG
    print("  [OK] PSSG imported")
    
    from src.models.decoder_cls import DecoderCls
    from src.models.decoder_recon import DecoderRecon
    print("  [OK] Decoders imported")
    
    from src.channel.awgn import AWGNChannel
    print("  [OK] AWGN Channel imported")
    
    from src.models.full_pipeline import SemanticCommSystem
    print("  [OK] Full Pipeline imported")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model = SemanticCommSystem(task='reconstruction', pretrained_encoder=False)
    x = torch.randn(2, 3, 256, 256)
    snr = torch.tensor([10.0, 5.0])
    
    result = model(x, snr_db=snr, rate=0.5)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {result['output'].shape}")
    print(f"  Latent shape: {result['z'].shape}")
    
    assert result['output'].shape == (2, 3, 256, 256), "Output shape mismatch!"
    assert result['z'].shape == (2, 256), "Latent shape mismatch!"
    
    print("\n✓ All tests PASSED!")
    
except Exception as e:
    print(f"\n✗ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
