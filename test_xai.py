"""Quick test for XAI pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
print("Testing XAI-guided Semantic Communication Pipeline...")

try:
    from src.models.xai_pipeline import XAIGuidedSemanticComm, OCRInstanceLevel
    print("  [OK] XAI Pipeline imported")
    
    from src.models.prenet import PreNet
    print("  [OK] Pre-Net imported")
    
    # Test XAI pipeline
    print("\nCreating XAI-guided model...")
    model = XAIGuidedSemanticComm(
        task='reconstruction',
        pretrained_encoder=False,
        use_xai=True
    )
    model.eval()
    print("  [OK] XAI model created")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    snr = torch.tensor([10.0, 5.0])
    
    result = model(x, snr_db=snr, rate=0.5, use_xai=False)  # XAI off for training mode
    
    print(f"\n  Input shape: {x.shape}")
    print(f"  Output shape: {result['output'].shape}")
    print(f"  Importance shape: {result['importance'].shape}")
    
    # Test Pre-Net prediction
    pred = model.predict_quality(x, snr)
    print(f"  Pre-Net prediction shape: {pred.shape}")
    
    # Test rate selector
    print("\nTesting instance-level rate selector...")
    ocr = OCRInstanceLevel(model, threshold=0.5)
    selected_rate, pred = ocr.select_rate(x[:1], snr[:1])
    print(f"  Selected rate: {selected_rate}")
    
    print("\n✓ All XAI tests PASSED!")
    
except Exception as e:
    print(f"\n✗ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
