"""
Smoke test for train_custom_jscc.py
Tests all components to ensure no errors before full training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

# Import all components from train_custom_jscc
from train_custom_jscc import (
    CustomEncoder,
    CustomDecoder,
    CustomJSCC,
    SRSC,
    PSSG,
    AWGNChannel,
    FiLMBlock,
    ResBlock,
    TeacherGradCAM,
    PerceptualLoss,
    compute_psnr,
    compute_ssim,
    compute_msssim,
    get_loss_weights,
)

def test_resblock():
    print("Testing ResBlock...", end=" ")
    block = ResBlock(128)
    x = torch.randn(2, 128, 32, 32)
    y = block(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    print("✓")

def test_encoder():
    print("Testing CustomEncoder (MEDIUM)...", end=" ")
    encoder = CustomEncoder(out_channels=256)
    x = torch.randn(2, 3, 256, 256)
    z = encoder(x)
    assert z.shape == (2, 256, 16, 16), f"Shape mismatch: {z.shape}"
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ ({num_params:,} params)")
    return encoder

def test_decoder():
    print("Testing CustomDecoder (MEDIUM)...", end=" ")
    decoder = CustomDecoder(in_channels=256)
    z = torch.randn(2, 256, 16, 16)
    x = decoder(z, rate=0.5, snr_db=10.0)
    assert x.shape == (2, 3, 256, 256), f"Shape mismatch: {x.shape}"
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ ({num_params:,} params)")
    return decoder

def test_srsc():
    print("Testing SR-SC...", end=" ")
    srsc = SRSC(256)
    z = torch.randn(2, 256, 16, 16)
    z_selected, mask, k = srsc(z, keep_ratio=0.5)
    assert z_selected.shape == z.shape
    assert k == 128
    print(f"✓ (k={k})")

def test_pssg():
    print("Testing PSSG...", end=" ")
    pssg = PSSG()
    z = torch.randn(2, 256, 16, 16)
    mask = torch.ones(2, 256, 1, 1)
    z_norm, scale = pssg(z, mask)
    assert z_norm.shape == z.shape
    print("✓")

def test_awgn():
    print("Testing AWGN Channel...", end=" ")
    channel = AWGNChannel()
    channel.train()
    z = torch.randn(2, 256, 16, 16)
    z_noisy = channel(z, snr_db=10.0)
    assert z_noisy.shape == z.shape
    print("✓")

def test_film():
    print("Testing FiLM Block...", end=" ")
    film = FiLMBlock(256)
    z = torch.randn(2, 256, 16, 16)
    z_cond = film(z, rate=0.5, snr=10.0)
    assert z_cond.shape == z.shape
    print("✓")

def test_full_model():
    print("Testing CustomJSCC (full pipeline)...", end=" ")
    model = CustomJSCC(num_channels=256)
    x = torch.randn(2, 3, 256, 256)
    x_hat = model(x, snr_db=10.0, rate=0.5)
    
    if isinstance(x_hat, dict):
        x_hat = x_hat['output']
    
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape}"
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ ({num_params:,} params)")
    return model

def test_psnr():
    print("Testing compute_psnr...", end=" ")
    x = torch.randn(2, 3, 256, 256)
    x_hat = x + 0.1 * torch.randn_like(x)
    psnr = compute_psnr(x_hat, x)
    assert psnr.shape == (2,)
    assert psnr.mean() > 0
    print(f"✓ (PSNR={psnr.mean():.2f} dB)")

def test_ssim():
    print("Testing compute_ssim...", end=" ")
    x = torch.rand(2, 3, 256, 256)
    x_hat = x + 0.1 * torch.randn_like(x)
    x_hat = x_hat.clamp(0, 1)
    ssim = compute_ssim(x_hat, x)
    assert ssim.shape == (2,)
    assert 0 <= ssim.mean() <= 1
    print(f"✓ (SSIM={ssim.mean():.4f})")

def test_msssim():
    print("Testing compute_msssim...", end=" ")
    x = torch.rand(2, 3, 256, 256)
    x_hat = x + 0.05 * torch.randn_like(x)
    x_hat = x_hat.clamp(0, 1)
    msssim = compute_msssim(x_hat.clone(), x.clone())
    assert msssim.shape == (2,)
    assert 0 <= msssim.mean() <= 1
    print(f"✓ (MS-SSIM={msssim.mean():.4f})")

def test_loss_scheduling():
    print("Testing loss scheduling...", end=" ")
    for epoch in [1, 5, 15, 25, 50, 100]:
        weights = get_loss_weights(epoch, 100)
        assert len(weights) == 4, "Should return 4 weights"
    print("✓")

def test_teacher_cam():
    print("Testing TeacherGradCAM...", end=" ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = TeacherGradCAM(device)
    x = torch.rand(2, 3, 256, 256).to(device)
    cam = teacher.get_cam(x)
    assert cam.shape[0] == 2
    assert cam.shape[1] == 1
    print(f"✓ (CAM shape={cam.shape})")
    return teacher

def test_perceptual_loss():
    print("Testing PerceptualLoss...", end=" ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    perceptual = PerceptualLoss(device)
    x = torch.rand(2, 3, 256, 256).to(device)
    x_hat = x + 0.1 * torch.randn_like(x)
    x_hat = x_hat.clamp(0, 1)
    loss = perceptual(x_hat, x)
    assert loss.ndim == 0  # Scalar
    print(f"✓ (Loss={loss.item():.4f})")
    return perceptual

def test_training_step():
    print("Testing training step (forward + backward)...", end=" ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CustomJSCC(num_channels=256).to(device)
    teacher_cam = TeacherGradCAM(device)
    perceptual_loss = PerceptualLoss(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Fake batch
    x = torch.rand(2, 3, 256, 256).to(device)
    
    optimizer.zero_grad()
    
    # Forward
    z = model.encoder(x)
    z_selected, mask, k = model.sr_sc(z, 0.5)
    z_norm, scale = model.pssg(z_selected, mask)
    z_noisy = model.channel(z_norm, 10.0)
    x_hat = model.decoder(z_noisy, 0.5, 10.0)
    
    # Losses
    mse = F.mse_loss(x_hat, x)
    ssim = compute_ssim(x_hat, x)
    msssim = compute_msssim(x_hat.clone(), x.clone())
    perc = perceptual_loss(x_hat, x)
    cam = teacher_cam.get_cam(x)
    
    # Total loss
    L = mse + 0.15 * (1 - msssim).mean() + 0.02 * perc
    
    # Backward
    L.backward()
    optimizer.step()
    
    print(f"✓ (Loss={L.item():.4f})")

def main():
    print("=" * 60)
    print("SMOKE TEST - train_custom_jscc.py")
    print("=" * 60)
    
    try:
        test_resblock()
        test_encoder()
        test_decoder()
        test_srsc()
        test_pssg()
        test_awgn()
        test_film()
        test_full_model()
        test_psnr()
        test_ssim()
        test_msssim()
        test_loss_scheduling()
        test_teacher_cam()
        test_perceptual_loss()
        test_training_step()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
