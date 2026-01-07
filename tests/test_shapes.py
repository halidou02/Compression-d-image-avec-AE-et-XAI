"""
Shape Tests for Semantic Communication System.

Validates that all modules produce correct output shapes.
"""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoder import SemanticEncoder, LightEncoder
from src.models.sr_sc import SRSC
from src.models.acc import ACC
from src.models.pssg import PSSG
from src.models.prenet import PreNet
from src.models.decoder_cls import DecoderCls
from src.models.decoder_recon import DecoderRecon
from src.channel.awgn import AWGNChannel


class TestEncoder:
    def test_resnet_encoder_shape(self):
        encoder = SemanticEncoder(pretrained=False)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 512, 8, 8), f"Expected (2, 512, 8, 8), got {out.shape}"
    
    def test_light_encoder_shape(self):
        encoder = LightEncoder()
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 512, 8, 8), f"Expected (2, 512, 8, 8), got {out.shape}"


class TestSRSC:
    def test_output_shape(self):
        sr_sc = SRSC(num_channels=512)
        features = torch.randn(2, 512, 8, 8)
        xa, importance = sr_sc(features, keep_ratio=0.5)
        assert xa.shape == features.shape
        assert importance.shape == (2, 512)
    
    def test_masking(self):
        sr_sc = SRSC(num_channels=512)
        features = torch.randn(2, 512, 8, 8)
        xa, _ = sr_sc(features, keep_ratio=0.5)
        # About 50% of channels should be non-zero
        non_zero = (xa.abs().sum(dim=(2, 3)) > 0).sum(dim=1)
        assert all(non_zero == 256), f"Expected 256 non-zero channels, got {non_zero}"


class TestACC:
    def test_output_shape(self):
        acc = ACC(num_channels=512)
        xa = torch.randn(2, 512, 8, 8)
        gamma = torch.tensor([10.0, 5.0])
        x_prime = acc(xa, gamma)
        assert x_prime.shape == xa.shape


class TestPSSG:
    def test_output_shape(self):
        pssg = PSSG(input_dim=512*8*8, smax=256)
        x_prime = torch.randn(2, 512, 8, 8)
        z, _ = pssg(x_prime, rate=1.0)
        assert z.shape == (2, 256)
    
    def test_rate_masking(self):
        pssg = PSSG(input_dim=512*8*8, smax=256)
        x_prime = torch.randn(2, 512, 8, 8)
        
        for rate in [0.1, 0.5, 1.0]:
            z, _ = pssg(x_prime, rate=rate)
            expected_nonzero = int(rate * 256)
            actual_nonzero = (z.abs() > 1e-8).sum(dim=1)
            assert all(actual_nonzero >= expected_nonzero - 1)


class TestChannel:
    def test_output_shape(self):
        channel = AWGNChannel(snr_db=10.0)
        z = torch.randn(2, 256)
        z_prime = channel(z)
        assert z_prime.shape == z.shape
    
    def test_noise_addition(self):
        channel = AWGNChannel(snr_db=10.0)
        z = torch.randn(2, 256)
        z_prime = channel(z)
        # Output should be different from input
        assert not torch.allclose(z_prime, z)


class TestDecoders:
    def test_cls_decoder_shape(self):
        decoder = DecoderCls(smax=256, num_classes=10)
        z = torch.randn(2, 256)
        logits = decoder(z)
        assert logits.shape == (2, 10)
    
    def test_recon_decoder_shape(self):
        decoder = DecoderRecon(smax=256, image_size=256)
        z = torch.randn(2, 256)
        x_hat = decoder(z)
        assert x_hat.shape == (2, 3, 256, 256)
    
    def test_recon_output_range(self):
        decoder = DecoderRecon(smax=256, image_size=256)
        z = torch.randn(2, 256)
        x_hat = decoder(z)
        assert x_hat.min() >= 0 and x_hat.max() <= 1


class TestPreNet:
    def test_output_shape_cls(self):
        prenet = PreNet(num_channels=512, task='classification')
        x_prime = torch.randn(2, 512, 8, 8)
        gamma = torch.tensor([10.0, 5.0])
        pred = prenet(x_prime, gamma)
        assert pred.shape == (2, 1)
    
    def test_output_shape_recon(self):
        prenet = PreNet(num_channels=512, task='reconstruction')
        x_prime = torch.randn(2, 512, 8, 8)
        gamma = torch.tensor([10.0, 5.0])
        pred = prenet(x_prime, gamma)
        assert pred.shape == (2, 2)  # PSNR, SSIM


def test_full_pipeline():
    """Smoke test for full pipeline."""
    from src.models.full_pipeline import SemanticCommSystem
    
    # Test reconstruction mode
    model = SemanticCommSystem(task='reconstruction', pretrained_encoder=False)
    x = torch.randn(2, 3, 256, 256)
    snr = torch.tensor([10.0, 5.0])
    
    result = model(x, snr_db=snr, rate=0.5)
    
    assert 'output' in result
    assert result['output'].shape == (2, 3, 256, 256)
    assert result['z'].shape == (2, 256)
    
    # Test classification mode
    model_cls = SemanticCommSystem(task='classification', num_classes=10, pretrained_encoder=False)
    result_cls = model_cls(x, snr_db=snr, rate=0.5)
    
    assert result_cls['output'].shape == (2, 10)
    
    print("All shape tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
