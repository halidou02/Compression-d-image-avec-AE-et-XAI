"""
Spatial-Preserving Semantic Encoder with Multi-Scale Outputs.

Returns:
- z0: [B, 256, 16, 16] - main features for transmission
- s1: [B, 256, 64, 64] - skip for U-Net decoder
- s2: [B, 512, 32, 32] - skip for U-Net decoder
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict


class SemanticEncoderUNet(nn.Module):
    """
    Encoder with multi-scale outputs for U-Net decoder skips.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
        freeze_bn: bool = True
    ):
        super().__init__()
        
        # Load ResNet-50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Stem
        self.stem = nn.Sequential(
            resnet.conv1,      # 256 -> 128
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # 128 -> 64
        )
        self.layer1 = resnet.layer1  # 64 -> 64, 256 ch (skip s1)
        self.layer2 = resnet.layer2  # 64 -> 32, 512 ch (skip s2)
        self.layer3 = resnet.layer3  # 32 -> 16, 1024 ch
        
        # Reduce channels: 1024 -> 256
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.GroupNorm(32, 512),  # GroupNorm instead of BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
        
        # Freeze BatchNorm in backbone for stability with variable rate/noise
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        """Freeze BatchNorm layers in backbone."""
        for module in [self.stem, self.layer1, self.layer2, self.layer3]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
    
    def train(self, mode=True):
        """Override train to keep BN frozen."""
        super().train(mode)
        self._freeze_bn()
        return self
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-scale outputs.
        
        Returns:
            dict with:
                - 'z0': [B, 256, 16, 16] main features
                - 's1': [B, 256, 64, 64] skip 1
                - 's2': [B, 512, 32, 32] skip 2
                - 's3': [B, 1024, 16, 16] for Grad-CAM (optional)
        """
        x = self.stem(x)           # [B, 64, 64, 64]
        s1 = self.layer1(x)        # [B, 256, 64, 64] - skip
        s2 = self.layer2(s1)       # [B, 512, 32, 32] - skip
        s3 = self.layer3(s2)       # [B, 1024, 16, 16] - Grad-CAM
        z0 = self.channel_reduce(s3)  # [B, 256, 16, 16] - transmit
        
        return {
            'z0': z0,
            's1': s1,
            's2': s2,
            's3': s3  # for Grad-CAM hooks
        }


# Keep old class for backward compatibility
class SemanticEncoder(SemanticEncoderUNet):
    """Alias for backward compatibility."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return only z0 for backward compatibility."""
        result = super().forward(x)
        return result['z0']


if __name__ == "__main__":
    encoder = SemanticEncoderUNet(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    result = encoder(x)
    print(f"z0: {result['z0'].shape}")  # [2, 256, 16, 16]
    print(f"s1: {result['s1'].shape}")  # [2, 256, 64, 64]
    print(f"s2: {result['s2'].shape}")  # [2, 512, 32, 32]
    print(f"s3: {result['s3'].shape}")  # [2, 1024, 16, 16]
