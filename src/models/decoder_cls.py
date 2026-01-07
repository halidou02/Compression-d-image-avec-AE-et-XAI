"""
Classification Decoder.

Takes latent vector z and produces class logits.

Input: z [B, Smax]
Output: logits [B, num_classes]
"""
import torch
import torch.nn as nn


class DecoderCls(nn.Module):
    """
    Classification decoder: MLP + Softmax.
    """
    
    def __init__(
        self,
        smax: int = 256,
        num_classes: int = 10,
        hidden_dims: list = None
    ):
        """
        Args:
            smax: Input latent dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        in_dim = smax
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vector [B, Smax]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        return self.classifier(z)


class DecoderClsFromFeatures(nn.Module):
    """
    Classification decoder that works directly on spatial features.
    Uses global average pooling + MLP.
    """
    
    def __init__(
        self,
        num_channels: int = 512,
        num_classes: int = 10,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            
        Returns:
            logits: [B, num_classes]
        """
        x = self.gap(features).squeeze(-1).squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    # Test decoder
    decoder = DecoderCls(smax=256, num_classes=10)
    z = torch.randn(2, 256)
    logits = decoder(z)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {logits.shape}")
