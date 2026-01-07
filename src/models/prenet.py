"""
Pre-Net: Instance-level performance predictor.

Predicts expected performance (accuracy or PSNR/SSIM) based on 
latent statistics and channel SNR.

Used for instance-level rate selection.
"""
import torch
import torch.nn as nn
from typing import Tuple


class ResidualFCBlock(nn.Module):
    """
    Residual Fully Connected Block (RFCB).
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out


class PreNet(nn.Module):
    """
    Instance-level performance predictor.
    
    Predicts expected accuracy/quality from latent statistics + SNR.
    """
    
    def __init__(
        self,
        num_channels: int = 2048,
        hidden_dim: int = 128,
        num_rfcb: int = 2,
        task: str = 'classification',  # 'classification' or 'reconstruction'
        dropout: float = 0.1
    ):
        """
        Args:
            num_channels: Number of input feature channels
            hidden_dim: Hidden dimension
            num_rfcb: Number of RFCB blocks
            task: Task type for output format
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.task = task
        
        # Input: mu [B, C], sigma [B, C], gamma [B, 1] -> [B, 2C + 1]
        input_dim = 2 * num_channels + 1
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )
        
        # RFCB blocks
        self.rfcb_blocks = nn.ModuleList([
            ResidualFCBlock(hidden_dim, dropout)
            for _ in range(num_rfcb)
        ])
        
        # Output head
        if task == 'classification':
            # Output: predicted accuracy (or probability of acc >= threshold)
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Accuracy in [0, 1]
            )
        else:
            # Output: predicted PSNR/SSIM
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 2)  # [PSNR, SSIM]
            )
    
    def compute_stats(self, x_prime: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute channel-wise statistics.
        
        Args:
            x_prime: Features [B, C, H, W]
            
        Returns:
            mu: Channel means [B, C]
            sigma: Channel stds [B, C]
        """
        mu = x_prime.mean(dim=(2, 3))  # [B, C]
        sigma = x_prime.std(dim=(2, 3))  # [B, C]
        return mu, sigma
    
    def forward(
        self,
        x_prime: torch.Tensor,
        gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_prime: Features [B, C, H, W]
            gamma: SNR in dB [B, 1] or [B,]
            
        Returns:
            prediction: [B, 1] for classification, [B, 2] for reconstruction
        """
        # Compute statistics
        mu, sigma = self.compute_stats(x_prime)
        
        # Ensure gamma shape
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        
        # Concatenate inputs
        x = torch.cat([mu, sigma, gamma], dim=1)  # [B, 2C + 1]
        
        # Process through network
        x = self.input_proj(x)
        
        for block in self.rfcb_blocks:
            x = block(x)
        
        # Output prediction
        out = self.output_head(x)
        
        return out


class PreNetWithRate(nn.Module):
    """
    Extended Pre-Net that also takes rate as input.
    More accurate for rate-dependent predictions.
    """
    
    def __init__(
        self,
        num_channels: int = 512,
        hidden_dim: int = 128,
        num_rfcb: int = 2,
        task: str = 'classification'
    ):
        super().__init__()
        
        # Input: mu, sigma, gamma, rate
        input_dim = 2 * num_channels + 2
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim)
        )
        
        self.rfcb_blocks = nn.ModuleList([
            ResidualFCBlock(hidden_dim)
            for _ in range(num_rfcb)
        ])
        
        if task == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, 2)
            )
        
        self.task = task
    
    def forward(
        self,
        x_prime: torch.Tensor,
        gamma: torch.Tensor,
        rate: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with rate input.
        
        Args:
            x_prime: [B, C, H, W]
            gamma: [B, 1] SNR
            rate: [B, 1] transmission rate
        """
        mu = x_prime.mean(dim=(2, 3))
        sigma = x_prime.std(dim=(2, 3))
        
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        if rate.dim() == 1:
            rate = rate.unsqueeze(1)
        
        x = torch.cat([mu, sigma, gamma, rate], dim=1)
        x = self.input_proj(x)
        
        for block in self.rfcb_blocks:
            x = block(x)
        
        return self.output_head(x)


if __name__ == "__main__":
    # Test PreNet
    prenet = PreNet(num_channels=512, task='classification')
    
    x_prime = torch.randn(2, 512, 8, 8)
    gamma = torch.tensor([10.0, 5.0])
    
    pred = prenet(x_prime, gamma)
    print(f"Input shape: {x_prime.shape}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Predictions: {pred.squeeze().tolist()}")
