"""
Progressive U-Net Decoder with Skip Connections and Refinement.

Features:
- U-Net style upsampling with skip connections from encoder
- GroupNorm for stability with variable rate/noise
- FiLM conditioning on (rate, SNR)
- Progressive refinement: channels [0:128] base, [128:192] refine1, [192:256] refine2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            cond: [B, cond_dim]
        """
        params = self.fc(cond)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResFiLMBlock(nn.Module):
    """Residual block with FiLM conditioning and GroupNorm."""
    
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 64, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.film1 = FiLM(cond_dim, out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.film1(h, cond)
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.film2(h, cond)
        h = F.silu(h)
        
        return h + self.skip(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + ResFiLM."""
    
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int = 64):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.block = ResFiLMBlock(out_ch, out_ch, cond_dim=cond_dim)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.block(x, cond)


class ProgressiveUNetDecoder(nn.Module):
    """
    Progressive U-Net Decoder with refinement groups.
    
    Uses channels [0:128] for base, [128:192] for refine1, [192:256] for refine2.
    This ensures rate=1.0 is actually better than rate=0.5.
    """
    
    def __init__(self, cond_dim: int = 64):
        super().__init__()
        
        # Condition embedding: (rate_norm, snr_norm) -> cond_dim
        self.condition_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, cond_dim),
            nn.SiLU()
        )
        
        # Skip reducers
        self.s2_reduce = nn.Conv2d(512, 128, 1)
        self.s1_reduce = nn.Conv2d(256, 64, 1)
        
        # Base stream: uses channels [0:128]
        self.base_init = nn.Conv2d(128, 256, 3, padding=1)
        self.up1 = UpBlock(256, 128, 256, cond_dim)  # 16->32, concat s2
        self.up2 = UpBlock(256, 64, 128, cond_dim)   # 32->64, concat s1
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')  # 64->128
        self.block3 = ResFiLMBlock(128, 64, cond_dim)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')  # 128->256
        self.block4 = ResFiLMBlock(64, 64, cond_dim)
        
        # Base output
        self.base_out = nn.Conv2d(64, 3, 3, padding=1)
        
        # Refine1 stream: uses channels [128:192]
        self.ref1_init = nn.Conv2d(64, 64, 3, padding=1)
        self.ref1_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        
        # Refine2 stream: uses channels [192:256]
        self.ref2_init = nn.Conv2d(64, 64, 3, padding=1)
        self.ref2_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
    
    def _group_gate(self, K: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """Continuous gating for channel group [a:b]."""
        g = (K.float() - a) / float(b - a)
        return g.clamp(0, 1).view(-1, 1, 1, 1)
    
    def forward(
        self,
        z: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        rate: float,
        snr_db: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: [B, 256, 16, 16] noisy features after AWGN
            s1: [B, 256, 64, 64] encoder skip
            s2: [B, 512, 32, 32] encoder skip
            rate: compression rate (0.1-1.0)
            snr_db: [B] SNR values
        """
        B = z.size(0)
        device = z.device
        
        # Compute K (number of active channels)
        K = torch.full((B,), int(round(rate * 256)), device=device)
        
        # Normalize conditions
        rate_tensor = torch.full((B,), rate, device=device)
        rate_norm = (rate_tensor - 0.1) / 0.9  # [0.1-1.0] -> [0-1]
        snr_norm = snr_db / 20.0  # [0-20] -> [0-1]
        condition = torch.stack([rate_norm, snr_norm], dim=1)  # [B, 2]
        cond = self.condition_embed(condition)  # [B, 64]
        
        # Split z into groups
        z_base = z[:, :128]       # [B, 128, 16, 16]
        z_ref1 = z[:, 128:192]    # [B, 64, 16, 16]
        z_ref2 = z[:, 192:256]    # [B, 64, 16, 16]
        
        # Reduce skips
        s2_red = self.s2_reduce(s2)  # [B, 128, 32, 32]
        s1_red = self.s1_reduce(s1)  # [B, 64, 64, 64]
        
        # === BASE STREAM ===
        x = self.base_init(z_base)           # [B, 256, 16, 16]
        x = self.up1(x, s2_red, cond)        # [B, 256, 32, 32]
        x = self.up2(x, s1_red, cond)        # [B, 128, 64, 64]
        x = self.block3(self.up3(x), cond)   # [B, 64, 128, 128]
        x = self.block4(self.up4(x), cond)   # [B, 64, 256, 256]
        
        base = torch.sigmoid(self.base_out(x))  # [B, 3, 256, 256]
        
        # === REFINE STREAMS ===
        # Gates based on K
        g1 = self._group_gate(K, 128, 192)  # active when K > 128
        g2 = self._group_gate(K, 192, 256)  # active when K > 192
        
        # Refine1: uses z_ref1
        r1 = self.ref1_init(z_ref1)
        delta1 = self.ref1_up(r1) * g1  # [B, 3, 256, 256]
        
        # Refine2: uses z_ref2
        r2 = self.ref2_init(z_ref2)
        delta2 = self.ref2_up(r2) * g2  # [B, 3, 256, 256]
        
        # Combine: base + refinements
        out = (base + delta1 + delta2).clamp(0, 1)
        
        return out


if __name__ == "__main__":
    # Test decoder
    decoder = ProgressiveUNetDecoder()
    
    z = torch.randn(2, 256, 16, 16)
    s1 = torch.randn(2, 256, 64, 64)
    s2 = torch.randn(2, 512, 32, 32)
    snr = torch.tensor([10.0, 15.0])
    
    out = decoder(z, s1, s2, rate=0.5, snr_db=snr)
    print(f"Output shape: {out.shape}")  # [2, 3, 256, 256]
    
    out_full = decoder(z, s1, s2, rate=1.0, snr_db=snr)
    print(f"Output (rate=1.0) shape: {out_full.shape}")
    
    # Count params
    total = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {total:,}")
