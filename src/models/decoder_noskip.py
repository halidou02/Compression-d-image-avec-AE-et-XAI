"""
Self-Contained Decoder (No Skip Leak).

The decoder ONLY receives z_noisy from the channel.
It generates its own multi-scale features for upsampling.
This ensures the rate actually controls information flow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.fc(cond)
        gamma, beta = params.chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ResFiLMBlock(nn.Module):
    """Residual block with FiLM and GroupNorm."""
    
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
        h = F.silu(self.film1(self.gn1(self.conv1(x)), cond))
        h = F.silu(self.film2(self.gn2(self.conv2(h)), cond))
        return h + self.skip(x)


class UpBlock(nn.Module):
    """Upsample + ResFiLM (no skip from encoder)."""
    
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 64):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block = ResFiLMBlock(out_ch, out_ch, cond_dim=cond_dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return self.block(x, cond)


class SelfContainedDecoder(nn.Module):
    """
    Decoder that ONLY uses z_noisy - no skip connections from encoder.
    
    This is physically correct for JSCC: all info must go through the channel.
    Uses progressive channel groups for rate-dependent refinement.
    """
    
    def __init__(self, in_channels: int = 256, cond_dim: int = 64):
        super().__init__()
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, cond_dim),
            nn.SiLU()
        )
        
        # Base stream: uses channels [0:128]
        self.base_init = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU()
        )
        self.up1 = UpBlock(256, 256, cond_dim)   # 16 -> 32
        self.up2 = UpBlock(256, 128, cond_dim)   # 32 -> 64
        self.up3 = UpBlock(128, 64, cond_dim)    # 64 -> 128
        self.up4 = UpBlock(64, 64, cond_dim)     # 128 -> 256
        self.base_out = nn.Conv2d(64, 3, 3, padding=1)
        
        # Refine1: uses channels [128:192] - adds detail
        self.ref1_process = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Bounded residual
        )
        
        # Refine2: uses channels [192:256] - adds texture
        self.ref2_process = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Bounded residual
        )
        
        # Residual scaling (learnable)
        self.ref1_scale = nn.Parameter(torch.tensor(0.1))
        self.ref2_scale = nn.Parameter(torch.tensor(0.1))
    
    def _group_gate(self, K: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """Continuous gating for channel group [a:b]."""
        g = (K.float() - a) / float(b - a)
        return g.clamp(0, 1).view(-1, 1, 1, 1)
    
    def forward(
        self,
        z: torch.Tensor,
        rate: float,
        snr_db: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z: [B, 256, 16, 16] noisy features (ONLY input)
            rate: compression rate
            snr_db: [B] SNR values
        """
        B = z.size(0)
        device = z.device
        
        K = torch.full((B,), int(round(rate * 256)), device=device)
        
        # Condition
        rate_norm = (rate - 0.1) / 0.9
        snr_norm = snr_db / 20.0
        condition = torch.stack([
            torch.full((B,), rate_norm, device=device),
            snr_norm
        ], dim=1)
        cond = self.condition_embed(condition)
        
        # Split z into groups
        z_base = z[:, :128]       # [B, 128, 16, 16]
        z_ref1 = z[:, 128:192]    # [B, 64, 16, 16]
        z_ref2 = z[:, 192:256]    # [B, 64, 16, 16]
        
        # === BASE STREAM ===
        x = self.base_init(z_base)
        x = self.up1(x, cond)  # 32
        x = self.up2(x, cond)  # 64
        x = self.up3(x, cond)  # 128
        x = self.up4(x, cond)  # 256
        base = torch.sigmoid(self.base_out(x))
        
        # === REFINE STREAMS (gated by rate) ===
        g1 = self._group_gate(K, 128, 192)
        g2 = self._group_gate(K, 192, 256)
        
        delta1 = self.ref1_process(z_ref1) * self.ref1_scale * g1
        delta2 = self.ref2_process(z_ref2) * self.ref2_scale * g2
        
        out = (base + delta1 + delta2).clamp(0, 1)
        return out


if __name__ == "__main__":
    decoder = SelfContainedDecoder()
    z = torch.randn(2, 256, 16, 16)
    snr = torch.tensor([10.0, 15.0])
    
    for rate in [0.25, 0.5, 0.75, 1.0]:
        out = decoder(z, rate=rate, snr_db=snr)
        print(f"rate={rate}: output {out.shape}")
    
    total = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder params: {total:,}")
