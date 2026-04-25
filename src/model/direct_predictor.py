import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return x + hidden


class LatentNextStatePredictor(nn.Module):
    """
    Simple residual predictor over two consecutive latent states.

    This is intentionally much simpler than the DEL solver path and is meant as
    a strong sanity-check baseline for whether the latent representation itself is
    learnable.
    """

    def __init__(self, latent_channels, hidden_channels=256, num_blocks=4):
        super().__init__()
        self.in_proj = nn.Conv2d(latent_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(int(num_blocks))])
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.out_norm = nn.GroupNorm(groups, hidden_channels)
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, q_prev, q_curr):
        hidden = torch.cat([q_prev, q_curr], dim=1)
        hidden = self.in_proj(hidden)
        hidden = self.blocks(hidden)
        return self.out_proj(F.silu(self.out_norm(hidden)))


__all__ = ["LatentNextStatePredictor"]
