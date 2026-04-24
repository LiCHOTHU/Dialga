import math

import torch
import torch.nn as nn


class LeWMPatchAutoencoder(nn.Module):
    """
    LeWM-inspired learned patch encoder with a lightweight image decoder.

    This is not a verbatim copy of LeWM. It keeps the main ideas relevant to this
    repo: direct pixel encoding, learned patch tokens, transformer mixing, and a
    trainable latent map that can be regularized with SIGReg.
    """

    def __init__(
        self,
        image_size=128,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        latent_channels=48,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.embed_dim = int(embed_dim)
        self.latent_channels = int(latent_channels)

        self.patch_embed = nn.Conv2d(
            in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.embed_dim * float(mlp_ratio)),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(self.embed_dim)

        self.projector = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.latent_channels, kernel_size=1),
        )

        hidden = max(self.latent_channels, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, hidden * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden * 4, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected image batch with shape (B, C, H, W), got {tuple(x.shape)}.")

        tokens = self.patch_embed(x)
        batch_size = tokens.shape[0]
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.out_norm(tokens)
        tokens = tokens.transpose(1, 2).reshape(
            batch_size,
            self.embed_dim,
            self.grid_size,
            self.grid_size,
        )
        return self.projector(tokens)

    def decode(self, latent):
        if latent.dim() != 4:
            raise ValueError(f"Expected latent batch with shape (B, C, H, W), got {tuple(latent.shape)}.")
        return self.decoder(latent)


__all__ = ["LeWMPatchAutoencoder"]
