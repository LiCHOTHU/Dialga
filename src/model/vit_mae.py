"""ViT-MAE over a Wan-latent chunk — the simple-transformer + MAE test.

Motivation (2026-06-28 reference audit, [[reference-audit-2026-06-28]]): every prior
encoder globally pooled the (48,9,8,8) latent into one vector BEFORE the code formed,
which (a) destroyed spatial structure -> blur + low participation ratio, and (b) made
masked-modelling inert (masking the input of a globally-pooled encoder doesn't remove
information from the latent). VidTwin/MAETok/Slot-Attention all keep a spatial axis.

This module does the canonical MAE the right way and stays entirely in Wan-latent
space (NO DINO target):
  - tokenize: one token per (t,h,w) cell of the Wan latent -> 9*8*8 = 576 tokens, dim 48.
  - mask a fraction; the ENCODER sees ONLY visible tokens (true removal, He et al. 2021).
  - a light decoder + mask tokens reconstructs the MASKED Wan-latent tokens.
  - representation = encoder token outputs (NO pool). Probes read a pooled summary, but
    the bottleneck itself keeps the full spatio-temporal token grid.

I/O: input latent (B, C=48, T=9, H=8, W=8). Loss = MSE on masked tokens (Wan-latent
values). encode(x) returns per-token features (B, 576, dim) for probing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    """Pre-norm transformer block (MHSA + MLP)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VitMAE(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        t_lat: int = 9,
        spatial: int = 8,
        dim: int = 256,
        depth: int = 6,
        n_heads: int = 4,
        dec_dim: int = 128,
        dec_depth: int = 4,
        dec_heads: int = 4,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.t_lat = int(t_lat)
        self.spatial = int(spatial)
        self.n_tokens = self.t_lat * self.spatial * self.spatial
        self.dim = int(dim)
        self.mask_ratio = float(mask_ratio)

        # tokenize: one (t,h,w) cell -> one token of dim latent_ch
        self.patch_embed = nn.Linear(self.latent_ch, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        self.encoder = nn.ModuleList(
            [Block(dim, n_heads) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(dim)

        # decoder
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dec_dim))
        self.decoder = nn.ModuleList(
            [Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_head = nn.Linear(dec_dim, self.latent_ch)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # ---- tokenization helpers -------------------------------------------
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, N=T*H*W, C)."""
        if x.dim() != 5:
            raise ValueError(f"expected (B,C,T,H,W), got {tuple(x.shape)}")
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

    def detokenize(self, tok: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, T, H, W)."""
        B, N, C = tok.shape
        T, H, W = self.t_lat, self.spatial, self.spatial
        return tok.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    # ---- MAE masking (He et al. 2021 random shuffle) --------------------
    @staticmethod
    def random_masking(x: torch.Tensor, mask_ratio: float):
        """x: (B, N, D). Keep len_keep visible tokens per sample.
        Returns (x_visible, mask, ids_restore) where mask: 1=masked, 0=kept."""
        B, N, D = x.shape
        len_keep = int(N * (1.0 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)               # ascending -> keep first
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_vis, mask, ids_restore

    # ---- forward (training): returns loss + diagnostics -----------------
    def forward(self, latent: torch.Tensor, mask_ratio: float | None = None) -> dict:
        mr = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        target = self.tokenize(latent)                   # (B, N, C)
        x = self.patch_embed(target) + self.pos_embed    # (B, N, dim)

        x_vis, mask, ids_restore = self.random_masking(x, mr)
        for blk in self.encoder:
            x_vis = blk(x_vis)
        x_vis = self.enc_norm(x_vis)

        # decoder: re-insert mask tokens, restore order, add dec pos emb
        h = self.dec_embed(x_vis)                        # (B, len_keep, dec_dim)
        B, _, Dd = h.shape
        n_mask = self.n_tokens - h.shape[1]
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        h = torch.cat([h, mask_tokens], dim=1)           # (B, N, dec_dim)
        h = torch.gather(h, 1, ids_restore.unsqueeze(-1).expand(-1, -1, Dd))
        h = h + self.dec_pos_embed
        for blk in self.decoder:
            h = blk(h)
        h = self.dec_norm(h)
        pred = self.dec_head(h)                          # (B, N, C)

        # loss on MASKED tokens only (canonical MAE)
        per_tok = (pred - target).pow(2).mean(dim=-1)    # (B, N)
        denom = mask.sum().clamp(min=1.0)
        loss_masked = (per_tok * mask).sum() / denom
        loss_all = per_tok.mean()
        return {"loss": loss_masked, "loss_all": loss_all,
                "pred": pred, "mask": mask, "target": target}

    # ---- representation (no masking, no pool) ---------------------------
    @torch.no_grad()
    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        """Full-view encoder tokens for probing: (B, N=576, dim). No pool."""
        target = self.tokenize(latent)
        x = self.patch_embed(target) + self.pos_embed
        for blk in self.encoder:
            x = blk(x)
        return self.enc_norm(x)

    @torch.no_grad()
    def reconstruct(self, latent: torch.Tensor, mask_ratio: float | None = None) -> torch.Tensor:
        """Full reconstructed Wan latent (B, C, T, H, W). Masked positions use
        the decoder prediction; visible positions use the ground-truth token
        (standard MAE eval — the encoder never compresses visible tokens)."""
        out = self.forward(latent, mask_ratio=mask_ratio)
        pred, mask, target = out["pred"], out["mask"], out["target"]
        m = mask.unsqueeze(-1)
        filled = target * (1 - m) + pred * m
        return self.detokenize(filled)


__all__ = ["VitMAE"]
