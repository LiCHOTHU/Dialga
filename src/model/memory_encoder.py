"""MemoryEncoder — DIALGA's factorized encoder, run over a WHOLE video with a
persistent static memory.

Identical trunks / heads / decoder-interface to LatentEncoder3D (so an arm with
mem_mode='none' reproduces the current per-chunk behaviour and the comparison is
single-variable); the only change is that the static grid is produced by
StaticMemory, which may carry state across chunks.

    forward(seq (B, K, C, T, H, W)) ->
        z_static_grids : (B, K, c_s, g,  g )   grid used to decode each chunk
        z_dyn          : (B, K, T,  d_dyn)
        mem_final      : (B, c_hidden, g, g)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.camera_pose import CameraConditioner
from src.model.latent_encoder import _conv3d_trunk
from src.model.static_memory import StaticMemory


class MemoryEncoder(nn.Module):
    def __init__(self, latent_ch: int = 48, hidden_ch: int = 192,
                 d_static: int = 96, static_grid: int = 4,
                 d_dyn: int = 256, dyn_grid: int = 8,
                 mem_update: str = "none", mem_collapse: str = "mean",
                 n_groups: int = 8, zero_mean_dyn: bool = False,
                 d_pose: int = 0, pose_dim: int = 3, chunk_size_lat: int = 9):
        super().__init__()
        # zero_mean_dyn: remove z_dyn's temporal mean inside the encoder, so the
        # dynamics code cannot represent anything CONSTANT over the chunk. Measured
        # motivation: with the free-form code, zeroing z_static costs only ~10% recon
        # while zeroing z_dyn costs ~43% -- z_dyn (2304 floats at full 8x8) simply
        # re-encodes the whole frame and leaves z_static with no job. Projecting it
        # onto the zero-mean subspace forces every persistent structure through
        # z_static: the division of labour becomes architectural, not hoped-for.
        self.zero_mean_dyn = bool(zero_mean_dyn)
        g2, g2d = static_grid ** 2, dyn_grid ** 2
        if d_static % g2 or d_dyn % g2d:
            raise ValueError("d_static/d_dyn must divide their grid areas")
        self.c_static, self.c_dyn = d_static // g2, d_dyn // g2d
        self.static_grid, self.dyn_grid = static_grid, dyn_grid
        self.d_static, self.d_dyn = d_static, d_dyn

        self.trunk_static = _conv3d_trunk(latent_ch, hidden_ch, n_groups)
        self.trunk_dyn = _conv3d_trunk(latent_ch, hidden_ch, n_groups)
        self.d_pose = int(d_pose)
        self.cc = (CameraConditioner(pose_dim=pose_dim, d_pose=d_pose)
                   if d_pose > 0 else None)
        self.mem = StaticMemory(update=mem_update, collapse=mem_collapse,
                                ch=hidden_ch, grid=static_grid, d_pose=d_pose,
                                n_frames=chunk_size_lat)
        self.proj_static = nn.Conv2d(hidden_ch, self.c_static, 1)
        self.proj_dyn = nn.Conv2d(hidden_ch, self.c_dyn, 1)

    def encode_chunk(self, x: torch.Tensor, mem, pose_emb=None, pose_raw=None):
        """x : (B, C, T, H, W). Returns (z_static_grid, z_dyn, mem)."""
        B, _, T, H, W = x.shape
        g, gd = self.static_grid, self.dyn_grid

        hs = self.trunk_static(x)                                   # (B,hid,T,H,W)
        hs_pf = F.adaptive_avg_pool2d(
            hs.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W), (g, g)
        ).reshape(B, T, -1, g, g)                                   # (B,T,hid,g,g)
        mem, grid = self.mem(mem, hs_pf, pose_emb, pose_raw)
        z_static_grid = self.proj_static(grid)                      # (B,c_s,g,g)

        hd = self.trunk_dyn(x)
        hd_pf = F.adaptive_avg_pool2d(
            hd.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W), (gd, gd))
        z_dyn = self.proj_dyn(hd_pf).reshape(B, T, self.d_dyn)      # (B,T,d_dyn)
        if self.zero_mean_dyn:
            z_dyn = z_dyn - z_dyn.mean(dim=1, keepdim=True)
        return z_static_grid, z_dyn, mem

    def forward(self, seq: torch.Tensor, pose=None):
        """seq : (B,K,C,T,H,W); pose : (B,K,T,pose_dim) or None.

        Pose is relativised to the VIDEO's first frame (not each chunk's own frame
        0), so every chunk's static grid is expressed in ONE common reference --
        the DUSt3R move, and the precondition for accumulating them into a single
        scene memory."""
        mem, grids, dyns = None, [], []
        pe = rel_pose = None
        if self.cc is not None and pose is not None:
            B, K, T, P = pose.shape
            anchor = pose[:, 0, 0:1]                              # (B,1,P) video frame 0
            rel = self.cc.relative_to(pose.reshape(B, K * T, P), anchor)
            pe = self.cc.embed(rel).reshape(B, K, T, self.d_pose)
            rel_pose = rel.reshape(B, K, T, P)
        for k in range(seq.shape[1]):
            gsz, zd, mem = self.encode_chunk(
                seq[:, k], mem, None if pe is None else pe[:, k],
                None if rel_pose is None else rel_pose[:, k])
            grids.append(gsz)
            dyns.append(zd)
        return torch.stack(grids, 1), torch.stack(dyns, 1), mem


__all__ = ["MemoryEncoder"]
