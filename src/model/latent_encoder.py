"""LatentEncoder3D — chunk-wise encoder over one Wan-latent chunk.

v5.1.1 semantics: ONE chunk in -> z_static (time-collapsed) + z_dyn (per-frame).

The 20-vid overfit at v5.1 (time-collapsed z_dyn) plateaued at L_recon~0.019
even with 4x capacity, because the decoder must produce 9 distinct frames
from a single static-in-time z_dyn. Per-frame z_dyn restores temporal
information for reconstruction while keeping z_static global (the identity
channel that InfoNCE pulls together across chunks).

Two independent Conv3d trunks (no shared backbone) so the encoder cannot
trivially route identity into both heads via the same hidden features.

Input :  (B, C=48, T_lat=9, H=8, W=8)
Output:  dict with
            z_static : (B, D_s=32)         time-pooled
            z_dyn    : (B, T_lat, D_d=16)  per-frame (H,W pooled only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attn_pool import AttentionPool
from src.model.camera_pose import (PoseGridAggregator, PlaneSweepAggregator,
                                    WorldMemoryAggregator)


def _conv3d_trunk(in_ch: int, hidden_ch: int, n_groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, hidden_ch, kernel_size=3, padding=1),
        nn.GroupNorm(n_groups, hidden_ch),
        nn.SiLU(),
        nn.Conv3d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
        nn.GroupNorm(n_groups, hidden_ch),
        nn.SiLU(),
    )


class LatentEncoder3D(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        hidden_ch: int = 32,
        d_static: int = 32,
        d_dyn: int = 16,
        n_groups: int = 8,
        shared_trunk: bool = False,
        use_layer_norm: bool = False,
        pool_type: str = "mean",
        n_queries: int = 8,
        n_heads: int = 4,
        static_grid: int = 4,
        d_pose: int = 0,
        chunk_size_lat: int = 9,
        static_agg: str = "sweep",
        dyn_spatial: bool = False,
        dyn_grid: int = 8,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.hidden_ch = int(hidden_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.d_pose = int(d_pose)   # v5.8: camera-pose channel into z_dyn (encode-inverts)
        self.chunk_size_lat = int(chunk_size_lat)
        self.static_agg = str(static_agg)   # v5.8 fix: "sweep" (plane-sweep) | "conv" (legacy)
        # v5.9 recon fix (literature: VidTwin/Hi-VAE/Cosmos all keep a per-frame
        # SPATIAL axis): z_dyn as a per-frame (c_dyn, gd, gd) grid instead of a
        # global vector. A global per-frame vector can only produce a rank-limited,
        # spatially-uniform per-frame delta -> caps real-video recon (~16 dB DROID).
        self.dyn_spatial = bool(dyn_spatial)
        self.dyn_grid = int(dyn_grid)
        if self.dyn_spatial:
            g2d = self.dyn_grid * self.dyn_grid
            if self.d_dyn % g2d != 0:
                raise ValueError(f"dyn_spatial: d_dyn ({self.d_dyn}) must be divisible "
                                 f"by dyn_grid^2 ({g2d})")
            self.c_dyn = self.d_dyn // g2d
        self.shared_trunk = bool(shared_trunk)
        self.use_layer_norm = bool(use_layer_norm)
        if pool_type not in ("mean", "attn", "slot", "spatial"):
            raise ValueError(f"pool_type must be 'mean', 'attn', 'slot' or 'spatial', got {pool_type!r}")
        self.pool_type = pool_type
        self.n_queries = int(n_queries)

        # v5.7 spatial-grid z_static: keep an image-like (c, g, g) static latent
        # instead of collapsing the cell grid to one vector (the global-pool
        # bottleneck the latent-size sweep proved was rate-independent). The
        # SAME d_static budget is reshaped as static_grid x static_grid cells of
        # c_static channels (default 4x4x6 = 96), so "what is where" survives.
        self.static_grid = int(static_grid)
        if self.pool_type == "spatial":
            g2 = self.static_grid * self.static_grid
            if self.d_static % g2 != 0:
                raise ValueError(f"d_static ({self.d_static}) must be divisible by "
                                 f"static_grid^2 ({g2}) for pool_type='spatial'")
            self.c_static = self.d_static // g2

        self.trunk_static = _conv3d_trunk(latent_ch, hidden_ch, n_groups)
        if self.shared_trunk:
            # Both heads pool from the same trunk's hidden tensor (ablation 2).
            self.trunk_dyn = self.trunk_static
        else:
            self.trunk_dyn = _conv3d_trunk(latent_ch, hidden_ch, n_groups)

        # v5.2: multi-query attention pooling replaces the global mean-pool that
        # diluted per-object cells ~25:1 (see attn_pool.py). The pool outputs
        # hidden_ch so the existing head Linear keeps the same interface.
        if self.pool_type == "attn":
            self.pool_static = AttentionPool(hidden_ch, out_dim=hidden_ch,
                                             n_queries=n_queries, n_heads=n_heads)
            self.pool_dyn = AttentionPool(hidden_ch, out_dim=hidden_ch,
                                          n_queries=n_queries, n_heads=n_heads)
        elif self.pool_type == "slot":
            # v5.3: z_static keeps the M-slot axis (object-separated identity).
            # z_dyn stays a single per-frame motion vector (global dynamics).
            self.pool_static = AttentionPool(hidden_ch, out_dim=hidden_ch,
                                             n_queries=n_queries, n_heads=n_heads,
                                             slot_mode=True)
            self.pool_dyn = AttentionPool(hidden_ch, out_dim=hidden_ch,
                                          n_queries=n_queries, n_heads=n_heads)
        elif self.pool_type == "spatial":
            # z_static: time-collapse the static trunk, adaptively pool the 8x8
            # cell grid down to (static_grid x static_grid), then 1x1 conv to
            # c_static channels -> the spatial code (B, c_static, g, g). z_dyn
            # keeps the per-frame global pool (motion is low-dim).
            self.proj_static = nn.Conv2d(hidden_ch, self.c_static, kernel_size=1)
            # v5.8 camera: MULTI-FRAME pose-conditioned aggregation collapses the
            # per-frame static grids to ONE camera-canonical grid, so z_static is
            # stable under a moving camera (encoder inverts; decoder re-applies).
            # Toy-selected (toy_bridge.py): a learned intrinsics-free conditioner
            # matches an explicit homography plane-sweep and beats single-frame
            # warping under parallax. Zero-init residual over the mean -> starts as
            # the plain mean-collapse. Only built with pose (d_pose>0).
            if self.d_pose > 0:
                # v5.8 FIX (2026-07-28): plane-sweep aggregator recovers depth from
                # the known-pose window (3x more camera-invariant than the conv one,
                # which fed pose-but-not-depth and HURT invariance). "conv" keeps the
                # legacy PoseGridAggregator for ablation.
                if self.static_agg == "world":
                    # MosaicMem-form robust world memory: median accumulation ->
                    # rejects the manipulated (moving) object from z_static.
                    self.pose_agg = WorldMemoryAggregator(hidden_ch, self.d_pose,
                                                          n_frames=self.chunk_size_lat,
                                                          grid=self.static_grid)
                elif self.static_agg == "sweep":
                    self.pose_agg = PlaneSweepAggregator(hidden_ch, self.d_pose,
                                                         n_frames=self.chunk_size_lat,
                                                         grid=self.static_grid)
                else:
                    self.pose_agg = PoseGridAggregator(hidden_ch, self.d_pose,
                                                       n_frames=self.chunk_size_lat,
                                                       grid=self.static_grid)

        self.head_static = nn.Linear(hidden_ch, d_static)
        # v5.8: pose channel concatenated to the per-frame dyn feature before the
        # head, so z_dyn is produced in a camera-canonical frame (encode-inverts).
        # d_pose=0 -> Linear(hidden_ch, d_dyn), byte-identical to legacy.
        self.head_dyn = nn.Linear(hidden_ch + self.d_pose, d_dyn)
        # v5.9: spatial z_dyn head — 1x1 conv over the per-frame (gd,gd) feature grid
        # -> (c_dyn, gd, gd), flattened to d_dyn. Carries per-frame per-location change.
        if self.dyn_spatial:
            self.proj_dyn = nn.Conv2d(hidden_ch + self.d_pose, self.c_dyn, kernel_size=1)

        # v5.1.2 ckpts have these LN layers; pre-v5.1.2 ckpts (v5.pt / v5_best.pt)
        # do not. Opt-in via use_layer_norm so both populations load cleanly.
        if self.use_layer_norm:
            self.norm_static = nn.LayerNorm(d_static)
            self.norm_dyn = nn.LayerNorm(d_dyn)
            with torch.no_grad():
                self.norm_dyn.weight.fill_(15.0 / (d_dyn ** 0.5))

    def forward(self, latent_chunk: torch.Tensor,
                pose_emb: "torch.Tensor | None" = None) -> dict:
        """
        latent_chunk : (B, C, T_lat, H, W)
        pose_emb     : (B, T_lat, d_pose)  camera-pose embedding (v5.8; required
                       iff d_pose>0, and only supported for pool_type='mean').
        Returns dict with
            z_static : (B, D_s)
            z_dyn    : (B, T_lat, D_d)   per-frame
        """
        if latent_chunk.dim() != 5:
            raise ValueError(f"expected 5D (B,C,T,H,W), got {tuple(latent_chunk.shape)}")
        if self.d_pose > 0:
            if self.pool_type not in ("mean", "spatial"):
                raise NotImplementedError("camera pose (d_pose>0) is wired for pool_type='mean' or 'spatial'")
            if pose_emb is None or pose_emb.shape[-1] != self.d_pose:
                raise ValueError(f"d_pose={self.d_pose} needs pose_emb (B,T,{self.d_pose}), "
                                 f"got {None if pose_emb is None else tuple(pose_emb.shape)}")
        elif pose_emb is not None:
            raise ValueError("pose_emb passed but encoder built with d_pose=0")

        if self.shared_trunk:
            h_s_feat = self.trunk_static(latent_chunk)               # (B, hidden_ch, T, H, W)
            h_d_feat = h_s_feat
        else:
            h_s_feat = self.trunk_static(latent_chunk)
            h_d_feat = self.trunk_dyn(latent_chunk)

        B, C, T, H, W = h_s_feat.shape
        z_slots = None
        if self.pool_type == "slot":
            # z_static keeps the M-slot axis: pool returns (B, M, hidden_ch);
            # the head maps each slot to D_s -> z_slots (B, M, D_s). A mean over
            # slots gives a compat global z_static for InfoNCE / attrs / probes.
            cells_s = h_s_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            slots_s = self.pool_static(cells_s)                      # (B, M, hidden_ch)
            z_slots = self.head_static(slots_s)                      # (B, M, D_s)
            z_static = z_slots.mean(dim=1)                           # (B, D_s)
            cells_d = (h_d_feat.permute(0, 2, 3, 4, 1)
                       .reshape(B * T, H * W, C))
            h_d = self.pool_dyn(cells_d).reshape(B, T, C)            # (B, T, hidden_ch)
            z_dyn = self.head_dyn(h_d)                               # (B, T, D_d)
            if self.use_layer_norm:
                z_static = self.norm_static(z_static)
                z_dyn = self.norm_dyn(z_dyn)
            return {"z_static": z_static, "z_dyn": z_dyn, "z_slots": z_slots}

        if self.pool_type == "spatial":
            # z_static keeps a spatial grid. Pool each FRAME's 8x8 cells to (g,g),
            # then (camera) realign every frame to a canonical viewpoint via the
            # pose warp, THEN collapse time -> a viewpoint-stable canonical grid.
            g = self.static_grid
            h_s_pf = (F.adaptive_avg_pool2d(
                        h_s_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), (g, g))
                      .reshape(B, T, C, g, g))                       # (B, T, hidden, g, g)
            if self.d_pose > 0:
                h_s_map = self.pose_agg(h_s_pf, pose_emb)            # multi-frame -> canonical grid
            else:
                h_s_map = h_s_pf.mean(dim=1)                         # (B, hidden, g, g) canonical
            z_static_grid = self.proj_static(h_s_map)                # (B, c_static, g, g)
            z_static = z_static_grid.flatten(1)                      # (B, d_static)
            # z_dyn: per-frame global motion; concat pose so dynamics is also
            # camera-canonical (matches the mean path when d_pose>0).
            if self.dyn_spatial:
                # v5.9: per-frame SPATIAL z_dyn grid (c_dyn, gd, gd) -> flat d_dyn.
                gd = self.dyn_grid
                h_d_map = F.adaptive_avg_pool2d(
                    h_d_feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), (gd, gd))  # (B*T,hidden,gd,gd)
                if self.d_pose > 0:
                    pe = pose_emb.reshape(B * T, self.d_pose, 1, 1).expand(B * T, self.d_pose, gd, gd)
                    h_d_map = torch.cat([h_d_map, pe], dim=1)
                z_dyn_grid = self.proj_dyn(h_d_map).reshape(B, T, self.c_dyn, gd, gd)
                z_dyn = z_dyn_grid.flatten(2)                        # (B, T, D_d)
            else:
                h_d = h_d_feat.mean(dim=(3, 4)).permute(0, 2, 1).contiguous()  # (B, T, hidden)
                if self.d_pose > 0:
                    h_d = torch.cat([h_d, pose_emb], dim=-1)         # (B, T, hidden+d_pose)
                z_dyn = self.head_dyn(h_d)                           # (B, T, D_d)
            if self.use_layer_norm:
                z_dyn = self.norm_dyn(z_dyn)
            return {"z_static": z_static, "z_dyn": z_dyn, "z_static_grid": z_static_grid}

        if self.pool_type == "attn":
            # z_static: attention-pool over ALL T*H*W cells.
            cells_s = h_s_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            h_s = self.pool_static(cells_s)                          # (B, hidden_ch)
            # z_dyn: per-frame attention-pool over H*W cells.
            cells_d = (h_d_feat.permute(0, 2, 3, 4, 1)
                       .reshape(B * T, H * W, C))
            h_d = self.pool_dyn(cells_d).reshape(B, T, C)            # (B, T, hidden_ch)
        else:
            h_s = h_s_feat.mean(dim=(2, 3, 4))                       # (B, hidden_ch)
            h_d = h_d_feat.mean(dim=(3, 4)).permute(0, 2, 1).contiguous()  # (B, T, hidden_ch)

        if self.d_pose > 0:
            # concat the per-frame pose embedding so the head can subtract the
            # known ego-motion out of z_dyn (camera-canonical dynamics).
            h_d = torch.cat([h_d, pose_emb], dim=-1)                 # (B, T, hidden+d_pose)

        z_static = self.head_static(h_s)                             # (B, D_s)
        z_dyn = self.head_dyn(h_d)                                   # (B, T, D_d)
        if self.use_layer_norm:
            z_static = self.norm_static(z_static)
            z_dyn = self.norm_dyn(z_dyn)
        return {"z_static": z_static, "z_dyn": z_dyn}


__all__ = ["LatentEncoder3D"]
