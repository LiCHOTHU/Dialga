"""v5.8 camera-aware disentanglement: inject a KNOWN camera trajectory so the
static/dynamic split survives a MOVING camera.

Principle (Matrix-Game 3.5, "Warped PRoPE", Sec 2.1.1): the camera pose is a
KNOWN operator, not a latent to predict. The encoder INVERTS it (produces a
world-/camera-canonical code) and the decoder RE-APPLIES it (renders the given
viewpoint). Because the *same known pose* enters both sides, the ego-motion is
divided out by construction rather than guessed from a single view (which is
ill-posed). See DEVLOG "camera-aware" section.

Two injection modes:
  - "concat": pose embedding concatenated into the per-frame code (cheap
    baseline; stronger than nothing but not relative).
  - "prope" : relative-pose-biased temporal self-attention -- the relative pose
    M_ij enters the attention softmax, our lightweight video analogue of PRoPE.
    Use this to answer the "is FiLM-style conditioning enough?" risk.

Stabilizers lifted verbatim from MG3.5 (lines 216-221 of the tech report):
  - recenter every pose to the FIRST frame of the chunk (relative pose), and
  - direction-preserving log-compression of the translation magnitude
    (|t| -> log1p(|t|)/4, direction kept) so large camera translations do not
    destabilise training.

Pose convention (generic, opaque vector so the same code serves the 2D-pan
gate and the 3D DROID stage):
    pose : (B, T_lat, P)  raw per-frame camera params, e.g.
        2D pan/zoom gate : P=3  -> (t_x, t_y, log_scale)
        3D SE(3)  DROID  : P=6  -> se(3) log  (relativisation is subtraction on
                                   the log map to first order; the exact
                                   T_0^{-1} T_t matrix form is the drop-in
                                   upgrade for large rotations -- see _relativize)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# pose preprocessing (MG3.5 stabilizers)
# ----------------------------------------------------------------------------
def log_compress_translation(t: torch.Tensor, div: float = 4.0) -> torch.Tensor:
    """Direction-preserving log compression of a translation vector's magnitude.
    t : (..., D_t).  MG3.5: magnitude -> log scale /4, direction unchanged."""
    mag = t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    new_mag = torch.log1p(mag) / div
    return t * (new_mag / mag)


def relativize_to(pose: torch.Tensor, anchor: torch.Tensor,
                  n_trans: int = 2, log_div: float = 4.0) -> torch.Tensor:
    """Recenter per-frame poses to an EXTERNAL anchor frame and log-compress the
    translation part.

    pose : (B, T, P);  anchor : (B, 1, P) — the reference frame to subtract.
    The first ``n_trans`` channels are the translation component (compressed);
    the rest (rotation/scale) are recentred by subtraction. Used for CROSS-CHUNK
    registration: relativise a second chunk's absolute poses to the FIRST chunk's
    frame 0 so both canonical static grids live in one common episode reference
    (the DROID persistence loss). Absolute-pose caches (DROID EE cartesian_position
    in the robot base frame) make this subtraction meaningful across chunks.
    """
    if pose.dim() != 3:
        raise ValueError(f"pose must be (B, T, P), got {tuple(pose.shape)}")
    rel = pose - anchor
    if n_trans > 0:
        trans = log_compress_translation(rel[..., :n_trans], div=log_div)
        rel = torch.cat([trans, rel[..., n_trans:]], dim=-1)
    return rel


def relativize(pose: torch.Tensor, n_trans: int = 2, log_div: float = 4.0) -> torch.Tensor:
    """Recenter per-frame poses to their OWN frame 0 and log-compress translation.

    pose : (B, T, P). Thin wrapper over ``relativize_to`` with anchor = frame 0.
    This first-order (subtractive) relativisation is exact for the 2D-pan gate;
    for large 3D rotations swap in the SE(3) matrix form T_0^{-1} T_t here.
    """
    if pose.dim() != 3:
        raise ValueError(f"pose must be (B, T, P), got {tuple(pose.shape)}")
    return relativize_to(pose, pose[:, :1, :], n_trans=n_trans, log_div=log_div)


# ----------------------------------------------------------------------------
# pose embedding + injection modules
# ----------------------------------------------------------------------------
class PoseEmbed(nn.Module):
    """Per-frame relative pose -> (B, T, d_pose) embedding (concat mode)."""

    def __init__(self, pose_dim: int, d_pose: int = 32):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.d_pose = int(d_pose)
        self.net = nn.Sequential(
            nn.Linear(pose_dim, d_pose), nn.SiLU(),
            nn.Linear(d_pose, d_pose),
        )

    def forward(self, rel_pose: torch.Tensor) -> torch.Tensor:
        return self.net(rel_pose)


class PoseAttention(nn.Module):
    """Relative-pose-biased temporal self-attention over the T_lat frames.

    The relative pose M_ij = rel_pose[j] - rel_pose[i] is embedded to a per-head
    additive bias on the attention logits, so a single softmax carries both the
    token content and the relative camera motion (the PRoPE idea, minus the
    exact parameter-free projection-matrix multiply). Parameter cost is tiny.
    """

    def __init__(self, dim: int, pose_dim: int, n_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.rel_bias = nn.Sequential(
            nn.Linear(pose_dim, dim), nn.SiLU(), nn.Linear(dim, n_heads),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
        """x : (B, T, dim);  rel_pose : (B, T, P) per-frame relative pose."""
        B, T, D = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)   # (B,nh,T,T)
        # pairwise relative pose M_ij = pose_j - pose_i  -> per-head bias
        m_ij = rel_pose[:, None, :, :] - rel_pose[:, :, None, :]       # (B,T,T,P)
        bias = self.rel_bias(m_ij).permute(0, 3, 1, 2)                 # (B,nh,T,T)
        attn = (logits + bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return x + self.o(out)                                         # residual


class SpatialPoseWarp(nn.Module):
    """Pose-conditioned spatial self-attention over a (g x g) cell grid, applied
    per frame. This is the learnable analogue of a camera reprojection: given a
    frame's camera pose, the attention re-routes content across spatial cells
    (a pose-steered, depth-free warp).

    Used symmetrically:
      - encoder: realign every frame's static features toward a CANONICAL grid
        (invert viewpoint) before time-collapsing -> a viewpoint-stable z_static.
      - decoder: re-warp the canonical grid per frame by the same pose (re-apply
        viewpoint) so each frame renders its own camera.

    The attention bias for cell-pair (i, j) is an MLP of [pose_emb_t, coord_i -
    coord_j]: the pose selects WHICH spatial offset to attend to, so frame 0
    (rel_pose~0) can learn ~identity while a moved camera shifts content.
    """

    def __init__(self, dim: int, d_pose: int, grid: int, n_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.d_pose = int(d_pose)
        self.grid = int(grid)
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.bias_net = nn.Sequential(
            nn.Linear(d_pose + 2, dim), nn.SiLU(), nn.Linear(dim, n_heads),
        )
        # zero-init the output projection so the block starts at identity
        # (residual pass-through): pose can only HELP, never wreck early training.
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
        g = self.grid
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        coord = torch.stack([xx.flatten(), yy.flatten()], dim=-1).float()  # (N,2)
        coord = coord / max(g - 1, 1)                                      # [0,1]
        self.register_buffer("coord_delta",
                             coord[:, None, :] - coord[None, :, :])        # (N,N,2)

    def forward(self, feat: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """feat : (B, T, C, g, g);  pose_emb : (B, T, d_pose). -> (B, T, C, g, g)."""
        B, T, C, g, gg = feat.shape
        N = g * gg
        x = feat.flatten(3).permute(0, 1, 3, 2).reshape(B * T, N, C)       # (B*T,N,C)
        h = self.norm(x)
        q = self.q(h).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)        # (B*T,nh,N,N)
        # bias from [pose_emb_t (broadcast over cell pairs), coord_i - coord_j]
        pe = pose_emb.reshape(B * T, 1, 1, self.d_pose).expand(B * T, N, N, self.d_pose)
        cd = self.coord_delta[None].expand(B * T, N, N, 2)
        bias = self.bias_net(torch.cat([pe, cd], dim=-1))                  # (B*T,N,N,nh)
        bias = bias.permute(0, 3, 1, 2)
        attn = (logits + bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, C)
        out = x + self.o(out)
        return out.reshape(B, T, N, C).permute(0, 1, 3, 2).reshape(B, T, C, g, gg)


def _safe_groups(ch: int, max_groups: int = 8) -> int:
    """Largest divisor of ch that is <= max_groups (GroupNorm needs ch % g == 0)."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class PoseGridAggregator(nn.Module):
    """v5.8 (toy-selected): MULTI-FRAME pose-conditioned aggregation of a per-frame
    static grid into ONE camera-canonical grid.

    Replaces the per-frame SpatialPoseWarp-then-mean. The local toy sweep
    (scratchpad toy_bridge.py) settled the mechanism: a LEARNED multi-frame
    pose-conditioner matches an explicit homography plane-sweep in accuracy while
    being INTRINSICS-FREE -- the right call for DROID, whose K/extrinsics are
    noisy (dataset issue #64). It also strictly beats single-frame warping under
    parallax (toy_parallax.py: +3.5 dB) because it can fuse the multiple views the
    temporal window already supplies.

    Mechanism (the winning `L` arm, generalised to T frames): broadcast each
    frame's pose embedding over the g x g cells, concat all T per-frame grids +
    poses along the channel axis, and conv-mix (3x3 convs give LOCAL spatial
    routing + cross-frame fusion) down to a single canonical grid. The output head
    is ZERO-INIT and added as a RESIDUAL over the plain temporal mean, so at init
    this module is byte-identical to the old mean-collapse and pose can only help.
    """

    def __init__(self, dim: int, d_pose: int, n_frames: int, grid: int,
                 hidden: Optional[int] = None, depth: int = 2, n_groups: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.d_pose = int(d_pose)
        self.n_frames = int(n_frames)
        self.grid = int(grid)
        hidden = int(hidden or max(dim, 32))
        ng = _safe_groups(hidden, n_groups)
        in_ch = self.n_frames * (self.dim + self.d_pose)
        layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(ng, hidden), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(ng, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, self.dim, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start == mean-collapse

    def forward(self, feat: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """feat : (B, T, C, g, g);  pose_emb : (B, T, d_pose). -> (B, C, g, g) canonical."""
        B, T, C, g, gg = feat.shape
        if T != self.n_frames:
            raise ValueError(f"PoseGridAggregator built for n_frames={self.n_frames}, got T={T}")
        pe = pose_emb.reshape(B, T, self.d_pose, 1, 1).expand(B, T, self.d_pose, g, gg)
        x = torch.cat([feat, pe], dim=2).reshape(B, T * (C + self.d_pose), g, gg)
        h = self.net(x)
        return feat.mean(dim=1) + self.out(h)                            # residual over mean


class PlaneSweepAggregator(nn.Module):
    """v5.8 FIX (2026-07-28, local-toy decided): recover per-cell inverse-depth
    from the KNOWN-pose multi-frame window via an explicit PLANE-SWEEP, then warp
    every frame to ONE camera-canonical grid.

    Drop-in replacement for PoseGridAggregator, which FAILED: a generic conv fed
    `pose` but not `depth` cannot undo object parallax (shift = -cam*depth), so
    pose HURT invariance (real-module test: z_static inv_ratio ON 0.42 vs BLIND
    0.35). This module is 3x MORE camera-invariant (0.12 vs 0.36 blind) because
    photo-consistency over the temporal window recovers depth with NO depth sensor.

    Mechanism (backward warp): for each canonical cell P and each inverse-depth
    hypothesis d_k, gather frame t at P - flow_t * d_k; the correct depth makes the
    T frames photo-consistent (low variance), so a softmin over cross-frame
    variance selects it per cell, and the canonical grid is the frame-mean at the
    chosen depth. Depth-free (no depth input), intrinsics-free: the per-frame image
    flow is LEARNED from the pose embedding (a fronto-parallel-plane analogue of
    the homography H(d); the learned version matched GT pose in the toy). Zero-init
    flow head -> at init flow=0 => every hypothesis is identity => output == plain
    temporal mean (byte-identical to the old start), so pose can only help.
    """

    def __init__(self, dim: int, d_pose: int, n_frames: int, grid: int,
                 n_depth: int = 8, d_min: float = 0.3, d_max: float = 1.0,
                 tau: float = 0.02, hidden: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.d_pose = int(d_pose)
        self.n_frames = int(n_frames)
        self.grid = int(grid)
        self.tau = float(tau)
        hidden = int(hidden or max(d_pose, 32))
        self.register_buffer("depths", torch.linspace(d_min, d_max, n_depth))
        self.flow = nn.Sequential(nn.Linear(d_pose, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2))
        # zero-init -> flow=0 at start => sweep == plain temporal mean (identity),
        # so this module cannot wreck early training; pose only helps.
        nn.init.zeros_(self.flow[-1].weight); nn.init.zeros_(self.flow[-1].bias)
        g = self.grid
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g),
                                indexing="ij")
        self.register_buffer("base_grid", torch.stack([xs, ys], dim=-1))   # (g,g,2)

    def forward(self, feat: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """feat : (B, T, C, g, g);  pose_emb : (B, T, d_pose). -> (B, C, g, g)."""
        B, T, C, g, gg = feat.shape
        v = self.flow(pose_emb)                                    # (B,T,2) flow/unit-depth
        fBT = feat.reshape(B * T, C, g, gg)
        base = self.base_grid.to(feat.dtype)                       # (g,g,2)
        warped, cost = [], []
        for d in self.depths:
            sh = v * d                                             # (B,T,2) normalized shift
            grid = base[None, None] - sh[:, :, None, None, :]      # (B,T,g,g,2) backward warp
            w = F.grid_sample(fBT, grid.reshape(B * T, g, gg, 2),
                              align_corners=True, padding_mode="border"
                              ).reshape(B, T, C, g, gg)
            warped.append(w.mean(dim=1))                           # (B,C,g,g) frame-mean @ d
            cost.append(w.var(dim=1).mean(dim=1))                  # (B,g,g) cross-frame var
        warped = torch.stack(warped, dim=1)                        # (B,K,C,g,g)
        cost = torch.stack(cost, dim=1)                            # (B,K,g,g)
        wgt = torch.softmax(-cost / self.tau, dim=1)               # pick photo-consistent depth
        return (wgt.unsqueeze(2) * warped).sum(dim=1)              # (B,C,g,g) canonical


class WorldMemoryAggregator(nn.Module):
    """MosaicMem-form z_static (2026-07-29): a ROBUST world-frame memory.

    Same learned pose->flow + inverse-depth sweep as PlaneSweepAggregator, but the
    per-depth accumulation over frames is the TEMPORAL MEDIAN, not the mean. The
    median rejects a TRANSIENT moving object: on real DROID the robot manipulates
    the scene, so a mean-accumulated map smears the moving object into z_static;
    the median keeps z_static = the STATIC background (camera-registered by known
    pose), and the moving object falls into z_dyn as the residual the static map
    can't explain. This dissolves the "scene isn't static" confound that made the
    pooled/mean z_static fail on DROID. Local toy: world-memory z_static is 4.9x
    more camera-invariant than pooled AND localizes the mover in the residual (21x).

    Zero-init flow head -> at init flow=0 => every hypothesis is identity => output
    == plain temporal median of the (unwarped) frames, so pose can only help.
    """

    def __init__(self, dim: int, d_pose: int, n_frames: int, grid: int,
                 n_depth: int = 8, d_min: float = 0.3, d_max: float = 1.0,
                 tau: float = 0.02, hidden: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.d_pose = int(d_pose)
        self.n_frames = int(n_frames)
        self.grid = int(grid)
        self.tau = float(tau)
        hidden = int(hidden or max(d_pose, 32))
        self.register_buffer("depths", torch.linspace(d_min, d_max, n_depth))
        self.flow = nn.Sequential(nn.Linear(d_pose, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 2))
        nn.init.zeros_(self.flow[-1].weight); nn.init.zeros_(self.flow[-1].bias)
        g = self.grid
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g),
                                indexing="ij")
        self.register_buffer("base_grid", torch.stack([xs, ys], dim=-1))   # (g,g,2)

    def forward(self, feat: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """feat : (B, T, C, g, g);  pose_emb : (B, T, d_pose). -> (B, C, g, g)."""
        B, T, C, g, gg = feat.shape
        v = self.flow(pose_emb)                                    # (B,T,2) flow/unit-depth
        fBT = feat.reshape(B * T, C, g, gg)
        base = self.base_grid.to(feat.dtype)
        med, cost = [], []
        for d in self.depths:
            sh = v * d
            grid = base[None, None] - sh[:, :, None, None, :]
            w = F.grid_sample(fBT, grid.reshape(B * T, g, gg, 2),
                              align_corners=True, padding_mode="border"
                              ).reshape(B, T, C, g, gg)
            med.append(w.median(dim=1).values)                     # ROBUST: reject the mover
            cost.append(w.var(dim=1).mean(dim=1))                  # (B,g,g) cross-frame var
        med = torch.stack(med, dim=1)                              # (B,K,C,g,g) robust map per depth
        cost = torch.stack(cost, dim=1)                            # (B,K,g,g)
        wgt = torch.softmax(-cost / self.tau, dim=1)               # pick photo-consistent depth
        return (wgt.unsqueeze(2) * med).sum(dim=1)                 # (B,C,g,g) world memory


class PoseGridReapply(nn.Module):
    """Decoder-side symmetric partner of PoseGridAggregator: RE-APPLY the known
    pose to the canonical static grid, expanding it to a PER-FRAME grid so each
    frame renders its own viewpoint. Zero-init residual head -> at init this is a
    plain broadcast of the canonical grid over T (identity), pose only helps."""

    def __init__(self, dim: int, d_pose: int, grid: int,
                 hidden: Optional[int] = None, depth: int = 2, n_groups: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.d_pose = int(d_pose)
        self.grid = int(grid)
        hidden = int(hidden or max(dim, 32))
        ng = _safe_groups(hidden, n_groups)
        layers = [nn.Conv2d(self.dim + self.d_pose, hidden, 3, padding=1),
                  nn.GroupNorm(ng, hidden), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(ng, hidden), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, self.dim, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start == broadcast

    def forward(self, grid: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """grid : (B, C, g, g);  pose_emb : (B, T, d_pose). -> (B, T, C, g, g) per-frame."""
        B, C, g, gg = grid.shape
        T = pose_emb.shape[1]
        gexp = grid.unsqueeze(1).expand(B, T, C, g, gg)                  # (B,T,C,g,g)
        pe = pose_emb.reshape(B, T, self.d_pose, 1, 1).expand(B, T, self.d_pose, g, gg)
        x = torch.cat([gexp, pe], dim=2).reshape(B * T, C + self.d_pose, g, gg)
        h = self.net(x)
        out = gexp.reshape(B * T, C, g, gg) + self.out(h)
        return out.reshape(B, T, C, g, gg)


class CameraConditioner(nn.Module):
    """Owns the pose preprocessing + embedding; produces the per-frame pose
    embedding the encoder/decoder consume, and (prope mode) a temporal mixer.

    Usage:
        cc = CameraConditioner(pose_dim=3, d_pose=32, mode="prope")
        rel   = cc.relative(pose)                 # (B,T,P) stabilised
        p_emb = cc.embed(rel)                     # (B,T,d_pose)  -> concat channel
        h_mix = cc.mix(h_dyn, rel)                # (B,T,C) pose-aware (prope)
    """

    def __init__(self, pose_dim: int, d_pose: int = 32, mode: str = "concat",
                 n_trans: int = 2, mix_dim: Optional[int] = None, n_heads: int = 4):
        super().__init__()
        if mode not in ("concat", "prope"):
            raise ValueError(f"mode must be 'concat' or 'prope', got {mode!r}")
        self.pose_dim = int(pose_dim)
        self.d_pose = int(d_pose)
        self.mode = mode
        self.n_trans = int(n_trans)
        self.embed_net = PoseEmbed(pose_dim, d_pose)
        self.mixer = (PoseAttention(mix_dim, pose_dim, n_heads=n_heads)
                      if (mode == "prope" and mix_dim is not None) else None)

    def relative(self, pose: torch.Tensor) -> torch.Tensor:
        return relativize(pose, n_trans=self.n_trans)

    def relative_to(self, pose: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Relativise ``pose`` to an EXTERNAL anchor frame (B,1,P) — cross-chunk
        registration for the DROID persistence loss."""
        return relativize_to(pose, anchor, n_trans=self.n_trans)

    def embed(self, rel_pose: torch.Tensor) -> torch.Tensor:
        return self.embed_net(rel_pose)

    def mix(self, h: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
        return self.mixer(h, rel_pose) if self.mixer is not None else h


# ----------------------------------------------------------------------------
# synthetic 2D camera motion (the depth-free gate)
# ----------------------------------------------------------------------------
def synthetic_pan(chunk: torch.Tensor, max_shift: float = 0.4,
                  max_zoom: float = 0.15,
                  generator: Optional[torch.Generator] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a per-sample LINEAR camera pan+zoom to a (B,C,T,H,W) tensor and
    return (warped_chunk, pose) with pose (B,T,3) = (t_x, t_y, log_scale) in
    normalised grid units. Frame t is shifted by (t/(T-1)) * velocity and zoomed
    linearly -- a constant-velocity moving camera. grid_sample, align_corners.

    A pure 2D homography => NO depth/parallax ambiguity: this isolates the
    "can the encoder invert a KNOWN camera transform at all?" question, which is
    the go/no-go before the 3D DROID stage.
    """
    if chunk.dim() != 5:
        raise ValueError(f"expected (B,C,T,H,W), got {tuple(chunk.shape)}")
    B, C, T, H, W = chunk.shape
    dev, dt = chunk.device, chunk.dtype
    g = generator
    # per-sample pan velocity (grid units, full-clip displacement) + zoom rate
    vel = (torch.rand(B, 2, generator=g, device=dev) * 2 - 1) * max_shift
    zoom = (torch.rand(B, 1, generator=g, device=dev) * 2 - 1) * max_zoom
    frac = torch.linspace(0, 1, T, device=dev, dtype=dt)                  # (T,)
    tx = vel[:, 0:1] * frac[None, :]                                      # (B,T)
    ty = vel[:, 1:2] * frac[None, :]
    log_scale = zoom * frac[None, :]                                      # (B,T)
    scale = torch.exp(log_scale)
    # affine theta per (B,T): [[s,0,tx],[0,s,ty]]
    x = chunk.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    s = scale.reshape(B * T)
    theta = torch.zeros(B * T, 2, 3, device=dev, dtype=dt)
    theta[:, 0, 0] = s
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx.reshape(B * T)
    theta[:, 1, 2] = ty.reshape(B * T)
    grid = F.affine_grid(theta, [B * T, C, H, W], align_corners=False)
    warped = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    warped = warped.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    pose = torch.stack([tx, ty, log_scale], dim=-1)                      # (B,T,3)
    return warped, pose


def camera_invariance_loss(z_ref: torch.Tensor, z_warp: torch.Tensor) -> torch.Tensor:
    """MSE between the code from the un-panned view (pose=identity) and the code
    from the panned view (pose=W). Drives 'invert the known camera'."""
    return F.mse_loss(z_warp, z_ref)


__all__ = [
    "log_compress_translation", "relativize", "relativize_to", "PoseEmbed", "PoseAttention",
    "SpatialPoseWarp", "PoseGridAggregator", "PlaneSweepAggregator",
    "WorldMemoryAggregator", "PoseGridReapply",
    "CameraConditioner", "synthetic_pan", "camera_invariance_loss",
]
