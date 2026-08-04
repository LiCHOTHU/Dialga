"""LatentDecoder — per-frame Wan-latent decoder.

v5.1.1 semantics: (z_static (B, D_s), z_dyn (B, T, D_d)) -> (B, C, T, H, W).

Each frame t is decoded from (z_static, z_dyn[t]) independently — no
temporal mixing inside the decoder. Per-frame z_dyn is required to make
per-frame reconstruction tractable (v5.1's time-collapsed z_dyn plateaued
at L_recon ~0.02 on a 20-vid overfit because one static-in-time vector
cannot encode 9 distinct frame contents).

Linear lift per frame, one refining 2D conv, 1x1 zero-init projection back
to latent_ch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.camera_pose import PoseGridReapply


class LatentDecoder(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 32,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
        d_pose: int = 0,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)
        self.d_pose = int(d_pose)   # v5.8: camera-pose channel (decode-applies)

        in_dim = self.d_static + self.d_dyn + self.d_pose
        spatial = self.spatial_size * self.spatial_size
        self.in_proj = nn.Linear(in_dim, hidden_ch * spatial)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, hidden_ch),
            nn.SiLU(),
        )
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_static: torch.Tensor, z_dyn: torch.Tensor,
                pose_emb: "torch.Tensor | None" = None) -> torch.Tensor:
        """
        z_static : (B, D_s)
        z_dyn    : (B, T_lat, D_d)
        pose_emb : (B, T_lat, d_pose)  camera-pose embedding (v5.8; required iff
                   d_pose>0). The decoder RE-APPLIES the known viewpoint here.
        Returns:
            latent : (B, latent_ch, T_lat, H, W)
        """
        if z_static.dim() != 2:
            raise ValueError(f"z_static must be (B, D_s), got {tuple(z_static.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B, T, D_d), got {tuple(z_dyn.shape)}")
        B, T, D_d = z_dyn.shape
        if D_d != self.d_dyn:
            raise ValueError(f"z_dyn last dim {D_d} != d_dyn {self.d_dyn}")

        z_s_exp = z_static.unsqueeze(1).expand(B, T, -1)               # (B, T, D_s)
        parts = [z_s_exp, z_dyn]
        if self.d_pose > 0:
            if pose_emb is None or pose_emb.shape[-1] != self.d_pose:
                raise ValueError(f"d_pose={self.d_pose} needs pose_emb (B,T,{self.d_pose}), "
                                 f"got {None if pose_emb is None else tuple(pose_emb.shape)}")
            parts.append(pose_emb)
        elif pose_emb is not None:
            raise ValueError("pose_emb passed but decoder built with d_pose=0")
        x = torch.cat(parts, dim=-1)                                   # (B, T, D_s+D_d[+d_pose])
        x = x.reshape(B * T, -1)                                       # (BT, D_s+D_d)
        h = self.in_proj(x)                                            # (BT, hidden*H*W)
        h = h.reshape(B * T, self.hidden_ch, self.spatial_size, self.spatial_size)
        h = self.refine(h)
        out = self.out_proj(h)                                         # (BT, C, H, W)
        out = out.reshape(B, T, self.latent_ch,
                          self.spatial_size, self.spatial_size)
        return out.permute(0, 2, 1, 3, 4).contiguous()                 # (B, C, T, H, W)


class SpatialBroadcastDecoder(nn.Module):
    """v5.2 decoder: broadcast the latent over the 8x8 grid + coordinate
    channels, then a deeper conv stack decides WHERE to place content.

    LatentDecoder maps the latent to the 8x8 grid through a single Linear lift
    (one matrix inverting position out of a pooled vector). The internals
    inspection showed motion under-rendered (76% of GT temporal amplitude):
    position has to be re-synthesised spatially but the only pathway is that
    lift. The spatial-broadcast scheme (Watters et al. 2019) instead tiles the
    latent across every grid cell and appends fixed (x, y) coordinate channels,
    so the conv stack can read "this cell is at (x, y)" and route content there
    through translation-aware convolutions, with real depth to do it.

    Per frame t the input plane is [z_static (tiled); z_dyn[t] (tiled);
    coords(2)] -> (D_s + D_d + 2, H, W); convs refine; 1x1 zero-init head ->
    latent_ch. Same I/O contract as LatentDecoder.
    """

    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 32,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
        depth: int = 3,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)

        # Fixed normalized coordinate grid (2, H, W), registered so .to(device)
        # moves it and it is excluded from the optimizer.
        s = self.spatial_size
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=0))  # (2, H, W)

        in_ch = self.d_static + self.d_dyn + 2
        layers = [nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
                  nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                       nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        self.refine = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_static: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        if z_static.dim() != 2:
            raise ValueError(f"z_static must be (B, D_s), got {tuple(z_static.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B, T, D_d), got {tuple(z_dyn.shape)}")
        B, T, D_d = z_dyn.shape
        if D_d != self.d_dyn:
            raise ValueError(f"z_dyn last dim {D_d} != d_dyn {self.d_dyn}")
        s = self.spatial_size

        z_s = z_static.unsqueeze(1).expand(B, T, -1).reshape(B * T, self.d_static)
        z_d = z_dyn.reshape(B * T, self.d_dyn)
        zcat = torch.cat([z_s, z_d], dim=-1)                           # (BT, D_s+D_d)
        plane = zcat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, s, s)  # (BT, D, H, W)
        coords = self.coords.unsqueeze(0).expand(B * T, -1, -1, -1)    # (BT, 2, H, W)
        h = torch.cat([plane, coords], dim=1)
        h = self.refine(h)
        out = self.out_proj(h)                                         # (BT, C, H, W)
        out = out.reshape(B, T, self.latent_ch, s, s)
        return out.permute(0, 2, 1, 3, 4).contiguous()                 # (B, C, T, H, W)


class SlotDecoder(nn.Module):
    """v5.3 object-centric decoder: composite M slots by learned alpha masks.

    SpatialBroadcastDecoder broadcasts ONE global code over the grid, so it
    cannot place two objects at different positions distinctly — the root of
    the entanglement/blur. SlotDecoder instead decodes EACH of the M slots
    independently (slot vector tiled over the 8x8 grid + (x,y) coords, conv
    stack), producing a per-slot latent plane plus a per-slot alpha logit, then
    composites the slots with a softmax over the slot axis (Slot Attention /
    SlotDiffusion decoding). The per-frame z_dyn is fed to every slot so its
    alpha mask can move across the chunk (global motion modulating object
    placement). I/O contract: (z_slots (B, M, D_s), z_dyn (B, T, D_d)) ->
    (B, latent_ch, T, H, W).

    Zero-init output head (content + alpha): content starts at 0 and alpha at 0
    (uniform composite of zeros) so the decoder emits 0 at init, matching the
    zero-init reconstruction convention of LatentDecoder.
    """

    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 32,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
        depth: int = 3,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)

        s = self.spatial_size
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=0))  # (2, H, W)

        in_ch = self.d_static + self.d_dyn + 2
        layers = [nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
                  nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                       nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        self.refine = nn.Sequential(*layers)
        # +1 output channel = per-slot alpha logit for the compositing softmax.
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch + 1, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_slots: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        if z_slots.dim() != 3:
            raise ValueError(f"z_slots must be (B, M, D_s), got {tuple(z_slots.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B, T, D_d), got {tuple(z_dyn.shape)}")
        B, M, D_s = z_slots.shape
        T, D_d = z_dyn.shape[1], z_dyn.shape[2]
        if D_s != self.d_static:
            raise ValueError(f"z_slots dim {D_s} != d_static {self.d_static}")
        if D_d != self.d_dyn:
            raise ValueError(f"z_dyn last dim {D_d} != d_dyn {self.d_dyn}")
        s = self.spatial_size

        slots = z_slots.unsqueeze(1).expand(B, T, M, D_s)              # (B, T, M, D_s)
        dyn = z_dyn.unsqueeze(2).expand(B, T, M, D_d)                  # (B, T, M, D_d)
        x = torch.cat([slots, dyn], dim=-1).reshape(B * T * M, D_s + D_d)
        plane = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, s, s)     # (BTM, D, H, W)
        coords = self.coords.unsqueeze(0).expand(B * T * M, -1, -1, -1)
        h = torch.cat([plane, coords], dim=1)
        h = self.refine(h)
        o = self.out_proj(h)                                          # (BTM, C+1, H, W)
        o = o.reshape(B, T, M, self.latent_ch + 1, s, s)
        content = o[:, :, :, : self.latent_ch]                        # (B, T, M, C, H, W)
        alpha = o[:, :, :, self.latent_ch:]                           # (B, T, M, 1, H, W)
        w = alpha.softmax(dim=2)                                      # compose over slots
        out = (content * w).sum(dim=2)                                # (B, T, C, H, W)
        return out.permute(0, 2, 1, 3, 4).contiguous()                # (B, C, T, H, W)


class SpatialGridDecoder(nn.Module):
    """v5.7 decoder: consume a SPATIAL z_static grid (B, c, g, g).

    SpatialBroadcastDecoder tiles ONE global static vector over every cell, so
    "what is where" must be re-synthesised from a position-free code — the exact
    failure the latent-size sweep pinned on the encoder's global pool. Here the
    encoder hands over an image-like static grid (c_static channels on a g x g
    lattice); the decoder upsamples it to the Wan-latent 8x8 resolution so each
    output cell reads the LOCAL static code for its own region, then fuses the
    per-frame z_dyn (broadcast) and fixed (x, y) coords through a conv stack.

    I/O contract: (z_static_grid (B, c_static, g, g), z_dyn (B, T, D_d)) ->
    (B, latent_ch, T, H, W). Zero-init head keeps the emit-0-at-init convention.
    """

    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 96,
        static_grid: int = 4,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
        depth: int = 3,
        d_pose: int = 0,
        dyn_spatial: bool = False,
        dyn_grid: int = 8,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.static_grid = int(static_grid)
        g2 = self.static_grid * self.static_grid
        if self.d_static % g2 != 0:
            raise ValueError(f"d_static ({self.d_static}) must be divisible by "
                             f"static_grid^2 ({g2})")
        self.c_static = self.d_static // g2
        self.d_dyn = int(d_dyn)
        # v5.9: consume z_dyn as a per-frame SPATIAL grid (c_dyn, gd, gd) upsampled
        # to the latent lattice, instead of broadcasting one global vector to every
        # cell. This is what lets the decoder render per-frame per-location change.
        self.dyn_spatial = bool(dyn_spatial)
        self.dyn_grid = int(dyn_grid)
        if self.dyn_spatial:
            g2d = self.dyn_grid * self.dyn_grid
            if self.d_dyn % g2d != 0:
                raise ValueError(f"dyn_spatial: d_dyn ({self.d_dyn}) must be divisible "
                                 f"by dyn_grid^2 ({g2d})")
            self.c_dyn = self.d_dyn // g2d
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)
        self.d_pose = int(d_pose)   # v5.8: RE-APPLY the known pose per frame

        s = self.spatial_size
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=0))  # (2, H, W)

        # v5.8 camera: expand the ONE canonical static grid to a PER-FRAME grid by
        # re-applying the known pose (symmetric partner of the encoder aggregator),
        # so each frame renders its own viewpoint. Zero-init residual -> starts as a
        # plain broadcast. Only built with pose (d_pose>0).
        if self.d_pose > 0:
            self.pose_reapply = PoseGridReapply(self.c_static, self.d_pose, self.static_grid)

        in_ch = self.c_static + (self.c_dyn if self.dyn_spatial else self.d_dyn) + 2
        layers = [nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
                  nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
                       nn.GroupNorm(n_groups, hidden_ch), nn.SiLU()]
        self.refine = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_static_grid: torch.Tensor, z_dyn: torch.Tensor,
                pose_emb: "torch.Tensor | None" = None) -> torch.Tensor:
        if z_static_grid.dim() != 4:
            raise ValueError(f"z_static_grid must be (B, c, g, g), got {tuple(z_static_grid.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B, T, D_d), got {tuple(z_dyn.shape)}")
        B, C, gh, gw = z_static_grid.shape
        if C != self.c_static:
            raise ValueError(f"z_static_grid channels {C} != c_static {self.c_static}")
        T, D_d = z_dyn.shape[1], z_dyn.shape[2]
        if D_d != self.d_dyn:
            raise ValueError(f"z_dyn last dim {D_d} != d_dyn {self.d_dyn}")
        if self.d_pose > 0:
            if pose_emb is None or pose_emb.shape[-1] != self.d_pose:
                raise ValueError(f"d_pose={self.d_pose} needs pose_emb (B,T,{self.d_pose}), "
                                 f"got {None if pose_emb is None else tuple(pose_emb.shape)}")
        elif pose_emb is not None:
            raise ValueError("pose_emb passed but decoder built with d_pose=0")
        s = self.spatial_size

        if self.d_pose > 0:
            # RE-APPLY the known pose: canonical grid -> per-frame grids, then
            # upsample each frame to the Wan-latent 8x8 lattice.
            pf = self.pose_reapply(z_static_grid, pose_emb)             # (B, T, c, g, g)
            grid_up = F.interpolate(pf.reshape(B * T, C, gh, gw), size=(s, s),
                                    mode="bilinear", align_corners=False)   # (B*T, c, H, W)
        else:
            # upsample the static grid to the Wan-latent 8x8 lattice, then broadcast
            # over the T frames.
            grid_up = F.interpolate(z_static_grid, size=(s, s),
                                    mode="bilinear", align_corners=False)   # (B, c, H, W)
            grid_up = grid_up.unsqueeze(1).expand(B, T, C, s, s).reshape(B * T, C, s, s)
        if self.dyn_spatial:
            # per-frame spatial grid -> upsample to the latent lattice (carries
            # per-location change), instead of broadcasting one vector everywhere.
            zdg = z_dyn.reshape(B * T, self.c_dyn, self.dyn_grid, self.dyn_grid)
            z_d = F.interpolate(zdg, size=(s, s), mode="bilinear", align_corners=False)
        else:
            z_d = z_dyn.reshape(B * T, self.d_dyn, 1, 1).expand(-1, -1, s, s)
        coords = self.coords.unsqueeze(0).expand(B * T, -1, -1, -1)
        h = torch.cat([grid_up, z_d, coords], dim=1)
        h = self.refine(h)
        out = self.out_proj(h)                                          # (BT, C, H, W)
        out = out.reshape(B, T, self.latent_ch, s, s)
        return out.permute(0, 2, 1, 3, 4).contiguous()                  # (B, C, T, H, W)


def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """DiT-style sinusoidal timestep embedding for scalar t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0     # scale [0,1] -> [0,1000]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _FiLMConvBlock(nn.Module):
    """Conv -> GroupNorm -> FiLM(t) -> SiLU. FiLM scale/shift come from the
    timestep embedding (adaLN-style modulation, the DiT conditioning path)."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, n_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(n_groups, out_ch)
        self.film = nn.Linear(temb_dim, 2 * out_ch)
        nn.init.zeros_(self.film.weight)                            # start as identity modulation
        nn.init.zeros_(self.film.bias)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(h))
        scale, shift = self.film(temb).chunk(2, dim=-1)            # (BT, 2*out_ch)
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        return self.act(h)


class FlowMatchingDecoder(nn.Module):
    """v5.8 generative decoder: a rectified-flow (MinRF) velocity net over the
    Wan latent chunk, conditioned on our compact code + a diffusion timestep.

    Ported from VideoFlexTok (Apple/EPFL, ICML 2026): a deterministic MSE
    decoder learns the CONDITIONAL MEAN of the target latent, so any residual
    ambiguity in the code renders as blur. A flow decoder instead learns the
    velocity field of a linear noise->data path and SAMPLES a sharp latent,
    escaping the mean-blur ceiling. Training uses no VAE decode (velocity MSE in
    latent space); sharp pixels appear only at sampling time.

    Conditioning mirrors SpatialGridDecoder (bilinear-up static grid broadcast
    over T + per-frame z_dyn + coords), with the NOISED target latent as extra
    input channels and the timestep injected via FiLM in every conv block.

    Per-frame (2D over the 8x8 lattice, shared across T): input plane =
    [x_sigma (latent_ch); static grid (c_static); z_dyn[t] (d_dyn); coords(2)].
    I/O for the velocity net:
        forward(x_sigma (B,C,T,H,W), sigma (B,), z_static_grid (B,c,g,g),
                z_dyn (B,T,D_d)) -> velocity (B,C,T,H,W).
    """

    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 96,
        static_grid: int = 4,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
        depth: int = 3,
        temb_dim: int = 256,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.static_grid = int(static_grid)
        g2 = self.static_grid * self.static_grid
        if self.d_static % g2 != 0:
            raise ValueError(f"d_static ({self.d_static}) must be divisible by "
                             f"static_grid^2 ({g2})")
        self.c_static = self.d_static // g2
        self.d_dyn = int(d_dyn)
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)
        self.temb_dim = int(temb_dim)

        s = self.spatial_size
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], dim=0))   # (2, H, W)

        self.temb_mlp = nn.Sequential(
            nn.Linear(self.temb_dim, self.temb_dim), nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )

        in_ch = self.latent_ch + self.c_static + self.d_dyn + 2
        blocks = [_FiLMConvBlock(in_ch, hidden_ch, self.temb_dim, n_groups)]
        for _ in range(max(0, depth - 1)):
            blocks.append(_FiLMConvBlock(hidden_ch, hidden_ch, self.temb_dim, n_groups))
        self.blocks = nn.ModuleList(blocks)

        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)                           # predict 0 velocity at init
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_sigma: torch.Tensor, sigma: torch.Tensor,
                z_static_grid: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        if x_sigma.dim() != 5:
            raise ValueError(f"x_sigma must be (B,C,T,H,W), got {tuple(x_sigma.shape)}")
        if z_static_grid.dim() != 4:
            raise ValueError(f"z_static_grid must be (B,c,g,g), got {tuple(z_static_grid.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B,T,D_d), got {tuple(z_dyn.shape)}")
        B, C, T, gh, gw = x_sigma.shape
        s = self.spatial_size
        if z_static_grid.shape[1] != self.c_static:
            raise ValueError(f"grid channels {z_static_grid.shape[1]} != c_static {self.c_static}")
        if z_dyn.shape[1] != T:
            raise ValueError(f"z_dyn T {z_dyn.shape[1]} != x_sigma T {T}")

        # timestep embedding, broadcast to (B*T, temb_dim)
        temb = self.temb_mlp(_sinusoidal_embedding(sigma, self.temb_dim))   # (B, temb_dim)
        temb = temb.unsqueeze(1).expand(B, T, self.temb_dim).reshape(B * T, self.temb_dim)

        x = x_sigma.permute(0, 2, 1, 3, 4).reshape(B * T, C, s, s)          # (BT, C, H, W)
        grid_up = F.interpolate(z_static_grid, size=(s, s),
                                mode="bilinear", align_corners=False)       # (B, c, H, W)
        grid_up = grid_up.unsqueeze(1).expand(B, T, self.c_static, s, s
                                              ).reshape(B * T, self.c_static, s, s)
        z_d = z_dyn.reshape(B * T, self.d_dyn, 1, 1).expand(-1, -1, s, s)
        coords = self.coords.unsqueeze(0).expand(B * T, -1, -1, -1)
        h = torch.cat([x, grid_up, z_d, coords], dim=1)

        for blk in self.blocks:
            h = blk(h, temb)
        v = self.out_proj(h)                                               # (BT, C, H, W)
        v = v.reshape(B, T, C, s, s).permute(0, 2, 1, 3, 4).contiguous()
        return v                                                           # (B, C, T, H, W)


class FlowMatcher:
    """MinRF / rectified-flow training + sampling helpers for FlowMatchingDecoder.

    Verbatim mechanism from VideoFlexTok's MinRFNoiseModule + MinRFPipeline:
      train: sigma = sigmoid(N(0,1)); x_sigma = sigma*eps + (1-sigma)*x0;
             velocity target v = eps - x0; loss = MSE(net(x_sigma, sigma, cond), v).
      sample: x = eps; for t in {1, ..., 1/N}: x = x - (1/N) * net(x, t, cond).
    """

    @staticmethod
    def add_noise(x0: torch.Tensor, ln: bool = True):
        """Returns (x_sigma, sigma, v_target) for a clean latent x0 (B,C,T,H,W)."""
        B = x0.shape[0]
        eps = torch.randn_like(x0)
        if ln:
            sigma = torch.sigmoid(torch.randn(B, device=x0.device))
        else:
            sigma = torch.rand(B, device=x0.device)
        s = sigma.view(B, *([1] * (x0.dim() - 1)))
        x_sigma = s * eps + (1.0 - s) * x0
        v_target = eps - x0
        return x_sigma, sigma, v_target

    @staticmethod
    @torch.no_grad()
    def sample(net: FlowMatchingDecoder, z_static_grid: torch.Tensor, z_dyn: torch.Tensor,
               shape, steps: int = 50, device=None, generator=None) -> torch.Tensor:
        """Euler-integrate the flow from noise to a clean Wan latent chunk."""
        device = device or z_static_grid.device
        x = torch.randn(shape, device=device, generator=generator)
        B = shape[0]
        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            t = i / steps
            sigma = torch.full((B,), t, device=device)
            v = net(x, sigma, z_static_grid, z_dyn)
            x = x - dt * v
        return x


__all__ = ["LatentDecoder", "SpatialBroadcastDecoder", "SlotDecoder", "SpatialGridDecoder",
           "FlowMatchingDecoder", "FlowMatcher"]
