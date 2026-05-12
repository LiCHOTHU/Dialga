"""Overfit a Wan2.2-latent flow-matching decoder on the 5-video CLEVRER set.

Pipeline
--------
  1. Load frozen Wan 2.2 VAE (TI2V-5B) — encodes GT frames to (B,48,2,8,8)
     latents and decodes sampled latents back to pixels.
  2. Load frozen encoder + (optionally) Lagrangian from a stage1.pt checkpoint
     to read GT or encoder-predicted (q, z_static).
  3. Train WanLatentFlowDecoder with linear-path flow matching:
        x_t = (1 - t) * x_0 + t * x_1,  x_0 ~ N(0, I),  x_1 = z_norm
        loss = || v_pred - (x_1 - x_0) ||^2
  4. Eval: Euler-sample from N(0, I), denormalize, decode through frozen VAE,
     save GT / VAE round-trip / decoder-sample grids.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import copy

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.slot_lagrangian import SlotQueryEncoder, LatentSIGRegEncoder
from src.model.wan_flow_decoder import WanLatentFlowDecoder
from scripts.train_slot import encode_window


class EMA:
    """Polyak-averaged copy of a module. Updated in-place each step."""

    def __init__(self, module, decay=0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(module).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source):
        d = self.decay
        for p_ema, p in zip(self.module.parameters(), source.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1 - d)
        for b_ema, b in zip(self.module.buffers(), source.buffers()):
            b_ema.copy_(b)


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def build_encoder(ckpt, attr_dim, d_static, device):
    cfg = ckpt["config"]
    enc_type = str(cfg["model"]["encoder_type"])
    common = dict(
        image_size=int(cfg["dataset"]["image_size"]),
        patch_size=int(cfg["model"]["patch_size"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        depth=int(cfg["model"]["encoder_depth"]),
        num_heads=int(cfg["model"]["num_heads"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
        max_objects=int(cfg["dataset"]["max_objects"]),
        attr_dim=int(attr_dim),
        num_state_dims=int(cfg["model"]["num_state_dims"]),
    )
    # encoder may have a z_static head
    try:
        if enc_type == "slot":
            encoder = SlotQueryEncoder(d_static=d_static, **common)
        else:
            encoder = LatentSIGRegEncoder(latent_dim=int(cfg["model"]["latent_dim"]),
                                          d_static=d_static, **common)
    except TypeError:
        # older checkpoint without d_static
        if enc_type == "slot":
            encoder = SlotQueryEncoder(**common)
        else:
            encoder = LatentSIGRegEncoder(latent_dim=int(cfg["model"]["latent_dim"]), **common)
    encoder.load_state_dict(ckpt["encoder_state_dict"]); encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def load_wan_vae(model_id, dtype, device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def latent_norm_buffers(vae, device, dtype):
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_video(vae, frames):
    """frames: (B, T, 3, H, W). Returns (B, C, T_lat, H_lat, W_lat) un-normalized."""
    B, T = frames.shape[:2]
    video = frames.permute(0, 2, 1, 3, 4).contiguous()              # (B, 3, T, H, W)
    return vae.encode(video).latent_dist.mode()


@torch.no_grad()
def decode_latent(vae, latent):
    """latent: (B, C, T_lat, H_lat, W_lat). Returns frames (B, T_out, 3, H, W)."""
    out = vae.decode(latent).sample                                  # (B, 3, T_out, H, W)
    return out.permute(0, 2, 1, 3, 4)


def velocities_from_positions(positions):
    """Finite-difference velocity. positions: (B, T, K, 2). Returns same shape."""
    v = torch.zeros_like(positions)
    v[:, 1:] = positions[:, 1:] - positions[:, :-1]
    v[:, 0] = v[:, 1]                                                 # mirror first
    return v


def unnormalize(t):
    return ((t + 1.0) / 2.0).clamp(0, 1)


def save_grid(rows, path, nrow):
    from torchvision.utils import make_grid, save_image
    grid = make_grid([unnormalize(f.cpu()) for f in rows], nrow=nrow, padding=2)
    save_image(grid, path)


# -------------------------------------------------------------------------
# training
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to stage1.pt or stage2.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cond_source", choices=["gt_attrs", "z_static"], default="z_static",
                    help="'gt_attrs' uses raw GT one-hot color/shape (purest decoder test); "
                         "'z_static' uses the encoder's video-pooled identity latent.")
    ap.add_argument("--use_gt_pos", action="store_true",
                    help="Use GT positions instead of encoder-predicted (default off → encoder).")
    ap.add_argument("--use_i0", action="store_true", default=True,
                    help="Condition on first-frame Wan latent.")
    ap.add_argument("--no_i0", dest="use_i0", action="store_false")
    ap.add_argument("--num_sample_steps", type=int, default=32)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay for the decoder. Set to 0 to disable EMA.")
    ap.add_argument("--time_dist", choices=["uniform", "logitnorm"], default="logitnorm",
                    help="t sampling: uniform [0,1] or sigmoid(N(0,1)) which concentrates mass near 0.5.")
    ap.add_argument("--lr_min_ratio", type=float, default=0.05,
                    help="cosine schedule floor as a fraction of lr (0=decay to zero).")
    ap.add_argument("--device", default="cuda")
    # wandb
    ap.add_argument("--wandb", action="store_true", help="enable wandb logging")
    ap.add_argument("--wandb_project", default="dialga")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_tags", default="", help="comma-separated tags")
    # data overrides — let CLI bump max_videos / windows beyond what's in the slot ckpt cfg
    ap.add_argument("--max_videos", type=int, default=-1, help="override cfg.training.max_videos")
    ap.add_argument("--data_seed", type=int, default=-1, help="override cfg.training.seed for dataset sampling")
    ap.add_argument("--ckpt_every", type=int, default=10, help="save decoder.pt every N epochs (rolling)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint + dataset
    print("Loading checkpoint:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    pos_norm = float(cfg["dataset"]["pos_normalize"])
    d_static = int(cfg["model"].get("d_static", 16))

    eff_max_videos = int(cfg["training"]["max_videos"]) if args.max_videos < 0 else int(args.max_videos)
    eff_seed = int(cfg["training"]["seed"]) if args.data_seed < 0 else int(args.data_seed)
    dataset = ClevrerPairedDataset(
        data_dir=str(cfg["dataset"]["data_dir"]),
        annotation_dir=str(cfg["dataset"]["annotation_dir"]),
        split=str(cfg["dataset"]["split"]),
        window_length=int(cfg["training"]["window_length"]),
        frames_per_video=int(cfg["dataset"]["video_num_frames"]),
        windows_per_video=int(cfg["training"]["windows_per_video"]),
        max_videos=eff_max_videos,
        max_objects=int(cfg["dataset"]["max_objects"]),
        coordinate_mode=str(cfg["dataset"]["coordinate_mode"]),
        image_size=int(cfg["dataset"]["image_size"]),
        seed=eff_seed,
    )
    attr_dim = dataset.attr_dim
    print(f"Dataset: {len(dataset)} windows | attr_dim={attr_dim} | d_static={d_static} "
          f"| max_videos={eff_max_videos} | seed={eff_seed}")

    # ---- wandb (optional)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                group=args.wandb_group,
                tags=[t for t in args.wandb_tags.split(",") if t],
                config={**vars(args), "max_videos": eff_max_videos, "data_seed": eff_seed},
                dir=str(out_dir),
            )
            print(f"wandb init: project={args.wandb_project} name={args.wandb_name} group={args.wandb_group}")
        except Exception as e:
            print(f"wandb init failed: {e!r} — continuing without")
            wandb_run = None

    # encoder + Wan VAE
    encoder = build_encoder(ckpt, attr_dim, d_static, device)
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"VAE params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M (frozen)")
    lat_mean, lat_std = latent_norm_buffers(vae, device, torch.float32)

    # decoder feature dim — depends on conditioning source
    cond_feat_dim = attr_dim if args.cond_source == "gt_attrs" else d_static
    decoder = WanLatentFlowDecoder(
        latent_channels=48, latent_grid=8, latent_T=2,
        d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks,
        t_dim=args.d_model,
        d_static=cond_feat_dim,
        max_objects=int(cfg["dataset"]["max_objects"]),
        window_length=int(cfg["training"]["window_length"]),
        use_i0=bool(args.use_i0),
    ).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Flow decoder params: {n_dec/1e6:.2f}M | cond_source={args.cond_source} "
          f"| use_gt_pos={args.use_gt_pos} | use_i0={args.use_i0}")

    loader = DataLoader(
        dataset, batch_size=int(args.batch_size), shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        collate_fn=paired_collate, persistent_workers=True,
    )

    optim = AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=max(args.epochs, 1),
                              eta_min=float(args.lr) * float(args.lr_min_ratio))
    ema = EMA(decoder, decay=float(args.ema_decay)) if args.ema_decay > 0 else None
    print(f"Training: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}, "
          f"ema_decay={args.ema_decay}, time_dist={args.time_dist}, "
          f"lr_min_ratio={args.lr_min_ratio}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        sums = {"flow": 0.0}
        steps = 0
        for batch in loader:
            frames = batch["frames"].to(device).to(dtype)             # (B, T, 3, H, W)
            gt_pos = batch["positions"].to(device).float() / pos_norm
            visib = batch["visibility"].to(device).float()
            attrs = batch["attrs"].to(device).float()
            B, T = frames.shape[:2]

            with torch.no_grad():
                # latent target
                z = encode_video(vae, frames).float()                  # (B, 48, 2, 8, 8)
                z_norm = (z - lat_mean) / lat_std

                # i0 latent: encode first frame as a single-frame video
                if args.use_i0:
                    f0 = frames[:, :1]                                 # (B, 1, 3, H, W)
                    z_i0 = encode_video(vae, f0).float()               # (B, 48, 1, 8, 8)
                    z_i0_norm = (z_i0 - lat_mean) / lat_std
                else:
                    z_i0_norm = None

                # conditioning state
                if args.cond_source == "gt_attrs":
                    z_cond = attrs                                     # (B, K, A)
                else:
                    enc_out = encode_window(encoder, frames.float(), attrs)
                    if isinstance(enc_out, tuple):
                        # (positions, z_static_per_frame[, ...])
                        q_enc = enc_out[0]
                        zs = enc_out[1]
                        # video-pool z_static with visibility weights
                        w = visib.unsqueeze(-1)
                        z_cond = (zs * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
                    else:
                        q_enc = enc_out
                        z_cond = attrs                                  # fallback
                if args.use_gt_pos:
                    q = gt_pos
                else:
                    if args.cond_source == "gt_attrs":
                        enc_out = encode_window(encoder, frames.float(), attrs)
                        q = enc_out[0] if isinstance(enc_out, tuple) else enc_out
                    else:
                        q = q_enc
                v = velocities_from_positions(q)

            # flow-matching loss
            x1 = z_norm
            x0 = torch.randn_like(x1)
            if args.time_dist == "uniform":
                t = torch.rand(B, device=device)
            else:
                t = torch.sigmoid(torch.randn(B, device=device))
            t_b = t.view(B, 1, 1, 1, 1)
            x_t = (1 - t_b) * x0 + t_b * x1
            v_target = x1 - x0

            v_pred = decoder(x_t, t, q, v, z_cond, visib, z_i0_norm)
            flow_loss = F.mse_loss(v_pred, v_target)

            optim.zero_grad(set_to_none=True)
            flow_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optim.step()
            if ema is not None:
                ema.update(decoder)

            sums["flow"] += flow_loss.item()
            steps += 1

        sched.step()
        avg_flow = sums["flow"] / max(steps, 1)
        if wandb_run is not None:
            wandb_run.log({"train/flow": avg_flow, "lr": optim.param_groups[0]["lr"]}, step=epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"[ep {epoch:4d}/{args.epochs}] flow {avg_flow:.5f} "
                  f"| lr {optim.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        ckpt_every = int(getattr(args, "ckpt_every", 10))
        if ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == args.epochs):
            save_blob = {
                "decoder_state_dict": decoder.state_dict(),
                "args": vars(args),
                "config": cfg,
                "epoch": epoch,
            }
            if ema is not None:
                save_blob["ema_decoder_state_dict"] = ema.module.state_dict()
            torch.save(save_blob, out_dir / "decoder.pt")
            print(f"  -> decoder ckpt @ ep {epoch} saved to {out_dir / 'decoder.pt'}")

    # ---- eval: one window per video, sample with Euler ----
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= 5:
            break
    if len(seen) == 0:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= 5:
                break
    samples = list(seen.values())
    batch = paired_collate(samples)
    frames = batch["frames"].to(device).to(dtype)
    gt_pos = batch["positions"].to(device).float() / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device).float()
    N, T = frames.shape[:2]

    sample_module = ema.module if ema is not None else decoder
    sample_module.eval()
    decoder.eval()
    with torch.no_grad():
        z = encode_video(vae, frames).float()
        ceiling = decode_latent(vae, z.to(dtype))                       # ground truth roundtrip
        if args.use_i0:
            z_i0 = encode_video(vae, frames[:, :1]).float()
            z_i0_norm = (z_i0 - lat_mean) / lat_std
        else:
            z_i0_norm = None
        if args.cond_source == "gt_attrs":
            z_cond = attrs
            q = gt_pos if args.use_gt_pos else encode_window(encoder, frames.float(), attrs)
            if isinstance(q, tuple): q = q[0]
        else:
            enc_out = encode_window(encoder, frames.float(), attrs)
            q_enc = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            zs = enc_out[1] if isinstance(enc_out, tuple) and len(enc_out) > 1 else None
            if zs is not None:
                w = visib.unsqueeze(-1)
                z_cond = (zs * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
            else:
                z_cond = attrs
            q = gt_pos if args.use_gt_pos else q_enc
        v = velocities_from_positions(q)

        # Euler integration from t=0 (noise) to t=1 (data)
        x = torch.randn(N, 48, 2, 8, 8, device=device)
        n_steps = int(args.num_sample_steps)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_cur = ts[i].expand(N)
            dt = (ts[i + 1] - ts[i]).item()
            v_pred = sample_module(x, t_cur, q, v, z_cond, visib, z_i0_norm)
            x = x + dt * v_pred
        z_sample = x * lat_std + lat_mean
        rec = decode_latent(vae, z_sample.to(dtype))                    # (N, T_out, 3, H, W)

    rec = rec.float()
    ceiling = ceiling.float()
    frames_f = frames.float()
    T_out = rec.shape[1]
    T_cmp = min(T, T_out)

    pixel_mse = (rec[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
    ceil_mse = (ceiling[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
    print(f"\nFinal eval pixel MSE (decoder) : {pixel_mse:.6f}")
    print(f"Wan-VAE round-trip ceiling MSE: {ceil_mse:.6f}")
    if wandb_run is not None:
        wandb_run.summary["eval/pixel_mse"] = pixel_mse
        wandb_run.summary["eval/ceiling_mse"] = ceil_mse
        wandb_run.summary["eval/x_ceiling"] = pixel_mse / max(ceil_mse, 1e-12)

    for i, s in enumerate(samples):
        gt_row = [frames_f[i, t] for t in range(T_cmp)]
        ceil_row = [ceiling[i, t] for t in range(T_cmp)]
        rec_row = [rec[i, t] for t in range(T_cmp)]
        save_grid(gt_row + ceil_row + rec_row,
                  out_dir / f"video_{s['video_id']}_grid.png", nrow=T_cmp)
    print(f"Grids saved under {out_dir}/  (rows: GT / Wan-VAE round-trip / decoder sample)")


if __name__ == "__main__":
    main()
