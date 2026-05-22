"""Cache CLEVRER videos as Wan-VAE latent tensors.

The trajectory training pipeline operates in Wan-latent space (the frozen
Wan-2.2 VAE handles pixel encoding/decoding). Encoding the dataset once
up-front amortizes the expensive Wan-VAE forward pass.

Output layout:
    <out_dir>/
        metadata.json        — list of (video_id, start_frame, window_length)
        latents/<idx>.pt     — one tensor per training window, shape (C, T_lat, H_lat, W_lat)

Usage:
    python scripts/cache_wan_latents.py \\
        --data_dir datasets/CLEVRER/train_video \\
        --annotation_dir datasets/CLEVRER/annotations \\
        --max_videos 30 --windows_per_video 4 --window_length 12 \\
        --out_dir outputs/cache/wan_30vid_W12 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate


def load_wan_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    """Load the Wan-2.2 VAE in eval mode, frozen."""
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_window(vae, frames, device, dtype):
    """frames: (T, 3, H, W) in [-1, 1] → latent (C, T_lat, H_lat, W_lat).

    Wan VAE expects (B, C, T, H, W) input. We do one video at a time.
    """
    x = frames.unsqueeze(0).to(device).to(dtype)                # (1, T, 3, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()                   # (1, 3, T, H, W)
    out = vae.encode(x)
    # diffusers returns an AutoencoderKLOutput with .latent_dist (DiagonalGaussian).
    # We deterministically take the mean (no sampling) — keeps the cache
    # reproducible across runs.
    if hasattr(out, "latent_dist"):
        z = out.latent_dist.mean
    else:
        z = out.latents
    return z.squeeze(0).cpu().float()                           # (C, T_lat, H_lat, W_lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str,
                    default="datasets/CLEVRER/train_video")
    ap.add_argument("--annotation_dir", type=str,
                    default="datasets/CLEVRER/annotations")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_videos", type=int, default=30)
    ap.add_argument("--windows_per_video", type=int, default=4)
    ap.add_argument("--window_length", type=int, default=12,
                    help="Frames per training window. Longer = stronger temporal signal for z_dyn.")
    ap.add_argument("--frames_per_video", type=int, default=128)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--deterministic_starts", type=str, default=None,
                    help="Comma-separated start_frame offsets per video, e.g. '0,33,66'. "
                         "If set, overrides windows_per_video random sampling.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    det_starts = None
    if args.deterministic_starts:
        det_starts = [int(s) for s in args.deterministic_starts.split(",")]
        print(f"[data] deterministic starts: {det_starts}")

    # Build the dataset — gives us (video_id, start_frame, frames, state) windows.
    ds = ClevrerPairedDataset(
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        window_length=args.window_length,
        frames_per_video=args.frames_per_video,
        windows_per_video=args.windows_per_video,
        max_videos=args.max_videos,
        max_objects=args.max_objects,
        coordinate_mode="world_xy",
        image_size=args.image_size,
        seed=args.seed,
        deterministic_starts=det_starts,
    )
    print(f"[data] {len(ds)} windows over {args.max_videos} videos "
          f"(W={args.window_length} frames each)")

    print(f"[vae ] loading {args.model_id} (this can take a minute)")
    vae = load_wan_vae(args.model_id, dtype, device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[vae ] loaded; {n_params/1e6:.1f}M params; device={device}; dtype={dtype}")

    metadata = []
    t0 = time.time()
    n_skipped = 0
    for idx in range(len(ds)):
        out_path = out_dir / "latents" / f"{idx:06d}.pt"
        if out_path.exists():
            n_skipped += 1
            if n_skipped == 1:
                print(f"[resume] skipping existing latents…")
            if (idx + 1) % 1000 == 0:
                print(f"[resume] skipped {idx+1} so far")
            cached_blob = torch.load(out_path, map_location="cpu", weights_only=False)
            metadata.append({
                "idx": idx,
                "path": str(out_path.relative_to(out_dir)),
                "video_id": int(cached_blob["video_id"]),
                "start_frame": int(cached_blob["start_frame"]),
                "latent_shape": list(cached_blob["latent"].shape),
            })
            continue
        s = ds[idx]
        frames = s["frames"]                                    # (T, 3, H, W)
        z = encode_window(vae, frames, device, dtype)           # (C, T_lat, H_lat, W_lat)
        # We also stash the (small) per-window metadata that the training
        # script will need: positions, visibility, attrs, slot_mask, collisions.
        # Frames themselves are NOT cached (large, and we recompute via Wan
        # decoder if we need pixels).
        torch.save({
            "latent": z,                                         # the main payload
            "positions": s["positions"],
            "velocities": s["velocities"],
            "visibility": s["visibility"],
            "slot_mask": s["slot_mask"],
            "attrs": s["attrs"],
            "object_ids": s["object_ids"],
            "collision_mask": s["collision_mask"],
            "collisions": s["collisions"],
            "video_id": int(s["video_id"]),
            "start_frame": int(s["start_frame"]),
        }, out_path)
        metadata.append({
            "idx": idx,
            "path": str(out_path.relative_to(out_dir)),
            "video_id": int(s["video_id"]),
            "start_frame": int(s["start_frame"]),
            "latent_shape": list(z.shape),
        })
        if (idx + 1) % 10 == 0 or idx == len(ds) - 1:
            elapsed = time.time() - t0
            print(f"  cached {idx+1}/{len(ds)}  "
                  f"latent shape {tuple(z.shape)}  "
                  f"{elapsed:.1f}s  ({(idx+1)/elapsed:.2f} win/s)")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "args": vars(args),
            "n_windows": len(metadata),
            "windows": metadata,
        }, f, indent=2)
    print(f"\n[done] {len(metadata)} windows cached to {out_dir}")
    print(f"       metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
