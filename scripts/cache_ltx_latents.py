"""Cache CLEVRER videos as LTX-2 VAE latent tensors.

LTX-2's VAE compresses more aggressively than Wan 2.2 (deeper spatial and
temporal downsampling, 128 latent channels), which gives us a smaller and
more semantic dense latent to operate on. The bottleneck-on-top-of-VAE then
needs to do less work, and the latent itself already carries motion priors.

Output layout (mirrors cache_wan_latents.py):
    <out_dir>/
        metadata.json
        latents/<idx>.pt

Usage:
    python scripts/cache_ltx_latents.py \\
        --max_videos 5 --windows_per_video 4 --window_length 12 \\
        --out_dir outputs/cache/ltx_5vid_W12 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset


def load_ltx2_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    """Load the LTX-2 VAE in eval mode, frozen."""
    from diffusers import AutoencoderKLLTX2Video
    vae = AutoencoderKLLTX2Video.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_window(vae, frames, device, dtype):
    """frames: (T, 3, H, W) in [-1, 1] → latent (C, T_lat, H_lat, W_lat).

    LTX VAE expects (B, C, T, H, W). We do one video at a time.
    """
    x = frames.unsqueeze(0).to(device).to(dtype)                # (1, T, 3, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()                   # (1, 3, T, H, W)
    out = vae.encode(x)
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
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--windows_per_video", type=int, default=4)
    ap.add_argument("--window_length", type=int, default=12)
    ap.add_argument("--frames_per_video", type=int, default=128)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", type=str, default="Lightricks/LTX-2")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

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
    )
    print(f"[data] {len(ds)} windows over {args.max_videos} videos "
          f"(W={args.window_length} frames each)")

    print(f"[vae ] loading {args.model_id}")
    vae = load_ltx2_vae(args.model_id, dtype, device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[vae ] loaded; {n_params/1e6:.1f}M params; device={device}; dtype={dtype}")

    # Encode the first window to learn the latent shape
    s0 = ds[0]
    z0 = encode_window(vae, s0["frames"], device, dtype)
    print(f"[vae ] latent shape per window: {tuple(z0.shape)}  "
          f"(input was {tuple(s0['frames'].shape)})")
    print(f"[vae ] compression: spatial {args.image_size}/{z0.shape[-1]}× "
          f"× temporal {args.window_length}/{z0.shape[1]}× "
          f"× channels {z0.shape[0]}")

    metadata = []
    t0 = time.time()
    for idx in range(len(ds)):
        s = ds[idx]
        z = encode_window(vae, s["frames"], device, dtype) if idx > 0 else z0
        out_path = out_dir / "latents" / f"{idx:06d}.pt"
        torch.save({
            "latent": z,
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
        if (idx + 1) % 5 == 0 or idx == len(ds) - 1:
            elapsed = time.time() - t0
            print(f"  cached {idx+1}/{len(ds)}  "
                  f"{elapsed:.1f}s  ({(idx+1)/elapsed:.2f} win/s)")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "args": vars(args),
            "n_windows": len(metadata),
            "windows": metadata,
            "backbone": "LTX-2",
        }, f, indent=2)
    print(f"\n[done] {len(metadata)} windows cached to {out_dir}")


if __name__ == "__main__":
    main()
