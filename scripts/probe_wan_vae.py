"""Probe Wan 2.2 VAE: download, encode a CLEVRER window, decode, save grid.

Goal: confirm latent shape and visual round-trip quality before designing a
flow-matching decoder that targets this latent space.

Run: river env. e.g.
    /storage/.../envs/river/bin/python scripts/probe_wan_vae.py \
        --ckpt /path/to/stage1.pt --out_dir outputs/wan_vae_probe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate


def unnormalize(t):
    return ((t + 1.0) / 2.0).clamp(0, 1)


def save_grid(rows, path, nrow):
    from torchvision.utils import make_grid, save_image
    grid = make_grid([unnormalize(f.cpu()) for f in rows], nrow=nrow, padding=2)
    save_image(grid, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to stage1.pt or stage2.pt — only used to mirror its dataset config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_videos", type=int, default=5)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Wan VAE from", args.model_id, "subfolder=vae")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"VAE params: {sum(p.numel() for p in vae.parameters())/1e6:.1f}M")
    print("VAE config keys:", {k: getattr(vae.config, k, None) for k in
                                ["latent_channels", "z_dim", "temperal_downsample",
                                 "scaling_factor", "shift_factor"]})

    # Mirror the CLEVRER dataset cfg from the checkpoint so the probe runs on
    # the same windows as the slot-state pipeline.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg["dataset"]["data_dir"]),
        annotation_dir=str(cfg["dataset"]["annotation_dir"]),
        split=str(cfg["dataset"]["split"]),
        window_length=int(cfg["training"]["window_length"]),
        frames_per_video=int(cfg["dataset"]["video_num_frames"]),
        windows_per_video=int(cfg["training"]["windows_per_video"]),
        max_videos=int(cfg["training"]["max_videos"]),
        max_objects=int(cfg["dataset"]["max_objects"]),
        coordinate_mode=str(cfg["dataset"]["coordinate_mode"]),
        image_size=int(cfg["dataset"]["image_size"]),
        seed=int(cfg["training"]["seed"]),
    )

    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= args.num_videos:
            break
    if len(seen) == 0:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= args.num_videos:
                break
    samples = list(seen.values())
    batch = paired_collate(samples)
    frames = batch["frames"].to(device).to(dtype)              # (N, T, 3, H, W) in [-1, 1]
    N, T, _, H, W = frames.shape
    print(f"Window batch: N={N}, T={T}, H={H}, W={W}, dtype={frames.dtype}")

    # Wan VAE expects (B, 3, T, H, W).
    video = frames.permute(0, 2, 1, 3, 4).contiguous()         # (N, 3, T, H, W)
    print("Encoder input shape:", tuple(video.shape))

    with torch.no_grad():
        enc = vae.encode(video).latent_dist
        latent = enc.mode()                                     # (N, C_lat, T_lat, H_lat, W_lat)
        print("Latent shape:", tuple(latent.shape), "dtype:", latent.dtype)
        # Wan VAE has shift+scale for normalization
        latents_mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype) \
            .view(1, -1, 1, 1, 1) if hasattr(vae.config, "latents_mean") else None
        latents_std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype) \
            .view(1, -1, 1, 1, 1) if hasattr(vae.config, "latents_std") else None
        if latents_mean is not None:
            print("latents_mean[:5]:", latents_mean.flatten()[:5].tolist())
            print("latents_std[:5]:",  latents_std.flatten()[:5].tolist())
        print("latent stats — mean:", latent.float().mean().item(),
              "std:", latent.float().std().item(),
              "min:", latent.float().min().item(),
              "max:", latent.float().max().item())

        # Round-trip
        recon = vae.decode(latent).sample                       # (N, 3, T, H, W)
    recon = recon.float().permute(0, 2, 1, 3, 4)                # (N, T_out, 3, H, W)
    frames_f = frames.float()
    T_out = recon.shape[1]
    T_cmp = min(T, T_out)
    print(f"Decoder output T_out={T_out}; comparing first {T_cmp} frames")

    pixel_mse = (recon[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
    print(f"\nWan-VAE round-trip pixel MSE (ceiling): {pixel_mse:.6f}")

    for i, s in enumerate(samples):
        gt_row = [frames_f[i, t] for t in range(T_cmp)]
        rec_row = [recon[i, t] for t in range(T_cmp)]
        save_grid(gt_row + rec_row,
                  out_dir / f"video_{s['video_id']}_grid.png", nrow=T_cmp)
    print(f"Saved roundtrip grids to {out_dir}/  (rows: GT / Wan-VAE round-trip, T={T_cmp})")


if __name__ == "__main__":
    main()
