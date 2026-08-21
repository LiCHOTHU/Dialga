"""Cache Something-Something-v2 clips as Wan-VAE latent tensors.

SSv2 clips are short real-world hand-object videos (12 fps, ~240p, median ~45
frames). We reuse the DIALGA latent pipeline: each clip is decoded, center-
cropped to a square and resized to `image_size`, normalized to [-1, 1], and
Wan-encoded at 3 temporal windows of W=33 frames each (-> T_lat=9). The 3
windows are placed at evenly-spaced starts in [0, L-W]; for short clips they
overlap, which is fine (same-clip InfoNCE positive + a later chunk for the
lightly-weighted forward-dynamics term).

Output layout (identical convention to the CLEVRER/DROID caches):
    <out_dir>/
        metadata.json        — {args, n_windows, windows:[{idx,path,video_id,start_frame}]}
        latents/<idx>.pt     — {latent (C,T_lat,H,W), video_id, start_frame, win_idx, label_id}

Usage:
    python scripts/cache_wan_ssv2.py \\
        --video_dir .../ssv2/videos_extracted/20bn-something-something-v2 \\
        --label_json .../ssv2/labels.json \\
        --split_json .../ssv2/train.json \\
        --max_videos 10000 --image_size 128 --window_frames 33 \\
        --out_dir .../cache/ssv2_10000vid_W33 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_wan_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_window(vae, frames, device, dtype):
    """frames: (T, 3, H, W) in [-1, 1] -> latent (C, T_lat, H_lat, W_lat)."""
    x = frames.unsqueeze(0).to(device).to(dtype)      # (1, T, 3, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()         # (1, 3, T, H, W)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.squeeze(0).cpu().float()


def read_clip(path: str) -> np.ndarray | None:
    """Decode a webm to (T, H, W, 3) uint8 RGB. Returns None on failure."""
    import av
    try:
        container = av.open(path)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
    except Exception as e:  # corrupt / unreadable clip
        print(f"[warn] decode failed {Path(path).name}: {e}")
        return None
    if not frames:
        return None
    return np.stack(frames, axis=0)


def prep_frames(clip: np.ndarray, start: int, W: int, size: int) -> torch.Tensor:
    """clip (T,H,W,3) uint8 -> (W,3,size,size) float in [-1,1], center-crop square."""
    seg = clip[start:start + W]                                   # (W,H,Wd,3)
    t = torch.from_numpy(seg).float().permute(0, 3, 1, 2) / 255.  # (W,3,H,Wd)
    _, _, H, Wd = t.shape
    s = min(H, Wd)
    top, left = (H - s) // 2, (Wd - s) // 2
    t = t[:, :, top:top + s, left:left + s]                       # center square
    t = torch.nn.functional.interpolate(t, size=(size, size),
                                         mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0                                          # [-1,1]


def window_starts(L: int, W: int, n: int) -> list[int]:
    """n evenly-spaced start offsets in [0, L-W] (overlap allowed for short clips)."""
    if L < W:
        return []
    hi = L - W
    if n == 1 or hi == 0:
        return [0] * n
    return [round(i * hi / (n - 1)) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", type=str, required=True,
                    help="dir of <id>.webm clips")
    ap.add_argument("--label_json", type=str, required=True,
                    help="labels.json (template-string -> class-id)")
    ap.add_argument("--split_json", type=str, required=True,
                    help="train.json / validation.json (list of {id,label,template,...})")
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all")
    ap.add_argument("--windows_per_video", type=int, default=3)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    W, size, K = args.window_frames, args.image_size, args.windows_per_video

    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    label_map = json.loads(Path(args.label_json).read_text())   # template -> str(id)
    split = json.loads(Path(args.split_json).read_text())       # list of dicts
    rng = np.random.RandomState(args.seed)
    rng.shuffle(split)

    vdir = Path(args.video_dir)
    print(f"[data] split has {len(split)} clips; window_frames={W} K={K} size={size}")

    print(f"[vae ] loading {args.model_id}")
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"[vae ] loaded on {device} dtype={dtype}")

    metadata = []
    win_idx = 0
    n_clips = 0
    n_skip_short = 0
    t0 = time.time()
    for entry in split:
        if args.max_videos and n_clips >= args.max_videos:
            break
        vid = int(entry["id"])
        template = entry.get("template", "").replace("[", "").replace("]", "")
        label_id = int(label_map.get(template, -1))
        path = vdir / f"{vid}.webm"
        if not path.exists():
            continue
        clip = read_clip(str(path))
        if clip is None:
            continue
        L = clip.shape[0]
        starts = window_starts(L, W, K)
        if not starts:
            n_skip_short += 1
            continue
        for s in starts:
            out_path = out_dir / "latents" / f"{win_idx:06d}.pt"
            if out_path.exists():                               # resume: skip encoded
                metadata.append({"idx": win_idx,
                                 "path": str(out_path.relative_to(out_dir)),
                                 "video_id": vid, "start_frame": int(s)})
                win_idx += 1
                continue
            frames = prep_frames(clip, s, W, size)              # (W,3,size,size)
            z = encode_window(vae, frames, device, dtype)       # (C,T_lat,H,W)
            torch.save({"latent": z, "video_id": vid, "start_frame": int(s),
                        "win_idx": win_idx, "label_id": label_id}, out_path)
            metadata.append({"idx": win_idx,
                             "path": str(out_path.relative_to(out_dir)),
                             "video_id": vid, "start_frame": int(s)})
            win_idx += 1
        n_clips += 1
        if n_clips % 50 == 0:
            el = time.time() - t0
            print(f"  {n_clips} clips / {win_idx} windows  "
                  f"{el:.0f}s ({n_clips/el:.2f} clip/s)  skip_short={n_skip_short}", flush=True)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_windows": len(metadata),
                   "windows": metadata}, f)
    print(f"\n[done] {n_clips} clips -> {len(metadata)} windows in {out_dir}")
    print(f"       skipped_short={n_skip_short}")


if __name__ == "__main__":
    main()
