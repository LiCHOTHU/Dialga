"""Cache LIBERO-90 episodes as Wan-VAE latent chunks (W=33).

Mirrors scripts/cache_wan_latents.py (CLEVRER) so the downstream training
pipeline can consume LIBERO and CLEVRER caches interchangeably.

Per episode, slide W=33 non-overlapping windows starting at frame 0; drop the
tail if shorter than W. Median episode length is 141 frames so most episodes
yield 4 chunks; total over the 4500-episode dataset is ~18k chunks (~2 GB).

Output layout:
    <out_dir>/
        metadata.json              — args + per-chunk index
        latents/<idx:06d>.pt       — one blob per chunk

Blob keys:
    latent             (48, 9, 8, 8) float32   — Wan-VAE mean
    actions            (33, 7)       float32   — per-frame delta-EE (xyz, rpy, gripper)
    gripper_state      (33,)         float32   — = actions[:, 6], stored explicitly for convenience
    gripper_event      ()            bool      — did gripper flip sign inside this chunk?
    task_id            int                     — 0..89, label-encoded
    task_name          str                     — human-readable task slug
    prompt             str                     — natural-language goal text
    episode_id         int                     — 0..49 within the task
    global_episode_id  int                     — 0..4499 across the dataset
    start_frame        int                     — frame offset of this chunk inside the episode
    num_frames_total   int                     — full episode length
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_video


W = 33  # Wan VAE temporal stride: T_lat = (T-1)//4 + 1 → T=33 gives T_lat=9.


def load_wan_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_chunk(vae, frames_uint8: torch.Tensor, device, dtype) -> torch.Tensor:
    """frames_uint8: (T, H, W, 3) uint8 → latent (C, T_lat, H_lat, W_lat) float32."""
    # uint8 [0, 255] → float [-1, 1], then permute to (1, 3, T, H, W)
    x = frames_uint8.to(device).to(dtype)
    x = (x / 127.5) - 1.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.squeeze(0).cpu().float()


def gripper_event_in_chunk(actions_chunk: np.ndarray) -> bool:
    """True if gripper toggles sign anywhere in the chunk.

    LIBERO action[:, 6] is binary {-1, +1} (open / closed). A sign flip within
    the chunk is a discrete event we can supervise the EventHead with as a
    drop-in for the CLEVRER collision GT.
    """
    g = actions_chunk[:, 6]
    if g.size < 2:
        return False
    return bool(np.any(g[:-1] * g[1:] < 0))


def discover_episodes(jsonl_path: Path) -> list[dict]:
    eps = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            eps.append(r)
    eps.sort(key=lambda r: (r["task"], r["demo_id"]))
    return eps


def build_task_id_map(episodes: list[dict]) -> dict[str, int]:
    task_names = sorted({e["task"] for e in episodes})
    return {t: i for i, t in enumerate(task_names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", type=str,
                    default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90_processed",
                    help="Directory with libero_90.jsonl + video_tile/ + actions/")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Destination for latents/ + metadata.json (use /storage/scratch1).")
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="If >0, only encode the first N episodes (debug).")
    ap.add_argument("--start_episode", type=int, default=0,
                    help="Skip the first N episodes — for resuming across array jobs.")
    ap.add_argument("--end_episode", type=int, default=0,
                    help="Stop before this episode index (0 = end of dataset).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    libero_root = Path(args.libero_root)
    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(libero_root / "libero_90.jsonl")
    task_id_map = build_task_id_map(episodes)
    print(f"[data] {len(episodes)} episodes across {len(task_id_map)} tasks")

    if args.end_episode == 0:
        args.end_episode = len(episodes)
    if args.max_episodes > 0:
        args.end_episode = min(args.end_episode, args.start_episode + args.max_episodes)
    episodes_slice = episodes[args.start_episode:args.end_episode]
    print(f"[slice] processing episodes [{args.start_episode}, {args.end_episode}) "
          f"= {len(episodes_slice)} episodes")

    # Plan all chunks upfront so resume + progress logging stay simple.
    plan: list[dict] = []
    task_demo_counter: dict[str, int] = {}  # local episode_id within each task
    for global_ep_id, ep in enumerate(episodes):
        local_ep_id = task_demo_counter.setdefault(ep["task"], 0)
        task_demo_counter[ep["task"]] = local_ep_id + 1
        if global_ep_id < args.start_episode or global_ep_id >= args.end_episode:
            continue
        n = ep["num_frames"]
        n_chunks = n // W
        for c in range(n_chunks):
            plan.append({
                "global_episode_id": global_ep_id,
                "episode_id": local_ep_id,
                "task": ep["task"],
                "prompt": ep["prompt"],
                "video_path": ep["visual_input"],
                "action_path": ep["action"],
                "start_frame": c * W,
                "num_frames_total": n,
            })
    print(f"[plan] {len(plan)} chunks queued (W={W} non-overlapping)")

    print(f"[vae ] loading {args.model_id} (≈1 minute)")
    vae = load_wan_vae(args.model_id, dtype, device)
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"[vae ] {n_params/1e6:.1f}M params, device={device}, dtype={dtype}")

    metadata = []
    t0 = time.time()
    n_skipped = 0

    # Cache the last-decoded episode so successive chunks of the same episode
    # don't re-decode the whole video.
    last_path = None
    last_video = None
    last_actions = None

    for i, item in enumerate(plan):
        out_path = out_dir / "latents" / f"{i:06d}.pt"
        if out_path.exists():
            blob = torch.load(out_path, map_location="cpu", weights_only=False)
            metadata.append({
                "idx": i,
                "path": str(out_path.relative_to(out_dir)),
                "global_episode_id": int(blob["global_episode_id"]),
                "task_id": int(blob["task_id"]),
                "episode_id": int(blob["episode_id"]),
                "start_frame": int(blob["start_frame"]),
                "latent_shape": list(blob["latent"].shape),
            })
            n_skipped += 1
            continue

        # Lazy-load the video + action file for this episode, reuse across chunks.
        if item["video_path"] != last_path:
            v, _, _ = read_video(str(libero_root / item["video_path"]), pts_unit="sec")
            last_video = v
            last_actions = np.load(libero_root / item["action_path"])
            last_path = item["video_path"]
            assert last_video.shape[0] == last_actions.shape[0] == item["num_frames_total"], (
                f"length mismatch on {item['video_path']}: "
                f"vid={last_video.shape[0]} act={last_actions.shape[0]} "
                f"jsonl={item['num_frames_total']}"
            )

        s = item["start_frame"]
        frames_chunk = last_video[s:s + W]                  # (33, 128, 128, 3) uint8
        actions_chunk = last_actions[s:s + W]               # (33, 7) float32
        z = encode_chunk(vae, frames_chunk, device, dtype)  # (48, 9, 8, 8)

        torch.save({
            "latent":            z,
            "actions":           torch.from_numpy(actions_chunk).float(),
            "gripper_state":     torch.from_numpy(actions_chunk[:, 6]).float(),
            "gripper_event":     torch.tensor(gripper_event_in_chunk(actions_chunk)),
            "task_id":           int(task_id_map[item["task"]]),
            "task_name":         item["task"],
            "prompt":            item["prompt"],
            "episode_id":        int(item["episode_id"]),
            "global_episode_id": int(item["global_episode_id"]),
            "start_frame":       int(item["start_frame"]),
            "num_frames_total":  int(item["num_frames_total"]),
        }, out_path)

        metadata.append({
            "idx": i,
            "path": str(out_path.relative_to(out_dir)),
            "global_episode_id": int(item["global_episode_id"]),
            "task_id": int(task_id_map[item["task"]]),
            "episode_id": int(item["episode_id"]),
            "start_frame": int(item["start_frame"]),
            "latent_shape": list(z.shape),
        })

        if (i + 1) % 100 == 0 or i == len(plan) - 1:
            elapsed = time.time() - t0
            new = i + 1 - n_skipped
            rate = new / elapsed if new > 0 else 0
            eta = (len(plan) - i - 1) / rate if rate > 0 else float("nan")
            print(f"  cached {i+1}/{len(plan)}  skipped={n_skipped}  "
                  f"latent {tuple(z.shape)}  "
                  f"{elapsed:.1f}s  ({rate:.2f} chunk/s, ETA {eta/60:.1f} min)")

    with (out_dir / "metadata.json").open("w") as f:
        json.dump({
            "args": vars(args),
            "n_chunks": len(metadata),
            "task_id_map": task_id_map,
            "windows": metadata,
        }, f, indent=2)
    print(f"\n[done] {len(metadata)} chunks cached to {out_dir}")
    print(f"       metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
