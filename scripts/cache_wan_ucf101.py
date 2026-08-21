"""Cache UCF101 clips as Wan-VAE latent tensors (same format as cache_wan_ssv2.py).

UCF101 layout: <root>/UCF-101/<ClassName>/v_<Class>_g##_c##.avi, label = class.
Split lists: ucfTrainTestlist/{trainlist01,testlist01}.txt; classInd.txt maps
class-id <-> name. Output cache is dataset-agnostic (metadata.json with
video_id/start_frame + per-blob label_id), consumed by SSv2ChunkPairs via
train_v5 --dataset ssv2.

Usage:
    python scripts/cache_wan_ucf101.py \\
        --video_root .../ucf101/UCF-101 \\
        --split_list .../ucf101/ucfTrainTestlist/trainlist01.txt \\
        --class_ind  .../ucf101/ucfTrainTestlist/classInd.txt \\
        --max_videos 8000 --out_dir .../cache/ucf101_train_W33 --device cuda
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
# reuse the SSv2 helpers (identical decode / window / encode logic)
from scripts.cache_wan_ssv2 import (load_wan_vae, encode_window, read_clip,
                                    prep_frames, window_starts)


def parse_split(split_list: str, class_ind: str):
    """Return list of (rel_path, label_id_0based)."""
    name2id = {}
    for line in Path(class_ind).read_text().splitlines():
        if line.strip():
            i, name = line.split()
            name2id[name] = int(i) - 1                      # 0-based
    items = []
    for line in Path(split_list).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]                               # "Class/v_xxx.avi [id]"
        cls = rel.split("/")[0]
        items.append((rel, name2id.get(cls, -1)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True, help="UCF-101 dir (class subdirs)")
    ap.add_argument("--split_list", required=True)
    ap.add_argument("--class_ind", required=True)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--windows_per_video", type=int, default=3)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    W, size, K = args.window_frames, args.image_size, args.windows_per_video

    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    items = parse_split(args.split_list, args.class_ind)
    np.random.RandomState(args.seed).shuffle(items)
    root = Path(args.video_root)
    print(f"[data] split has {len(items)} clips; W={W} K={K} size={size}")

    print(f"[vae ] loading {args.model_id}")
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"[vae ] loaded on {device} dtype={dtype}")

    metadata = []
    win_idx = n_clips = n_skip = 0
    t0 = time.time()
    for vid_i, (rel, label_id) in enumerate(items):
        if args.max_videos and n_clips >= args.max_videos:
            break
        path = root / rel
        if not path.exists():
            continue
        clip = read_clip(str(path))
        if clip is None:
            continue
        starts = window_starts(clip.shape[0], W, K)
        if not starts:
            n_skip += 1
            continue
        for s in starts:
            out_path = out_dir / "latents" / f"{win_idx:06d}.pt"
            if out_path.exists():
                metadata.append({"idx": win_idx, "path": str(out_path.relative_to(out_dir)),
                                 "video_id": vid_i, "start_frame": int(s)})
                win_idx += 1
                continue
            frames = prep_frames(clip, s, W, size)
            z = encode_window(vae, frames, device, dtype)
            torch.save({"latent": z, "video_id": vid_i, "start_frame": int(s),
                        "win_idx": win_idx, "label_id": int(label_id)}, out_path)
            metadata.append({"idx": win_idx, "path": str(out_path.relative_to(out_dir)),
                             "video_id": vid_i, "start_frame": int(s)})
            win_idx += 1
        n_clips += 1
        if n_clips % 50 == 0:
            el = time.time() - t0
            print(f"  {n_clips} clips / {win_idx} windows  {el:.0f}s "
                  f"({n_clips/el:.2f} clip/s) skip_short={n_skip}", flush=True)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_windows": len(metadata),
                   "windows": metadata}, f)
    print(f"\n[done] {n_clips} clips -> {len(metadata)} windows in {out_dir} (skip_short={n_skip})")


if __name__ == "__main__":
    main()
