"""scripts/extract_trajectory_gt.py — TODO 0.2.

For each val video_id (matching the trainer's val_frac=0.2, seed=42 split),
read the CLEVRER annotation JSON and produce:

    per-chunk positions   (n_chunks=3, max_objects=8, 2)
    per-chunk velocities  (n_chunks=3, max_objects=8, 2)
    per-chunk collision   (n_chunks=3,) bool — any collision in chunk_pred?
    slot_mask             (max_objects,) bool

Chunk frames: deterministic starts [0, 33, 66], window length 33 pixel frames
each. Positions/velocities are averaged across the 33 pixel frames in a chunk.

Output:
    outputs/trajectory_gt_val.pt  (dict keyed by video_id)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import (
    COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB,
    _annotation_subdir, _encode_attributes,
)


CHUNK_STARTS = [0, 33, 66]
CHUNK_LEN = 33
MAX_OBJECTS = 8


def _video_id_to_annotation_path(video_id: int, ann_root: Path, split: str) -> Path:
    """Return the CLEVRER annotation JSON path for `video_id`.

    CLEVRER layout: annotations/<split>/annotation_NNNNN-NNNNN/annotation_<id>.json
    """
    lower = (video_id // 1000) * 1000
    upper = lower + 1000
    sub = f"annotation_{lower:05d}-{upper:05d}"
    candidates = [
        ann_root / split / sub / f"annotation_{video_id:05d}.json",
        ann_root / sub / f"annotation_{video_id:05d}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"annotation not found for video_id={video_id}; tried {candidates}")


def _gt_for_one_video(ann: dict) -> dict:
    """Parse one CLEVRER annotation -> per-chunk positions/velocities + collision flags."""
    obj_props = ann["object_property"]
    # Slot ordering matches clevrer_states._encode_attributes: sorted by object_id
    sorted_props = sorted(obj_props, key=lambda item: item["object_id"])
    object_id_to_slot = {}
    for slot_idx, prop in enumerate(sorted_props):
        if slot_idx >= MAX_OBJECTS:
            break
        object_id_to_slot[int(prop["object_id"])] = slot_idx
    slot_mask = torch.zeros(MAX_OBJECTS, dtype=torch.bool)
    for slot_idx in object_id_to_slot.values():
        slot_mask[slot_idx] = True

    motion = ann.get("motion_trajectory", [])
    num_frames = min(len(motion), CHUNK_STARTS[-1] + CHUNK_LEN)
    positions = torch.zeros(num_frames, MAX_OBJECTS, 2, dtype=torch.float32)
    visibility = torch.zeros(num_frames, MAX_OBJECTS, dtype=torch.bool)
    for f_idx, frame_info in enumerate(motion[:num_frames]):
        for obj in frame_info.get("objects", []):
            oid = int(obj["object_id"])
            if oid not in object_id_to_slot:
                continue
            slot = object_id_to_slot[oid]
            loc = obj["location"]  # 3D world
            positions[f_idx, slot, 0] = float(loc[0])
            positions[f_idx, slot, 1] = float(loc[1])
            visibility[f_idx, slot] = bool(obj.get("inside_camera_view", True))

    # Collisions per chunk: True if any annotated collision falls inside that chunk's pixel-frame range
    # (using the chunk that contains the collision; the gate_GT for "chunk_pred" used in training is
    # mirrored from the cache, so here we record per-chunk collisions in absolute terms.)
    chunk_collision = torch.zeros(len(CHUNK_STARTS), dtype=torch.bool)
    for event in ann.get("collision", []) or []:
        f = int(event.get("frame_id", -1))
        if f < 0:
            continue
        for ci, start in enumerate(CHUNK_STARTS):
            if start <= f < start + CHUNK_LEN:
                chunk_collision[ci] = True

    # Per-chunk position/velocity: average over the 33 pixel frames inside the chunk
    n_chunks = len(CHUNK_STARTS)
    per_chunk_positions = torch.zeros(n_chunks, MAX_OBJECTS, 2)
    per_chunk_velocities = torch.zeros(n_chunks, MAX_OBJECTS, 2)
    per_chunk_visibility = torch.zeros(n_chunks, MAX_OBJECTS, dtype=torch.bool)
    for ci, start in enumerate(CHUNK_STARTS):
        end = start + CHUNK_LEN
        if start >= num_frames:
            continue
        end_use = min(end, num_frames)
        block_pos = positions[start:end_use]            # (T, K, 2)
        block_vis = visibility[start:end_use]
        per_chunk_visibility[ci] = block_vis.any(dim=0) & slot_mask
        # Mean position weighted by per-frame visibility; falls back to slot_mask if invisible
        weight = block_vis.float().unsqueeze(-1)        # (T, K, 1)
        denom = weight.sum(dim=0).clamp_min(1.0)        # (K, 1)
        masked_pos = block_pos * weight                  # (T, K, 2)
        per_chunk_positions[ci] = masked_pos.sum(dim=0) / denom
        # Velocity: (mean position at end) - (mean position at start). Approximate.
        # Use last-3 frame mean vs first-3 frame mean to be a bit more robust.
        n_used = end_use - start
        if n_used >= 2:
            head = block_pos[:max(1, n_used // 3)].mean(dim=0)
            tail = block_pos[-max(1, n_used // 3):].mean(dim=0)
            per_chunk_velocities[ci] = (tail - head) / max(1.0, n_used - 1)
    return {
        "per_chunk_positions":  per_chunk_positions,
        "per_chunk_velocities": per_chunk_velocities,
        "per_chunk_visibility": per_chunk_visibility,
        "chunk_collision":      chunk_collision,
        "slot_mask":            slot_mask,
        "n_objects":            int(slot_mask.sum().item()),
    }


def _val_video_ids(cache_dir: Path, val_frac: float, seed: int) -> list[int]:
    """Reproduce the trainer's val split (val_frac=0.2 seed=42), returning sorted val video_ids."""
    meta = json.loads((cache_dir / "metadata.json").read_text())
    by_vid: dict[int, list[int]] = {}
    for w in meta["windows"]:
        vid = int(w["video_id"])
        by_vid.setdefault(vid, []).append(int(w["start_frame"]))
    # Filter to videos with all 3 chunks (matches ClevrerChunkPairs default)
    by_vid = {v: starts for v, starts in by_vid.items() if len(starts) >= 3}
    vids = sorted(by_vid.keys())
    rng = random.Random(seed)
    rng.shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac))
    return sorted(vids[:n_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--annotation_dir",
                    default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations")
    ap.add_argument("--split", default="train",
                    help="CLEVRER annotation split where videos live (cache was built from train).")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/trajectory_gt_val.pt")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    ann_root = Path(args.annotation_dir)

    val_ids = _val_video_ids(cache_dir, args.val_frac, args.seed)
    print(f"[split] val_frac={args.val_frac} seed={args.seed} -> {len(val_ids)} val videos")

    out: dict[int, dict] = {}
    t0 = time.time()
    missing = []
    for i, vid in enumerate(val_ids):
        try:
            ap_path = _video_id_to_annotation_path(vid, ann_root, args.split)
        except FileNotFoundError:
            missing.append(vid)
            continue
        ann = json.loads(ap_path.read_text())
        out[vid] = _gt_for_one_video(ann)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(val_ids)} ({elapsed:.1f}s)")
    print(f"[done] parsed {len(out)} videos in {time.time()-t0:.1f}s; missing {len(missing)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"per_video": out, "val_ids": val_ids, "missing": missing,
                "args": vars(args)}, out_path)
    print(f"[done] wrote {out_path}")

    # Quick sanity stats
    if out:
        any_v = next(iter(out.values()))
        print(f"  one entry shapes: pos={tuple(any_v['per_chunk_positions'].shape)} "
              f"vel={tuple(any_v['per_chunk_velocities'].shape)} "
              f"collision={tuple(any_v['chunk_collision'].shape)}")
        coll_rate = sum(int(v["chunk_collision"].any()) for v in out.values()) / len(out)
        print(f"  any-collision rate (per video, across 3 chunks): {coll_rate:.3f}")


if __name__ == "__main__":
    main()
