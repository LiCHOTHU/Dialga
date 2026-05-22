"""ClevrerChunkPairs — paired-chunk sampler for v5.1.

Reads the W=33 Wan-latent cache (T_lat=9 per blob, 3 blobs per video at
deterministic start_frames [0, 33, 66]).

Each __getitem__ returns one training sample:

    chunk_obs    : (48, 9, 8, 8)   observed chunk
    chunk_pred   : (48, 9, 8, 8)   temporally-adjacent next chunk
    chunk_obs_b  : (48, 9, 8, 8)   different chunk from the same video (InfoNCE +)
    gate_GT      : scalar bool     1 if any collision happens in chunk_pred
    attrs        : (K, A)          one-hot (color|material|shape), per-video
    slot_mask    : (K,)            bool
    video_id     : int
    start_frame  : int             of chunk_obs

For each video with 3 chunks at offsets [s0, s1, s2] (= [0, 33, 66]) we emit
two pairs:
    pair A:  obs=s0, pred=s1, b=s2
    pair B:  obs=s1, pred=s2, b=s0

Split is by video_id; train/val never share videos.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _gate_GT_from_collision_mask(collision_mask: torch.Tensor) -> bool:
    """collision_mask : (W_pix, K) bool. Returns scalar bool — True if any
    pixel-frame inside this chunk has a collision on any slot."""
    return bool(collision_mask.any().item())


class ClevrerChunkPairs(Dataset):
    def __init__(
        self,
        cache_dir,
        split: str = "all",
        val_frac: float = 0.0,
        seed: int = 42,
        max_videos: int = 0,
        require_chunks_per_video: int = 3,
    ):
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        windows = meta["windows"]
        self.cache_dir = cache_dir
        self.windows = windows

        # Group windows by video_id, sorted by start_frame
        by_vid: dict[int, list[int]] = {}
        for i, w in enumerate(windows):
            by_vid.setdefault(int(w["video_id"]), []).append(i)
        for vid in list(by_vid.keys()):
            by_vid[vid] = sorted(by_vid[vid], key=lambda i: int(windows[i]["start_frame"]))
            if len(by_vid[vid]) < require_chunks_per_video:
                del by_vid[vid]

        # train/val split by video_id (deterministic)
        if val_frac > 0 and split in ("train", "val"):
            vids = sorted(by_vid.keys())
            rng = random.Random(seed)
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * val_frac))
            val_vids = set(vids[:n_val])
            if split == "train":
                keep = {v: ws for v, ws in by_vid.items() if v not in val_vids}
            else:
                keep = {v: ws for v, ws in by_vid.items() if v in val_vids}
        else:
            keep = by_vid

        if max_videos > 0 and max_videos < len(keep):
            kept_vids = sorted(keep.keys())
            rng_cap = random.Random(seed)
            chosen = sorted(rng_cap.sample(kept_vids, max_videos))
            keep = {v: keep[v] for v in chosen}

        # Build the pair index. For each kept video, emit two pairs:
        #   (obs=ws[0], pred=ws[1], b=ws[2])
        #   (obs=ws[1], pred=ws[2], b=ws[0])
        pairs: list[tuple[int, int, int]] = []
        for vid in sorted(keep.keys()):
            ws = keep[vid]
            # only first 3 chunks per video are used; if more exist, ignore extras
            ws = ws[:3]
            pairs.append((ws[0], ws[1], ws[2]))
            pairs.append((ws[1], ws[2], ws[0]))
        self.pairs = pairs

        # Probe one blob to discover T_lat, W_pix
        probe = torch.load(cache_dir / windows[0]["path"], map_location="cpu", weights_only=False)
        self.T_lat = int(probe["latent"].shape[1])
        self.W_pix = int(probe["collision_mask"].shape[0])

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int) -> dict:
        w = self.windows[idx]
        return torch.load(self.cache_dir / w["path"], map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> dict:
        i_obs, i_pred, i_b = self.pairs[idx]
        obs = self._load(i_obs)
        pred = self._load(i_pred)
        b = self._load(i_b)
        # invariants
        assert obs["video_id"] == pred["video_id"] == b["video_id"], \
            "paired-chunk sampler broke video_id invariant"
        return {
            "chunk_obs":   obs["latent"],
            "chunk_pred":  pred["latent"],
            "chunk_obs_b": b["latent"],
            "gate_GT":     torch.tensor(_gate_GT_from_collision_mask(pred["collision_mask"]),
                                        dtype=torch.float32),
            "attrs":       obs["attrs"],
            "slot_mask":   obs["slot_mask"],
            "video_id":    int(obs["video_id"]),
            "start_frame": int(obs["start_frame"]),
        }


def chunk_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif isinstance(v0, int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch]
    return out


__all__ = ["ClevrerChunkPairs", "chunk_collate"]
