"""SSv2ChunkPairs — paired-chunk sampler for Something-Something-v2.

Reads the Wan-latent cache built by scripts/cache_wan_ssv2.py. Each clip is
Wan-encoded at 3 temporal windows (W=33 frames -> T_lat=9). SSv2 clips are
short (median ~45 frames @ 12fps), so the 3 windows generally OVERLAP; they
still give a valid same-clip InfoNCE positive for L_consist and a temporally
later chunk for the (lightly-weighted) forward-dynamics term.

SSv2 has no CLEVRER-style attribute / collision / pose labels, so those
supervision targets are returned as inert placeholders and train_v5 must run
with --lambda_attrs 0 --lambda_event_aux 0 --lambda_gate 0 and WITHOUT
--use_camera_pose (this mirrors DroidChunkPairs). The integer action-class
label is carried through as `label_id` for the downstream frozen-feature
action-recognition probe.

Pairing (per clip, 3 windows sorted by start_frame -> [w0, w1, w2]):
    pair A:  obs=w0, pred=w1, b=w2
    pair B:  obs=w1, pred=w2, b=w0
Split is by clip id; train/val never share clips.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SSv2ChunkPairs(Dataset):
    def __init__(self, cache_dir, split: str = "all", val_frac: float = 0.0,
                 seed: int = 42, max_videos: int = 0,
                 require_chunks_per_video: int = 3, dino_cache_dir=None, **_ignored):
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        self.cache_dir = cache_dir
        self.windows = meta["windows"]
        # optional DINOv2 patch-feature cache (scripts/cache_dino_ssv2.py) for
        # semantic distillation (train_v5 --lambda_mae); rows aligned to windows.
        self.dino_cache_dir = Path(dino_cache_dir) if dino_cache_dir else None
        self._dino_mm = None
        if self.dino_cache_dir is not None:
            di = json.loads((self.dino_cache_dir / "index.json").read_text())
            self._dino_shape = tuple(di["shape"])

        by_vid: dict[int, list[int]] = {}
        for i, w in enumerate(self.windows):
            by_vid.setdefault(int(w["video_id"]), []).append(i)
        for vid in list(by_vid.keys()):
            by_vid[vid] = sorted(by_vid[vid],
                                 key=lambda i: int(self.windows[i]["start_frame"]))
            if len(by_vid[vid]) < require_chunks_per_video:
                del by_vid[vid]

        if val_frac > 0 and split in ("train", "val"):
            vids = sorted(by_vid.keys())
            rng = random.Random(seed)
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * val_frac))
            val_vids = set(vids[:n_val])
            keep = {v: w for v, w in by_vid.items()
                    if (v in val_vids) == (split == "val")}
        else:
            keep = by_vid

        if max_videos > 0 and max_videos < len(keep):
            chosen = sorted(random.Random(seed).sample(sorted(keep), max_videos))
            keep = {v: keep[v] for v in chosen}

        pairs: list[tuple[int, int, int]] = []
        for vid in sorted(keep):
            ws = keep[vid][:3]
            pairs.append((ws[0], ws[1], ws[2]))
            pairs.append((ws[1], ws[2], ws[0]))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int) -> dict:
        return torch.load(self.cache_dir / self.windows[idx]["path"],
                          map_location="cpu", weights_only=False)

    def _dino_features(self, win_idx: int) -> torch.Tensor:
        import numpy as np
        if self._dino_mm is None:
            self._dino_mm = np.memmap(self.dino_cache_dir / "features.f16.bin",
                                      dtype=np.float16, mode="r", shape=self._dino_shape)
        return torch.from_numpy(np.asarray(self._dino_mm[win_idx])).float()

    def __getitem__(self, idx: int) -> dict:
        i_obs, i_pred, i_b = self.pairs[idx]
        obs, pred, b = self._load(i_obs), self._load(i_pred), self._load(i_b)
        assert obs["video_id"] == pred["video_id"] == b["video_id"]
        sample = {
            "chunk_obs":   obs["latent"],
            "chunk_pred":  pred["latent"],
            "chunk_obs_b": b["latent"],
            # inert placeholders (losses disabled) — keep the batch contract
            "gate_GT":     torch.zeros((), dtype=torch.float32),
            "attrs":       torch.zeros(1, 1, dtype=torch.float32),
            "slot_mask":   torch.zeros(1, dtype=torch.bool),
            "label_id":    int(obs.get("label_id", -1)),
            "video_id":    int(obs["video_id"]),
            "start_frame": int(obs["start_frame"]),
        }
        if self.dino_cache_dir is not None:
            sample["dino_obs"] = self._dino_features(i_obs)   # (9,8,8,D)
        return sample


__all__ = ["SSv2ChunkPairs"]
