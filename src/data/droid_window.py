"""DroidChunkPairs — paired-chunk sampler for the v5.8 moving-camera experiment.

Reads the DROID wrist-camera Wan-latent cache built by
scripts/extract_droid_wrist.py. Each blob carries the KNOWN per-frame camera
pose (`pose`, the EE cartesian_position at the 9 latent frames) and camera
velocity (`vel`). DROID has no CLEVRER-style attribute / collision labels, so
those supervision targets are returned as inert placeholders — train_v5 must run
with --lambda_attrs 0 --lambda_event_aux 0 --lambda_gate 0 (the camera gate
already forces the latent-recon config).

Pairing (per episode, chunks sorted by start_frame): for k>=3 chunks emit the
k-2 sliding triples (i, i+1, i+2):
    obs = chunk i, pred = chunk i+1 (temporal next), b = chunk i+2 (InfoNCE
    positive — a different chunk of the SAME episode/scene).
The pose for each of the three chunks travels with it, so the encoder inverts
and the decoder re-applies the real camera trajectory of that specific chunk.

Split is by episode_id; train/val never share episodes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DroidChunkPairs(Dataset):
    def __init__(self, cache_dir, split: str = "all", val_frac: float = 0.0,
                 seed: int = 42, max_videos: int = 0, load_frames: bool = False,
                 **_ignored):
        self.load_frames = load_frames
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        self.cache_dir = cache_dir
        self.windows = meta["windows"]

        by_ep: dict[int, list[int]] = {}
        for i, w in enumerate(self.windows):
            by_ep.setdefault(int(w["episode_id"]), []).append(i)
        for ep in list(by_ep.keys()):
            by_ep[ep] = sorted(by_ep[ep], key=lambda i: int(self.windows[i]["start_frame"]))
            if len(by_ep[ep]) < 3:
                del by_ep[ep]

        if val_frac > 0 and split in ("train", "val"):
            eps = sorted(by_ep.keys())
            rng = random.Random(seed)
            rng.shuffle(eps)
            n_val = max(1, int(len(eps) * val_frac))
            val_eps = set(eps[:n_val])
            keep = {e: w for e, w in by_ep.items()
                    if (e in val_eps) == (split == "val")}
        else:
            keep = by_ep

        if max_videos > 0 and max_videos < len(keep):
            chosen = sorted(random.Random(seed).sample(sorted(keep), max_videos))
            keep = {e: keep[e] for e in chosen}

        pairs: list[tuple[int, int, int]] = []
        for ep in sorted(keep):
            ws = keep[ep]
            for i in range(len(ws) - 2):
                pairs.append((ws[i], ws[i + 1], ws[i + 2]))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int) -> dict:
        return torch.load(self.cache_dir / self.windows[idx]["path"],
                          map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> dict:
        i_obs, i_pred, i_b = self.pairs[idx]
        obs, pred, b = self._load(i_obs), self._load(i_pred), self._load(i_b)
        assert obs["episode_id"] == pred["episode_id"] == b["episode_id"]
        out = {
            "chunk_obs":   obs["latent"],
            "chunk_pred":  pred["latent"],
            "chunk_obs_b": b["latent"],
            "pose_obs":    obs["pose"],     # (9, 6) known camera trajectory
            "pose_pred":   pred["pose"],
            "pose_b":      b["pose"],
            "vel_obs":     obs["vel"],      # (9, 6) for the offline readout
            # inert placeholders (losses disabled): keep the batch contract
            "gate_GT":     torch.zeros((), dtype=torch.float32),
            "attrs":       torch.zeros(1, 1, dtype=torch.float32),
            "slot_mask":   torch.zeros(1, dtype=torch.bool),
            "video_id":    int(obs["episode_id"]),
            "start_frame": int(obs["start_frame"]),
        }
        if self.load_frames and "frames" in obs:  # (33,128,128,3) u8 GT — for pixel PSNR
            out["frames_obs"] = obs["frames"]
        return out


__all__ = ["DroidChunkPairs"]
