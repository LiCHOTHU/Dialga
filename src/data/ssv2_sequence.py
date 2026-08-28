"""SSv2 sequence dataset: all K windows of one clip, in temporal order.

Mirrors ClevrerSequence so the same trainer serves both. SSv2 has no attribute,
position or pose labels, so the only ground truth carried through is the action
class for the frozen-feature probe.

Why SSv2 is the harder and more honest testbed for a static-scene memory:
  * the camera is HANDHELD and genuinely moves, so the scene really does leave and
    re-enter the frame -- unlike CLEVRER, where the camera is fixed and the static
    content is nearly identical across clips;
  * but the pose is UNKNOWN, so an explicit pose-warped memory cannot be used. The
    patch memory has to align by attention alone. That is the pose-free question:
    does retrieve-and-compose still pay when nothing tells you where things went?
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SSv2Sequence(Dataset):
    def __init__(self, cache_dir, n_chunks: int = 4, max_videos: int = 0,
                 split: str = "train", val_frac: float = 0.15,
                 preload: bool = False):
        self.cache_dir = Path(cache_dir)
        self.n_chunks = int(n_chunks)
        index = _build_index(self.cache_dir)

        vids = sorted(v for v, ws in index.items() if len(ws) >= n_chunks)
        n_val = max(1, int(len(vids) * val_frac))
        val_vids = set(vids[:: max(1, len(vids) // n_val)][:n_val])
        if split == "train":
            vids = [v for v in vids if v not in val_vids]
        elif split == "val":
            vids = [v for v in vids if v in val_vids]
        if max_videos:
            vids = vids[:max_videos]
        self.videos = [(v, sorted(index[v])[:n_chunks]) for v in vids]

        self.cache = None
        if preload:
            self.cache = [self._read(i) for i in range(len(self.videos))]
            for c in self.cache:
                c["latents"] = c["latents"].half()

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, i: int) -> dict:
        if self.cache is not None:
            d = dict(self.cache[i])
            d["latents"] = d["latents"].float()
            return d
        return self._read(i)

    def _read(self, i: int) -> dict:
        vid, ws = self.videos[i]
        lat, label = [], -1
        for _, path in ws:
            b = torch.load(self.cache_dir / path, map_location="cpu",
                           weights_only=False)
            lat.append(b["latent"].float())
            label = int(b.get("label_id", -1))
        return {"latents": torch.stack(lat),
                "label_id": torch.tensor(label),
                "start_frames": torch.tensor([int(sf) for sf, _ in ws]),
                "video_id": torch.tensor(vid)}


def _build_index(cache_dir: Path) -> dict[int, list[tuple[int, str]]]:
    meta = cache_dir / "metadata.json"
    idx = defaultdict(list)
    if meta.exists():
        blob = json.loads(meta.read_text())
        rows = blob["windows"] if isinstance(blob, dict) else blob
        for r in rows:
            idx[int(r["video_id"])].append((int(r["start_frame"]), r["path"]))
        return idx
    for f in sorted((cache_dir / "latents").glob("*.pt")):
        b = torch.load(f, map_location="cpu", weights_only=False)
        idx[int(b["video_id"])].append(
            (int(b["start_frame"]), str(f.relative_to(cache_dir))))
    return idx


__all__ = ["SSv2Sequence"]
