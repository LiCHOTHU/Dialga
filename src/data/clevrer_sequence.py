"""CLEVRER sequence dataset: ALL chunks of one video, in temporal order.

The paired dataset (clevrer_window.py) serves (obs, pred, b) triples, which is
enough for one forward-dynamics step but gives the model no way to carry state
across a video. A static-scene memory has to see chunks IN ORDER, so this
dataset makes one item = one video = (K, C, T, H, W).

Item:
    latents    : (K, C, T, H, W)  K chunks in start_frame order
    attrs      : (Kobj, A)        one-hot colour/material/shape per object slot
    slot_mask  : (Kobj,)          which slots are real objects
    speeds     : (Kobj,)          mean per-object speed over the video (GT), used
                                  to split objects into moving vs stationary
    video_id   : int
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ClevrerSequence(Dataset):
    def __init__(self, cache_dir, n_chunks: int = 4, max_videos: int = 0,
                 split: str = "train", val_frac: float = 0.1, preload: bool = False):
        self.cache_dir = Path(cache_dir)
        self.n_chunks = int(n_chunks)
        index = _build_index(self.cache_dir)

        vids = sorted(v for v, ws in index.items() if len(ws) >= n_chunks)
        # deterministic video-level split (no chunk of a val video is ever trained on)
        n_val = max(1, int(len(vids) * val_frac))
        val_vids = set(vids[::max(1, len(vids) // n_val)][:n_val])
        if split == "train":
            vids = [v for v in vids if v not in val_vids]
        elif split == "val":
            vids = [v for v in vids if v in val_vids]
        if max_videos:
            vids = vids[:max_videos]
        self.videos = [(v, sorted(index[v])[:n_chunks]) for v in vids]
        # The whole cache is ~5 GB; holding it in RAM as fp16 turns each epoch from
        # 4 x len(videos) disk reads into pure compute (this box has 60 GB).
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
        lat, attrs, slot_mask, speeds = [], None, None, None
        for _, path in ws:
            b = torch.load(self.cache_dir / path, map_location="cpu",
                           weights_only=False)
            lat.append(b["latent"].float())
            if attrs is None:
                attrs, slot_mask = b["attrs"].float(), b["slot_mask"].bool()
            if speeds is None:
                p = b["positions"].float()                  # (W, Kobj, 2)
                speeds = (p[1:] - p[:-1]).norm(dim=-1).mean(dim=0)   # (Kobj,)
        return {
            "latents": torch.stack(lat),                    # (K, C, T, H, W)
            "attrs": attrs, "slot_mask": slot_mask, "speeds": speeds,
            "video_id": torch.tensor(vid),
        }


def _build_index(cache_dir: Path) -> dict[int, list[tuple[int, str]]]:
    """video_id -> [(start_frame, relpath)]. Uses metadata.json when the cache run
    has finished; otherwise scans the shards once and memoises its own index so a
    partially-built cache is still usable."""
    meta = cache_dir / "metadata.json"
    if meta.exists():
        blob = json.loads(meta.read_text())
        # cache_wan_latents writes {"args":..., "n_windows":..., "windows":[...]};
        # tolerate a bare list too.
        rows = blob["windows"] if isinstance(blob, dict) else blob
        idx = defaultdict(list)
        for r in rows:
            idx[int(r["video_id"])].append((int(r["start_frame"]), r["path"]))
        return idx

    side = cache_dir / "seq_index.json"
    files = sorted((cache_dir / "latents").glob("*.pt"))
    if side.exists():
        cached = json.loads(side.read_text())
        if cached.get("n_files") == len(files):
            return {int(k): [tuple(x) for x in v]
                    for k, v in cached["index"].items()}
    idx = defaultdict(list)
    for f in files:
        b = torch.load(f, map_location="cpu", weights_only=False)
        idx[int(b["video_id"])].append(
            (int(b["start_frame"]), str(f.relative_to(cache_dir))))
    side.write_text(json.dumps({"n_files": len(files), "index": idx}))
    return idx


__all__ = ["ClevrerSequence"]
