"""LiberoChunkPairs — paired-chunk sampler for LIBERO-90.

Parallel of src/data/clevrer_window.py for the Wan-VAE LIBERO cache emitted by
scripts/encode_libero_wan.py. The training pipeline (train_v5*.py) is agnostic
to which dataset it consumes — both yield the same chunk_obs/chunk_pred/
chunk_obs_b/gate_GT keys.

Per-episode chunks are formed via W=33 non-overlapping windows during caching.
Episodes have variable chunk counts (mode 3-4, max 11). For each episode with
C >= 3 chunks, this sampler emits (C-1) pairs:

    pair i:  obs = chunks[i], pred = chunks[i+1], b = random chunks[j]
                                                       (j != i, i+1)

`b` is chosen ONCE at __init__ via a deterministic RNG, so two epochs see the
same (obs, pred, b) triple — matches the CLEVRER recipe.

Splits ----------------------------------------------------------------------
Two split axes:
  - `task_holdout_ids`: a fixed list of task_ids reserved for a cross-task
    evaluation probe (never appear in train / val).
  - within the remaining tasks, train / val split by `episode_id` modulo
    `val_every_k` (deterministic; no info leak between train and val).

Returned dict (chunk_collate-compatible with src/data/clevrer_window.py):
    chunk_obs            : (48, 9, 8, 8)   observed chunk
    chunk_pred           : (48, 9, 8, 8)   next chunk (target of L_fwd)
    chunk_obs_b          : (48, 9, 8, 8)   InfoNCE positive (other chunk, same ep)
    actions_obs          : (33, 7)         per-frame action over obs window
    actions_pred         : (33, 7)         per-frame action over pred window
    gripper_state_obs    : (33,)           = actions_obs[:, 6]
    gate_GT              : scalar float    1 if gripper flips inside pred chunk
                                           (drop-in for CLEVRER's collision-gate)
    task_id              : int (0..89)
    episode_id           : int (within-task, 0..49)
    global_episode_id    : int (0..4499)
    start_frame          : int (of obs)
    prompt               : str             natural-language task description
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _make_holdout_task_ids(n_tasks: int, n_holdout: int, seed: int) -> list[int]:
    """Deterministic split: shuffle task_ids by seed, take the first n_holdout."""
    rng = random.Random(seed)
    ids = list(range(n_tasks))
    rng.shuffle(ids)
    return sorted(ids[:n_holdout])


class LiberoChunkPairs(Dataset):
    def __init__(
        self,
        cache_dir,
        split: str = "train",
        n_holdout_tasks: int = 10,
        task_holdout_seed: int = 0,
        val_every_k: int = 10,
        pair_seed: int = 42,
        max_episodes: int = 0,
        require_chunks_per_video: int = 3,
    ):
        """
        split:
            "train"   — non-holdout tasks, episode_id % val_every_k != 0
            "val"     — non-holdout tasks, episode_id % val_every_k == 0
            "heldout" — only the held-out task_ids (for cross-task action probe)
            "all"     — everything (for debug / full caching)
        """
        assert split in ("train", "val", "heldout", "all"), f"bad split: {split}"
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        windows = meta["windows"]
        n_tasks = len(meta["task_id_map"])
        self.cache_dir = cache_dir
        self.windows = windows
        self.task_id_map = meta["task_id_map"]
        self.task_id_to_name = {v: k for k, v in self.task_id_map.items()}
        self.holdout_task_ids = set(_make_holdout_task_ids(
            n_tasks, n_holdout_tasks, task_holdout_seed))

        # Group windows by global_episode_id; sort within episode by start_frame.
        by_ep: dict[int, list[int]] = {}
        ep_meta: dict[int, dict] = {}
        for i, w in enumerate(windows):
            ep = int(w["global_episode_id"])
            by_ep.setdefault(ep, []).append(i)
            ep_meta.setdefault(ep, {
                "task_id": int(w["task_id"]),
                "episode_id": int(w["episode_id"]),
            })
        for ep in by_ep:
            by_ep[ep] = sorted(by_ep[ep], key=lambda i: int(windows[i]["start_frame"]))

        # Apply split filter.
        kept = {}
        for ep, idxs in by_ep.items():
            tid = ep_meta[ep]["task_id"]
            ep_local = ep_meta[ep]["episode_id"]
            in_holdout = tid in self.holdout_task_ids
            if split == "heldout":
                if not in_holdout: continue
            elif split == "all":
                pass
            else:
                if in_holdout: continue
                is_val_ep = (ep_local % val_every_k == 0)
                if split == "train" and is_val_ep: continue
                if split == "val" and not is_val_ep: continue
            if len(idxs) < require_chunks_per_video: continue
            kept[ep] = idxs

        if max_episodes > 0 and max_episodes < len(kept):
            ordered_eps = sorted(kept.keys())
            rng_cap = random.Random(pair_seed)
            chosen = sorted(rng_cap.sample(ordered_eps, max_episodes))
            kept = {ep: kept[ep] for ep in chosen}

        # Emit (C-1) consecutive triples per episode. b chosen deterministically
        # from chunks not in {obs, pred}.
        pair_rng = random.Random(pair_seed)
        pairs: list[tuple[int, int, int, int]] = []  # (i_obs, i_pred, i_b, ep)
        for ep in sorted(kept.keys()):
            ws = kept[ep]
            C = len(ws)
            for i in range(C - 1):
                others = [ws[j] for j in range(C) if j != i and j != i + 1]
                i_b = pair_rng.choice(others) if others else ws[i + 1]  # degenerate fallback
                pairs.append((ws[i], ws[i + 1], i_b, ep))
        self.pairs = pairs
        self.kept_episodes = list(kept.keys())

        # Probe one blob for shape sanity.
        probe = torch.load(cache_dir / windows[0]["path"], map_location="cpu", weights_only=False)
        self.T_lat = int(probe["latent"].shape[1])
        self.W_pix = int(probe["actions"].shape[0])

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int) -> dict:
        w = self.windows[idx]
        return torch.load(self.cache_dir / w["path"], map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> dict:
        i_obs, i_pred, i_b, ep = self.pairs[idx]
        obs = self._load(i_obs)
        pred = self._load(i_pred)
        b = self._load(i_b)
        assert obs["global_episode_id"] == pred["global_episode_id"] == b["global_episode_id"], (
            "paired-chunk sampler broke episode invariant"
        )
        return {
            "chunk_obs":   obs["latent"],
            "chunk_pred":  pred["latent"],
            "chunk_obs_b": b["latent"],
            "actions_obs":       obs["actions"],
            "actions_pred":      pred["actions"],
            "gripper_state_obs": obs["gripper_state"],
            "gate_GT":     pred["gripper_event"].float(),
            "attrs":       torch.zeros(1, 1, dtype=torch.float32),
            "slot_mask":   torch.zeros(1, dtype=torch.bool),
            "task_id":     int(obs["task_id"]),
            "episode_id":  int(obs["episode_id"]),
            "global_episode_id": int(obs["global_episode_id"]),
            "start_frame": int(obs["start_frame"]),
            "prompt":      obs["prompt"],
        }


class LiberoChunkPairsWithPixels(LiberoChunkPairs):
    """Same as LiberoChunkPairs but each item also carries 3 pixel chunks of
    shape (33, 3, 128, 128) in [-1, 1] for L_pixel / VAE-unfreeze experiments.

    Loads frames from the libero_90 mp4 dir referenced by `visual_input` in
    libero_90.jsonl. A small per-process LRU caches decoded videos so the 3
    chunks of one episode in nearby batches don't re-decode the whole mp4.
    """

    def __init__(self, cache_dir, libero_root, image_size: int = 128, **kwargs):
        super().__init__(cache_dir, **kwargs)
        from pathlib import Path as _P
        self.libero_root = _P(libero_root)
        self.image_size = int(image_size)
        # gid -> mp4 relative path (from libero_90.jsonl)
        ep_video: dict[int, str] = {}
        with (self.libero_root / "libero_90.jsonl").open() as f:
            for gid, line in enumerate(f):
                r = json.loads(line)
                ep_video[gid] = r["visual_input"]
        self.ep_video = ep_video

    def _load_chunk_pixels(self, gid: int, start_frame: int) -> torch.Tensor:
        """Returns (W_pix, 3, 128, 128) in [-1, 1]."""
        from torchvision.io import read_video
        path = str(self.libero_root / self.ep_video[gid])
        if getattr(self, "_video_path", None) != path:
            v, _, _ = read_video(path, pts_unit="sec")
            self._video = v          # (T, 128, 128, 3) uint8
            self._video_path = path
        frames = self._video[start_frame:start_frame + self.W_pix]
        f = frames.float() / 127.5 - 1.0
        return f.permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        i_obs, i_pred, i_b, ep = self.pairs[idx]
        s_obs  = int(self.windows[i_obs ]["start_frame"])
        s_pred = int(self.windows[i_pred]["start_frame"])
        s_b    = int(self.windows[i_b   ]["start_frame"])
        sample["pix_obs"]   = self._load_chunk_pixels(ep, s_obs)
        sample["pix_pred"]  = self._load_chunk_pixels(ep, s_pred)
        sample["pix_obs_b"] = self._load_chunk_pixels(ep, s_b)
        return sample


def libero_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif isinstance(v0, int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        elif isinstance(v0, float):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        else:
            out[k] = [b[k] for b in batch]
    return out


__all__ = ["LiberoChunkPairs", "LiberoChunkPairsWithPixels", "libero_collate"]
