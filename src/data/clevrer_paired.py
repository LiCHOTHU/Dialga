"""
Combined frames + states dataset for the slot-Lagrangian pipeline.

Each item is one window of W consecutive frames from a CLEVRER video, paired
with the per-object ground-truth state from the matching annotation:

    frames     : (W, 3, H, W) float in [-1, 1]
    positions  : (W, K, 2)    world XY positions (zero-filled for unused slots)
    velocities : (W, K, 2)    world XY velocities (zero-filled for unused slots)
    visibility : (W, K)       bool, inside_camera_view per frame per slot
    slot_mask  : (K,)         bool, which slots correspond to real objects in the scene
    attrs      : (K, A)       one-hot color/material/shape per slot
    object_ids : (K,)         int64, object_id per slot (-1 for unused)
    video_id   : int          scene_index
    start_frame: int          first frame index in the video

The slot ordering matches sorted-by-object_id, identical to ClevrerStateDataset.
"""

from __future__ import annotations

import glob
import os
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .clevrer_states import ClevrerStateDataset


def _resolve_frame_provider(data_dir):
    """Decide whether `data_dir` holds extracted frame folders or mp4 files."""
    if any(os.path.isdir(p) and glob.glob(os.path.join(p, "frame_*.png"))
           for p in glob.glob(os.path.join(data_dir, "*"))):
        return "frames"
    if glob.glob(os.path.join(data_dir, "**", "video_*.mp4"), recursive=True):
        return "videos"
    raise FileNotFoundError(
        f"Could not find CLEVRER frame folders or mp4 files under {data_dir}."
    )


class ClevrerPairedDataset(Dataset):
    """
    Iterate over (window, state) pairs.

    Built on top of `ClevrerStateDataset` for the annotation side; the frames
    side reuses the same video_path the state dataset already resolves.
    """

    def __init__(
        self,
        data_dir,
        annotation_dir,
        split="train",
        window_length=6,
        frames_per_video=128,
        windows_per_video=4,
        max_videos=0,
        max_objects=8,
        coordinate_mode="world_xy",
        image_size=128,
        seed=0,
        deterministic_starts=None,
    ):
        self.data_dir = data_dir
        self.window_length = int(window_length)
        self.frames_per_video = int(frames_per_video)
        self.windows_per_video = int(windows_per_video)
        self.max_videos = int(max_videos)
        self.image_size = int(image_size)
        self.seed = int(seed)
        # If provided, list of exact start_frame offsets to emit per video
        # (e.g. [0, 33, 66] for v5.1). Overrides windows_per_video random sampling.
        self.deterministic_starts = (
            tuple(int(s) for s in deterministic_starts)
            if deterministic_starts is not None else None
        )
        if self.window_length < 3:
            raise ValueError("window_length must be at least 3 (DEL needs triples).")

        self.state_dataset = ClevrerStateDataset(
            annotation_dir=annotation_dir,
            split=split,
            traj_len=self.window_length,
            stride=1,
            frames_per_video=self.frames_per_video,
            max_objects=int(max_objects),
            coordinate_mode=coordinate_mode,
            use_inside_camera_view_mask=True,
            video_dir=data_dir,
        )
        self.attr_dim = self.state_dataset.attr_dim
        self.max_objects = self.state_dataset.max_objects

        self.frame_provider = _resolve_frame_provider(data_dir)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = self._build_index()

    def _build_index(self):
        import random
        rng = random.Random(self.seed)

        # Group state-dataset windows by annotation_path (= one video).
        per_video = {}
        for win_idx, (ann_path, start) in enumerate(self.state_dataset.samples):
            per_video.setdefault(str(ann_path), []).append((win_idx, start))

        video_keys = sorted(per_video.keys())
        if self.max_videos > 0 and self.max_videos < len(video_keys):
            video_keys = sorted(rng.sample(video_keys, k=self.max_videos))

        samples = []
        for key in video_keys:
            entries = per_video[key]
            if self.deterministic_starts is not None:
                by_start = {int(s): w for w, s in entries}
                chosen = []
                for s in self.deterministic_starts:
                    if s in by_start:
                        chosen.append((by_start[s], s))
                    # silently skip if a requested start isn't valid for this video
            elif self.windows_per_video >= len(entries):
                chosen = entries
            else:
                chosen = sorted(rng.sample(entries, k=self.windows_per_video))
            for win_idx, _start in chosen:
                samples.append(win_idx)
        return samples

    @staticmethod
    @lru_cache(maxsize=4)
    def _read_full_video_frames(video_path):
        try:
            from torchvision.io import read_video
            frames, _, _ = read_video(video_path, pts_unit="sec")
            if frames.shape[0] > 0:
                return tuple(frame.cpu().numpy() for frame in frames)
        except Exception:
            pass
        import imageio.v2 as imageio
        reader = imageio.get_reader(video_path)
        try:
            frames = [frame for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}.")
        return tuple(frames)

    def _load_frames_window(self, video_path, start, end):
        """Returns a list of PIL.Image for indices [start, end)."""
        if self.frame_provider == "frames":
            video_id = re.search(r"video_(\d+)\.mp4$", video_path).group(1)
            folder = os.path.join(self.data_dir, f"video_{int(video_id):05d}")
            paths = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
            paths = paths[start:end]
            return [Image.open(p).convert("RGB") for p in paths]

        all_frames = self._read_full_video_frames(video_path)
        chunk = all_frames[start:end]
        return [Image.fromarray(f).convert("RGB") for f in chunk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        win_idx = self.samples[index]
        state = self.state_dataset[win_idx]
        start = int(state["start_frame"])
        end = start + self.window_length
        video_path = state["video_path"]
        if video_path is None:
            raise RuntimeError("video_path is None; ClevrerPairedDataset needs a video_dir.")

        pil_frames = self._load_frames_window(video_path, start, end)
        if len(pil_frames) != self.window_length:
            raise RuntimeError(
                f"Expected {self.window_length} frames for window {win_idx}, got {len(pil_frames)}."
            )
        frames = torch.stack([self.transform(f) for f in pil_frames], dim=0)

        return {
            "frames": frames,
            "positions": state["positions"],
            "velocities": state["velocities"],
            "visibility": state["mask"],
            "slot_mask": state["slot_mask"],
            "attrs": state["object_attrs"],
            "object_ids": state["object_ids"],
            "collision_mask": state["collision_mask"],     # (W, K) bool
            "collisions": state["collisions"],             # list[(t,i,j)]
            "video_id": int(state["scene_index"]),
            "start_frame": start,
        }


def paired_collate(batch):
    """Stack tensors; keep video_id/start_frame as 1D long tensors."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        elif isinstance(batch[0][k], int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            out[k] = [b[k] for b in batch]
    return out


__all__ = ["ClevrerPairedDataset", "paired_collate"]
