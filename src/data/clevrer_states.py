import json
import re
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset


COLOR_VOCAB = [
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "purple",
    "cyan",
    "yellow",
]
MATERIAL_VOCAB = ["rubber", "metal"]
SHAPE_VOCAB = ["sphere", "cube", "cylinder"]


def _one_hot(value, vocabulary):
    vector = torch.zeros(len(vocabulary), dtype=torch.float32)
    if value not in vocabulary:
        raise ValueError(f"Unknown categorical value '{value}'. Expected one of {vocabulary}.")
    vector[vocabulary.index(value)] = 1.0
    return vector


def _encode_attributes(object_properties, max_objects):
    attr_dim = len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)
    attrs = torch.zeros(max_objects, attr_dim, dtype=torch.float32)
    object_ids = torch.full((max_objects,), -1, dtype=torch.long)

    for slot_idx, prop in enumerate(sorted(object_properties, key=lambda item: item["object_id"])):
        if slot_idx >= max_objects:
            raise ValueError(
                f"Scene contains {len(object_properties)} objects but max_objects={max_objects}."
            )
        attrs[slot_idx] = torch.cat(
            [
                _one_hot(prop["color"], COLOR_VOCAB),
                _one_hot(prop["material"], MATERIAL_VOCAB),
                _one_hot(prop["shape"], SHAPE_VOCAB),
            ]
        )
        object_ids[slot_idx] = int(prop["object_id"])

    return attrs, object_ids


def _centered_difference(values):
    velocities = torch.zeros_like(values)
    if values.shape[0] < 2:
        return velocities

    velocities[1:-1] = 0.5 * (values[2:] - values[:-2])
    velocities[0] = values[1] - values[0]
    velocities[-1] = values[-1] - values[-2]
    return velocities


def _annotation_subdir(scene_index):
    lower = (int(scene_index) // 1000) * 1000
    upper = lower + 1000
    return f"annotation_{lower:05d}-{upper:05d}"


def _video_subdir(video_filename):
    match = re.search(r"video_(\d+)\.mp4$", video_filename)
    if match is None:
        return None
    video_index = int(match.group(1))
    lower = (video_index // 1000) * 1000
    upper = lower + 1000
    return f"video_{lower:05d}-{upper:05d}"


class ClevrerStateDataset(Dataset):
    """
    Returns fixed-length trajectory windows from CLEVRER annotations.

    The public CLEVRER annotations expose 3D object locations and velocities but
    not camera matrices, so this loader currently uses a configurable 2D slice of
    the world coordinates (`world_xy` by default). The image-plane projection hook
    can be added once camera calibration is available.
    """

    def __init__(
        self,
        annotation_dir,
        split="train",
        traj_len=33,
        stride=4,
        frames_per_video=128,
        max_objects=8,
        coordinate_mode="world_xy",
        use_inside_camera_view_mask=True,
        video_dir=None,
    ):
        self.annotation_dir = Path(annotation_dir)
        self.split = split
        self.traj_len = int(traj_len)
        self.stride = int(stride)
        self.frames_per_video = int(frames_per_video)
        self.max_objects = int(max_objects)
        self.coordinate_mode = coordinate_mode
        self.use_inside_camera_view_mask = bool(use_inside_camera_view_mask)
        self.video_dir = Path(video_dir) if video_dir else None
        self.attr_dim = len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)

        if self.traj_len < 2:
            raise ValueError("traj_len must be at least 2.")
        if self.frames_per_video < self.traj_len:
            raise ValueError(
                f"frames_per_video={self.frames_per_video} must be >= traj_len={self.traj_len}."
            )
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")

        self.annotation_paths = self._discover_annotation_files()
        if not self.annotation_paths:
            raise FileNotFoundError(
                "No CLEVRER annotation json files were found under "
                f"{self.annotation_dir}. Download the annotation split into a path like "
                f"$SCRATCH/Dialga/CLEVRER/annotations/{self.split}/annotation_00000-01000/*.json."
            )

        self.samples = self._build_window_index()

    def _discover_annotation_files(self):
        split_root = self.annotation_dir / self.split
        patterns = [
            split_root.glob("annotation_*/*.json"),
            split_root.glob("*.json"),
            self.annotation_dir.glob("annotation_*/*.json"),
            self.annotation_dir.glob("*.json"),
        ]
        if self.split:
            patterns.append(self.annotation_dir.glob(f"{self.split}/annotation_*/*.json"))
        discovered = []
        seen = set()
        for pattern in patterns:
            for path in sorted(pattern):
                resolved = path.resolve()
                if resolved not in seen:
                    discovered.append(path)
                    seen.add(resolved)
        return discovered

    def _build_window_index(self):
        windows = []
        last_start = self.frames_per_video - self.traj_len
        for annotation_path in self.annotation_paths:
            for start_idx in range(0, last_start + 1, self.stride):
                windows.append((annotation_path, start_idx))
        return windows

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_annotation(annotation_path):
        with Path(annotation_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _extract_2d_location(self, location):
        if self.coordinate_mode == "world_xy":
            return float(location[0]), float(location[1])
        if self.coordinate_mode == "world_xz":
            return float(location[0]), float(location[2])
        raise ValueError(
            f"Unsupported coordinate_mode='{self.coordinate_mode}'. "
            "Use 'world_xy' or 'world_xz'."
        )

    def _build_scene_tensors(self, annotation):
        object_attrs, object_ids = _encode_attributes(
            annotation["object_property"],
            max_objects=self.max_objects,
        )
        object_id_to_slot = {
            int(object_id): slot_idx
            for slot_idx, object_id in enumerate(object_ids.tolist())
            if object_id >= 0
        }

        num_frames = min(len(annotation["motion_trajectory"]), self.frames_per_video)
        positions = torch.zeros(num_frames, self.max_objects, 2, dtype=torch.float32)
        visibility_mask = torch.zeros(num_frames, self.max_objects, dtype=torch.bool)
        slot_mask = object_ids >= 0

        for frame_idx, frame_info in enumerate(annotation["motion_trajectory"][:num_frames]):
            for obj_info in frame_info["objects"]:
                object_id = int(obj_info["object_id"])
                if object_id not in object_id_to_slot:
                    continue
                slot_idx = object_id_to_slot[object_id]
                positions[frame_idx, slot_idx] = torch.tensor(
                    self._extract_2d_location(obj_info["location"]),
                    dtype=torch.float32,
                )
                visibility_mask[frame_idx, slot_idx] = bool(obj_info["inside_camera_view"])

        velocities = _centered_difference(positions)
        velocities = velocities * slot_mask.to(velocities.dtype).view(1, self.max_objects, 1)

        if self.use_inside_camera_view_mask:
            mask = visibility_mask
        else:
            mask = slot_mask.view(1, self.max_objects).expand(num_frames, -1)

        # Collision tensor: (num_frames, max_objects) bool — slot i is involved in a
        # collision at frame f if any annotated collision has frame_id == f and
        # involves the corresponding object_id.
        collision_slot = torch.zeros(num_frames, self.max_objects, dtype=torch.bool)
        # Per-frame pair list (for impulse training); a list of length num_frames,
        # each entry is a list of (slot_i, slot_j) tuples.
        collision_pairs_per_frame = [[] for _ in range(num_frames)]
        for event in annotation.get("collision", []) or []:
            f = int(event.get("frame_id", -1))
            if f < 0 or f >= num_frames:
                continue
            ids = event.get("object_ids", [])
            if len(ids) != 2:
                continue
            i_id, j_id = int(ids[0]), int(ids[1])
            if i_id not in object_id_to_slot or j_id not in object_id_to_slot:
                continue
            si = object_id_to_slot[i_id]
            sj = object_id_to_slot[j_id]
            collision_slot[f, si] = True
            collision_slot[f, sj] = True
            collision_pairs_per_frame[f].append((si, sj))

        return (positions, velocities, mask, slot_mask, object_attrs,
                object_ids, collision_slot, collision_pairs_per_frame)

    def _resolve_video_path(self, annotation):
        if self.video_dir is None:
            return None
        subdir = _video_subdir(annotation["video_filename"])
        if subdir is None:
            return str(self.video_dir / annotation["video_filename"])
        return str(self.video_dir / subdir / annotation["video_filename"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        annotation_path, start_idx = self.samples[index]
        annotation = self._load_annotation(str(annotation_path))
        (positions, velocities, mask, slot_mask, object_attrs, object_ids,
         collision_slot, collision_pairs_per_frame) = self._build_scene_tensors(annotation)

        end_idx = start_idx + self.traj_len
        # Per-window list of collision pair tuples with frame indices remapped
        # to be window-local (0..traj_len-1).
        window_collisions = []
        for f in range(start_idx, end_idx):
            for (si, sj) in collision_pairs_per_frame[f]:
                window_collisions.append((f - start_idx, int(si), int(sj)))

        return {
            "positions": positions[start_idx:end_idx],
            "velocities": velocities[start_idx:end_idx],
            "mask": mask[start_idx:end_idx],
            "slot_mask": slot_mask.clone(),
            "object_attrs": object_attrs.clone(),
            "object_ids": object_ids.clone(),
            "collision_mask": collision_slot[start_idx:end_idx].clone(),  # (W, K) bool
            "collisions": window_collisions,                              # list[(t, i, j)]
            "scene_index": int(annotation["scene_index"]),
            "video_filename": annotation["video_filename"],
            "video_path": self._resolve_video_path(annotation),
            "annotation_path": str(annotation_path),
            "start_frame": start_idx,
            "end_frame": end_idx - 1,
        }
