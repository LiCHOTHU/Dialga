import glob
import os
import re
from functools import lru_cache

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ClevrerTripletDataset(Dataset):
    def __init__(
        self,
        data_dir="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video",
        video_num_frames=128,
    ):
        """
        Supports two CLEVRER layouts:
        1. Extracted frames: data_dir/video_00001/frame_0000.png
        2. Raw videos:      data_dir/video_00000-01000/video_00000.mp4
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"CLEVRER data directory does not exist: {data_dir}")

        self.data_dir = data_dir
        self.video_num_frames = int(video_num_frames)
        if self.video_num_frames < 3:
            raise ValueError("video_num_frames must be at least 3.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.frame_folders = self._discover_frame_folders()
        self.video_files = self._discover_video_files()

        if self.frame_folders:
            self.storage_mode = "frames"
            self.valid_triplets, self.frame_folder_triplet_ranges = self._build_frame_index()
            if not self.valid_triplets:
                raise RuntimeError(f"No valid CLEVRER triplets were found under {data_dir}")
        elif self.video_files:
            self.storage_mode = "videos"
            self.triplets_per_video = self.video_num_frames - 2
        else:
            raise RuntimeError(
                "Could not find CLEVRER frame folders or CLEVRER mp4 files under "
                f"{data_dir}."
            )

    def _discover_frame_folders(self):
        frame_folders = []
        for candidate in sorted(glob.glob(os.path.join(self.data_dir, "*"))):
            if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "frame_*.png")):
                frame_folders.append(candidate)
        return frame_folders

    def _discover_video_files(self):
        return sorted(
            path
            for path in glob.glob(
                os.path.join(self.data_dir, "**", "video_*.mp4"), recursive=True
            )
            if os.path.isfile(path)
        )

    def _build_frame_index(self):
        valid_triplets = []
        frame_folder_triplet_ranges = []
        frame_pattern = re.compile(r"frame_(\d+)\.png$")

        for vid_path in self.frame_folders:
            start_idx = len(valid_triplets)
            frame_paths = sorted(glob.glob(os.path.join(vid_path, "frame_*.png")))
            indexed_frames = {}

            for frame_path in frame_paths:
                match = frame_pattern.search(os.path.basename(frame_path))
                if match is None:
                    continue
                indexed_frames[int(match.group(1))] = frame_path

            if len(indexed_frames) < 3:
                continue

            for frame_id in sorted(indexed_frames):
                prev_path = indexed_frames.get(frame_id - 1)
                curr_path = indexed_frames.get(frame_id)
                next_path = indexed_frames.get(frame_id + 1)
                if prev_path and curr_path and next_path:
                    valid_triplets.append((prev_path, curr_path, next_path))

            end_idx = len(valid_triplets)
            frame_folder_triplet_ranges.append((start_idx, end_idx))

        return valid_triplets, frame_folder_triplet_ranges

    @staticmethod
    @lru_cache(maxsize=8)
    def _cached_full_video_frames_torchvision(video_path):
        from torchvision.io import read_video

        frames, _, _ = read_video(video_path, pts_unit="sec")
        if frames.shape[0] == 0:
            raise RuntimeError(f"No frames were decoded from {video_path}.")
        return tuple(frame.cpu().numpy() for frame in frames)

    def _read_triplet_from_video_with_torchvision(self, video_path, center_idx):
        frames = self._cached_full_video_frames_torchvision(video_path)
        if len(frames) <= center_idx + 1:
            raise IndexError(
                f"Video {video_path} has only {len(frames)} frames, but index "
                f"{center_idx} was requested."
            )
        triplet = frames[center_idx - 1 : center_idx + 2]
        return [Image.fromarray(frame) for frame in triplet]

    def _read_full_video_with_torchvision(self, video_path):
        frames = self._cached_full_video_frames_torchvision(video_path)
        return [Image.fromarray(frame) for frame in frames]

    def _read_triplet_from_video_with_imageio(self, video_path, center_idx):
        try:
            import imageio
        except ImportError as exc:
            raise ImportError("imageio is not available to decode CLEVRER videos.") from exc

        reader = imageio.get_reader(video_path)
        try:
            frames = [
                Image.fromarray(reader.get_data(center_idx - 1)),
                Image.fromarray(reader.get_data(center_idx)),
                Image.fromarray(reader.get_data(center_idx + 1)),
            ]
        finally:
            reader.close()
        return frames

    def _read_full_video_with_imageio(self, video_path):
        try:
            import imageio
        except ImportError as exc:
            raise ImportError("imageio is not available to decode CLEVRER videos.") from exc

        reader = imageio.get_reader(video_path)
        try:
            frames = [Image.fromarray(frame) for frame in reader]
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"No frames were decoded from {video_path}.")
        return frames

    def _load_video_triplet(self, video_path, center_idx):
        try:
            return self._read_triplet_from_video_with_torchvision(video_path, center_idx)
        except Exception:
            try:
                return self._read_triplet_from_video_with_imageio(video_path, center_idx)
            except Exception as imageio_error:
                raise RuntimeError(
                    "Failed to decode CLEVRER mp4 data. Install a working video backend "
                    "for torchvision, or install `imageio-ffmpeg`, or pre-extract frames."
                ) from imageio_error

    def _load_video_sequence(self, video_path):
        try:
            return self._read_full_video_with_torchvision(video_path)
        except Exception:
            try:
                return self._read_full_video_with_imageio(video_path)
            except Exception as imageio_error:
                raise RuntimeError(
                    "Failed to decode a full CLEVRER video sequence. Install a working "
                    "video backend for torchvision, or install `imageio-ffmpeg`, or "
                    "pre-extract frames."
                ) from imageio_error

    def num_sequences(self):
        if self.storage_mode == "frames":
            return len(self.frame_folders)
        return len(self.video_files)

    def get_video_sequence(self, video_index=0, max_frames=None):
        """
        Returns a full transformed video tensor with shape (T, C, H, W).
        This is used for visualization/evaluation rather than training.
        """
        total_sequences = self.num_sequences()
        if total_sequences == 0:
            raise RuntimeError("No CLEVRER videos are available for visualization.")

        video_index = max(0, min(int(video_index), total_sequences - 1))

        if self.storage_mode == "frames":
            frame_paths = sorted(
                glob.glob(os.path.join(self.frame_folders[video_index], "frame_*.png"))
            )
            if max_frames is not None:
                frame_paths = frame_paths[:max_frames]
            if not frame_paths:
                raise RuntimeError(
                    f"No frames were found under {self.frame_folders[video_index]}."
                )
            frames = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as frame_img:
                    frames.append(self.transform(frame_img.convert("RGB")))
            return torch.stack(frames, dim=0)

        frames = self._load_video_sequence(self.video_files[video_index])
        if max_frames is not None:
            frames = frames[:max_frames]
        return torch.stack(
            [self.transform(frame.convert("RGB")) for frame in frames],
            dim=0,
        )

    def get_triplet_indices_for_video(self, video_index=0):
        """
        Returns dataset indices covering one full video's valid triplet trajectory.
        """
        total_sequences = self.num_sequences()
        if total_sequences == 0:
            raise RuntimeError("No CLEVRER videos are available for overfit indexing.")

        video_index = max(0, min(int(video_index), total_sequences - 1))

        if self.storage_mode == "frames":
            start_idx, end_idx = self.frame_folder_triplet_ranges[video_index]
            if end_idx <= start_idx:
                raise RuntimeError(
                    f"Frame folder {self.frame_folders[video_index]} does not contain any valid triplets."
                )
            return list(range(start_idx, end_idx))

        start_idx = video_index * self.triplets_per_video
        end_idx = start_idx + self.triplets_per_video
        return list(range(start_idx, end_idx))

    def __len__(self):
        if self.storage_mode == "frames":
            return len(self.valid_triplets)
        return len(self.video_files) * self.triplets_per_video

    def __getitem__(self, idx):
        if self.storage_mode == "frames":
            prev_path, curr_path, next_path = self.valid_triplets[idx]
            with Image.open(prev_path) as prev_img:
                o_prev = self.transform(prev_img.convert("RGB"))
            with Image.open(curr_path) as curr_img:
                o_curr = self.transform(curr_img.convert("RGB"))
            with Image.open(next_path) as next_img:
                o_next = self.transform(next_img.convert("RGB"))
            return o_prev, o_curr, o_next

        video_idx = idx // self.triplets_per_video
        center_idx = (idx % self.triplets_per_video) + 1
        video_path = self.video_files[video_idx]
        prev_img, curr_img, next_img = self._load_video_triplet(video_path, center_idx)
        o_prev = self.transform(prev_img.convert("RGB"))
        o_curr = self.transform(curr_img.convert("RGB"))
        o_next = self.transform(next_img.convert("RGB"))
        return o_prev, o_curr, o_next
