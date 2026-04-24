import random

import torch
from torch.utils.data import Dataset

from .clevrer_dataset import ClevrerTripletDataset


class ClevrerSequenceWindowDataset(Dataset):
    """
    Returns short contiguous frame windows from CLEVRER videos.

    This is used to train the latent dynamics model on the same kind of
    autoregressive windows that it sees during rollout, instead of only isolated
    triplets.
    """

    def __init__(
        self,
        data_dir,
        window_length=6,
        max_frames=128,
        windows_per_video=4,
        max_videos=0,
        seed=0,
    ):
        self.base_dataset = ClevrerTripletDataset(
            data_dir=data_dir,
            video_num_frames=max(max_frames, window_length),
        )
        self.window_length = int(window_length)
        self.max_frames = int(max_frames)
        self.windows_per_video = int(windows_per_video)
        self.max_videos = int(max_videos)
        self.seed = int(seed)

        if self.window_length < 3:
            raise ValueError("window_length must be at least 3.")
        if self.windows_per_video <= 0:
            raise ValueError("windows_per_video must be > 0.")

        self.samples = self._build_index()

    def _build_index(self):
        rng = random.Random(self.seed)
        samples = []
        video_indices = list(range(self.base_dataset.num_sequences()))
        if self.max_videos > 0 and self.max_videos < len(video_indices):
            video_indices = sorted(rng.sample(video_indices, k=self.max_videos))

        for video_index in video_indices:
            full_length = self.base_dataset.get_video_sequence(
                video_index=video_index,
                max_frames=self.max_frames,
            ).shape[0]
            if full_length < self.window_length:
                continue
            max_start = full_length - self.window_length
            if max_start == 0:
                samples.append((video_index, 0))
                continue

            if self.windows_per_video >= max_start + 1:
                starts = list(range(max_start + 1))
            else:
                starts = sorted(rng.sample(range(max_start + 1), k=self.windows_per_video))
            samples.extend((video_index, start) for start in starts)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_index, start = self.samples[index]
        sequence = self.base_dataset.get_video_sequence(
            video_index=video_index,
            max_frames=self.max_frames,
        )
        window = sequence[start : start + self.window_length]
        if window.shape[0] != self.window_length:
            raise RuntimeError("Failed to extract a full CLEVRER sequence window.")
        return window


__all__ = ["ClevrerSequenceWindowDataset"]
