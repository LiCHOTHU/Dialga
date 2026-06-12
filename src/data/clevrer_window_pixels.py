"""ClevrerChunkPairsWithPixels — paired-chunk sampler that also returns raw
33-frame pixel chunks alongside the cached latents.

Used for the v5.1.2 Wan-VAE unfreeze experiments (Exp 2/3/4):
  - Exp 2 (vae-enc unfrozen): need raw pixels to re-encode through trainable
    Wan-VAE encoder during training.
  - Exp 3 (vae-dec unfrozen): need raw pixels as the pixel-space target for
    L_pixel = MSE(VAE-dec(pred_latent), pixels).
  - Exp 4 (both): both reasons.

Preprocessing must match scripts/cache_wan_latents.py exactly so re-encoded
latents are comparable to cached ones:
    PIL.Image -> Resize(image_size, image_size) -> ToTensor() -> Normalize([0.5]*3, [0.5]*3)

Mp4s live at:
    <video_dir>/video_AAAAA-BBBBB/video_NNNNN.mp4
where AAAAA-BBBBB is a 1000-video range like 00000-01000.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .clevrer_window import ClevrerChunkPairs


@lru_cache(maxsize=64)
def _read_full_video_frames(video_path: str) -> tuple:
    """Cache the decoded frames of one mp4 — small in PIL form, big speed win
    when 3 chunks of the same video are sampled in nearby steps."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(video_path)
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}.")
    return tuple(frames)


def _resolve_video_path(video_dir: Path, video_id: int) -> Path:
    """Return /<video_dir>/video_AAAAA-BBBBB/video_NNNNN.mp4 — the CLEVRER
    train_video layout. AAAAA is video_id rounded down to nearest 1000."""
    base = (video_id // 1000) * 1000
    folder = f"video_{base:05d}-{base + 1000:05d}"
    name = f"video_{video_id:05d}.mp4"
    return video_dir / folder / name


class ClevrerChunkPairsWithPixels(ClevrerChunkPairs):
    """Same as ClevrerChunkPairs but each item also carries 3 pixel chunks of
    shape (W_pix=33, 3, image_size, image_size) in [-1, 1]."""

    def __init__(self, cache_dir, video_dir, image_size: int = 128, **kwargs):
        super().__init__(cache_dir, **kwargs)
        self.video_dir = Path(video_dir)
        self.image_size = int(image_size)
        # MUST match cache_wan_latents.py preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_chunk_pixels(self, video_id: int, start_frame: int) -> torch.Tensor:
        """Returns (W_pix, 3, H, W) in [-1, 1]."""
        path = _resolve_video_path(self.video_dir, video_id)
        all_frames = _read_full_video_frames(str(path))
        chunk = all_frames[start_frame: start_frame + self.W_pix]
        if len(chunk) != self.W_pix:
            raise RuntimeError(
                f"vid={video_id} start={start_frame}: expected {self.W_pix} frames, "
                f"got {len(chunk)} (total in video: {len(all_frames)})"
            )
        pil = [Image.fromarray(f).convert("RGB") for f in chunk]
        return torch.stack([self.transform(im) for im in pil], dim=0)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        vid = sample["video_id"]
        # All three chunks (obs / pred / b) come from windows in self.pairs[idx].
        # ClevrerChunkPairs stored their indices via self.pairs; we re-fetch the
        # start_frame from windows to avoid re-loading the latent blobs.
        i_obs, i_pred, i_b = self.pairs[idx]
        s_obs  = int(self.windows[i_obs ]["start_frame"])
        s_pred = int(self.windows[i_pred]["start_frame"])
        s_b    = int(self.windows[i_b   ]["start_frame"])
        sample["pix_obs"]   = self._load_chunk_pixels(vid, s_obs)
        sample["pix_pred"]  = self._load_chunk_pixels(vid, s_pred)
        sample["pix_obs_b"] = self._load_chunk_pixels(vid, s_b)
        return sample


__all__ = ["ClevrerChunkPairsWithPixels"]
