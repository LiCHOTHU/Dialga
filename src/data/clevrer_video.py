from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ClevrerVideoDataset(Dataset):
    def __init__(self, video_dir, max_frames=None, image_size=(128, 128)):
        self.video_dir = Path(video_dir)
        self.max_frames = max_frames
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        self.video_paths = sorted(self.video_dir.glob("**/video_*.mp4"))
        if not self.video_paths:
            raise FileNotFoundError(f"No CLEVRER videos found under {self.video_dir}.")

    def __len__(self):
        return len(self.video_paths)

    @staticmethod
    def _decode_video(video_path):
        try:
            from torchvision.io import read_video

            frames, _, _ = read_video(str(video_path), pts_unit="sec")
            if frames.shape[0] == 0:
                raise RuntimeError(f"No frames decoded from {video_path}.")
            return [Image.fromarray(frame.cpu().numpy()) for frame in frames]
        except Exception:
            try:
                import imageio.v2 as imageio

                reader = imageio.get_reader(str(video_path))
                try:
                    frames = [Image.fromarray(frame) for frame in reader]
                finally:
                    reader.close()
                if not frames:
                    raise RuntimeError(f"No frames decoded from {video_path}.")
                return frames
            except Exception as exc:
                raise RuntimeError(
                    "Failed to decode CLEVRER video. Install torchvision video backends "
                    "or imageio-ffmpeg."
                ) from exc

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        frames = self._decode_video(video_path)
        if self.max_frames is not None:
            frames = frames[: self.max_frames]
        frame_tensor = torch.stack(
            [self.transform(frame.convert("RGB")) for frame in frames],
            dim=0,
        )
        return {
            "frames": frame_tensor,
            "video_path": str(video_path),
            "video_filename": video_path.name,
        }
