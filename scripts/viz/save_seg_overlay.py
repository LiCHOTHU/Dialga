"""Render ONE CLEVRER video with its object/background segmentation so the mask
can be eyeballed. Uses the EXACT same _video_foreground_masks() that the
object-centric pixel loss consumes (temporal-median background subtraction),
so what you see is literally what the loss weights.

Output: a side-by-side mp4  [ original | mask | red overlay ]  at image_size.
"""
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import (
    _read_full_video_frames, _video_foreground_masks, _resolve_video_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="/storage/project/r-agarg35-0/lwang831/"
                    "dataset/CLEVRER/train_video")
    ap.add_argument("--video_id", type=int, default=12)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.035)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="outputs/seg_overlay.mp4")
    args = ap.parse_args()

    path = _resolve_video_path(Path(args.video_dir), args.video_id)
    print(f"[video] {path}")
    frames = _read_full_video_frames(str(path))
    masks = _video_foreground_masks(str(path), args.image_size, args.thr)
    S = args.image_size

    fg_fracs, panels = [], []
    for f, m in zip(frames, masks):
        rgb = np.asarray(Image.fromarray(f).convert("RGB").resize(
            (S, S), Image.BILINEAR))                       # (S,S,3) uint8
        mask = m.astype(bool)
        fg_fracs.append(mask.mean())
        mask_vis = (np.stack([mask] * 3, -1) * 255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[mask] = (0.45 * overlay[mask] +
                         0.55 * np.array([255, 0, 0])).astype(np.uint8)
        gap = np.full((S, 4, 3), 255, np.uint8)
        panels.append(np.concatenate([rgb, gap, mask_vis, gap, overlay], axis=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), panels, fps=args.fps)
    print(f"[saved] {out}  ({len(panels)} frames, layout: original | mask | overlay)")
    print(f"[fg]   mean foreground = {np.mean(fg_fracs)*100:.2f}%  "
          f"(min {min(fg_fracs)*100:.2f}%, max {max(fg_fracs)*100:.2f}%)  thr={args.thr}")


if __name__ == "__main__":
    main()
