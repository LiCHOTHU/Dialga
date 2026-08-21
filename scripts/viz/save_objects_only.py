"""Segment ALL objects (moving AND static) in a CLEVRER video and save an
"objects-only" video (background blacked out).

Why not temporal-median (per-video)? A static object is present in EVERY frame
of its own video, so the per-video median absorbs it into the background and it
is missed. Fix: CLEVRER has a FIXED camera + FIXED background scene; only object
placement is randomized per video. So the per-pixel median over frames sampled
from MANY DIFFERENT videos is the true empty-floor plate. Any single object only
occupies a given pixel in a small fraction of videos, so the cross-video median
is clean floor everywhere -> objects that are static within the target video are
still caught.

Outputs under outputs/:
  objects_only_v<ID>.mp4   : [ original | objects-only (bg black) ] side by side
  global_bg_plate.png      : the estimated empty-floor background (sanity)
"""
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import _read_full_video_frames, _resolve_video_path


def resize_rgb(frame, S):
    return np.asarray(Image.fromarray(frame).convert("RGB").resize(
        (S, S), Image.BILINEAR)).astype(np.float32) / 255.


def build_global_bg(video_dir, S, n_videos, frames_per_video, exclude_id):
    """Per-pixel median over frames from many DIFFERENT videos -> empty floor."""
    samples = []
    vid = 0
    used = 0
    while used < n_videos and vid < n_videos * 4:
        if vid != exclude_id:
            try:
                fr = _read_full_video_frames(str(_resolve_video_path(video_dir, vid)))
                idxs = np.linspace(0, len(fr) - 1, frames_per_video).astype(int)
                for i in idxs:
                    samples.append(resize_rgb(fr[i], S))
                used += 1
            except Exception as e:
                print(f"  skip vid {vid}: {e}")
        vid += 1
    arr = np.stack(samples)                       # (n_videos*fpv, S, S, 3)
    bg = np.median(arr, axis=0)                   # (S, S, 3) empty floor plate
    print(f"[bg] global background from {used} videos x {frames_per_video} frames "
          f"= {len(samples)} plates")
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="/storage/project/r-agarg35-0/lwang831/"
                    "dataset/CLEVRER/train_video")
    ap.add_argument("--video_id", type=int, default=12)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--thr", type=float, default=0.04)
    ap.add_argument("--n_bg_videos", type=int, default=50)
    ap.add_argument("--frames_per_video", type=int, default=6)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="outputs/objects_only.mp4")
    args = ap.parse_args()
    vd = Path(args.video_dir); S = args.image_size

    bg = build_global_bg(vd, S, args.n_bg_videos, args.frames_per_video, args.video_id)
    Image.fromarray((bg * 255).astype(np.uint8)).save("outputs/global_bg_plate.png")

    path = _resolve_video_path(vd, args.video_id)
    print(f"[video] {path}")
    frames = _read_full_video_frames(str(path))

    fg_fracs, panels = [], []
    for f in frames:
        rgb = resize_rgb(f, S)                              # (S,S,3) [0,1]
        diff = np.abs(rgb - bg).mean(-1)                    # (S,S) deviation from floor
        m = diff > args.thr
        m = ndi.binary_closing(m, iterations=1)
        m = ndi.binary_fill_holes(m)
        m = ndi.binary_opening(m, iterations=1)             # drop speckle/shadow edges
        m = ndi.binary_dilation(m, iterations=1)
        fg_fracs.append(m.mean())
        rgb_u8 = (rgb * 255).astype(np.uint8)
        obj_only = np.zeros_like(rgb_u8)
        obj_only[m] = rgb_u8[m]                             # objects on black bg
        gap = np.full((S, 4, 3), 255, np.uint8)
        panels.append(np.concatenate([rgb_u8, gap, obj_only], axis=1))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), panels, fps=args.fps)
    print(f"[saved] {out}  ({len(panels)} frames, layout: original | objects-only)")
    print(f"[fg]   mean foreground = {np.mean(fg_fracs)*100:.2f}%  "
          f"(min {min(fg_fracs)*100:.2f}%, max {max(fg_fracs)*100:.2f}%)  thr={args.thr}")


if __name__ == "__main__":
    main()
