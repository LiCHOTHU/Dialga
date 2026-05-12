"""Phase-1 sanity test: run the inertial-baseline event extractor on **GT**
positions for the 5-video subset and compare to CLEVRER GT collision events.

This isolates the extractor's correctness from any encoder error. If F1 is
high here, we know the extractor + threshold + spatial filter are sane; if it
is high here but low when run on encoder positions, the encoder is the
bottleneck.

Usage:
    python scripts/test_event_extractor.py \
        --max_videos 5 --z_threshold 3.0 --contact_distance 0.8 --time_tolerance 2

The ClevrerStateDataset path is read from defaults pointing at the project
data dir. Annotations only — no frame loading.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import ClevrerStateDataset
from src.dynamics.events import (
    extract_inertial_events_single,
    compare_events_to_gt,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_dir", type=str,
                   default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--frames_per_video", type=int, default=128)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument("--z_threshold", type=float, default=3.0,
                   help="MAD-based z-score threshold on per-video accel magnitudes.")
    p.add_argument("--contact_distance", type=float, default=0.8,
                   help="Spatial-filter distance in world units (CLEVRER spheres r ~ 0.4).")
    p.add_argument("--time_tolerance", type=int, default=2,
                   help="Frames of slack when matching to GT collisions.")
    p.add_argument("--no_neighbor_filter", action="store_true",
                   help="Disable the spatial coincidence filter.")
    p.add_argument("--min_temporal_extent", type=int, default=1)
    p.add_argument("--nms_window", type=int, default=3)
    p.add_argument("--min_participants", type=int, default=2,
                   help="Drop events with fewer than this many participants. "
                        "Default 2 enforces Newton's 3rd law for collisions.")
    p.add_argument("--abs_floor", type=float, default=1e-3,
                   help="Absolute lower bound on the per-video threshold; "
                        "prevents collapse to 0 on mostly-static videos.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="event_extractor_test.json")
    args = p.parse_args()

    ds = ClevrerStateDataset(
        annotation_dir=args.annotation_dir,
        split=args.split,
        traj_len=args.frames_per_video,
        stride=args.frames_per_video,
        frames_per_video=args.frames_per_video,
        max_objects=args.max_objects,
        coordinate_mode="world_xy",
        use_inside_camera_view_mask=True,
        video_dir=None,
    )

    rng = random.Random(args.seed)
    if 0 < args.max_videos < len(ds):
        idxs = sorted(rng.sample(range(len(ds)), k=args.max_videos))
    else:
        idxs = list(range(len(ds)))

    print(f"Dataset: {len(ds)} full-video trajectories total; testing on {len(idxs)} videos.")
    print(f"  z_threshold={args.z_threshold}, contact_distance={args.contact_distance},")
    print(f"  require_neighbor={not args.no_neighbor_filter}, min_temporal_extent={args.min_temporal_extent},")
    print(f"  nms_window={args.nms_window}, time_tolerance={args.time_tolerance}\n")

    per_video = []
    total_tp = total_fp = total_fn = 0
    n_videos_with_gt = 0

    for idx in idxs:
        sample = ds[idx]
        q = sample["positions"]               # (W, K, 2) world XY
        alpha = sample["mask"].float()        # (W, K)
        gt_events = list(sample["collisions"])  # [(t, i, j), ...]
        scene = int(sample["scene_index"])

        events = extract_inertial_events_single(
            q, alpha,
            z_threshold=args.z_threshold,
            contact_distance=args.contact_distance,
            require_neighbor=not args.no_neighbor_filter,
            min_temporal_extent=args.min_temporal_extent,
            nms_window=args.nms_window,
            min_participants=args.min_participants,
            abs_floor=args.abs_floor,
        )

        cmp = compare_events_to_gt(
            events, gt_events,
            time_tolerance=args.time_tolerance,
            require_pair_overlap=True,
        )
        cmp["scene_index"] = scene
        cmp["events"] = [(e.t, list(e.participants), round(e.magnitude, 4)) for e in events]
        cmp["gt_events"] = list(gt_events)
        per_video.append(cmp)

        total_tp += cmp["tp"]
        total_fp += cmp["fp"]
        total_fn += cmp["fn"]
        if cmp["n_gt"] > 0:
            n_videos_with_gt += 1

        print(f"  video {scene:5d}: extracted={cmp['n_extracted']:2d}, gt={cmp['n_gt']:2d}, "
              f"TP={cmp['tp']:2d} FP={cmp['fp']:2d} FN={cmp['fn']:2d} | "
              f"P={cmp['precision']:.2f} R={cmp['recall']:.2f} F1={cmp['f1']:.2f}")

    p_o = total_tp / max(total_tp + total_fp, 1)
    r_o = total_tp / max(total_tp + total_fn, 1)
    f1_o = 2 * p_o * r_o / max(p_o + r_o, 1e-12)
    print(f"\n  OVERALL ({len(idxs)} videos, {n_videos_with_gt} with GT events):")
    print(f"    TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"    P={p_o:.3f}  R={r_o:.3f}  F1={f1_o:.3f}")

    out = {
        "args": vars(args),
        "overall": dict(tp=total_tp, fp=total_fp, fn=total_fn,
                        precision=p_o, recall=r_o, f1=f1_o,
                        n_videos=len(idxs), n_videos_with_gt=n_videos_with_gt),
        "per_video": per_video,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
