"""Probe |Δ²c| distribution from the motion teacher across the 5 videos
at GT collision frames vs random non-event frames. The goal: pick a threshold
that fires at collisions but stays quiet otherwise.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clevrer_paired import ClevrerPairedDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--attention_sigma", type=float, default=0.20)
    p.add_argument(
        "--data_dir",
        default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video",
    )
    p.add_argument(
        "--annotation_dir",
        default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations",
    )
    p.add_argument("--pos_norm", type=float, default=4.0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ClevrerPairedDataset(
        data_dir=args.data_dir, annotation_dir=args.annotation_dir,
        split="train", window_length=128, frames_per_video=128,
        windows_per_video=1, max_videos=args.max_videos, max_objects=8,
        coordinate_mode="world_xy", image_size=128, seed=0,
    )
    seen, idx_keep = [], []
    for i in range(len(ds)):
        v = int(ds[i]["video_id"])
        if v not in seen:
            seen.append(v); idx_keep.append(i)
        if len(idx_keep) >= args.max_videos: break

    coll_amag = []   # |Δ²c| at slots involved in GT collisions, near collision frames
    non_amag = []    # |Δ²c| at all other (t, k) cells with slot visible

    for idx in idx_keep:
        s = ds[idx]
        frames = s["frames"].to(device)            # (T, 3, H, W)
        gt_pos = s["positions"].to(device)          # (T, K, 2)
        alpha = s["visibility"].float().to(device)  # (T, K)
        gt_events = list(s["collisions"])
        vid = int(s["video_id"])

        T, _, H, W = frames.shape
        K = gt_pos.shape[1]

        q_norm = gt_pos / args.pos_norm

        # saliency
        fr_mean = frames.mean(dim=(-1, -2), keepdim=True)
        saliency = (frames - fr_mean).abs().mean(dim=1)  # (T, H, W)

        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        sigma = args.attention_sigma

        dx = gx.view(1, 1, H, W) - q_norm[..., 0].view(T, K, 1, 1)
        dy = gy.view(1, 1, H, W) - q_norm[..., 1].view(T, K, 1, 1)
        mask = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))  # (T, K, H, W)
        weights = saliency.unsqueeze(1) * mask                    # (T, K, H, W)
        W_sum = weights.sum(dim=(-1, -2)).clamp_min(1e-3)
        cx = (weights * gx.view(1, 1, H, W)).sum((-1, -2)) / W_sum  # (T, K)
        cy = (weights * gy.view(1, 1, H, W)).sum((-1, -2)) / W_sum

        cx_a = cx[2:] - 2 * cx[1:-1] + cx[:-2]
        cy_a = cy[2:] - 2 * cy[1:-1] + cy[:-2]
        a_mag = (cx_a.pow(2) + cy_a.pow(2)).sqrt()  # (T-2, K)
        vis = alpha[1:-1]                            # (T-2, K)

        print(f"\n=== video {vid} ===")
        print(f"  a_mag stats (visible only): "
              f"min={a_mag[vis > 0.5].min().item():.4f} "
              f"max={a_mag[vis > 0.5].max().item():.4f} "
              f"mean={a_mag[vis > 0.5].mean().item():.4f} "
              f"p99={a_mag[vis > 0.5].quantile(0.99).item():.4f}")
        for (t_gt, i, j) in gt_events:
            # a_mag is indexed (T-2,), so frame t corresponds to a_mag index t-1
            t_idx = t_gt - 1
            if 0 <= t_idx < a_mag.shape[0]:
                # take max over ±2 frames, slots i and j
                w_lo = max(0, t_idx - 2); w_hi = min(a_mag.shape[0], t_idx + 3)
                ai = a_mag[w_lo:w_hi, i].max().item() if i < K else float("nan")
                aj = a_mag[w_lo:w_hi, j].max().item() if j < K else float("nan")
                print(f"  collision t={t_gt} obj{i}↔{j}: a_mag[i]_max±2={ai:.4f}  a_mag[j]_max±2={aj:.4f}")
                coll_amag.extend([ai, aj])

        # accumulate non-event mass (skip slots/frames in ±5 around collisions)
        flag_off = torch.ones_like(a_mag, dtype=torch.bool)
        for (t_gt, i, j) in gt_events:
            t_idx = t_gt - 1
            w_lo = max(0, t_idx - 5); w_hi = min(a_mag.shape[0], t_idx + 6)
            if i < K: flag_off[w_lo:w_hi, i] = False
            if j < K: flag_off[w_lo:w_hi, j] = False
        non = a_mag[(flag_off) & (vis > 0.5)]
        print(f"  non-event a_mag (n={non.numel()}): "
              f"mean={non.mean().item():.4f} max={non.max().item():.4f} "
              f"p99={non.quantile(0.99).item():.4f}")
        non_amag.extend(non.flatten().tolist())

    coll_t = torch.tensor(coll_amag)
    non_t = torch.tensor(non_amag)
    print("\n=== overall ===")
    print(f"  COLLISION a_mag (n={coll_t.numel()}): "
          f"min={coll_t.min():.4f} median={coll_t.median():.4f} "
          f"mean={coll_t.mean():.4f} max={coll_t.max():.4f} "
          f"p25={coll_t.quantile(0.25):.4f}")
    print(f"  NONEVENT  a_mag (n={non_t.numel()}): "
          f"min={non_t.min():.4f} median={non_t.median():.4f} "
          f"mean={non_t.mean():.4f} max={non_t.max():.4f} "
          f"p99={non_t.quantile(0.99):.4f} p99.5={non_t.quantile(0.995):.4f}")
    # Suggest thresholds: anything below 99th percentile of non-events, above 25th of collisions.
    nq99 = non_t.quantile(0.99).item()
    cq25 = coll_t.quantile(0.25).item()
    print(f"\n  Recommended thresholds: non-event p99={nq99:.4f}, collision p25={cq25:.4f}")
    if nq99 < cq25:
        print(f"  ✅ clean separation → thresh ≈ {(nq99 + cq25)/2:.4f}")
    else:
        print(f"  ⚠️  thresholds overlap → consider relaxing abs_thresh below {cq25:.4f}, "
              f"accept some FP at {nq99:.4f}")


if __name__ == "__main__":
    main()
