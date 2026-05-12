"""Train EventHead ALONE on GT q + GT collision_mask. No encoder, no decoder.

Goal: confirm whether the head has capacity to overfit 20-video collision
labels in seconds, isolating the head's algorithmic correctness from
encoder/decoder coupling.

If this overfits to event_loss < 0.1 in <1 minute → head is fine, the
bottleneck is encoder-head coupling in train_slot.py.
If this also plateaus high → head architecture is structurally wrong.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset
from src.model.event_head import EventHead, build_event_features, dilate_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_videos", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_weight", type=float, default=50.0)
    p.add_argument("--dilation", type=int, default=2)
    p.add_argument("--include_neighbor", action="store_true")
    p.add_argument("--ignore_z_static", action="store_true",
                   help="If set, feed zeros for z_static (test if z_static helps or hurts)")
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
        if v not in seen: seen.append(v); idx_keep.append(i)
        if len(idx_keep) >= args.max_videos: break

    # preload into tensors
    qs, alphas, slot_masks, gt_event_masks = [], [], [], []
    for idx in idx_keep:
        s = ds[idx]
        gt_pos = s["positions"] / args.pos_norm                       # (T, K, 2)
        alpha = s["visibility"].float()                               # (T, K)
        slot_mask = s["slot_mask"].float()                            # (K,)
        # GT collision_mask: (T, K, K) — extract per-slot event mask
        # build from collisions list: list of (t, i, j)
        coll = torch.zeros(gt_pos.shape[0], gt_pos.shape[1], gt_pos.shape[1])
        for (t, i, j) in s["collisions"]:
            t = int(t); i = int(i); j = int(j)
            if 0 <= t < coll.shape[0] and i < coll.shape[1] and j < coll.shape[2]:
                coll[t, i, j] = 1.0
                coll[t, j, i] = 1.0
        per_slot = (coll.sum(dim=-1) > 0).float()                     # (T, K)
        per_slot = dilate_label(per_slot.unsqueeze(0), args.dilation).squeeze(0)
        qs.append(gt_pos); alphas.append(alpha)
        slot_masks.append(slot_mask); gt_event_masks.append(per_slot)

    qs = torch.stack(qs).to(device)                                    # (B, T, K, 2)
    alphas = torch.stack(alphas).to(device)
    slot_masks = torch.stack(slot_masks).to(device)
    gt_event_masks = torch.stack(gt_event_masks).to(device)            # (B, T, K)

    n_pos = (gt_event_masks > 0.5).float().sum().item()
    total = float(gt_event_masks.numel())
    print(f"Loaded {len(idx_keep)} videos. positive rate = {n_pos}/{total} = {n_pos/total*100:.2f}%")

    D = 2
    if args.include_neighbor:
        head_in = 3 * D + 2
    else:
        head_in = 3 * D
    head = EventHead(num_state_dims=head_in, d_static=16, hidden=64, kernel_size=5, depth=2)
    head.to(device)

    z_static = torch.zeros(qs.shape[0], qs.shape[2], 16, device=device)
    if not args.ignore_z_static:
        # use a learnable per-slot identity (10×8 = 80 dims max trivial)
        z_static = torch.randn(qs.shape[0], qs.shape[2], 16, device=device, requires_grad=False)
        # actually for true overfit, give each (video, slot) a unique identity
        torch.manual_seed(0)
        z_static = torch.randn(qs.shape[0], qs.shape[2], 16, device=device, requires_grad=False)

    print(f"head params: {sum(p.numel() for p in head.parameters())}")

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    pos_w = torch.tensor(args.pos_weight, device=device)

    feat = build_event_features(qs, include_neighbor=args.include_neighbor)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        head.train()
        opt.zero_grad()
        logits = head(feat, z_static)                                  # (B, T, K)
        # per-(B, T, K) mask: real slots only
        ev_mask = slot_masks.unsqueeze(1).expand_as(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, gt_event_masks, pos_weight=pos_w, reduction="none",
        )
        loss = (bce * ev_mask).sum() / ev_mask.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
        if ep <= 10 or ep % 10 == 0:
            with torch.no_grad():
                pred = (logits > 0).float() * ev_mask
                tgt = (gt_event_masks > 0.5).float() * ev_mask
                tp = ((pred > 0) & (tgt > 0)).float().sum().item()
                fp = ((pred > 0) & (tgt == 0)).float().sum().item()
                fn = ((pred == 0) & (tgt > 0)).float().sum().item()
                p_v = tp / max(tp + fp, 1)
                r_v = tp / max(tp + fn, 1)
                f1 = 2*p_v*r_v/max(p_v+r_v, 1e-9)
            print(f"  ep {ep:4d}  loss={loss.item():.4f}  TP={tp:.0f} FP={fp:.0f} FN={fn:.0f}  P={p_v:.3f} R={r_v:.3f} F1={f1:.3f}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
