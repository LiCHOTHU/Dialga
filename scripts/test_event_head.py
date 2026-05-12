"""Phase-3 inference test for the supervised EventHead.

Loads a checkpoint trained with `lambda_event > 0`, runs the encoder + event
head on full 128-frame test trajectories, thresholds logits, groups flagged
frames into events, and compares against CLEVRER GT collisions.

Two modes:
  (a) --in_distribution: use videos that were in the training subsample
      (matched by training's seed/max_videos). Validates that supervision
      worked.
  (b) default: use a fresh seed/max_videos to test held-out generalization.

Usage:
    python scripts/test_event_head.py \\
        --ckpt /path/to/stage1.pt \\
        --max_videos 5 --in_distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset
from src.dynamics.events import compare_events_to_gt, Event
from src.model.event_head import EventHead, build_event_features
from src.model.slot_lagrangian import SlotQueryEncoder, LatentSIGRegEncoder


def build_models_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    m = cfg["model"]
    d = cfg["dataset"]
    attr_dim = 13
    common = dict(
        image_size=int(d["image_size"]),
        patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["encoder_depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        max_objects=int(d["max_objects"]),
        attr_dim=attr_dim,
        num_state_dims=int(m["num_state_dims"]),
        d_static=int(m.get("d_static", 16)),
    )
    if str(m["encoder_type"]) == "slot":
        enc = SlotQueryEncoder(**common)
    else:
        enc = LatentSIGRegEncoder(latent_dim=int(m["latent_dim"]), **common)
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.to(device).eval()

    if "event_head_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint has no event_head_state_dict — was "
                           "lambda_event > 0 during training?")
    # event_head input shape — base num_state_dims unless the ckpt explicitly
    # records a wider input (e.g. [q, v, a] = 3*D from the diff-features mode).
    head_state_dims = int(m.get("event_head_num_state_dims", m["num_state_dims"]))
    head = EventHead(
        num_state_dims=head_state_dims,
        d_static=int(m.get("d_static", 16)),
        hidden=int(m.get("event_hidden", 64)),
        kernel_size=int(m.get("event_kernel", 5)),
        depth=int(m.get("event_depth", 2)),
    )
    head.load_state_dict(ckpt["event_head_state_dict"])
    head.to(device).eval()
    event_input_mode = str(ckpt.get("event_input_mode",
                                    m.get("event_input_mode", "q")))
    return enc, head, event_input_mode


@torch.no_grad()
def encode_video(encoder, frames, attrs, device, chunk=32):
    """frames: (T, 3, H, W); attrs: (K, A) — one video. Chunked."""
    frames = frames.to(device); attrs = attrs.to(device)
    T = frames.shape[0]
    out_q, out_z = [], []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        attrs_rep = attrs.unsqueeze(0).expand(e - s, -1, -1)
        q, z = encoder(frames[s:e], attrs_rep)
        out_q.append(q); out_z.append(z)
    return torch.cat(out_q, dim=0), torch.cat(out_z, dim=0)


def logits_to_events(
    logits: torch.Tensor,            # (T, K) per-(t,slot) event logit
    alpha: torch.Tensor,             # (T, K) visibility
    q_world: torch.Tensor,           # (T, K, 2) for spatial filter
    *,
    threshold: float = 0.0,          # logit threshold (sigmoid > 0.5 ⇔ logit > 0)
    contact_distance: float = 0.8,
    require_neighbor: bool = True,
    nms_window: int = 3,
    min_participants: int = 2,
):
    """Convert per-(t, slot) logits into an event list using the same
    NMS / spatial / participant pipeline as the unsupervised extractor."""
    T, K = logits.shape
    device = logits.device

    flagged = (logits > threshold) & (alpha > 0.5)               # (T, K)

    if require_neighbor:
        for t in range(T):
            if not flagged[t].any():
                continue
            qq = q_world[t]
            d_mat = (qq.unsqueeze(1) - qq.unsqueeze(0)).norm(dim=-1)
            eye = torch.eye(K, dtype=torch.bool, device=device)
            at = alpha[t] > 0.5
            valid_pair = at.unsqueeze(1) & at.unsqueeze(0) & ~eye
            has_neighbor = ((d_mat < contact_distance) & valid_pair).any(dim=-1)
            flagged[t] = flagged[t] & has_neighbor

    if nms_window > 1:
        for k in range(K):
            t = 0
            while t < T:
                if flagged[t, k]:
                    flagged[t + 1:min(t + nms_window, T), k] = False
                    t += nms_window
                else:
                    t += 1

    raw = []
    for t in range(T):
        slots = flagged[t].nonzero(as_tuple=False).flatten().tolist()
        if len(slots) < min_participants:
            continue
        mag = float(torch.sigmoid(logits[t, slots]).max().item())
        raw.append(Event(t=t, participants=slots, magnitude=mag))

    if not raw:
        return raw
    by_mag = sorted(range(len(raw)), key=lambda i: -raw[i].magnitude)
    keep = [True] * len(raw)
    for i in by_mag:
        if not keep[i]:
            continue
        for j in range(len(raw)):
            if i == j or not keep[j]:
                continue
            if abs(raw[j].t - raw[i].t) <= nms_window and raw[j].magnitude < raw[i].magnitude:
                if set(raw[i].participants) & set(raw[j].participants):
                    keep[j] = False
    events = [raw[i] for i in range(len(raw)) if keep[i]]
    events.sort(key=lambda e: e.t)
    return events


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str,
                   default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video")
    p.add_argument("--annotation_dir", type=str,
                   default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--frames_per_video", type=int, default=128)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0,
                   help="If --in_distribution, the dataset uses the training "
                        "seed and we slice the first --max_videos from it. "
                        "Otherwise we draw a fresh held-out subsample.")
    p.add_argument("--in_distribution", action="store_true",
                   help="Use videos from the training subsample (in-dist).")
    p.add_argument("--use_gt_q_inference", action="store_true",
                   help="Diagnostic: feed GT q (not encoder q_pred) to the head at inference.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Logit threshold (>0 means sigmoid>0.5).")
    p.add_argument("--contact_distance", type=float, default=0.8)
    p.add_argument("--time_tolerance", type=int, default=2)
    p.add_argument("--nms_window", type=int, default=3)
    p.add_argument("--min_participants", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="event_head_test.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    encoder, head, event_input_mode = build_models_from_ckpt(ckpt, device)
    pos_norm = float(ckpt["config"]["dataset"]["pos_normalize"])
    print(f"  encoder + head loaded; pos_normalize={pos_norm}; "
          f"event_input_mode={event_input_mode}; device={device}")

    if args.in_distribution:
        train_max = int(ckpt["config"]["training"]["max_videos"])
        train_seed = int(ckpt["config"]["training"]["seed"])
        ds = ClevrerPairedDataset(
            data_dir=args.data_dir,
            annotation_dir=args.annotation_dir,
            split=args.split,
            window_length=args.frames_per_video,
            frames_per_video=args.frames_per_video,
            windows_per_video=1,
            max_videos=train_max,
            max_objects=args.max_objects,
            coordinate_mode="world_xy",
            image_size=args.image_size,
            seed=train_seed,
        )
        # take just the first --max_videos distinct video_ids
        seen = []
        idx_keep = []
        for i in range(len(ds)):
            vid = int(ds[i]["video_id"])
            if vid not in seen:
                seen.append(vid)
                idx_keep.append(i)
            if len(idx_keep) >= args.max_videos:
                break
        idx_keep_set = set(idx_keep)
        idx_iter = sorted(idx_keep_set)
        print(f"\nIn-distribution subset: {len(idx_iter)} videos from training subsample (seed={train_seed}, max_videos={train_max})")
    else:
        ds = ClevrerPairedDataset(
            data_dir=args.data_dir,
            annotation_dir=args.annotation_dir,
            split=args.split,
            window_length=args.frames_per_video,
            frames_per_video=args.frames_per_video,
            windows_per_video=1,
            max_videos=args.max_videos,
            max_objects=args.max_objects,
            coordinate_mode="world_xy",
            image_size=args.image_size,
            seed=args.seed,
        )
        idx_iter = list(range(len(ds)))
        print(f"\nHeld-out subset: {len(idx_iter)} videos (seed={args.seed})")

    print(f"Threshold: logit>{args.threshold}, contact={args.contact_distance}, "
          f"time_tol={args.time_tolerance}, min_participants={args.min_participants}\n")

    print(f"  {'video':>5}  {'enc_RMS':>7}  {'extracted':>9} {'gt':>3}  "
          f"{'TP':>3} {'FP':>3} {'FN':>3}  {'P':>5} {'R':>5} {'F1':>5}")

    per_video = []
    tp_tot = fp_tot = fn_tot = 0

    for idx in idx_iter:
        s = ds[idx]
        frames = s["frames"]
        gt_pos = s["positions"]
        alpha = s["visibility"].float()
        attrs = s["attrs"]
        gt_events = list(s["collisions"])
        scene = int(s["video_id"])

        q_norm, z = encode_video(encoder, frames, attrs, device)        # (T, K, D), (T, K, d_s)
        # pool z_static over time (visibility-weighted, like training)
        a = alpha.to(device).unsqueeze(-1)
        z_video = (z * a).sum(dim=0) / a.sum(dim=0).clamp_min(1.0)       # (K, d_s)
        # event head wants (B, T, K, D), (B, K, d_s); apply diff features if
        # the ckpt was trained with them.
        if args.use_gt_q_inference:
            head_q = (gt_pos / pos_norm).to(device)                        # (T, K, 2) normalized
            head_input = head_q.unsqueeze(0)
        else:
            head_input = q_norm.unsqueeze(0)
        if event_input_mode == "qva":
            head_input = build_event_features(head_input)
        elif event_input_mode == "qva_nn":
            head_input = build_event_features(head_input, include_neighbor=True)
        with torch.no_grad():
            logits = head(head_input, z_video.unsqueeze(0))               # (1, T, K)
        logits = logits.squeeze(0).cpu()
        q_world_full = (q_norm * pos_norm).cpu()

        m = alpha > 0.5
        if m.any():
            mse_world = (((q_world_full - gt_pos).pow(2).sum(-1) * m).sum()
                         / m.sum().clamp_min(1.0)).item()
            enc_rms = mse_world ** 0.5
        else:
            enc_rms = float("nan")

        events = logits_to_events(
            logits, alpha, q_world_full,
            threshold=args.threshold,
            contact_distance=args.contact_distance,
            require_neighbor=True,
            nms_window=args.nms_window,
            min_participants=args.min_participants,
        )
        cmp = compare_events_to_gt(
            events, gt_events,
            time_tolerance=args.time_tolerance,
            require_pair_overlap=True,
        )
        print(f"  {scene:5d}  {enc_rms:7.4f}  {cmp['n_extracted']:>9} {cmp['n_gt']:>3}  "
              f"{cmp['tp']:>3} {cmp['fp']:>3} {cmp['fn']:>3}  "
              f"{cmp['precision']:.3f} {cmp['recall']:.3f} {cmp['f1']:.3f}")
        tp_tot += cmp["tp"]; fp_tot += cmp["fp"]; fn_tot += cmp["fn"]
        per_video.append({
            "scene_index": scene,
            "enc_rms_world": enc_rms,
            "n_extracted": cmp["n_extracted"], "n_gt": cmp["n_gt"],
            "tp": cmp["tp"], "fp": cmp["fp"], "fn": cmp["fn"],
            "precision": cmp["precision"], "recall": cmp["recall"], "f1": cmp["f1"],
            "events": [(e.t, list(e.participants), round(e.magnitude, 4)) for e in events],
            "gt_events": list(gt_events),
        })

    p_o = tp_tot / max(tp_tot + fp_tot, 1)
    r_o = tp_tot / max(tp_tot + fn_tot, 1)
    f1_o = 2 * p_o * r_o / max(p_o + r_o, 1e-12)
    print(f"\n  OVERALL ({len(idx_iter)} videos):")
    print(f"    TP={tp_tot}  FP={fp_tot}  FN={fn_tot}")
    print(f"    P={p_o:.3f}  R={r_o:.3f}  F1={f1_o:.3f}")

    Path(args.output).write_text(json.dumps({
        "args": vars(args), "ckpt": args.ckpt,
        "overall": dict(tp=tp_tot, fp=fp_tot, fn=fn_tot,
                        precision=p_o, recall=r_o, f1=f1_o),
        "per_video": per_video,
    }, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
