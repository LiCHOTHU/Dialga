"""Phase-2 test: run the inertial-baseline event extractor on encoder-
predicted positions (from a trained Stage-1 checkpoint) and compare against
CLEVRER GT, side-by-side with the GT-positions baseline.

Tests whether the encoder's q is clean enough that events still extract
correctly. If F1 on encoder-q is close to F1 on GT-q, the encoder is
sufficiently faithful for the v3 event layer.

Usage:
    python scripts/test_event_extractor_phase2.py \\
        --ckpt /storage/project/r-agarg35-0/lwang831/outputs/dialga/slot_300_scale_300_0509b/stage1.pt \\
        --max_videos 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset
from src.dynamics.events import (
    extract_inertial_events_single,
    extract_velocity_jump_events_single,
    compare_events_to_gt,
)
from src.model.slot_lagrangian import SlotQueryEncoder, LatentSIGRegEncoder


def build_encoder_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    m = cfg["model"]
    d = cfg["dataset"]
    attr_dim = 13                                  # CLEVRER: color 8 + material 2 + shape 3
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
    return enc


@torch.no_grad()
def encode_video(encoder, frames, attrs, device, chunk=32):
    """frames: (T, 3, H, W); attrs: (K, A) — one video. Chunked to avoid OOM."""
    frames = frames.to(device)
    attrs = attrs.to(device)
    T = frames.shape[0]
    out_q, out_z = [], []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        attrs_rep = attrs.unsqueeze(0).expand(e - s, -1, -1)
        q, z = encoder(frames[s:e], attrs_rep)
        out_q.append(q.cpu()); out_z.append(z.cpu())
    return torch.cat(out_q, dim=0), torch.cat(out_z, dim=0)


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z_threshold", type=float, default=3.0)
    p.add_argument("--contact_distance", type=float, default=0.8)
    p.add_argument("--time_tolerance", type=int, default=2)
    p.add_argument("--abs_floor", type=float, default=1e-3)
    p.add_argument("--smooth_kernel", type=int, default=1,
                   help="Triangular smoothing kernel size on q before 2nd diff "
                        "(odd, >=1; 1 = off). Useful for encoder noise.")
    p.add_argument("--detector", type=str, default="inertial",
                   choices=["inertial", "velocity_jump"],
                   help="'inertial' = 2nd-diff (Newton's 1st-law residual); "
                        "'velocity_jump' = ||v_post - v_pre|| with median "
                        "windowing (more robust to encoder noise).")
    p.add_argument("--vj_window", type=int, default=3,
                   help="Velocity-jump detector: frames on each side for "
                        "median-velocity estimate.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="event_extractor_phase2.json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    encoder = build_encoder_from_ckpt(ckpt, device)
    pos_norm = float(ckpt["config"]["dataset"]["pos_normalize"])
    print(f"  encoder loaded; pos_normalize={pos_norm}; device={device}")

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
    print(f"\nDataset: {len(ds)} videos, {args.frames_per_video} frames each.")
    print(f"Extractor: z_thr={args.z_threshold}, contact={args.contact_distance}, "
          f"time_tol={args.time_tolerance}, abs_floor={args.abs_floor}\n")

    print(f"  {'video':>5}  {'src':>3}  {'enc_RMS':>7}  {'extracted':>9} {'gt':>3}  "
          f"{'TP':>3} {'FP':>3} {'FN':>3}  {'P':>5} {'R':>5} {'F1':>5}")

    per_video = []
    totals = {"GT": [0, 0, 0], "enc": [0, 0, 0]}

    for idx in range(len(ds)):
        sample = ds[idx]
        frames = sample["frames"]
        gt_pos = sample["positions"]
        alpha = sample["visibility"].float()
        attrs = sample["attrs"]
        gt_events = list(sample["collisions"])
        scene = int(sample["video_id"])

        q_enc_norm, _ = encode_video(encoder, frames, attrs, device)
        q_enc_world = q_enc_norm * pos_norm

        m = alpha > 0.5
        if m.any():
            enc_mse_world = (((q_enc_world - gt_pos).pow(2).sum(-1) * m).sum()
                             / m.sum().clamp_min(1.0)).item()
            enc_rms = enc_mse_world ** 0.5
        else:
            enc_rms = float("nan")

        for src, q in (("GT", gt_pos), ("enc", q_enc_world)):
            if args.detector == "velocity_jump":
                events = extract_velocity_jump_events_single(
                    q, alpha,
                    window=args.vj_window,
                    z_threshold=args.z_threshold,
                    contact_distance=args.contact_distance,
                    require_neighbor=True,
                    nms_window=3,
                    min_participants=2,
                    abs_floor=args.abs_floor,
                )
            else:
                events = extract_inertial_events_single(
                    q, alpha,
                    z_threshold=args.z_threshold,
                    contact_distance=args.contact_distance,
                    require_neighbor=True,
                    min_temporal_extent=1,
                    nms_window=3,
                    min_participants=2,
                    abs_floor=args.abs_floor,
                    smooth_kernel=args.smooth_kernel,
                )
            cmp = compare_events_to_gt(
                events, gt_events,
                time_tolerance=args.time_tolerance,
                require_pair_overlap=True,
            )
            print(f"  {scene:5d}  {src:>3}  {enc_rms:7.4f}  {cmp['n_extracted']:>9} {cmp['n_gt']:>3}  "
                  f"{cmp['tp']:>3} {cmp['fp']:>3} {cmp['fn']:>3}  "
                  f"{cmp['precision']:.3f} {cmp['recall']:.3f} {cmp['f1']:.3f}")
            totals[src][0] += cmp["tp"]
            totals[src][1] += cmp["fp"]
            totals[src][2] += cmp["fn"]
            per_video.append({
                "scene_index": scene, "src": src,
                "enc_rms_world": enc_rms,
                "n_extracted": cmp["n_extracted"], "n_gt": cmp["n_gt"],
                "tp": cmp["tp"], "fp": cmp["fp"], "fn": cmp["fn"],
                "precision": cmp["precision"], "recall": cmp["recall"], "f1": cmp["f1"],
                "events": [(e.t, list(e.participants), round(e.magnitude, 4)) for e in events],
                "gt_events": list(gt_events),
            })

    print("\n  OVERALL:")
    for src in ("GT", "enc"):
        tp, fp, fn = totals[src]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        print(f"    {src:>3}:  TP={tp:3d} FP={fp:3d} FN={fn:3d}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    out = {"args": vars(args), "totals": totals, "per_video": per_video}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
