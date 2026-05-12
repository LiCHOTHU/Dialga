"""Diagnose why T=128 motion teacher gives F1=0 at inference despite
near-zero training event loss.

Reports for each in-distribution video:
  (a) logit stats from BOTH the GT-trained head and the motion-trained head
      on the SAME full-video inference path
  (b) offline motion-teacher labels evaluated on the full T=128 video using
      the SAME GT-q anchor used at training time, with the same params
      (abs_thresh=0.05, hard_binarize=true)
  (c) how those motion labels align (or not) with the GT collision frames

This isolates three failure modes:
  M1. Motion head trained but logits never cross 0 at inference
      (distribution-shift between training and inference inputs).
  M2. Logits cross 0 but get filtered by min_participants=2 (single-slot
      firings — motion teacher labels only the impacted slot's centroid).
  M3. Logits + extraction work, but the labels are at frames offset
      from GT collisions beyond time_tolerance.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset
from src.dynamics.events import compare_events_to_gt, Event
from src.dynamics.pixel_event_teacher import motion_centroid_event_soft
from src.model.event_head import EventHead, build_event_features
from src.model.slot_lagrangian import SlotQueryEncoder, LatentSIGRegEncoder


def build_models(ckpt, device):
    cfg = ckpt["config"]
    m, d = cfg["model"], cfg["dataset"]
    common = dict(
        image_size=int(d["image_size"]),
        patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["encoder_depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        max_objects=int(d["max_objects"]),
        attr_dim=13,
        num_state_dims=int(m["num_state_dims"]),
        d_static=int(m.get("d_static", 16)),
    )
    enc = SlotQueryEncoder(**common) if str(m["encoder_type"]) == "slot" \
        else LatentSIGRegEncoder(latent_dim=int(m["latent_dim"]), **common)
    enc.load_state_dict(ckpt["encoder_state_dict"]); enc.to(device).eval()

    head_state_dims = int(m.get("event_head_num_state_dims", m["num_state_dims"]))
    head = EventHead(
        num_state_dims=head_state_dims,
        d_static=int(m.get("d_static", 16)),
        hidden=int(m.get("event_hidden", 64)),
        kernel_size=int(m.get("event_kernel", 5)),
        depth=int(m.get("event_depth", 2)),
    )
    head.load_state_dict(ckpt["event_head_state_dict"]); head.to(device).eval()
    mode = str(ckpt.get("event_input_mode", m.get("event_input_mode", "q")))
    return enc, head, mode


@torch.no_grad()
def encode_video(enc, frames, attrs, device, chunk=32):
    frames = frames.to(device); attrs = attrs.to(device)
    T = frames.shape[0]
    out_q, out_z = [], []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        attrs_rep = attrs.unsqueeze(0).expand(e - s, -1, -1)
        q, z = enc(frames[s:e], attrs_rep)
        out_q.append(q); out_z.append(z)
    return torch.cat(out_q, 0), torch.cat(out_z, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_ckpt", required=True)
    p.add_argument("--gt_ckpt", required=True)
    p.add_argument("--max_videos", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--data_dir",
        default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video",
    )
    p.add_argument(
        "--annotation_dir",
        default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_m = torch.load(args.motion_ckpt, map_location="cpu", weights_only=False)
    ckpt_g = torch.load(args.gt_ckpt, map_location="cpu", weights_only=False)
    enc_m, head_m, mode_m = build_models(ckpt_m, device)
    enc_g, head_g, mode_g = build_models(ckpt_g, device)
    pos_norm = float(ckpt_m["config"]["dataset"]["pos_normalize"])
    train_seed = int(ckpt_m["config"]["training"]["seed"])
    train_max = int(ckpt_m["config"]["training"]["max_videos"])

    # offline motion-teacher params (must match training)
    abs_thresh = float(ckpt_m["config"]["training"].get("motion_abs_thresh", 0.05))
    sharpness = float(ckpt_m["config"]["training"].get("self_event_sharpness", 8.0))
    att_sigma = float(ckpt_m["config"]["training"].get("pixel_event_attention_sigma", 0.20))
    hard = bool(ckpt_m["config"]["training"].get("event_teacher_hard", True))
    label_dilation = int(ckpt_m["config"]["training"].get("event_label_dilation", 3))
    use_gt_q = bool(ckpt_m["config"]["training"].get("motion_teacher_gt_q", True))
    print(f"Motion teacher params: abs_thresh={abs_thresh} sharpness={sharpness} "
          f"att_sigma={att_sigma} hard={hard} dilation={label_dilation} use_gt_q={use_gt_q}")

    ds = ClevrerPairedDataset(
        data_dir=args.data_dir, annotation_dir=args.annotation_dir,
        split="train", window_length=128, frames_per_video=128,
        windows_per_video=1, max_videos=train_max, max_objects=8,
        coordinate_mode="world_xy", image_size=128, seed=train_seed,
    )
    seen, idx_keep = [], []
    for i in range(len(ds)):
        v = int(ds[i]["video_id"])
        if v not in seen:
            seen.append(v); idx_keep.append(i)
        if len(idx_keep) >= args.max_videos: break

    for idx in idx_keep:
        s = ds[idx]
        frames = s["frames"]                 # (T, 3, H, W)
        gt_pos = s["positions"]               # (T, K, 2)  world
        alpha = s["visibility"].float()       # (T, K)
        attrs = s["attrs"]
        gt_events = list(s["collisions"])
        vid = int(s["video_id"])

        # --- inference logits for both checkpoints (encoders may differ)
        q_m, z_m = encode_video(enc_m, frames, attrs, device)
        q_g, z_g = encode_video(enc_g, frames, attrs, device)
        a = alpha.to(device).unsqueeze(-1)
        zv_m = (z_m * a).sum(0) / a.sum(0).clamp_min(1.0)
        zv_g = (z_g * a).sum(0) / a.sum(0).clamp_min(1.0)

        def head_forward(head, q, zv, mode):
            x = q.unsqueeze(0)
            if mode == "qva": x = build_event_features(x)
            return head(x, zv.unsqueeze(0)).squeeze(0)  # (T, K)

        lg_m = head_forward(head_m, q_m, zv_m, mode_m).cpu()
        lg_g = head_forward(head_g, q_g, zv_g, mode_g).cpu()

        # --- offline motion teacher labels on the full T=128 video.
        # Use GT q in [-1,1] normalized scale (training used motion_teacher_gt_q=true).
        gt_q_norm = (gt_pos / pos_norm).to(device)
        if use_gt_q:
            teacher_q = gt_q_norm.unsqueeze(0)
        else:
            teacher_q = q_m.unsqueeze(0)
        teacher_labels = motion_centroid_event_soft(
            frames.unsqueeze(0).to(device),
            teacher_q,
            alpha.unsqueeze(0).to(device),
            attention_sigma=att_sigma,
            z_thresh=2.0,
            sharpness=sharpness,
            label_dilation=label_dilation,
            abs_thresh=abs_thresh,
            hard_binarize=hard,
        ).squeeze(0).cpu()  # (T, K)

        # --- per-video summary
        print(f"\n=== Video {vid} ===")
        print(f"GT collisions ({len(gt_events)}): "
              + ", ".join(f"t={t}(obj{i},{j})" for (t, i, j) in gt_events))

        def logit_summary(name, lg):
            T_, K_ = lg.shape
            vmask = alpha > 0.5
            v = lg[vmask]
            n_pos = ((lg > 0) & vmask).sum().item()
            # per-frame: count of slots with logit > 0
            slots_per_frame = ((lg > 0) & vmask).sum(dim=1)
            frames_ge2 = (slots_per_frame >= 2).sum().item()
            top5_t = torch.topk(lg.flatten(), k=5).indices
            top5 = [(int(i) // K_, int(i) % K_, float(lg.flatten()[i].item()))
                    for i in top5_t]
            print(f"  [{name}] logits: min={v.min().item():+.2f} max={v.max().item():+.2f} "
                  f"mean={v.mean().item():+.3f} | (t,k) > 0: {n_pos} | frames ≥2 slots: {frames_ge2}")
            print(f"           top5 cells (t,k,logit): {top5}")

        logit_summary("motion-ckpt", lg_m)
        logit_summary("GT-ckpt    ", lg_g)

        # --- motion teacher firings on this full video
        T_ = teacher_labels.shape[0]
        K_ = teacher_labels.shape[1]
        # find frames where any slot teacher-label > 0.5
        slots_per_frame_lbl = (teacher_labels > 0.5).sum(dim=1)
        active_frames = (slots_per_frame_lbl > 0).nonzero(as_tuple=False).flatten().tolist()
        # condensed: list of contiguous runs
        runs = []
        if active_frames:
            s, prev = active_frames[0], active_frames[0]
            for t in active_frames[1:]:
                if t == prev + 1: prev = t
                else: runs.append((s, prev)); s = prev = t
            runs.append((s, prev))
        ge2 = (slots_per_frame_lbl >= 2).sum().item()
        print(f"  [motion teacher offline] active frames (any slot ≥0.5): {len(active_frames)} "
              f"| frames with ≥2 slots: {ge2}")
        print(f"           runs: {runs}")

        # alignment: for each GT collision t_gt, is there a motion label within ±2?
        if gt_events:
            for (t_gt, i, j) in gt_events:
                # look ±2 frames
                window = list(range(max(0, t_gt - 2), min(T_, t_gt + 3)))
                any_lbl = any(slots_per_frame_lbl[t].item() > 0 for t in window)
                two_lbl = any(slots_per_frame_lbl[t].item() >= 2 for t in window)
                slots_fired = set()
                for t in window:
                    if slots_per_frame_lbl[t].item() > 0:
                        slots_fired |= set((teacher_labels[t] > 0.5).nonzero().flatten().tolist())
                print(f"    GT t={t_gt} obj{i}↔{j} : motion any={any_lbl} "
                      f"≥2={two_lbl}  slots_within_±2={sorted(slots_fired)}")


if __name__ == "__main__":
    main()
