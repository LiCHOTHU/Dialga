"""Train ONLY the EventHead, with the encoder loaded frozen from a prior
converged Stage-1/Stage-2 checkpoint.

Why this script exists:
  Joint training of (encoder + event head) on 300 videos with lambda_event=1
  caused the BCE loss (~1.4) to dominate the state loss (~0.2), preventing the
  encoder from binding slots to objects. The 8317648 checkpoint (no event
  head) showed the encoder DOES converge cleanly given the same architecture,
  data, and epochs. So the cleanest path is: freeze the converged encoder and
  train just the event head against its outputs.

Inputs:
  --src_ckpt   Path to a .pt with encoder_state_dict (stage1.pt or stage2.pt)
  --output_dir Where to write the new combined checkpoint

Output:
  <output_dir>/event_head_only.pt — same blob shape as a stage1 ckpt with
                                   `event_head_state_dict` populated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.event_head import EventHead, build_event_features, dilate_label
from src.model.slot_lagrangian import SlotQueryEncoder, LatentSIGRegEncoder


def build_encoder_from_cfg(cfg, attr_dim, device):
    m, d = cfg["model"], cfg["dataset"]
    common = dict(
        image_size=int(d["image_size"]),
        patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["encoder_depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        max_objects=int(d["max_objects"]),
        attr_dim=int(attr_dim),
        num_state_dims=int(m["num_state_dims"]),
        d_static=int(m.get("d_static", 16)),
    )
    if str(m["encoder_type"]) == "slot":
        enc = SlotQueryEncoder(**common)
    else:
        enc = LatentSIGRegEncoder(latent_dim=int(m["latent_dim"]), **common)
    return enc.to(device)


@torch.no_grad()
def encode_window(encoder, frames, attrs):
    B, W = frames.shape[:2]
    flat = frames.reshape(B * W, *frames.shape[2:])
    attrs_rep = attrs.unsqueeze(1).expand(-1, W, -1, -1).reshape(B * W, *attrs.shape[1:])
    pos, z = encoder(flat, attrs_rep)
    return pos.view(B, W, *pos.shape[1:]), z.view(B, W, *z.shape[1:])


def pool_z_static(z_per_frame, visibility):
    w = visibility.unsqueeze(-1)
    num = (z_per_frame * w).sum(dim=1)
    den = w.sum(dim=1).clamp_min(1.0)
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_ckpt", required=True,
                   help="Stage-1 or stage-2 ckpt with converged encoder.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video")
    p.add_argument("--annotation_dir", default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations")
    p.add_argument("--split", default="train")
    p.add_argument("--max_videos", type=int, default=300,
                   help="Match the value used to train the source encoder so we hit "
                        "the same video subset (seed pinned).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window_length", type=int, default=6)
    p.add_argument("--windows_per_video", type=int, default=32,
                   help="High coverage so the head sees most collision frames.")
    p.add_argument("--frames_per_video", type=int, default=128)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--pos_weight", type=float, default=50.0)
    p.add_argument("--label_dilation", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=5,
                   help="Save event_head_only.pt every N epochs so a walltime "
                        "kill still leaves a usable ckpt.")
    p.add_argument("--use_diff_features", action="store_true",
                   help="Feed [q, v, a] (D*3) to the head instead of just q. "
                        "Gives Newton's-1st-law residual explicitly so the head "
                        "doesn't have to learn finite differencing.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.src_ckpt}")
    ckpt = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    pos_norm = float(cfg["dataset"]["pos_normalize"])

    ds = ClevrerPairedDataset(
        data_dir=args.data_dir,
        annotation_dir=args.annotation_dir,
        split=args.split,
        window_length=args.window_length,
        frames_per_video=args.frames_per_video,
        windows_per_video=args.windows_per_video,
        max_videos=args.max_videos,
        max_objects=args.max_objects,
        coordinate_mode="world_xy",
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"[data] {len(ds)} windows over {args.max_videos} videos (seed={args.seed})")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=paired_collate, persistent_workers=args.num_workers > 0,
    )

    encoder = build_encoder_from_cfg(cfg, ds.attr_dim, device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    print(f"[enc ] params={sum(p.numel() for p in encoder.parameters()):,} (frozen)")

    base_state_dims = int(cfg["model"]["num_state_dims"])
    head_state_dims = base_state_dims * 3 if args.use_diff_features else base_state_dims
    head = EventHead(
        num_state_dims=head_state_dims,
        d_static=int(cfg["model"].get("d_static", 16)),
        hidden=int(cfg["model"].get("event_hidden", 64)),
        kernel_size=int(cfg["model"].get("event_kernel", 5)),
        depth=int(cfg["model"].get("event_depth", 2)),
    ).to(device)
    head.train()
    print(f"[head] params={sum(p.numel() for p in head.parameters()):,}")

    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Quick sanity: how many positive frames does the dataset see per epoch?
    print(f"[head] BCE pos_weight={args.pos_weight}, label_dilation={args.label_dilation}")
    print(f"[opt ] lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        n_batches = 0
        sum_loss = 0.0
        sum_pos = 0
        sum_total = 0
        for it, batch in enumerate(loader, 1):
            frames = batch["frames"].to(device)
            attrs = batch["attrs"].to(device)
            visib = batch["visibility"].to(device).float()
            slot_mask = batch["slot_mask"].to(device).float()
            coll_mask = batch["collision_mask"].to(device).bool()
            B, W = frames.shape[:2]

            with torch.no_grad():
                q_pred, z_per = encode_window(encoder, frames, attrs)
                z_video = pool_z_static(z_per, visib)

            head_input = build_event_features(q_pred) if args.use_diff_features else q_pred

            optim.zero_grad(set_to_none=True)
            logits = head(head_input, z_video)                          # (B, W, K)
            target = dilate_label(coll_mask, args.label_dilation)
            mask = slot_mask.view(B, 1, -1).expand(B, W, -1)
            pw = torch.tensor(args.pos_weight, device=device)
            bce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw, reduction="none",
            )
            loss = (bce * mask).sum() / mask.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()

            n_batches += 1
            sum_loss += float(loss.item())
            sum_pos += int((target * mask).sum().item())
            sum_total += int(mask.sum().item())

            if args.log_every > 0 and it % args.log_every == 0:
                print(f"  ep {ep:3d} it {it:4d}  loss {loss.item():.4f}  "
                      f"pos_rate {sum_pos / max(sum_total, 1):.4f}")

        sched.step()
        avg = sum_loss / max(n_batches, 1)
        pos_rate = sum_pos / max(sum_total, 1)
        print(f"[ep {ep:3d}/{args.epochs}] loss {avg:.4f}  pos_rate {pos_rate:.4f}  "
              f"lr {optim.param_groups[0]['lr']:.2e}  ({time.time() - t0:.1f}s)")

        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            # Persist a stage1-compatible blob. We write the event-head input
            # mode into the saved config so the Phase-3 test script knows
            # whether to apply build_event_features at inference time.
            save_cfg = copy.deepcopy(cfg)
            save_cfg.setdefault("model", {})["event_input_mode"] = (
                "qva" if args.use_diff_features else "q"
            )
            save_cfg["model"]["event_head_num_state_dims"] = head_state_dims
            blob = {
                "encoder_state_dict": encoder.state_dict(),
                "event_head_state_dict": head.state_dict(),
                "config": save_cfg,
                "stage": 1,
                "epoch": ep,
                "trained_event_head_only": True,
                "src_ckpt": args.src_ckpt,
                "event_input_mode": "qva" if args.use_diff_features else "q",
            }
            if "lagrangian_state_dict" in ckpt:
                blob["lagrangian_state_dict"] = ckpt["lagrangian_state_dict"]
            if "impulse_state_dict" in ckpt:
                blob["impulse_state_dict"] = ckpt["impulse_state_dict"]
            out_ckpt = out_dir / "event_head_only.pt"
            torch.save(blob, out_ckpt)
            print(f"[save] ep {ep} -> {out_ckpt}")


if __name__ == "__main__":
    main()
