"""v5 identity probe (linear): does z_static encode modal color/material/shape?

Scene-level modal-attribute probe — the v5 analogue of probe_iter21_identity.

  z_static is global (B, D_s=32). With no slot dimension, we cannot probe
  per-slot identity. Instead we target the MODAL attribute per video:
  the most frequent color (resp. material, shape) across slots with
  slot_mask=True.

Protocol:
  1. Load v5 checkpoint (state_dict keyed "encoder").
  2. Iterate the trainer's val split (val_frac=0.2 by default, same seed=42).
  3. Encode each window with LatentEncoder3D on the observed half
     latent[:, :L_obs] -> z_static (B, D_s); average across the 4
     windows of each video.
  4. For each video, compute modal_{color, material, shape} from
     (attrs, slot_mask). One label per video per attribute group.
  5. 50/50 split BY video_id (seed=0). Train a single linear layer
     (logistic regression) per group on probe-train, evaluate probe-test.
  6. Report test acc vs the empirical majority baseline.

This probe is structurally weaker than the iter-21 per-slot probe — we are
testing scene-level rather than per-object identity (the v5 architectural
commitment is no-K).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.model.latent_encoder import LatentEncoder3D
from scripts.legacy.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color",    0,                                      len(COLOR_VOCAB)),
    ("material", len(COLOR_VOCAB),                       len(COLOR_VOCAB) + len(MATERIAL_VOCAB)),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB), len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)),
]


def modal_label(attrs: torch.Tensor, slot_mask: torch.Tensor, lo: int, hi: int) -> int:
    """Return the modal class index for one video's attribute group.

    attrs:     (K, A)   one-hot blocks
    slot_mask: (K,)     bool
    Returns the most-common class index in [0, hi-lo) across real slots,
    or -1 if no real slots.
    """
    real_k = slot_mask.bool()
    if not real_k.any():
        return -1
    block = attrs[real_k, lo:hi]            # (K_real, hi-lo)
    classes = block.argmax(dim=-1)          # (K_real,)
    counts = torch.bincount(classes, minlength=hi - lo)
    return int(counts.argmax().item())


@torch.no_grad()
def compute_z_static_per_video(encoder: LatentEncoder3D, loader, L_obs: int, device):
    """Encode windows -> z_static, average across the windows of each video.

    Returns dict[video_id] = {
        "z_static":  (D_s,)  averaged across windows
        "attrs":     (K, A)
        "slot_mask": (K,)    bool
    }

    Note: v5.1 encoder takes the FULL chunk (no L_obs slicing) and returns a
    dict {"z_static", "z_dyn"} with z_dyn collapsed over time. The L_obs
    argument is retained for backwards-compat with v5 checkpoints; if the
    encoder is v5.1, the full chunk is used regardless.
    """
    accum = {}
    for batch in loader:
        latent = batch["latent"].to(device)        # (B, C, T_lat, H, W)
        out = encoder(latent)                       # dict for v5.1
        z_static = out["z_static"] if isinstance(out, dict) else out[0]
        z_static = z_static.float().cpu()
        for b in range(latent.shape[0]):
            vid = int(batch["video_id"][b].item())
            entry = accum.setdefault(vid, {
                "sum": torch.zeros_like(z_static[b]),
                "n": 0,
                "attrs": batch["attrs"][b].clone(),
                "slot_mask": batch["slot_mask"][b].clone().bool(),
            })
            entry["sum"] += z_static[b]
            entry["n"] += 1
    return {
        vid: {
            "z_static": d["sum"] / max(d["n"], 1),
            "attrs": d["attrs"],
            "slot_mask": d["slot_mask"],
        }
        for vid, d in accum.items()
    }


def build_xy(per_video, video_ids, lo, hi):
    """Per-video (x = z_static, y = modal label) pairs across `video_ids`."""
    xs, ys = [], []
    for vid in video_ids:
        d = per_video[vid]
        y = modal_label(d["attrs"], d["slot_mask"], lo, hi)
        if y < 0:
            continue
        xs.append(d["z_static"])
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_linear_probe(x_tr, y_tr, x_te, y_te, num_classes,
                       epochs=2000, lr=1e-2, weight_decay=1e-4, device="cuda"):
    d_in = x_tr.shape[1]
    probe = nn.Linear(d_in, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best = {"train_acc": 0.0, "test_acc": 0.0, "ep": 0, "loss": float("inf")}
    for ep in range(epochs):
        probe.train()
        logits = probe(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                tr_acc = (probe(x_tr).argmax(-1) == y_tr).float().mean().item()
                te_acc = (probe(x_te).argmax(-1) == y_te).float().mean().item()
            if loss.item() < best["loss"]:
                best = {"train_acc": tr_acc, "test_acc": te_acc, "ep": ep, "loss": loss.item()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_val = CachedLatentDataset(args.cache_dir, split="val",
                                 val_frac=args.val_frac, seed=args.seed)
    print(f"[data] val windows: {len(ds_val)}  (val_frac={args.val_frac}, seed={args.seed})")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    L_obs = int(a.get("L_obs", 1))
    d_static = int(a.get("d_static", 32))
    d_dyn = int(a.get("d_dyn", 16))
    # `enc_hidden_ch` is the canonical key in v5.1+; fall back to legacy `hidden_ch`
    enc_hidden_ch = int(a.get("enc_hidden_ch", a.get("hidden_ch", 64)))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    print(f"[model] encoder loaded  L_obs={L_obs} d_static={d_static} d_dyn={d_dyn} enc_hidden={enc_hidden_ch}")

    loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    t0 = time.time()
    per_video = compute_z_static_per_video(enc, loader, L_obs, device)
    n_videos = len(per_video)
    print(f"[encode] {n_videos} val videos in {time.time()-t0:.1f}s")

    vid_list = sorted(per_video.keys())
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    tr_vids, te_vids = vid_list[:half], vid_list[half:]
    print(f"[probe] split: {len(tr_vids)} train vids / {len(te_vids)} test vids "
          f"(seed={args.probe_split_seed})")

    results = {}
    print(f"\n{'group':10s} {'classes':>8s} {'tr/te vids':>14s} {'chance':>8s} "
          f"{'majority':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs maj':>10s}")
    print("-" * 92)
    for name, lo, hi in ATTR_GROUPS:
        x_tr, y_tr = build_xy(per_video, tr_vids, lo, hi)
        x_te, y_te = build_xy(per_video, te_vids, lo, hi)
        num_classes = hi - lo
        chance = 1.0 / num_classes
        majority = (torch.bincount(y_te, minlength=num_classes).max().item()
                    / max(y_te.numel(), 1))
        best = train_linear_probe(
            x_tr, y_tr, x_te, y_te, num_classes,
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_wd, device=device,
        )
        print(f"{name:10s} {num_classes:>8d} "
              f"{x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{chance:>8.3f} {majority:>10.3f} "
              f"{best['train_acc']:>10.3f} {best['test_acc']:>10.3f} "
              f"{best['test_acc']-majority:>+10.3f}")
        results[name] = {
            "num_classes": num_classes,
            "n_train_videos": x_tr.shape[0],
            "n_test_videos": x_te.shape[0],
            "chance": chance,
            "majority_baseline": majority,
            "train_acc": best["train_acc"],
            "test_acc": best["test_acc"],
        }

    out_path = Path(args.ckpt).parent / "probe_v5_modal.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "scene-level modal-attribute linear probe; 50/50 video split",
            "target": "modal (most-frequent across visible slots) per attribute group",
            "cache_dir": args.cache_dir,
            "ckpt": args.ckpt,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "probe_split_seed": args.probe_split_seed,
            "L_obs": L_obs,
            "n_val_videos": n_videos,
            "n_probe_train_videos": len(tr_vids),
            "n_probe_test_videos": len(te_vids),
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
