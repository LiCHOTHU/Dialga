"""B2 — Position-vs-GT diagnostic + attention visualization.

For each slot query in the ep-5 encoder, extract attention over the 8×8 patch
grid (per layer), reduce to a predicted position via argmax, and compare to
the slot's GT position projected into the latent grid.

Also dumps PNG heatmaps for the first --viz_windows windows so we can
visually classify the binding pattern (object vs spatial-bin vs collapsed).

Run:
    python scripts/probe_slot_position_error.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \\
        --ckpt outputs/iter22c_attrs_20260519_030044/trajectory.pt \\
        --n_windows 200 --viz_windows 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.trajectory_encoder_v21 import TrajectoryEncoder
from scripts.legacy.train_trajectory import CachedLatentDataset, collate
from scripts.legacy.probes.probe_wan_perslot_gtpos import project_world_xy_to_pixel, IMG_SIZE


# ---------------------------------------------------------------- encoder hook


@torch.no_grad()
def encoder_with_attention(enc: TrajectoryEncoder, wan_latent: torch.Tensor):
    """Replays TrajectoryEncoder.forward and captures slot→patch attention
    at each transformer layer.

    Returns:
        z_static : (B, K, d_static)
        slot_to_patch_per_layer : list of (B, K, 64) — time-averaged head-averaged
                                  attention from static slot queries onto the 8×8
                                  spatial patch grid.
    """
    e = enc
    B, C, T, H, W = wan_latent.shape
    S = H * W
    K = e.K

    x = wan_latent.permute(0, 2, 3, 4, 1).reshape(B, T, S, C)
    x = e.input_proj(x)
    x = x + e.spatial_pos
    t_idx = torch.arange(T, device=x.device)
    x = x + e.temporal_pos(t_idx)[None, :, None, :]

    static_q = e.slot_queries.expand(B, -1, -1)
    frame_q = e.frame_slot_queries.expand(B, T, -1, -1)
    frame_q = frame_q + e.temporal_pos(t_idx)[None, :, None, :]

    patch_tokens = x.reshape(B, T * S, -1)
    frame_slot_tokens = frame_q.reshape(B, T * K, -1)
    tokens = torch.cat([static_q, frame_slot_tokens, patch_tokens], dim=1)

    patch_col_start = K + T * K          # patches start at this column

    slot_to_patch_per_layer = []
    for layer in e.encoder.layers:
        pre = layer.norm1(tokens)
        # need_weights + average_attn_weights=False → (B, n_heads, L, L)
        attn_out, attn_w = layer.self_attn(
            pre, pre, pre,
            need_weights=True, average_attn_weights=False,
        )
        # slice slot-rows × patch-cols
        slot_rows = attn_w[:, :, :K, patch_col_start:]                # (B, h, K, T*S)
        slot_to_patch = slot_rows.mean(dim=1)                         # (B, K, T*S) avg over heads
        slot_to_patch = slot_to_patch.reshape(B, K, T, S).mean(dim=2) # (B, K, S=64) avg over T
        # Re-normalize so each slot row sums to 1 (already approximately does
        # after marginalising; re-normalising removes the heads/T averaging drift).
        s = slot_to_patch.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_to_patch = slot_to_patch / s
        slot_to_patch_per_layer.append(slot_to_patch.cpu())

        tokens = tokens + layer.dropout1(attn_out)
        ff_in = layer.norm2(tokens)
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ff_in))))
        tokens = tokens + layer.dropout2(ff_out)

    tokens = e.out_norm(tokens)
    z_static = e.static_head(tokens[:, :K])
    return z_static, slot_to_patch_per_layer


# ---------------------------------------------------------------- GT positions


def gt_slot_latent_position(positions_w, visibility, slot_mask, latent_size=8):
    """Returns (K, 2) GT slot position in latent-grid coords (or NaN if no
    visible frame) and a (K,) bool indicating valid."""
    K = slot_mask.shape[0]
    pos_xy = np.full((K, 2), np.nan, dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    for k in range(K):
        if not bool(slot_mask[k]):
            continue
        vis = visibility[:, k].astype(bool)
        if not vis.any():
            continue
        pix = project_world_xy_to_pixel(positions_w[vis, k])     # (T_vis, 2)
        in_frame = ((pix >= 0) & (pix < IMG_SIZE)).all(axis=-1)
        chosen = pix[in_frame] if in_frame.any() else pix
        mean_pix = chosen.mean(axis=0)
        latent_xy = mean_pix / (IMG_SIZE / latent_size)          # → [0, 8]
        pos_xy[k] = latent_xy
        valid[k] = True
    return pos_xy, valid


# ---------------------------------------------------------------- assignment


def hungarian_assign(cost_mat):
    """Greedy assignment (we want pred slot → GT slot, K is small so this is fine)."""
    from scipy.optimize import linear_sum_assignment
    row, col = linear_sum_assignment(cost_mat)
    return row, col


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--viz_windows", type=int, default=10,
                    help="Number of windows to dump attention heatmaps for")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = CachedLatentDataset(args.cache_dir, split="val",
                             val_frac=args.val_frac, seed=args.seed)
    ds.windows = ds.windows[: args.n_windows]
    print(f"[data] using first {len(ds)} val windows")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    sample = ds[0]
    C, T, H, W = sample["latent"].shape
    enc = TrajectoryEncoder(
        latent_ch=C,
        K=int(a.get("K", 8)),
        d_model=int(a.get("d_model", 192)),
        n_heads=int(a.get("n_heads", 4)),
        n_layers=int(a.get("n_layers", 4)),
        T_max=max(T * 2, 16), spatial_size=H,
        d_static=int(a.get("d_static", 16)),
        d_dyn=int(a.get("d_dyn", 32)),
        dropout=0.0,
    ).to(device)
    enc.load_state_dict(ckpt.get("encoder_state_dict", ckpt.get("encoder", ckpt)))
    enc.eval()
    K = enc.K
    n_layers = len(enc.encoder.layers)
    print(f"[model] ep5 encoder loaded — K={K}  n_layers={n_layers}")

    # Iterate windows in mini-batches
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    t0 = time.time()

    # accumulators per layer:
    #   - argmax-position errors (after Hungarian assignment to GT)
    #   - attention entropy (low = sharp/localized, high = spread)
    #   - attention concentration (max prob mass)
    per_layer_errors = [[] for _ in range(n_layers)]
    per_layer_entropy = [[] for _ in range(n_layers)]
    per_layer_maxprob = [[] for _ in range(n_layers)]

    viz_payloads = []  # list of dicts for matplotlib dump

    n_done = 0
    for batch in loader:
        wan = batch["latent"].to(device)
        B = wan.shape[0]
        _, slot_to_patch_layers = encoder_with_attention(enc, wan)
        # slot_to_patch_layers[l]: (B, K, 64)
        positions_w = batch["positions"].numpy()
        visibility = batch["visibility"].numpy()
        slot_mask = batch["slot_mask"].numpy()
        attrs = batch["attrs"].numpy()
        video_ids = batch["video_id"].numpy()

        for b in range(B):
            gt_pos, valid = gt_slot_latent_position(
                positions_w[b], visibility[b], slot_mask[b], latent_size=H
            )
            if valid.sum() < 1:
                continue

            # Predicted positions from each layer
            pred_per_layer = []
            for l, s2p in enumerate(slot_to_patch_layers):
                attn = s2p[b].numpy()        # (K, 64)
                # Entropy and max-prob (over the 64 patches)
                ent_per_slot = (-(attn * np.log(attn.clip(min=1e-12))).sum(-1))   # (K,)
                per_layer_entropy[l].append(ent_per_slot)
                per_layer_maxprob[l].append(attn.max(-1))
                # argmax → (kx, ky) — patch index in row-major over the 8x8 grid.
                # spatial_pos is laid out as `view(H, W)` then flattened to S — confirm row-major.
                idx = attn.argmax(-1)
                kx = (idx % H).astype(np.float32) + 0.5
                ky = (idx // H).astype(np.float32) + 0.5
                pred_xy = np.stack([kx, ky], axis=-1)        # (K, 2)
                pred_per_layer.append(pred_xy)

                # Position error per layer: Hungarian-assign pred slots ↔ valid GT slots
                # to remove the slot-permutation ambiguity, then compute mean error.
                vk = np.where(valid)[0]
                if len(vk) == 0:
                    continue
                gt_v = gt_pos[vk]                            # (V, 2)
                # cost = pairwise euclidean (in latent units)
                cost = np.linalg.norm(pred_xy[:, None, :] - gt_v[None, :, :], axis=-1)  # (K, V)
                # We have K predicted slots and V GT slots (V <= K). Standard
                # rectangular assignment minimizes total cost.
                row, col = hungarian_assign(cost)
                # row indexes K (predicted), col indexes V (GT). Take only those rows
                # where col is a valid GT match.
                for r, c in zip(row, col):
                    per_layer_errors[l].append(float(cost[r, c]))

            # Visualization payload for first viz_windows
            if len(viz_payloads) < args.viz_windows:
                # Store last-layer attention + GT positions + attrs for plot
                viz_payloads.append({
                    "video_id": int(video_ids[b]),
                    "gt_pos": gt_pos.copy(),
                    "valid": valid.copy(),
                    "attrs": attrs[b].copy(),
                    "slot_mask": slot_mask[b].copy(),
                    "attn_per_layer": [
                        slot_to_patch_layers[l][b].numpy().reshape(K, H, W).copy()
                        for l in range(n_layers)
                    ],
                })
        n_done += B
    print(f"[done] processed {n_done} windows in {time.time()-t0:.1f}s")

    # ---------- summary stats ----------
    print("\n=== Per-layer attention diagnostics ===")
    print(f"{'layer':>6s} | {'err_mean':>10s} | {'err_med':>10s} | "
          f"{'err_p90':>10s} | {'entropy':>10s} | {'max_prob':>10s}  "
          f"(chance ≈ {np.sqrt(2)*H/3:.2f} px latent for uniform)")
    summary = {}
    for l in range(n_layers):
        errs = np.array(per_layer_errors[l])
        ent = np.concatenate(per_layer_entropy[l])
        mp = np.concatenate(per_layer_maxprob[l])
        em = errs.mean(); ed = np.median(errs); ep90 = np.percentile(errs, 90)
        print(f"{l:>6d} | {em:>10.3f} | {ed:>10.3f} | {ep90:>10.3f} | "
              f"{ent.mean():>10.3f} | {mp.mean():>10.3f}")
        summary[f"layer_{l}"] = {
            "n_assigned_slots": int(errs.size),
            "err_mean_latent_units": float(em),
            "err_median": float(ed),
            "err_p90": float(ep90),
            "attention_entropy_mean_nats": float(ent.mean()),
            "attention_max_prob_mean": float(mp.mean()),
        }
    # Random baseline: expected mean L2 dist between two uniform points in [0,8]^2
    # is ≈ 0.5214 * 8 ≈ 4.17.  Computed analytically: ∫∫|x-y| dx dy = 8*0.5214 ≈ 4.17.
    print(f"\n[ref] expected L2(uniform random pair in 8x8) ≈ 4.17")
    print(f"[ref] uniform attention entropy = log(64) ≈ {np.log(64):.3f} nats")
    print(f"[ref] uniform max prob          = 1/64    ≈ {1/64:.4f}")

    # ---------- attention heatmap viz ----------
    out_dir = Path(args.ckpt).parent
    viz_dir = out_dir / "slot_attention_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    color_vocab = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    material_vocab = ["rubber", "metal"]
    shape_vocab = ["sphere", "cube", "cylinder"]

    for i, vp in enumerate(viz_payloads):
        # grid: rows = K slots, cols = n_layers (heatmaps), final col = GT scatter overlay
        n_layers_v = len(vp["attn_per_layer"])
        fig, axes = plt.subplots(K, n_layers_v + 1, figsize=(2 * (n_layers_v + 1), 2 * K))
        if K == 1:
            axes = axes[None, :]
        for k in range(K):
            for l in range(n_layers_v):
                a = vp["attn_per_layer"][l][k]
                ax = axes[k, l]
                ax.imshow(a, cmap="hot", origin="upper", vmin=0,
                          vmax=max(a.max(), 1e-3))
                if vp["valid"][k]:
                    gx, gy = vp["gt_pos"][k]
                    ax.scatter([gx - 0.5], [gy - 0.5], c="lime", s=40,
                               marker="+", linewidths=2)
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_title(f"L{l}", fontsize=9)
                if l == 0:
                    if vp["slot_mask"][k]:
                        oh = vp["attrs"][k]
                        c = color_vocab[int(oh[:8].argmax())]
                        m = material_vocab[int(oh[8:10].argmax())]
                        s = shape_vocab[int(oh[10:13].argmax())]
                        ax.set_ylabel(f"k{k}\n{c[:3]} {m[:3]}\n{s[:3]}",
                                      fontsize=7, rotation=0, labelpad=18, va="center")
                    else:
                        ax.set_ylabel(f"k{k}\n(empty)", fontsize=7,
                                      rotation=0, labelpad=18, va="center")
            # GT-scatter only in final column
            ax_gt = axes[k, n_layers_v]
            ax_gt.set_xlim(0, 8); ax_gt.set_ylim(8, 0)
            ax_gt.set_xticks([]); ax_gt.set_yticks([])
            for kk in range(K):
                if vp["valid"][kk]:
                    gx, gy = vp["gt_pos"][kk]
                    col = "red" if kk == k else "gray"
                    sz = 80 if kk == k else 25
                    ax_gt.scatter([gx], [gy], c=col, s=sz)
                    ax_gt.annotate(str(kk), (gx, gy), fontsize=6)
            if k == 0:
                ax_gt.set_title("GT", fontsize=9)
        fig.suptitle(f"Window {i}  video={vp['video_id']}  "
                     f"K={K} slots × {n_layers_v} layers (last col = GT scatter, red=current slot)",
                     fontsize=10)
        plt.tight_layout()
        fig.savefig(viz_dir / f"window_{i:03d}_vid{vp['video_id']:05d}.png", dpi=80)
        plt.close(fig)
    print(f"\n[viz] saved {len(viz_payloads)} heatmaps → {viz_dir}")

    out_path = out_dir / "probe_slot_position_error.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "n_windows": n_done,
            "K": K,
            "n_layers": n_layers,
            "summary": summary,
            "refs": {
                "uniform_pair_L2": 4.17,
                "uniform_entropy_nats": float(np.log(64)),
                "uniform_max_prob": 1.0 / 64,
                "latent_grid_side": H,
            },
        }, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
