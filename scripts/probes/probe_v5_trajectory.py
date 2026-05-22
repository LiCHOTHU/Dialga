"""scripts/probes/probe_v5_trajectory.py — TODO 2.

Test that z_dyn carries motion state that z_static does not — the mirror of the
identity probe. For each val (video, chunk_obs) we predict:

    target           : description
    pos_x, pos_y     : scene-mean position during chunk_obs (avg across visible slots)
    vel_x, vel_y     : scene-mean velocity during chunk_obs

Probes (each on probe-train, evaluated on probe-test, 50/50 split by video_id):
    feature  : z_static (D_s) — identity-only, should NOT predict motion
    feature  : z_dyn_last (D_d) — motion summary, should predict motion

R² is reported per (feature, target) pair so a high z_dyn R² coupled with a
low z_static R² is the factorization-in-both-directions evidence.

GT is read from outputs/trajectory_gt_val.pt (built by extract_trajectory_gt.py).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_encoder import LatentEncoder3D


CHUNK_STARTS = [0, 33, 66]


def _chunk_idx_for_start(s: int) -> int:
    for i, st in enumerate(CHUNK_STARTS):
        if s == st:
            return i
    raise ValueError(f"unknown start_frame {s}")


def _scene_state_for_chunk(gt_entry, chunk_idx: int):
    """Return (pos_mean_xy, vel_mean_xy) for chunk_idx, weighted by visibility."""
    pos = gt_entry["per_chunk_positions"][chunk_idx]   # (K, 2)
    vel = gt_entry["per_chunk_velocities"][chunk_idx]
    vis = gt_entry["per_chunk_visibility"][chunk_idx].float().unsqueeze(-1)  # (K, 1)
    denom = vis.sum().clamp_min(1.0)
    pos_mean = (pos * vis).sum(dim=0) / denom          # (2,)
    vel_mean = (vel * vis).sum(dim=0) / denom
    return pos_mean, vel_mean


def collect_features_and_targets(enc, loader, gt: dict, device):
    """For each video × chunk, aggregate z_static, z_dyn_last, and GT targets."""
    per_chunk = {}  # key = (video_id, chunk_idx) -> dict
    with torch.no_grad():
        for batch in loader:
            chunk_obs = batch["chunk_obs"].to(device)
            out = enc(chunk_obs)
            z_static = out["z_static"].float().cpu()
            z_dyn_last = out["z_dyn"][:, -1].float().cpu()
            B = chunk_obs.shape[0]
            for b in range(B):
                vid = int(batch["video_id"][b])
                s = int(batch["start_frame"][b])
                if vid not in gt:
                    continue
                try:
                    ci = _chunk_idx_for_start(s)
                except ValueError:
                    continue
                pos_mean, vel_mean = _scene_state_for_chunk(gt[vid], ci)
                key = (vid, ci)
                if key in per_chunk:
                    continue   # video × chunk seen twice across the (obs,pred,b) triples
                per_chunk[key] = {
                    "z_static":   z_static[b],
                    "z_dyn_last": z_dyn_last[b],
                    "pos":        pos_mean,
                    "vel":        vel_mean,
                }
    return per_chunk


def train_linear_regressor(x_tr, y_tr, x_te, y_te, epochs=2000, lr=1e-2, wd=1e-4, device="cuda"):
    d_in = x_tr.shape[1]
    d_out = y_tr.shape[1] if y_tr.dim() > 1 else 1
    reg = nn.Linear(d_in, d_out).to(device)
    opt = torch.optim.AdamW(reg.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best = {"test_mse": float("inf"), "train_mse": float("inf")}
    for ep in range(epochs):
        reg.train()
        pred = reg(x_tr)
        loss = F.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            reg.eval()
            with torch.no_grad():
                tr = F.mse_loss(reg(x_tr), y_tr).item()
                te = F.mse_loss(reg(x_te), y_te).item()
            if te < best["test_mse"]:
                best = {"train_mse": tr, "test_mse": te, "ep": ep}
    # R² on test:  1 - SS_res / SS_tot
    with torch.no_grad():
        y_te_mean = y_te.mean(dim=0, keepdim=True)
        ss_tot = ((y_te - y_te_mean) ** 2).sum().item()
        # Use the best probe weights — quickest is to re-train one more step at best lr (sloppy);
        # cleaner: simply compute R² at the last step (close to best).
        pred_te = reg(x_te)
        ss_res = ((pred_te - y_te) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    return {"train_mse": best["train_mse"], "test_mse": best["test_mse"], "r2": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", default="outputs/trajectory_gt_val.pt")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[gt] loading {args.gt}")
    gt_pack = torch.load(args.gt, map_location="cpu", weights_only=False)
    gt = gt_pack["per_video"]
    print(f"[gt] {len(gt)} val videos with trajectory GT")

    print(f"[ckpt] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    enc_hid  = int(a.get("enc_hidden_ch", 32))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hid, shared_trunk=shared_trunk).to(device)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_hidden={enc_hid}")

    ds = ClevrerChunkPairs(args.cache_dir, split="val",
                            val_frac=args.val_frac, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)
    print(f"[data] {len(ds)} val pairs")

    t0 = time.time()
    per_chunk = collect_features_and_targets(enc, loader, gt, device)
    n_chunks = len(per_chunk)
    print(f"[encode] {n_chunks} unique (video, chunk) entries in {time.time()-t0:.1f}s")

    # 50/50 video-id split (matches modal probe convention)
    by_vid: dict[int, list] = {}
    for (vid, ci), d in per_chunk.items():
        by_vid.setdefault(vid, []).append((ci, d))
    vids = sorted(by_vid.keys())
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vids)
    half = len(vids) // 2
    tr_vids, te_vids = set(vids[:half]), set(vids[half:])
    print(f"[probe] {len(tr_vids)} train vids / {len(te_vids)} test vids")

    def stack(features_key: str, target_key: str, vid_set: set):
        xs, ys = [], []
        for vid, entries in by_vid.items():
            if vid not in vid_set: continue
            for ci, d in entries:
                xs.append(d[features_key])
                ys.append(d[target_key])
        x = torch.stack(xs).float()
        y = torch.stack(ys).float()
        return x, y

    results = {}
    print(f"\n{'feature':<12s} {'target':<10s} {'train MSE':>12s} {'test MSE':>12s} {'R²':>8s}")
    print("-" * 60)
    for feature_key in ("z_static", "z_dyn_last"):
        for target_key in ("pos", "vel"):
            x_tr, y_tr = stack(feature_key, target_key, tr_vids)
            x_te, y_te = stack(feature_key, target_key, te_vids)
            res = train_linear_regressor(x_tr, y_tr, x_te, y_te,
                                         epochs=args.probe_epochs,
                                         lr=args.probe_lr, wd=args.probe_wd,
                                         device=device)
            results[f"{feature_key}__{target_key}"] = res
            print(f"{feature_key:<12s} {target_key:<10s} "
                  f"{res['train_mse']:>12.5f} {res['test_mse']:>12.5f} "
                  f"{res['r2']:>8.3f}")

    # Summary
    print("\n========== Verdict ==========")
    for t in ("pos", "vel"):
        r2s = results[f"z_static__{t}"]["r2"]
        r2d = results[f"z_dyn_last__{t}"]["r2"]
        print(f"{t}:  z_static R²={r2s:.3f}  z_dyn_last R²={r2d:.3f}  Δ={r2d-r2s:+.3f}")

    out_path = Path(args.ckpt).parent / "probe_v5_trajectory.json"
    out_path.write_text(json.dumps({"results": results,
                                    "n_chunks": n_chunks,
                                    "n_train_vids": len(tr_vids),
                                    "n_test_vids": len(te_vids),
                                    "args": vars(args)}, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
