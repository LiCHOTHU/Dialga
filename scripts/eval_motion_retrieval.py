"""TODO 2 — motion retrieval: does z_dyn find chunks with similar motion?

Per-chunk retrieval (motion varies across the 3 chunks of one video). For each
query chunk, retrieves top-K nearest chunks by cosine in z_dyn space and scores
them against motion ground-truth derived from outputs/trajectory_gt_val.pt:

  - mean_speed     (scalar)   — mean object velocity magnitude over visible slots
  - direction_hist (8 bins)   — angular hist of velocity directions (moving only)
  - collision      (bool)     — collision present in chunk

SAME-VIDEO chunks are excluded from the gallery — without this filter the top-K
trivially returns the other 2 chunks of the same video (similar motion by
construction).

Baselines:
  - z_dyn_last : the claim (last-frame z_dyn)
  - z_dyn_mean : same channel, time-averaged
  - z_static   : wrong tool — should retrieve by content, not motion
  - wan_mean   : raw Wan-latent baseline
  - random     : chance

Output: table to stdout + JSON next to val_embeddings.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def chunk_idx(start_frame: int) -> int:
    # Cache uses deterministic chunk starts [0, 33, 66]
    return {0: 0, 33: 1, 66: 2}[start_frame]


def motion_descriptor(pos, vel, vis, collision):
    """Per-chunk descriptor. pos/vel (K, 2), vis (K,) bool, collision: bool/int."""
    real = vis.bool()
    if not real.any():
        return {"speed": 0.0, "dir_hist": torch.zeros(8), "collision": int(bool(collision))}
    v = vel[real]
    speeds = v.norm(dim=-1)
    mean_speed = float(speeds.mean().item())
    moving = speeds > 1e-4
    if moving.any():
        vm = v[moving]
        angles = torch.atan2(vm[:, 1], vm[:, 0])
        hist = torch.histc(angles, bins=8, min=-math.pi, max=math.pi)
        if hist.sum() > 0:
            hist = hist / hist.sum()
    else:
        hist = torch.zeros(8)
    return {"speed": mean_speed, "dir_hist": hist, "collision": int(bool(collision))}


def motion_sim(a: dict, b: dict, speed_scale: float) -> dict:
    speed_sim = max(0.0, 1.0 - abs(a["speed"] - b["speed"]) / max(speed_scale, 1e-6))
    da, db = a["dir_hist"], b["dir_hist"]
    if da.sum() > 0 and db.sum() > 0:
        dir_sim = float(F.cosine_similarity(da.unsqueeze(0), db.unsqueeze(0)).item())
    else:
        dir_sim = 0.0
    coll_sim = 1.0 if a["collision"] == b["collision"] else 0.0
    return {
        "speed": speed_sim,
        "direction": dir_sim,
        "collision": coll_sim,
        "combined": (speed_sim + dir_sim + coll_sim) / 3,
    }


def cosine_topk_filtered(query_feat, gallery_feat, k: int, gallery_vids, query_vid):
    """Top-k indices by cosine, excluding all entries with gallery_vids[i] == query_vid."""
    q = F.normalize(query_feat.unsqueeze(0), dim=-1)
    g = F.normalize(gallery_feat, dim=-1)
    sims = (q @ g.T).squeeze(0)
    mask = torch.tensor([v == query_vid for v in gallery_vids])
    sims[mask] = -float("inf")
    return sims.topk(k).indices.tolist()


def eval_method(name, feat, descs, chunk_keys, query_idxs, ks, speed_scale, rng):
    gallery_vids = [k[0] for k in chunk_keys]
    n = len(chunk_keys)
    per_k_combined = {k: [] for k in ks}
    per_dim_at_5 = {d: [] for d in ("speed", "direction", "collision")}
    max_k = max(ks + [5])

    for q_idx in query_idxs:
        q_vid = chunk_keys[q_idx][0]
        if name == "random":
            pool = [i for i in range(n) if gallery_vids[i] != q_vid]
            top = rng.sample(pool, max_k)
        else:
            top = cosine_topk_filtered(feat[q_idx], feat, max_k, gallery_vids, q_vid)
        q_desc = descs[q_idx]
        sims = [motion_sim(q_desc, descs[i], speed_scale) for i in top]
        for k in ks:
            per_k_combined[k].append(sum(s["combined"] for s in sims[:k]) / k)
        for d in per_dim_at_5:
            per_dim_at_5[d].append(sum(s[d] for s in sims[:5]) / 5)

    return {
        "combined_at_k": {k: sum(per_k_combined[k]) / len(per_k_combined[k]) for k in ks},
        "per_dim_at_5":  {d: sum(per_dim_at_5[d]) / len(per_dim_at_5[d]) for d in per_dim_at_5},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gt", default="outputs/trajectory_gt_val.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--query_seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    pack = torch.load(args.cache, map_location="cpu", weights_only=False)
    chunk_keys = pack["chunk_keys"]
    print(f"[cache] {args.cache}  ({len(chunk_keys)} chunks)")

    gt = torch.load(args.gt, map_location="cpu", weights_only=False)["per_video"]
    gt = {int(k): v for k, v in gt.items()}
    print(f"[gt] {len(gt)} videos with trajectory GT")

    # Per-chunk motion descriptors, in chunk_keys order
    descs = []
    keep_mask = []
    for (vid, sf) in chunk_keys:
        if vid not in gt:
            descs.append(None); keep_mask.append(False); continue
        ci = chunk_idx(sf)
        g = gt[vid]
        descs.append(motion_descriptor(
            g["per_chunk_positions"][ci], g["per_chunk_velocities"][ci],
            g["per_chunk_visibility"][ci], g["chunk_collision"][ci].item(),
        ))
        keep_mask.append(True)
    n_kept = sum(keep_mask)
    print(f"[gt] {n_kept}/{len(chunk_keys)} chunks have GT descriptors")

    # Normalize speed against the 95th percentile so similarity ∈ [0, 1] is meaningful
    speeds = torch.tensor([d["speed"] for d, k in zip(descs, keep_mask) if k])
    speed_scale = float(speeds.quantile(0.95).item())
    print(f"[norm] speed p50={speeds.median().item():.3f}  p95={speed_scale:.3f}  "
          f"max={speeds.max().item():.3f}  (used p95 as similarity normalizer)")

    # Empirical chance: how often does a random chunk match collision?
    coll_pos = sum(d["collision"] for d in descs if d is not None) / max(n_kept, 1)
    chance_coll = coll_pos ** 2 + (1 - coll_pos) ** 2
    print(f"[chance] collision-presence positive rate {coll_pos:.3f}, "
          f"random-pair match rate {chance_coll:.3f}\n")

    # Query selection (over chunks that have GT)
    eligible = [i for i, ok in enumerate(keep_mask) if ok]
    rng = random.Random(args.query_seed)
    query_idxs = rng.sample(eligible, min(args.n_queries, len(eligible)))
    print(f"[query] {len(query_idxs)} chunks, seed={args.query_seed}, "
          f"Ks={args.ks}, same-video chunks filtered from gallery\n")

    methods = {
        "z_dyn_last": pack["z_dyn_last"],
        "z_dyn_mean": pack["z_dyn_mean"],
        "z_static":   pack["z_static"],
        "wan_mean":   pack["wan_mean"],
        "random":     None,
    }
    results = {}
    for name, feat in methods.items():
        results[name] = eval_method(
            name, feat if feat is not None else torch.empty(len(chunk_keys), 1),
            descs, chunk_keys, query_idxs, args.ks, speed_scale, rng,
        )

    # ---------- Table 1: combined similarity@K ----------
    print("=== Motion retrieval: mean combined-similarity @ K  (higher better) ===")
    print(f"{'method':<14}" + "".join(f"  @{k:<3}" for k in args.ks))
    for name in methods:
        row = "".join(f"  {results[name]['combined_at_k'][k]:>5.3f}" for k in args.ks)
        print(f"{name:<14}{row}")

    # ---------- Table 2: per-dimension @ K=5 ----------
    print("\n=== Per-dimension motion similarity @ K=5 ===")
    print(f"{'method':<14}  speed  direction  collision")
    for name in methods:
        pd = results[name]["per_dim_at_5"]
        print(f"{name:<14}  {pd['speed']:>5.3f}  {pd['direction']:>9.3f}  {pd['collision']:>9.3f}")

    out_path = Path(args.out) if args.out else Path(args.cache).parent / "eval_motion_retrieval.json"
    out_path.write_text(json.dumps({
        "cache": args.cache,
        "ckpt": pack["ckpt"],
        "n_queries": len(query_idxs),
        "query_seed": args.query_seed,
        "ks": args.ks,
        "speed_scale_p95": speed_scale,
        "collision_chance_match": chance_coll,
        "results": results,
    }, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
