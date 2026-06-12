"""TODO 1 — content retrieval: does z_static find videos with similar objects?

Loads scripts/cache_val_embeddings.py output. For each query video, retrieves
top-K nearest videos by cosine in z_static space, then scores those K against
ground-truth attribute-set overlap (Jaccard) decoded from the dataset's
per-video (attrs, slot_mask).

Baselines (same retrieval skeleton, different feature):
  - z_dyn_mean    : wrong tool — should retrieve by motion, not by content
  - wan_mean      : raw Wan-latent baseline, dominated by pixel statistics
  - random        : chance baseline

Metrics:
  - precision@K @ Jaccard ≥ 0.5      (binary "hit")
  - mean Jaccard@K                    (continuous, more informative on small sets)
  - per-attribute: color / material / shape Jaccard at K=5

Output: table to stdout + JSON next to val_embeddings.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB


N_COLOR = len(COLOR_VOCAB)
N_MAT   = len(MATERIAL_VOCAB)
N_SHAPE = len(SHAPE_VOCAB)
OFF_MAT, OFF_SHAPE = N_COLOR, N_COLOR + N_MAT


def decode_attr_set(attrs: torch.Tensor, slot_mask: torch.Tensor) -> set[tuple[int, int, int]]:
    """attrs (K, A) one-hot blocks, slot_mask (K,) bool -> {(c, m, s), ...}."""
    real = slot_mask.bool()
    if not real.any():
        return set()
    block = attrs[real]
    c = block[:, :N_COLOR].argmax(dim=-1).tolist()
    m = block[:, OFF_MAT:OFF_MAT + N_MAT].argmax(dim=-1).tolist()
    s = block[:, OFF_SHAPE:OFF_SHAPE + N_SHAPE].argmax(dim=-1).tolist()
    return set(zip(c, m, s))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def project_attr_set(s: set, which: str) -> set:
    if which == "full":     return s
    idx = {"color": 0, "material": 1, "shape": 2}[which]
    return {t[idx] for t in s}


def cosine_topk(query: torch.Tensor, gallery: torch.Tensor, k: int,
                exclude: int | None = None) -> list[int]:
    """query (D,), gallery (N, D) -> indices of top-k by cosine, excluding `exclude`."""
    q = F.normalize(query.unsqueeze(0), dim=-1)
    g = F.normalize(gallery, dim=-1)
    sims = (q @ g.T).squeeze(0)
    if exclude is not None:
        sims[exclude] = -float("inf")
    return sims.topk(k).indices.tolist()


def eval_method(name: str, feat: torch.Tensor, attr_sets: list[set],
                query_idxs: list[int], k_values: list[int], threshold: float,
                rng: random.Random) -> dict:
    """Run retrieval for one method; return per-K aggregates plus per-attribute @K=5."""
    n_videos = feat.shape[0]
    per_k = {k: {"precision": [], "jaccard": []} for k in k_values}
    per_attr_at_5 = {a: [] for a in ("color", "material", "shape")}
    max_k = max(k_values + [5])

    for q_idx in query_idxs:
        q_attrs = attr_sets[q_idx]
        if name == "random":
            pool = list(range(n_videos)); pool.remove(q_idx)
            top = rng.sample(pool, max_k)
        else:
            top = cosine_topk(feat[q_idx], feat, max_k, exclude=q_idx)

        for k in k_values:
            retrieved_full = [jaccard(q_attrs, attr_sets[i]) for i in top[:k]]
            per_k[k]["precision"].append(sum(j >= threshold for j in retrieved_full) / k)
            per_k[k]["jaccard"].append(sum(retrieved_full) / k)

        for which in ("color", "material", "shape"):
            qp = project_attr_set(q_attrs, which)
            js = [jaccard(qp, project_attr_set(attr_sets[i], which)) for i in top[:5]]
            per_attr_at_5[which].append(sum(js) / 5)

    return {
        "precision_at_k": {k: sum(per_k[k]["precision"]) / len(per_k[k]["precision"]) for k in k_values},
        "jaccard_at_k":   {k: sum(per_k[k]["jaccard"])   / len(per_k[k]["jaccard"])   for k in k_values},
        "per_attr_at_5":  {a: sum(per_attr_at_5[a]) / len(per_attr_at_5[a]) for a in per_attr_at_5},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="val_embeddings.pt produced by scripts/cache_val_embeddings.py")
    ap.add_argument("--out", default=None,
                    help="Output .json; default <cache_dir>/eval_content_retrieval.json")
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--query_seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Jaccard cutoff for precision@K (full attribute set)")
    args = ap.parse_args()

    pack = torch.load(args.cache, map_location="cpu", weights_only=False)
    video_ids = pack["video_ids"]
    n_vid = len(video_ids)
    print(f"[cache] {args.cache}")
    print(f"[cache] {n_vid} videos, ckpt={pack['ckpt']}")

    # Decode per-video attribute sets
    attr_sets = []
    for vid in video_ids:
        d = pack["per_video"][vid]
        attr_sets.append(decode_attr_set(d["attrs"], d["slot_mask"]))
    set_sizes = [len(s) for s in attr_sets]
    print(f"[attrs] set sizes: min={min(set_sizes)} max={max(set_sizes)} "
          f"mean={sum(set_sizes)/len(set_sizes):.2f}")

    rng = random.Random(args.query_seed)
    query_idxs = rng.sample(range(n_vid), min(args.n_queries, n_vid))
    print(f"[query] {len(query_idxs)} videos, seed={args.query_seed}, "
          f"Ks={args.ks}, jaccard_threshold={args.threshold}\n")

    methods = {
        "z_static":   pack["video_z_static"],
        "z_dyn_mean": pack["video_z_dyn_mean"],
        "wan_mean":   pack["video_wan_mean"],
        "random":     None,
    }
    results = {}
    for name, feat in methods.items():
        results[name] = eval_method(
            name, feat if feat is not None else torch.empty(n_vid, 1),
            attr_sets, query_idxs, args.ks, args.threshold, rng,
        )

    # ---------- Table 1: full Jaccard, precision@K and mean Jaccard@K ----------
    print("=== Content retrieval: precision@K (Jaccard_full ≥ {:.2f}) ===".format(args.threshold))
    print(f"{'method':<14}" + "".join(f"  P@{k:<3}" for k in args.ks))
    for name in methods:
        row = "".join(f"  {results[name]['precision_at_k'][k]:>5.3f}" for k in args.ks)
        print(f"{name:<14}{row}")

    print("\n=== Content retrieval: mean Jaccard_full @ K (no threshold) ===")
    print(f"{'method':<14}" + "".join(f"  J@{k:<3}" for k in args.ks))
    for name in methods:
        row = "".join(f"  {results[name]['jaccard_at_k'][k]:>5.3f}" for k in args.ks)
        print(f"{name:<14}{row}")

    # ---------- Table 2: per-attribute Jaccard @ K=5 ----------
    print("\n=== Per-attribute Jaccard @ K=5 ===")
    print(f"{'method':<14}  color  material  shape")
    for name in methods:
        pa = results[name]["per_attr_at_5"]
        print(f"{name:<14}  {pa['color']:>5.3f}  {pa['material']:>8.3f}  {pa['shape']:>5.3f}")

    out_path = Path(args.out) if args.out else Path(args.cache).parent / "eval_content_retrieval.json"
    out_path.write_text(json.dumps({
        "cache": args.cache,
        "ckpt": pack["ckpt"],
        "n_queries": len(query_idxs),
        "query_seed": args.query_seed,
        "threshold": args.threshold,
        "ks": args.ks,
        "results": results,
        "attr_set_sizes": {"min": min(set_sizes), "max": max(set_sizes),
                           "mean": sum(set_sizes) / len(set_sizes)},
    }, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
