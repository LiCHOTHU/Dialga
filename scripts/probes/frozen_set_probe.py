"""Frozen-encoder set-prediction probe on z_static.

WHY THIS EXISTS
---------------
The in-training AttrsHead readout (train_v5.py `_modal_labels`) is not a valid
semantic metric:

  1. It collapses the per-slot attribute table into ONE modal class per video via
     `counts.argmax()`. CLEVRER videos hold ~3-6 objects with mostly DISTINCT
     colors, so the counts tie at 1 and argmax silently returns the LOWEST class
     index present. The label is therefore "the smallest colour index in the set"
     — an arbitrary set-statistic, memorisable but not predictable.
  2. It is trained JOINTLY with lambda_attrs=0.5 backpropping into the encoder, so
     it is neither self-supervised nor a clean probe.

Result: every arm's val CE bottoms near chance (ln8+ln2+ln3 = 3.87) around ep15
then diverges to ~16 (confidently wrong) while train CE -> 0.03. That is pure
memorisation and it cannot rank architectures.

THIS PROBE
----------
Standard representation-learning protocol instead:

  FROZEN encoder -> z_static (96 floats, budget-matched across arms)
                 -> [single Linear] -> BCE over PRESENCE of each class

Labels are permutation-invariant multi-hot sets ("which colours/materials/shapes
appear in this video"), which is well-posed and generalisable. Only the probe
trains; the encoder never receives gradient. Reported against a base-rate prior
baseline so "beats chance" is unambiguous.

z_static is the right tensor for every arm: pool=mean gives the 96-d global
vector, pool=spatial gives `z_static_grid.flatten(1)` which is also 96-d.

CPU-friendly: needs only the cached Wan latents (no VAE decode).
"""
from __future__ import annotations
import argparse, json, sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_encoder import LatentEncoder3D

N_COLOR, N_MATERIAL, N_SHAPE = len(COLOR_VOCAB), len(MATERIAL_VOCAB), len(SHAPE_VOCAB)
GROUPS = [("color", 0, N_COLOR),
          ("material", N_COLOR, N_COLOR + N_MATERIAL),
          ("shape", N_COLOR + N_MATERIAL, N_COLOR + N_MATERIAL + N_SHAPE)]
N_ALL = N_COLOR + N_MATERIAL + N_SHAPE


def presence_labels(attrs: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant multi-hot: which classes appear in this video.

    attrs     : (B, K, A) one-hot blocks
    slot_mask : (B, K) bool
    Returns   : (B, A) float in {0,1}
    """
    m = slot_mask.float().unsqueeze(-1)                   # (B, K, 1)
    return (attrs.float() * m).sum(dim=1).clamp(max=1.0)  # (B, A)


def build_encoder(a: Namespace, state: dict, device) -> LatentEncoder3D:
    enc = LatentEncoder3D(
        d_static=a.d_static, d_dyn=a.d_dyn, hidden_ch=a.enc_hidden_ch,
        use_layer_norm=("norm_static.weight" in state),
        shared_trunk=getattr(a, "shared_trunk", False),
        pool_type=getattr(a, "pool_type", "mean"),
        static_grid=int(getattr(a, "static_grid", 4) or 4),
        n_queries=getattr(a, "pool_queries", 8),
        n_heads=getattr(a, "pool_heads", 4),
    ).to(device)
    enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def embed_split(enc, a: Namespace, split: str, device, max_batches: int):
    ds = ClevrerChunkPairs(a.cache_dir, split=split, val_frac=a.val_frac,
                           seed=a.seed, max_videos=a.max_videos)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                    collate_fn=chunk_collate)
    Z, Y = [], []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        z = enc(b["chunk_obs"].to(device))["z_static"]     # (B, 96) for mean AND spatial
        Z.append(z.float().cpu())
        Y.append(presence_labels(b["attrs"], b["slot_mask"]))
    return torch.cat(Z), torch.cat(Y)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AP (area under precision-recall), sklearn-free."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")                        # degenerate class, undefined
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


def eval_probe(logits: torch.Tensor, Y: torch.Tensor) -> dict:
    s = logits.numpy()
    y = Y.numpy()
    out = {}
    for name, lo, hi in GROUPS:
        aps = [average_precision(s[:, c], y[:, c]) for c in range(lo, hi)]
        aps = [x for x in aps if not np.isnan(x)]
        out[name] = float(np.mean(aps)) if aps else float("nan")
    return out


def baseline_prior(Ytr: torch.Tensor, Yva: torch.Tensor) -> dict:
    """Base-rate prior: score every val sample by the TRAIN frequency of the class.

    Constant per class -> AP equals the class positive rate. This is the
    'no information in z' floor.
    """
    prior = Ytr.mean(dim=0, keepdim=True).expand_as(Yva)
    return eval_probe(prior, Yva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows, base = [], None

    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]
            a = Namespace(**a) if isinstance(a, dict) else a
            enc = build_encoder(a, ck["encoder"], device)

            Ztr, Ytr = embed_split(enc, a, "train", device, args.max_batches)
            Zva, Yva = embed_split(enc, a, "val", device, args.max_batches)
            # standardise on TRAIN stats only
            mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True) + 1e-6
            Ztr, Zva = (Ztr - mu) / sd, (Zva - mu) / sd
            print(f"[{label}] pool={getattr(a,'pool_type','mean')} d_static={a.d_static} "
                  f"train={tuple(Ztr.shape)} val={tuple(Zva.shape)}", flush=True)

            if base is None:
                base = baseline_prior(Ytr, Yva)

            probe = nn.Linear(Ztr.shape[1], N_ALL).to(device)
            opt = torch.optim.AdamW(probe.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
            lossf = nn.BCEWithLogitsLoss()
            Ztr_d, Ytr_d = Ztr.to(device), Ytr.to(device)
            Zva_d = Zva.to(device)
            best, best_ep = None, -1
            for ep in range(args.epochs):
                probe.train()
                opt.zero_grad()
                loss = lossf(probe(Ztr_d), Ytr_d)
                loss.backward()
                opt.step()
                if (ep + 1) % 10 == 0:
                    probe.eval()
                    with torch.no_grad():
                        m = eval_probe(probe(Zva_d).cpu(), Yva)
                    score = np.nanmean([m[g] for g, _, _ in GROUPS])
                    if best is None or score > best[0]:
                        best, best_ep = (score, m), ep + 1
            rows.append((label, ck.get("epoch"), getattr(a, "pool_type", "mean"),
                         best[1], best[0], best_ep))
            print(f"  -> mAP color={best[1]['color']:.3f} material={best[1]['material']:.3f} "
                  f"shape={best[1]['shape']:.3f} | mean={best[0]:.3f} (probe ep{best_ep})",
                  flush=True)
        except Exception as e:
            print(f"  {label:<22} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'='*76}\nFROZEN set-prediction probe on z_static (val mAP, higher=better)\n{'='*76}")
    print(f"{'run':<22}{'ep':>5}{'pool':>9}{'color':>9}{'material':>10}{'shape':>8}{'mean':>8}")
    print("-" * 76)
    if base:
        bm = np.nanmean([base[g] for g, _, _ in GROUPS])
        print(f"{'base-rate prior':<22}{'':>5}{'':>9}{base['color']:>9.3f}"
              f"{base['material']:>10.3f}{base['shape']:>8.3f}{bm:>8.3f}")
    for label, ep, pool, m, score, _ in sorted(rows, key=lambda r: -r[4]):
        print(f"{label:<22}{str(ep):>5}{pool:>9}{m['color']:>9.3f}"
              f"{m['material']:>10.3f}{m['shape']:>8.3f}{score:>8.3f}")
    print("\nA run only carries generalisable object identity if it beats the "
          "base-rate prior.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"baseline": base,
                   "runs": [{"label": l, "epoch": e, "pool": p, "mAP": m,
                             "mean": s, "probe_ep": pe} for l, e, p, m, s, pe in rows]},
                  open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")
    print("FROZEN_SET_PROBE_DONE")


if __name__ == "__main__":
    main()
