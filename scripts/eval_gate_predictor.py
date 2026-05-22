"""scripts/eval_gate_predictor.py — TODO 5.

Does the trained GatePredictor identify collision chunks at inference time
*without* the GT annotation? Report precision/recall/F1/AUC on val.

Targets: gate_GT (1 = collision happens in chunk_pred, 0 = no collision).
Predictor: sigmoid(gp(z_dyn_obs[:, -1])) > threshold.

Threshold scan to also report the best-F1 operating point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.event_head import GatePredictor
from src.model.latent_encoder import LatentEncoder3D


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC; assumes labels in {0, 1}, scores any real. Mann-Whitney U."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # rank-based AUC: AUC = (mean rank of positives - (n_pos+1)/2) / n_neg
    combined = np.concatenate([pos, neg])
    ranks = combined.argsort().argsort() + 1
    rank_sum_pos = ranks[: pos.size].sum()
    n_pos, n_neg = pos.size, neg.size
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _prf(pred_b: np.ndarray, gt_b: np.ndarray):
    tp = int(((pred_b == 1) & (gt_b == 1)).sum())
    fp = int(((pred_b == 1) & (gt_b == 0)).sum())
    fn = int(((pred_b == 0) & (gt_b == 1)).sum())
    tn = int(((pred_b == 0) & (gt_b == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v5_events.pt (with trained gate_predictor)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    enc_hid  = int(a.get("enc_hidden_ch", 32))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hid, shared_trunk=shared_trunk).to(device)
    gp = GatePredictor(d_dyn=d_dyn).to(device)
    enc.load_state_dict(ckpt["encoder"])
    gp.load_state_dict(ckpt["gate_predictor"])
    enc.eval(); gp.eval()
    print(f"[model] loaded encoder + gate_predictor from {args.ckpt}")

    ds = ClevrerChunkPairs(args.cache_dir, split="val",
                            val_frac=args.val_frac, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)

    scores, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            chunk_obs = batch["chunk_obs"].to(device)
            out = enc(chunk_obs)
            z_dyn_last = out["z_dyn"][:, -1]
            logits = gp(z_dyn_last)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["gate_GT"].numpy())
    print(f"[encode] {time.time()-t0:.1f}s")

    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(np.int64)

    auc = _auc_roc(scores, labels)
    base_rate = float(labels.mean())
    print(f"\nval base rate (P(gate=1)): {base_rate:.3f}")
    print(f"ROC AUC:                   {auc:.3f}")
    print()

    # Threshold scan
    print(f"{'thresh':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'TN':>6s} "
          f"{'prec':>7s} {'recall':>7s} {'F1':>7s} {'acc':>7s}")
    print("-" * 75)
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    threshold_results = {}
    best_f1 = (0.0, None)
    for t in thresholds:
        pred_b = (scores > t).astype(np.int64)
        m = _prf(pred_b, labels)
        threshold_results[f"{t:.2f}"] = m
        print(f"{t:>7.2f} {m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d} {m['tn']:>6d} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['accuracy']:>7.3f}")
        if m["f1"] > best_f1[0]:
            best_f1 = (m["f1"], t)
    print(f"\n[best F1] {best_f1[0]:.3f} at threshold={best_f1[1]:.2f}")

    out = {
        "auc": auc, "base_rate": base_rate, "n_chunks": int(labels.size),
        "best_f1": best_f1[0], "best_f1_threshold": best_f1[1],
        "thresholds": threshold_results,
        "args": vars(args),
    }
    out_path = Path(args.ckpt).parent / "gate_f1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
