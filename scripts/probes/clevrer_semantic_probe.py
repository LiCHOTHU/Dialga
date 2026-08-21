"""CLEVRER per-attribute semantic + cross-probe on the FINAL model (fills Q1/Q2).

Extracts frozen z_static and z_dyn on CLEVRER val and reports attribute-presence
mAP per group (colour 8 / material 2 / shape 3) for each slot:
  - z_static mAP  -> Q1 semantic rows + Q2 identity(zs) cell
  - z_dyn    mAP  -> Q2 identity-leakage(zd) cell (want ~chance)
Reusable build_encoder handles the spatial-z_dyn checkpoint. Writes JSON.

Usage:
  python scripts/probes/clevrer_semantic_probe.py --ckpt .../v5_best.pt \\
      --cache_dir .../wan_10000vid_W33 --max_videos 1500 --out .../clevrer_sem.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.ssv2_action_probe import build_encoder

GROUPS = {"color": slice(0, 8), "material": slice(8, 10), "shape": slice(10, 13)}


@torch.no_grad()
def extract(cache_dir, enc, device, max_videos):
    meta = json.loads((Path(cache_dir) / "metadata.json").read_text())
    wins = meta["windows"][:max_videos] if max_videos else meta["windows"]
    ZS, ZD, PR = [], [], []
    for w in wins:
        b = torch.load(Path(cache_dir) / w["path"], map_location="cpu", weights_only=False)
        o = enc(b["latent"].unsqueeze(0).to(device))
        ZS.append(o["z_static"][0].float().cpu().numpy())
        ZD.append(o["z_dyn"][0].float().mean(0).cpu().numpy())
        a = b["attrs"].float(); m = b["slot_mask"].bool()
        PR.append((a[m] > 0.5).any(0).numpy().astype(np.float32))
    return np.stack(ZS), np.stack(ZD), np.stack(PR)


def group_map(X, PR, tr, va):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    sc = StandardScaler().fit(X[tr]); Xt, Xv = sc.transform(X[tr]), sc.transform(X[va])
    out = {}
    for g, sl in GROUPS.items():
        aps = []
        for j in range(sl.start, sl.stop):
            y = PR[:, j]
            if 0 < y[tr].sum() < y[tr].shape[0]:
                clf = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xt, y[tr])
                aps.append(average_precision_score(y[va], clf.predict_proba(Xv)[:, 1]))
        out[g] = round(float(np.mean(aps)), 3) if aps else None
    out["mean"] = round(float(np.mean([v for v in out.values() if v is not None])), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=1500); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, _ = build_encoder(ck, dev); enc.load_state_dict(ck["encoder"]); enc.eval()
    ZS, ZD, PR = extract(args.cache_dir, enc, dev, args.max_videos)
    n = len(ZS); tr = slice(0, int(.8 * n)); va = slice(int(.8 * n), n)
    res = {"ckpt": args.ckpt, "n": n,
           "z_static": group_map(ZS, PR, tr, va),
           "z_dyn": group_map(ZD, PR, tr, va)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[Q1] z_static mAP: {res['z_static']}")
    print(f"[Q2 leakage] z_dyn mAP (want low): {res['z_dyn']}")


if __name__ == "__main__":
    main()
