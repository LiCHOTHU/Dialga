"""LIBERO frozen-feature action-readout probe (paper Table tab:q3_action).

The v2a downstream claim: DIALGA's dynamics code z_dyn is action-relevant. Freeze
the encoder, extract per-chunk features, and read out the 7-DoF action (ridge
regression -> MSE) and the gripper open/close (logistic -> accuracy), on held-out
LIBERO tasks. Compares which slot / baseline best predicts action:
  z_dyn (motion, expected best), z_static, raw Wan-mean, random-init.

Usage:
  python scripts/probes/libero_action_probe.py --ckpt .../libero_v59/v5_best.pt \\
      --cache_dir .../libero_90_wan_W33 --out .../q3_action.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.ssv2_action_probe import build_encoder
from src.data.libero_window import LiberoChunkPairs


@torch.no_grad()
def extract(ds, enc, rnd, device):
    F = {k: [] for k in ("z_dyn", "z_static", "wanmean", "random")}
    A, G = [], []
    for i in range(len(ds)):
        s = ds[i]
        lat = s["chunk_obs"].unsqueeze(0).to(device)
        o = enc(lat)
        F["z_dyn"].append(o["z_dyn"][0].float().mean(0).cpu().numpy())
        F["z_static"].append(o["z_static"][0].float().cpu().numpy())
        F["wanmean"].append(s["chunk_obs"].mean(dim=(1, 2, 3)).numpy())
        F["random"].append(rnd(lat)["z_dyn"][0].float().mean(0).cpu().numpy())
        act = s["actions_obs"].float()                      # (33,7)
        A.append(act.mean(0).numpy())                       # mean action over chunk
        G.append(int((act[:, -1] > 0).float().mean() > 0.5))  # gripper majority
    return {k: np.stack(v) for k, v in F.items()}, np.stack(A), np.array(G)


def probe(Xtr, Xte, Atr, Ate, Gtr, Gte):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); Xt, Xv = sc.transform(Xtr), sc.transform(Xte)
    mse = float(np.mean((Ridge(alpha=1.0).fit(Xt, Atr).predict(Xv) - Ate) ** 2))
    if len(np.unique(Gtr)) > 1:
        acc = float((LogisticRegression(max_iter=500).fit(Xt, Gtr).predict(Xv) == Gte).mean())
    else:
        acc = float((Gte == Gtr[0]).mean())
    return round(mse, 4), round(acc, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, _ = build_encoder(ck, dev); enc.load_state_dict(ck["encoder"]); enc.eval()
    rnd, _ = build_encoder(ck, dev); rnd.eval()

    tr = LiberoChunkPairs(args.cache_dir, split="train")
    te = LiberoChunkPairs(args.cache_dir, split="val")
    Ftr, Atr, Gtr = extract(tr, enc, rnd, dev)
    Fte, Ate, Gte = extract(te, enc, rnd, dev)
    print(f"[data] train={len(Atr)} val={len(Ate)} chunks")

    rows = {"random": "random-init $\\zd$", "wanmean": "Wan mean-pool",
            "z_static": "$\\zs$ (control)", "z_dyn": "$\\zd$ (ours)"}
    res = {"ckpt": args.ckpt}
    print(f"{'feature':22s}{'Act-MSE':>10s}{'Grip-acc':>10s}")
    for k in rows:
        mse, acc = probe(Ftr[k], Fte[k], Atr, Ate, Gtr, Gte)
        res[k] = {"dim": int(Ftr[k].shape[1]), "act_mse": mse, "grip_acc": acc}
        print(f"{k:22s}{mse:10.4f}{acc:10.3f}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
