"""Table: LIBERO action read-out from each code, on HELD-OUT tasks.

This is the control task for the factorization. Actions are motion, so a working split
predicts z_dyn > z_static here -- the mirror image of CLEVRER attributes, where
z_static should win. A model that merely compresses well has no reason to produce that
asymmetry, so the ORDERING is the result, not the absolute accuracy.

The target is WINDOW-LOCAL: for the chunk covering frames [s, s+W) the label is the
action actually commanded over those frames. This is what makes the test sharp. There is
one z_static for the whole demo, so it *cannot* say which window it is being asked
about; if it nonetheless predicts window-local actions as well as z_dyn does, the codes
are not split. Pooling the target over the whole demo -- and pooling z_dyn over time,
which discards precisely the signal actions are made of -- would hide that.

LIBERO actions are 7-dim: six continuous DoF (translation + rotation) and a BINARY
gripper in the last dim. They are scored separately, ridge for the six and logistic for
the gripper; regressing the gripper as a seventh continuous dim mixes the two scales.

Controls included, because an ablation number alone is not evidence:
  entangled   one shared conv trunk feeding both heads -- cannot route factors apart
  random      the committed architecture, untrained -- isolates training from prior
  wanmean     mean-pool of the raw frozen latent -- the trivial representation

Held-out protocol: whole TASKS are excluded from probe training, so the probe cannot
memorise a task's action statistics.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.ssv2_sequence import SSv2Sequence                   # noqa: E402
from scripts.local.eval_psnr import build                          # noqa: E402
from src.model.memory_encoder import MemoryEncoder                 # noqa: E402


def load_scenes(split_json):
    """video_id -> scene id. The scene is the STATIC target for the dissociation: it is a
    property of the environment that holds for the whole demo and is unchanged by what the
    arm does, so a correctly split z_static should read it and z_dyn should not."""
    rows = json.loads(Path(split_json).read_text())
    names, out = {}, {}
    for r in rows:
        m = re.match(r"([A-Z_]*SCENE\d+)", r["template"])
        nm = m.group(1) if m else "NA"
        out[int(r["id"])] = names.setdefault(nm, len(names))
    return out, {v: k for k, v in names.items()}


def load_actions(split_json, act_root):
    """video_id -> (T,7) action array."""
    rows = json.loads(Path(split_json).read_text())
    out = {}
    for r in rows:
        p = Path(act_root) / Path(r["action"]).name
        if p.exists():
            out[int(r["id"])] = np.load(p)
    return out


@torch.no_grad()
def features(enc, loader, dev, acts, W, scenes):
    """One row per (demo, chunk); z_dyn keeps its time axis."""
    ZS, ZD, ZDP, WM, A, G, TASK, SC = [], [], [], [], [], [], [], []
    for b in loader:
        seq = b["latents"].to(dev)
        g, z, _ = enc(seq)
        sf = b["start_frames"].numpy()
        for k in range(seq.shape[1]):
            keep = []
            for n, vid in enumerate(b["video_id"].tolist()):
                a = acts.get(int(vid))
                s = int(sf[n, k])
                if a is None or len(a) < s + 2:      # window must overlap the demo
                    continue
                w = a[s: s + W]
                A.append(w[:, :6].mean(0)); G.append(int(w[:, 6].mean() > 0))
                SC.append(scenes.get(int(vid), -1))
                keep.append(n)
            if not keep:
                continue
            kp = torch.tensor(keep, device=dev)
            ZS.append(g[kp, k].flatten(1).cpu())
            ZD.append(z[kp, k].flatten(1).cpu())          # FULL temporal code
            ZDP.append(z[kp, k].mean(1).cpu())            # time-pooled, for reference
            WM.append(seq[kp, k].mean(dim=(2, 3, 4)).cpu())
            TASK.extend([b["label_id"][n].item() for n in keep])
    return (torch.cat(ZS).numpy(), torch.cat(ZD).numpy(), torch.cat(ZDP).numpy(),
            torch.cat(WM).numpy(), np.stack(A), np.array(G), np.array(TASK),
            np.array(SC))


def probe(Xtr, Xte, Atr, Ate, Gtr, Gte):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    xt, xv = sc.transform(Xtr), sc.transform(Xte)
    pred = Ridge(alpha=1.0).fit(xt, Atr).predict(xv)
    mse = float(np.mean((pred - Ate) ** 2))
    # R^2 against predicting the training mean: MSE alone cannot say whether a probe
    # learned anything, since the action scale differs per split.
    r2 = float(1 - ((pred - Ate) ** 2).sum() / (((Ate - Atr.mean(0)) ** 2).sum() + 1e-12))
    if len(np.unique(Gtr)) > 1 and len(np.unique(Gte)) > 1:
        acc = float((LogisticRegression(max_iter=1000).fit(xt, Gtr).predict(xv) == Gte).mean())
    else:
        acc = float("nan")
    return mse, r2, acc


def scene_probe(Xtr, Xte, Str, Ste):
    """Accuracy on the STATIC target, plus the majority-class rate for reference."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), Str)
    acc = float((m.predict(sc.transform(Xte)) == Ste).mean())
    maj = float((Ste == np.bincount(Str).argmax()).mean())
    return acc, maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", default="outputs/cache/libero_W17")
    ap.add_argument("--split_json", default="datasets/libero/split.json")
    ap.add_argument("--act_root", default="/home/licho/libero_90_processed/actions")
    ap.add_argument("--holdout_tasks", type=int, default=15)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=4)
    ap.add_argument("--entangled", action="store_true")
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", default="outputs/logs/libero_probe.json")
    args = ap.parse_args()
    dev = torch.device("cuda")

    W = int(json.loads((Path(args.cache_dir) / "metadata.json").read_text())
            ["args"]["window_frames"])
    ds = SSv2Sequence(args.cache_dir, args.n_chunks, args.max_videos, "train", val_frac=0.0)
    dl = DataLoader(ds, batch_size=32, num_workers=4)
    acts = load_actions(args.split_json, args.act_root)
    print(f"[data] {len(ds)} demos x {args.n_chunks} chunks, W={W}, "
          f"{len(acts)} action files", flush=True)

    enc, _, a = build(args.ckpt, dev)
    if args.random_init or args.entangled:
        enc = MemoryEncoder(hidden_ch=a["enc_hidden_ch"], d_static=a["d_static"],
                            static_grid=a["static_grid"], d_dyn=a["d_dyn"],
                            dyn_grid=a["dyn_grid"], mem_update=a["mem_update"],
                            mem_collapse=a["mem_collapse"], d_pose=a["d_pose"],
                            chunk_size_lat=a.get("chunk_size_lat", 5),
                            shared_trunk=args.entangled).to(dev)
        if args.entangled:   # entangled needs its own trained weights; random does not
            st = torch.load(args.ckpt, map_location="cpu", weights_only=False)["enc"]
            try:
                enc.load_state_dict(st)
            except Exception:
                print("[warn] entangled ckpt shape mismatch; using random init", flush=True)
        enc.eval()

    scenes, scene_names = load_scenes(args.split_json)
    ZS, ZD, ZDP, WM, A, G, TASK, SC = features(enc, dl, dev, acts, W, scenes)
    tasks = np.unique(TASK)
    # Cap the holdout at a third of the tasks: with few distinct tasks an unclamped
    # request empties the probe's training set instead of holding anything out.
    n_hold = max(1, min(args.holdout_tasks, len(tasks) // 3))
    held = set(np.random.RandomState(0).permutation(tasks)[:n_hold].tolist())
    te = np.isin(TASK, list(held)); tr = ~te
    if tr.sum() < 20 or te.sum() < 20:
        raise SystemExit(f"[fatal] degenerate task split: {tr.sum()} train / {te.sum()} "
                         f"held-out windows over {len(tasks)} tasks -- lower --holdout_tasks")
    print(f"[split] {tr.sum()} train / {te.sum()} held-out windows "
          f"({n_hold} unseen tasks of {len(tasks)}); gripper base rate "
          f"{G[te].mean():.3f}\n", flush=True)

    # A held-out task whose scene never appears in training is unpredictable by
    # construction; score the static target only on scenes the probe has seen.
    seen = set(np.unique(SC[tr]).tolist())
    sm = np.array([s in seen for s in SC])
    te_s = te & sm
    print(f"[scene] {len(seen)} scenes seen in train; scoring {te_s.sum()} of "
          f"{te.sum()} held-out windows\n", flush=True)

    res = {}
    print(f"{'feature':<24}{'dim':>6}{'ACTION R2':>11}{'SCENE acc':>11}{'gripper':>9}")
    print('-' * 62)
    rows = (("z_static", ZS), ("z_dyn (full temporal)", ZD), ("z_dyn (time-pooled)", ZDP),
            ("z_static+z_dyn", np.concatenate([ZS, ZD], 1)), ("raw latent mean-pool", WM))
    maj = None
    for nm, X in rows:
        m, r2, acc = probe(X[tr], X[te], A[tr], A[te], G[tr], G[te])
        sacc, maj = scene_probe(X[tr], X[te_s], SC[tr], SC[te_s])
        res[nm] = {"mse_6dof": m, "r2": r2, "gripper_acc": acc,
                   "scene_acc": sacc, "dim": int(X.shape[1])}
        print(f"{nm:<24}{X.shape[1]:>6}{r2:>11.3f}{sacc:>11.3f}{acc:>9.3f}", flush=True)
    res["_scene_majority"] = maj
    res["_n_scenes"] = len(seen)
    Path(args.out).write_text(json.dumps({args.label: res}, indent=2))
    print(f"\n(scene majority-class baseline {maj:.3f}, {len(seen)} scenes)")
    print("\nDOUBLE DISSOCIATION: a real split needs z_dyn ABOVE z_static on the action"
          "\ntarget AND BELOW it on the scene target. One direction alone is also produced"
          "\nby a code that is merely bigger or merely better trained.")
    print("LIBERO_PROBE_OK")


if __name__ == "__main__":
    main()
