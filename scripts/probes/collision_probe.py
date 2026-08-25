"""Physical-reasoning task: collision detection on CLEVRER.

A collision is a pure *dynamics* event, so it should read from the dynamics code z_dyn,
NOT from appearance. We predict, per chunk, whether a collision occurs (from the cache's
collision_mask), with a frozen linear probe -- the same protocol as our other probes.

Methods (all frozen features):
  z_dyn      : our dynamics code (mean over time)  -- expected best
  z_static   : our static code (control: appearance, should be near base-rate)
  wanmean    : raw Wan-latent mean-pool
  wanflat    : full raw Wan latent (ceiling of what the substrate holds)
  random     : random-init encoder z_dyn
External RGB baselines (VideoMAE / VideoFlexTok) added by collision_probe_baselines
via --with_rgb + --video_root (they need pixels).

Metric: Average Precision + AUROC (collision presence is imbalanced), and F1 at 0.5.
Usage:
  python scripts/probes/collision_probe.py --ckpt .../ckpt.pt \
     --cache_dir .../wan_10000vid_W33 --max_videos 2000 --out .../collision.json
"""
from __future__ import annotations
import argparse, json, sys
from argparse import Namespace
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.clevrer_decode_baselines import build_our_encoder
from scripts.probes.clevrer_baselines_probe import mp4_path, build_extractor
from scripts.cache_wan_ssv2 import read_clip
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from torch.utils.data import DataLoader


@torch.no_grad()
def collect(cache_dir, split, a, enc, rnd, device, max_videos):
    ds = ClevrerChunkPairs(cache_dir, split=split, val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=max_videos)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=chunk_collate)
    F = {k: [] for k in ("z_dyn", "z_static", "wanmean", "wanflat", "random")}
    y, seen, keys = [], set(), []
    # collision GT isn't in the collate batch -> reload the blob per window
    cd = Path(cache_dir)
    win = {(int(w["video_id"]), int(w["start_frame"])): w["path"] for w in ds.windows}
    for b in dl:
        x = b["chunk_obs"].to(device)
        o = enc(x); ro = rnd(x)
        zs = o["z_static"].flatten(1).float().cpu().numpy()
        # NO pooling: flatten the full z_dyn (all frames + dims) so the specific-frame
        # collision signal is fully preserved for the probe.
        zd = o["z_dyn"].flatten(1).float().cpu().numpy()
        rd = ro["z_dyn"].flatten(1).float().cpu().numpy()
        wm = x.mean(dim=(2, 3, 4)).float().cpu().numpy()
        wf = x.flatten(1).float().cpu().numpy()
        for j in range(len(b["video_id"])):
            key = (int(b["video_id"][j]), int(b["start_frame"][j]))
            if key in seen:
                continue
            seen.add(key)
            blob = torch.load(cd / win[key], map_location="cpu", weights_only=False)
            coll = int(blob["collision_mask"].any().item()) if "collision_mask" in blob \
                else int(len(blob.get("collisions", [])) > 0)
            F["z_static"].append(zs[j]); F["z_dyn"].append(zd[j]); F["random"].append(rd[j])
            F["wanmean"].append(wm[j]); F["wanflat"].append(wf[j]); y.append(coll)
            keys.append(key)
    return {k: np.stack(v) for k, v in F.items()}, np.array(y), keys


@torch.no_grad()
def extract_rgb(keys, video_root, models, device, W=33):
    """RGB-baseline (VideoMAE/VideoFlexTok) features aligned to `keys` order."""
    out = {m: [] for m in models}
    exts = {m: build_extractor(m, device) for m in models}
    clip_id, clip = -1, None
    for i, (vid, sf) in enumerate(keys):
        if vid != clip_id:
            clip = read_clip(str(mp4_path(video_root, vid))); clip_id = vid
        for m in models:
            f = np.zeros(exts[m].dim, np.float32) if clip is None \
                else np.asarray(exts[m].feat(clip, [sf], W), np.float32)
            out[m].append(f)
        if (i + 1) % 400 == 0:
            print(f"  [rgb] {i+1}/{len(keys)}", flush=True)
    for e in exts.values():
        del e
    torch.cuda.empty_cache()
    return {m: np.stack(v) for m, v in out.items()}


def probe(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xva))[:, 1]
    return {"AP": round(float(average_precision_score(yva, p)), 4),
            "AUROC": round(float(roc_auc_score(yva, p)), 4),
            "F1": round(float(f1_score(yva, (p > 0.5).astype(int))), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=2000)
    ap.add_argument("--rgb_models", nargs="*", default=[],
                    help="RGB baselines (e.g. videomae videoflextok); needs --video_root")
    ap.add_argument("--video_root", default="")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_our_encoder(a, ck["encoder"], device)
    rnd = build_our_encoder(a, {k: v.clone() for k, v in ck["encoder"].items()}, device)
    for m in rnd.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rnd.eval()
    Ftr, ytr, ktr = collect(args.cache_dir, "train", a, enc, rnd, device, args.max_videos)
    Fva, yva, kva = collect(args.cache_dir, "val", a, enc, rnd, device, args.max_videos)
    methods = ["z_dyn", "z_static", "wanmean", "wanflat", "random"]
    if args.rgb_models and args.video_root:
        rtr = extract_rgb(ktr, args.video_root, args.rgb_models, device)
        rva = extract_rgb(kva, args.video_root, args.rgb_models, device)
        for m in args.rgb_models:
            Ftr[m] = rtr[m]; Fva[m] = rva[m]; methods.append(m)
    base = float(yva.mean())
    print(f"[data] train={len(ytr)} val={len(yva)} collision base-rate={base:.3f}", flush=True)
    res = {"base_rate": round(base, 4), "methods": {}}
    for m in methods:
        r = probe(Ftr[m], ytr, Fva[m], yva)
        res["methods"][m] = r
        print(f"  {m:<10} AP={r['AP']:.3f} AUROC={r['AUROC']:.3f} F1={r['F1']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nCOLLISION_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
