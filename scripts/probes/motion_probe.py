"""z_dyn motion probe: 'how does the video move?'

The dynamics code z_dyn should read MOTION (speed, motion energy, # moving objects)
and be INVARIANT to appearance. We test two things, all with frozen linear probes:

 (A) Motion content -- regress window-level motion descriptors from each frozen rep:
       mean_speed, max_speed, motion_energy, n_moving  (from GT velocities).
       Expect z_dyn >> z_static, and z_dyn to beat wanmean/random.

 (B) Appearance-shift robustness -- THE headline. Train the motion probe on objects
       of one MATERIAL (e.g. metal) and test on the other (rubber). z_dyn is
       appearance-blind (reads color at chance) so it should transfer; VideoMAE
       entangles appearance into motion, so it should drop more under the shift.
       Report in-domain R^2, shifted R^2, and the gap (drop) per method.

Methods: z_dyn (no pooling, flatten), z_static (control), wanmean, wanflat, random,
and optional RGB baselines (videomae/videoflextok) via --rgb_models + --video_root.

Usage:
  python scripts/probes/motion_probe.py --ckpt .../ckpt.pt \
     --cache_dir .../wan_10000vid_W33 --max_videos 2500 --out .../motion.json \
     [--rgb_models videomae --video_root .../train_video]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.clevrer_decode_baselines import build_our_encoder
from scripts.probes.clevrer_baselines_probe import mp4_path, build_extractor
from scripts.cache_wan_ssv2 import read_clip
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from torch.utils.data import DataLoader

TAU = 0.02  # speed threshold (normalized units) for "moving"


def motion_targets(vel, vis):
    """vel (T,N,2), vis (T,N) bool -> dict of window-level motion scalars + material split.
    speed per obj-frame = |vel|; aggregate only over visible obj-frames."""
    speed = np.linalg.norm(vel, axis=-1)               # (T,N)
    v = vis.astype(bool)
    if v.sum() == 0:
        return dict(mean_speed=0.0, max_speed=0.0, motion_energy=0.0, n_moving=0.0)
    sp = speed[v]
    # per-object peak speed over frames (obj visible in >=1 frame)
    obj_vis = v.any(0)                                  # (N,)
    peak = np.where(v, speed, 0.0).max(0)              # (N,)
    n_moving = float(((peak > TAU) & obj_vis).sum())
    return dict(mean_speed=float(sp.mean()),
                max_speed=float(sp.max()),
                motion_energy=float((speed**2 * v).sum() / max(v.sum(), 1)),
                n_moving=n_moving)


@torch.no_grad()
def collect(cache_dir, split, a, enc, rnd, device, max_videos):
    ds = ClevrerChunkPairs(cache_dir, split=split, val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=max_videos)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=chunk_collate)
    # NOTE: mean-pool dropped as a baseline (too weak to be meaningful). Real baselines
    # = wanflat (full-latent ceiling), pretrained encoders (RGB), + random floor.
    F = {k: [] for k in ("z_dyn", "z_static", "wanflat", "random")}
    Y = {k: [] for k in ("mean_speed", "max_speed", "motion_energy", "n_moving")}
    mat, keys, seen = [], [], set()
    cd = Path(cache_dir)
    win = {(int(w["video_id"]), int(w["start_frame"])): w["path"] for w in ds.windows}
    for b in dl:
        x = b["chunk_obs"].to(device)
        o = enc(x); ro = rnd(x)
        zs = o["z_static"].flatten(1).float().cpu().numpy()
        zd = o["z_dyn"].flatten(1).float().cpu().numpy()   # NO pooling
        rd = ro["z_dyn"].flatten(1).float().cpu().numpy()
        wf = x.flatten(1).float().cpu().numpy()
        for j in range(len(b["video_id"])):
            key = (int(b["video_id"][j]), int(b["start_frame"][j]))
            if key in seen:
                continue
            seen.add(key)
            blob = torch.load(cd / win[key], map_location="cpu", weights_only=False)
            vel = blob["velocities"].numpy(); vis = blob["visibility"].numpy()
            t = motion_targets(vel, vis)
            for k in Y:
                Y[k].append(t[k])
            # window material: majority material among real (slot_mask) objects.
            # attrs (N,13): color[0:8], material[8:10], shape[10:13]. material=1 -> metal.
            attrs = blob["attrs"].numpy(); sm = blob["slot_mask"].numpy().astype(bool)
            m = attrs[sm, 8:10].argmax(1) if sm.sum() else np.array([0])
            mat.append(int(m.mean() >= 0.5))   # 1 if majority metal else rubber
            F["z_static"].append(zs[j]); F["z_dyn"].append(zd[j]); F["random"].append(rd[j])
            F["wanflat"].append(wf[j]); keys.append(key)
    return ({k: np.stack(v) for k, v in F.items()},
            {k: np.array(v, np.float32) for k, v in Y.items()},
            np.array(mat), keys)


@torch.no_grad()
def extract_rgb(keys, video_root, models, device, W=33):
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


def r2(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    return round(float(r2_score(yva, m.predict(sc.transform(Xva)))), 4)


def acc(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return round(float((clf.predict(sc.transform(Xva)) == yva).mean()), 4)


def build_shortcut_idx(y, a, rng, n_cell, bias=0.9):
    """Return (train_idx, test_id_idx, test_ood_idx) index arrays.
    TRAIN: appearance a spuriously correlated with label y (aligned cells y==a dominate).
    TEST-ID: same correlation (y==a). TEST-OOD: reversed (y!=a) -> appearance shortcut
    now points to the WRONG label. All splits are label-balanced on y."""
    cell = {(yy, aa): rng.permutation(np.where((y == yy) & (a == aa))[0])
            for yy in (0, 1) for aa in (0, 1)}
    avail = min(len(v) for v in cell.values())
    n = min(n_cell, avail)
    if n < 15:
        return None
    m = max(1, int(round(n * (1 - bias) / bias)))          # conflict count in train
    # train: aligned cells get n, conflict cells get m  (corr strength ~ bias)
    tr = np.concatenate([cell[(1, 1)][:n], cell[(0, 0)][:n],
                         cell[(1, 0)][:m], cell[(0, 1)][:m]])
    # test-ID (aligned) and test-OOD (conflict) drawn from the REMAINING rows, y-balanced
    k = min(n // 3 + 5, avail - n)
    if k < 10:
        k = min(n, avail - m)
    id_ = np.concatenate([cell[(1, 1)][n:n + k], cell[(0, 0)][n:n + k]])
    ood = np.concatenate([cell[(1, 0)][m:m + k], cell[(0, 1)][m:m + k]])
    return tr, id_, ood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=2500)
    ap.add_argument("--rgb_models", nargs="*", default=[])
    ap.add_argument("--video_root", default="")
    ap.add_argument("--bias_sweep", type=float, nargs="*", default=[0.9],
                    help="spurious-correlation strengths to sweep (train appr~label corr)")
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

    Ftr, Ytr, Mtr, Ktr = collect(args.cache_dir, "train", a, enc, rnd, device, args.max_videos)
    Fva, Yva, Mva, Kva = collect(args.cache_dir, "val", a, enc, rnd, device, args.max_videos)
    methods = ["z_dyn", "z_static", "wanflat", "random"]
    if args.rgb_models and args.video_root:
        rtr = extract_rgb(Ktr, args.video_root, args.rgb_models, device)
        rva = extract_rgb(Kva, args.video_root, args.rgb_models, device)
        for m in args.rgb_models:
            Ftr[m] = rtr[m]; Fva[m] = rva[m]; methods.append(m)

    targets = ["mean_speed", "motion_energy"]     # clean motion targets (dropped noisy max/n)
    print(f"[data] train={len(Mtr)} val={len(Mva)} "
          f"metal_frac tr={Mtr.mean():.2f} va={Mva.mean():.2f}", flush=True)
    res = {"n_train": int(len(Mtr)), "n_val": int(len(Mva)),
           "content": {}, "shortcut": {}}

    # (A) motion content: does z_dyn read HOW things move? in-domain R^2 per target.
    print("== (A) motion content (R^2, in-domain) ==", flush=True)
    for m in methods:
        res["content"][m] = {t: r2(Ftr[m], Ytr[t], Fva[m], Yva[t]) for t in targets}
        print(f"  {m:<12} " + " ".join(f"{t}={res['content'][m][t]:.3f}" for t in targets), flush=True)

    # (B) spurious-correlation OOD test. Label = fast(1)/slow(0) by median mean_speed.
    #   TRAIN: appearance (metal/rubber) spuriously correlated with fast/slow.
    #   TEST-OOD: correlation reversed -> appearance shortcut points to WRONG label.
    # Entangled reps (videomae/wan_flat) exploit the shortcut and break OOD; z_dyn is
    # appearance-blind so it must use real motion -> should keep OOD accuracy. THE win.
    thr = float(np.median(Ytr["mean_speed"]))
    ytr_b = (Ytr["mean_speed"] > thr).astype(int); yva_b = (Yva["mean_speed"] > thr).astype(int)
    # Sweep spurious-correlation strength: as bias -> 1, entangled reps lean harder on
    # the appearance shortcut and should collapse OOD, while z_dyn (appearance-blind)
    # holds. Look for a CROSSOVER where z_dyn's acc_ood overtakes videomae's.
    for bias in args.bias_sweep:
        itr = build_shortcut_idx(ytr_b, Mtr, np.random.default_rng(0), n_cell=1200, bias=bias)
        iva = build_shortcut_idx(yva_b, Mva, np.random.default_rng(1), n_cell=1200, bias=bias)
        if itr is None or iva is None:
            print(f"== (B) shortcut bias={bias}: too few per cell, skipped ==", flush=True)
            continue
        tr_idx = itr[0]; id_idx = iva[1]; ood_idx = iva[2]
        corr = float(np.mean(ytr_b[tr_idx] == Mtr[tr_idx]))
        key = f"bias_{bias:.2f}"
        res["shortcut"][key] = {"corr": round(corr, 3), "n_tr": int(len(tr_idx)),
                                "n_ood": int(len(ood_idx)), "methods": {}}
        print(f"== (B) spurious-correlation OOD bias={bias} (corr={corr:.2f}, "
              f"n_tr={len(tr_idx)} n_ood={len(ood_idx)}) ==", flush=True)
        for m in methods:
            a_id = acc(Ftr[m][tr_idx], ytr_b[tr_idx], Fva[m][id_idx], yva_b[id_idx])
            a_ood = acc(Ftr[m][tr_idx], ytr_b[tr_idx], Fva[m][ood_idx], yva_b[ood_idx])
            res["shortcut"][key]["methods"][m] = {
                "acc_id": a_id, "acc_ood": a_ood, "drop": round(a_id - a_ood, 4)}
            print(f"  {m:<12} acc_id={a_id:.3f} acc_ood={a_ood:.3f} drop={a_id-a_ood:+.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nMOTION_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
