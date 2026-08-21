"""DINOv2 baseline on SSv2 (fills the DINOv2 row of Table tab:ssv2).

Per-frame DINOv2 features (facebook/dinov2-base, CLS token), mean-pooled over
sampled frames -> clip feature; frozen linear probe over 174 SSv2 classes, same
protocol as our SSv2 probe. Reads webm frames directly.

Usage:
  python scripts/probes/ssv2_dinov2_probe.py \\
     --video_root .../ssv2/videos_extracted/20bn-something-something-v2 \\
     --label_json .../ssv2/labels.json --train_json .../ssv2/train.json \\
     --val_json .../ssv2/validation.json --max_train 4000 --max_test 1500 \\
     --out .../ssv2_dinov2.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_ssv2 import read_clip, window_starts


def parse_split(split_json, label_json):
    lab = json.loads(Path(label_json).read_text())               # template -> str(id)
    items = []
    for e in json.loads(Path(split_json).read_text()):
        tmpl = e.get("template", "").replace("[", "").replace("]", "")
        items.append((str(e["id"]), int(lab.get(tmpl, -1))))
    return items


class DINOv2Extractor:
    dim = 768

    def __init__(self, device):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def feat(self, clip, starts, W, n=8):
        outs = []
        for s in starts:
            idx = np.linspace(s, s + W - 1, n).round().astype(int)
            idx = np.clip(idx, 0, clip.shape[0] - 1)
            f = torch.from_numpy(clip[idx]).float().permute(0, 3, 1, 2) / 255.
            f = F.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
            f = (f - self.mean) / self.std
            o = self.m(f.to(self.device)).last_hidden_state[:, 0]   # (n,768) CLS
            outs.append(o.mean(0).float().cpu().numpy())
        return np.mean(outs, axis=0)


def extract_split(ext, items, root, W, K, max_n, tag):
    root = Path(root); X, y = [], []; n = 0; t0 = time.time()
    for vid, lab in items:
        if max_n and n >= max_n:
            break
        if lab < 0:
            continue
        clip = read_clip(str(root / f"{vid}.webm"))
        if clip is None:
            continue
        starts = window_starts(clip.shape[0], W, K)
        if not starts:
            continue
        X.append(ext.feat(clip, starts, W)); y.append(lab); n += 1
        if n % 300 == 0:
            print(f"  [{tag}] {n} clips {time.time()-t0:.0f}s", flush=True)
    return np.stack(X), np.array(y)


def probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte)); cls = clf.classes_
    top1 = (cls[p.argmax(1)] == yte).mean()
    t5 = np.argsort(-p, 1)[:, :5]
    top5 = np.mean([yte[i] in cls[t5[i]] for i in range(len(yte))])
    return float(top1 * 100), float(top5 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2", choices=["dinov2", "videomae", "videoflextok"])
    ap.add_argument("--video_root", required=True); ap.add_argument("--label_json", required=True)
    ap.add_argument("--train_json", required=True); ap.add_argument("--val_json", required=True)
    ap.add_argument("--window_frames", type=int, default=33); ap.add_argument("--windows_per_video", type=int, default=3)
    ap.add_argument("--max_train", type=int, default=4000); ap.add_argument("--max_test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[96, 256, 352])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tr = parse_split(args.train_json, args.label_json); te = parse_split(args.val_json, args.label_json)
    np.random.RandomState(args.seed).shuffle(tr); np.random.RandomState(args.seed).shuffle(te)
    if args.model == "dinov2":
        ext = DINOv2Extractor(dev)
    else:
        from scripts.probes.ucf101_baselines_probe import VideoMAEExtractor, VideoFlexTokExtractor
        ext = {"videomae": VideoMAEExtractor, "videoflextok": VideoFlexTokExtractor}[args.model](dev)
    W, K = args.window_frames, args.windows_per_video
    Xtr, ytr = extract_split(ext, tr, args.video_root, W, K, args.max_train, "train")
    Xte, yte = extract_split(ext, te, args.video_root, W, K, args.max_test, "test")
    ncl = len(np.unique(np.concatenate([ytr, yte])))
    res = {"model": args.model, "full_dim": ext.dim, "n_train": len(ytr),
           "n_test": len(yte), "n_classes": ncl, "at_dim": {}}
    from sklearn.decomposition import PCA
    for d in [ext.dim] + [k for k in args.pca_dims if k < ext.dim]:
        if d == ext.dim:
            Xt, Xv = Xtr, Xte
        else:
            pca = PCA(n_components=d, random_state=0).fit(Xtr)
            Xt, Xv = pca.transform(Xtr), pca.transform(Xte)
        t1, t5 = probe(Xt, ytr, Xv, yte)
        res["at_dim"][str(d)] = {"top1": round(t1, 2), "top5": round(t5, 2)}
        print(f"[RESULT] {args.model} SSv2 @dim {d}: top1={t1:.2f}% top5={t5:.2f}%", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
