"""UCF101 frozen-feature action-recognition probe for EXTERNAL baselines.

Runs a video-representation baseline (VideoMAE or VideoFlexTok) on the SAME
UCF101 clips / splits our model uses, with the SAME protocol (frozen features ->
linear probe -> top-1/5 over 101 classes), so the comparison in the paper is
apples-to-apples. Reads .avi frames directly (baselines want RGB, not our Wan
latent); samples the same 3 windows/clip and averages them into one clip feature.

Feature per model (frozen, no fine-tuning):
  videomae     : MCG-NJU/videomae-base, 16 frames @224, mean over patch tokens (768-d)
  videoflextok : EPFL-VILAB/videoflextok_d18_d18_k600, 17 frames @128,
                 continuous PRE-quant encoder output `enc_packed_seq`
                 (mean over 2560 tokens -> 1152-d), captured via a forward hook.

Usage:
  python scripts/probes/ucf101_baselines_probe.py --model videomae \\
    --video_root .../ucf101/UCF-101 \\
    --split_dir  .../ucf101/ucfTrainTestlist \\
    --max_train 4000 --max_test 1500 --out .../ucf101_videomae.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_ssv2 import read_clip, window_starts


def parse_split(split_path, class_ind):
    name2id = {}
    for line in Path(class_ind).read_text().splitlines():
        if line.strip():
            i, name = line.split()
            name2id[name] = int(i) - 1
    items = []
    for line in Path(split_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        items.append((rel, name2id.get(rel.split("/")[0], -1)))
    return items


def sample_frames(clip, start, n, W):
    """Pick n frames evenly from the W-frame window [start:start+W]. -> (n,H,W,3) u8."""
    idx = np.linspace(start, start + W - 1, n).round().astype(int)
    idx = np.clip(idx, 0, clip.shape[0] - 1)
    return clip[idx]


class VideoMAEExtractor:
    dim = 768

    def __init__(self, device):
        from transformers import VideoMAEModel
        self.m = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def feat(self, clip, starts, W):
        outs = []
        for s in starts:
            f = sample_frames(clip, s, 16, W)                       # (16,H,W,3)
            t = torch.from_numpy(f).float().permute(0, 3, 1, 2) / 255.
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            t = t.unsqueeze(0)                                       # (1,16,3,224,224)
            t = (t - self.mean) / self.std
            o = self.m(t.to(self.device)).last_hidden_state          # (1,N,768)
            outs.append(o.mean(1).float().cpu().numpy()[0])
        return np.mean(outs, axis=0)


class VideoFlexTokExtractor:
    dim = 1152

    def __init__(self, device):
        sys.path.insert(0, "ml-videoflextok")
        from videoflextok.wrappers import VideoFlexTokFromHub
        self.m = VideoFlexTokFromHub.from_pretrained(
            "EPFL-VILAB/videoflextok_d18_d18_k600").to(device).eval()
        self.device = device
        self.T = int(getattr(self.m, "chunk_size", 17) or 17)
        self._cap = {}
        for n, mod in self.m.named_modules():
            if "fsq" in type(mod).__name__.lower():
                mod.register_forward_hook(self._hook)
                break

    def _hook(self, mod, args, result):
        d = args[0]
        if isinstance(d, dict) and "enc_packed_seq" in d:
            self._cap["z"] = d["enc_packed_seq"].detach()

    @torch.no_grad()
    def feat(self, clip, starts, W):
        outs = []
        for s in starts:
            f = sample_frames(clip, s, self.T, W)                    # (17,H,W,3)
            t = torch.from_numpy(f).float().permute(3, 0, 1, 2) / 255.  # (3,17,H,W)
            t = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)
            t = (t * 2 - 1).unsqueeze(0)                             # (1,3,17,128,128)
            self._cap.clear()
            _ = self.m.tokenize(t.to(self.device))
            z = self._cap["z"]                                        # (1,2560,1152)
            outs.append(z.mean(1).float().cpu().numpy()[0])
        return np.mean(outs, axis=0)


def extract_split(ext, items, root, W, K, max_n, tag):
    root = Path(root)
    X, y = [], []
    t0 = time.time()
    n = 0
    for rel, lab in items:
        if max_n and n >= max_n:
            break
        if lab < 0:
            continue
        clip = read_clip(str(root / rel))
        if clip is None:
            continue
        starts = window_starts(clip.shape[0], W, K)
        if not starts:
            continue
        X.append(ext.feat(clip, starts, W))
        y.append(lab)
        n += 1
        if n % 200 == 0:
            print(f"  [{tag}] {n} clips  {time.time()-t0:.0f}s", flush=True)
    return np.stack(X), np.array(y)


def probe(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    cls = clf.classes_
    top1 = (cls[proba.argmax(1)] == yte).mean()
    top5i = np.argsort(-proba, 1)[:, :5]
    top5 = np.mean([yte[i] in cls[top5i[i]] for i in range(len(yte))])
    return float(top1 * 100), float(top5 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["videomae", "videoflextok"])
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--windows_per_video", type=int, default=3)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[96, 256, 352],
                    help="also probe PCA-reduced-to-K features (matched-low-dim comparison)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sd = Path(args.split_dir)
    tr = parse_split(sd / "trainlist01.txt", sd / "classInd.txt")
    te = parse_split(sd / "testlist01.txt", sd / "classInd.txt")
    np.random.RandomState(args.seed).shuffle(tr)
    np.random.RandomState(args.seed).shuffle(te)

    ext = {"videomae": VideoMAEExtractor,
           "videoflextok": VideoFlexTokExtractor}[args.model](device)
    print(f"[model] {args.model} dim={ext.dim}")

    W, K = args.window_frames, args.windows_per_video
    Xtr, ytr = extract_split(ext, tr, args.video_root, W, K, args.max_train, "train")
    Xte, yte = extract_split(ext, te, args.video_root, W, K, args.max_test, "test")
    print(f"[extract] train {Xtr.shape} test {Xte.shape}")
    n_cls = len(np.unique(np.concatenate([ytr, yte])))
    res = {"model": args.model, "full_dim": ext.dim, "n_train": len(ytr),
           "n_test": len(yte), "n_classes": n_cls, "at_dim": {}}
    # full dim + PCA to matched low dims (the "same low-dim manifold" comparison)
    from sklearn.decomposition import PCA
    for d in [ext.dim] + [k for k in args.pca_dims if k < ext.dim]:
        if d == ext.dim:
            Xt, Xv = Xtr, Xte
        else:
            pca = PCA(n_components=d, random_state=0).fit(Xtr)
            Xt, Xv = pca.transform(Xtr), pca.transform(Xte)
        t1, t5 = probe(Xt, ytr, Xv, yte)
        res["at_dim"][str(d)] = {"top1": round(t1, 2), "top5": round(t5, 2)}
        print(f"[RESULT] {args.model} @dim {d}: top1={t1:.2f}% top5={t5:.2f}%", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
