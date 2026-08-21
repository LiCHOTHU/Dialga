"""Leg-2 (decodable): how much pixel-reconstructable information does each frozen
representation retain?  Fair, measured, no assertions.

Protocol (identical across methods): freeze the representation, train ONE
equal-capacity broadcast decoder head to map the per-window feature vector back to
the Wan latent x=(48,9,8,8) with MSE (the same objective train_v5 optimizes), and
report best val latent-MSE + Wan-decoded pixel PSNR on a held-out subset. A strong
semantic encoder (DINOv2/VideoMAE) that discards appearance will reconstruct the
latent WORSE than our code; a pure tokenizer (VideoFlexTok) will do well on pixels
but -- see the Leg-1 table -- pays for it in semantics. The paper's point is the
JOINT frontier: DIALGA is the only representation strong on BOTH axes at low rate.

Rows:
  ours          : z_static + z_dyn from our checkpoint (the representation we ship)
  videomae      : MCG-NJU/videomae-base clip feature (768)
  videoflextok  : continuous pre-quant encoder feature (1152)
  dinov2        : per-frame CLS mean (768)
All fed through the SAME decoder head + SAME train/val split.

Usage:
  python scripts/probes/clevrer_decode_baselines.py --ckpt .../ckpt.pt \
     --cache_dir .../wan_10000vid_W33 \
     --video_root .../CLEVRER/train_video \
     --methods ours videomae videoflextok dinov2 --max_videos 2000 \
     --pixel --out .../clevrer_decode_baselines.json
"""
from __future__ import annotations
import argparse, json, sys, time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_ssv2 import read_clip
from scripts.probes.clevrer_baselines_probe import mp4_path, build_extractor
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_encoder import LatentEncoder3D


class FeatBroadcastDecoder(nn.Module):
    """feature vector (D) -> Wan latent (48,9,8,8). Fixed capacity for all methods."""
    def __init__(self, d_in, hidden=384, T=9, H=8, W=8, C=48):
        super().__init__()
        self.T, self.H, self.W, self.C = T, H, W, C
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden))
        self.pos = nn.Parameter(torch.randn(1, hidden, T, H, W) * 0.02)
        self.conv = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv3d(hidden, C, 1))

    def forward(self, f):
        h = self.proj(f)[:, :, None, None, None] + self.pos      # (B,hidden,T,H,W)
        return self.conv(h)


def build_our_encoder(a, state, device):
    def g(k, d):
        return a[k] if k in a else d
    d_pose = int(g("d_pose", 0)) if g("use_camera_pose", False) else 0
    enc = LatentEncoder3D(
        d_static=int(g("d_static", 96)), d_dyn=int(g("d_dyn", 256)),
        hidden_ch=int(g("enc_hidden_ch", 192)),
        use_layer_norm=("norm_static.weight" in state),
        pool_type=g("pool_type", "mean"),
        static_grid=int(g("static_grid", 4) or 4),
        n_queries=int(g("pool_queries", 8)), n_heads=int(g("pool_heads", 4)),
        shared_trunk=bool(g("shared_trunk", False)),
        chunk_size_lat=int(g("chunk_size_lat", 9)),
        static_agg=g("static_agg", "conv"),
        dyn_spatial=bool(g("dyn_spatial", False)), dyn_grid=int(g("dyn_grid", 8)),
        d_pose=d_pose).to(device)
    enc.load_state_dict(state); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def our_feature(enc, x):
    z = enc(x)
    s = z["z_static"].flatten(1)
    d = z["z_dyn"].flatten(1)
    return torch.cat([s, d], dim=1)          # (B, d_static'+d_dyn')


WAN_METHODS = ("ours", "random", "wanmean", "wanflat")  # computed from the Wan latent


def gather(method, split, a, args, enc, device):
    """Return (feat[N,D] on cpu, wanlat[N,48,9,8,8] on cpu). One pass, dedup window.

    ablation methods (no external RGB): 'random' = random-init copy of our encoder
    (isolates whether *training*, not architecture, causes decodability); 'wanmean'
    = mean-pool of the frozen Wan latent (does our code beat trivial pooling on
    decode?); 'wanflat' = full latent (upper bound)."""
    ds = ClevrerChunkPairs(a["cache_dir"], split=split,
                           val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=args.max_videos)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=chunk_collate)
    ext = None if method in WAN_METHODS else build_extractor(method, device)
    rnd = None
    if method == "random":
        # same architecture, freshly initialised (weights NOT loaded)
        st = {k: v for k, v in torch.load(args.ckpt, map_location="cpu",
                                          weights_only=False)["encoder"].items()}
        rnd = build_our_encoder(a, st, device)
        for m in rnd.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        rnd.eval()
    seen = set()
    feats, lats = [], []
    clip_id, clip = -1, None
    for b in dl:
        x = b["chunk_obs"]
        if method == "ours":
            fb = our_feature(enc, x.to(device)).cpu()
        elif method == "random":
            fb = our_feature(rnd, x.to(device)).cpu()
        elif method == "wanmean":
            fb = x.mean(dim=(2, 3, 4))                        # (B,48)
        elif method == "wanflat":
            fb = x.flatten(1)                                 # (B,27648)
        for j in range(len(b["video_id"])):
            vid, sf = int(b["video_id"][j]), int(b["start_frame"][j])
            if (vid, sf) in seen:
                continue
            seen.add((vid, sf))
            if method in WAN_METHODS:
                feats.append(fb[j])
            else:
                if vid != clip_id:
                    clip = read_clip(str(mp4_path(args.video_root, vid))); clip_id = vid
                if clip is None:
                    continue
                feats.append(torch.from_numpy(ext.feat(clip, [sf], args.window_frames)).float())
            lats.append(x[j])
    if ext is not None:
        del ext; torch.cuda.empty_cache()
    return torch.stack(feats), torch.stack(lats)


def train_decoder(Ftr, Ltr, Fva, Lva, device, epochs, lr, hidden):
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
    Ftr, Fva = (Ftr - mu) / sd, (Fva - mu) / sd
    dec = FeatBroadcastDecoder(Ftr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    Ftr_d, Ltr_d = Ftr.to(device), Ltr.to(device)
    bs, n = 128, Ftr.shape[0]
    best = float("inf")
    for ep in range(epochs):
        dec.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(dec(Ftr_d[idx]), Ltr_d[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            dec.eval()
            with torch.no_grad():
                v = 0.0
                for i in range(0, Fva.shape[0], bs):
                    fv = Fva[i:i + bs].to(device); lv = Lva[i:i + bs].to(device)
                    v += F.mse_loss(dec(fv), lv, reduction="sum").item()
                v /= (Lva.numel())
            best = min(best, v)
    return best, dec, (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "videomae", "videoflextok", "dinov2"])
    ap.add_argument("--max_videos", type=int, default=2000)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dec_hidden", type=int, default=384)
    ap.add_argument("--pca_to", type=int, nargs="*", default=[],
                    help="also decode each method's features PCA-reduced to these dims "
                         "(matched-budget decode comparison, defuses the dim confound)")
    ap.add_argument("--pixel", action="store_true", help="also Wan-decode a val subset -> PSNR")
    ap.add_argument("--pixel_n", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_our_encoder(a, ck["encoder"], device)

    vae = None
    if args.pixel:
        try:
            from scripts.cache_wan_ssv2 import load_wan_vae
            vae = load_wan_vae(a.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
                               torch.bfloat16, device)
        except Exception as e:
            print(f"[warn] pixel decode unavailable ({type(e).__name__}: {e}); "
                  f"reporting latent-MSE only", flush=True)
            vae = None

    from sklearn.decomposition import PCA
    results = {}
    for method in args.methods:
        t0 = time.time()
        Ftr, Ltr = gather(method, "train", a, args, enc, device)
        Fva, Lva = gather(method, "val", a, args, enc, device)
        full_dim = int(Ftr.shape[1])
        # native full-dim decode + matched-budget PCA decodes (only dims < full)
        row = {"dim": full_dim, "n_train": int(Ftr.shape[0]),
               "n_val": int(Fva.shape[0]), "at_dim": {}}
        for d in [full_dim] + [k for k in args.pca_to if k < full_dim]:
            if d == full_dim:
                Zt, Zv = Ftr, Fva
            else:
                pca = PCA(n_components=d, random_state=0).fit(Ftr.numpy())
                Zt = torch.from_numpy(pca.transform(Ftr.numpy())).float()
                Zv = torch.from_numpy(pca.transform(Fva.numpy())).float()
            best, dec, norm = train_decoder(Zt, Ltr, Zv, Lva, device,
                                            args.epochs, args.lr, args.dec_hidden)
            entry = {"val_latent_mse": round(best, 6)}
            if args.pixel and vae is not None and d == full_dim:
                entry["val_pixel_psnr"] = round(
                    pixel_psnr(dec, norm, Zv, Lva, vae, device, args.pixel_n), 3)
            row["at_dim"][str(d)] = entry
            print(f"[{method}] @dim {d:<5} val_latent_mse={best:.6f}", flush=True)
        row["val_latent_mse"] = row["at_dim"][str(full_dim)]["val_latent_mse"]  # back-compat
        results[method] = row
        print(f"[{method}] full_dim={full_dim} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nDECODE_BASELINES_DONE", flush=True)


@torch.no_grad()
def pixel_psnr(dec, norm, Fva, Lva, vae, device, n):
    mu, sd = norm
    dec.eval()
    idx = torch.arange(min(n, Fva.shape[0]))
    f = ((Fva[idx] - mu.cpu()) / sd.cpu()).to(device)
    pred = dec(f)
    gt = Lva[idx].to(device)
    def dec_pix(lat):
        return vae.decode(lat.to(vae.dtype)).sample.float().clamp(-1, 1)
    p = dec_pix(pred); g = dec_pix(gt)
    mse = F.mse_loss(p, g, reduction="mean").item()
    return 10.0 * np.log10(4.0 / max(mse, 1e-10))   # range [-1,1] -> peak^2=4


if __name__ == "__main__":
    main()
