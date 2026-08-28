"""SSv2 matched-protocol decodability: our code vs frozen video encoders.

Same protocol as the CLEVRER table (sec:q3), moved to real video: freeze each
representation, train an IDENTICAL-capacity head to map it back to the Wan latent, then
decode to pixels with the frozen VAE and score against the source frames. Reporting only
latent MSE would leave the reconstruction claim untested in pixel space on real video,
which is where it matters.

Methods
  ours          z_static + z_dyn from a trained checkpoint
  wanflat       the full Wan latent -- the protocol's own ceiling
  wanmean       mean-pool of the latent -- the trivial representation
  videomae      MCG-NJU/videomae-base, mean-pooled over tokens (768)
  videoflextok  EPFL-VILAB/videoflextok_d18_d18_k600 pre-quant features (1152)
  dinov2        facebook/dinov2-base per-frame, mean-pooled (768)

Ground truth comes from the source .webm at the cached start_frame, so pixels are
compared against the real clip rather than against the VAE's own round trip. The VAE
round trip is reported as `wanflat`, which bounds every other row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_latents import load_wan_vae                  # noqa: E402
from scripts.local.eval_psnr import build, psnr, wan_decode          # noqa: E402


def read_window(path, start, W, size=128):
    """(W,3,size,size) in [-1,1] from a webm, or None."""
    import av
    try:
        c = av.open(str(path))
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
    except Exception:
        return None
    if len(fr) < start + W:
        return None
    x = torch.from_numpy(np.stack(fr[start:start + W])).float().permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x / 127.5 - 1.0


class Head(torch.nn.Module):
    """Equal-capacity decode head: feature -> Wan latent (48,T,8,8)."""
    def __init__(self, d_in, T, hidden=384, C=48, S=8):
        super().__init__()
        self.T, self.C, self.S = T, C, S
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, C * T * S * S))

    def forward(self, f):
        return self.net(f).reshape(-1, self.C, self.T, self.S, self.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="outputs/cache/ssv2_W17_full")
    ap.add_argument("--video_dir", default="datasets/ssv2/videos")
    ap.add_argument("--ckpt", default="outputs/FINAL_SSV2/ckpt.pt")
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "wanflat", "wanmean", "videomae", "dinov2"])
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--n_pixel", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the decode head; features are seed-independent")
    ap.add_argument("--feat_cache", default="",
                    help="dir to cache extracted features; RGB encoders re-decode every "
                         "webm otherwise, which dominates the runtime of a seed sweep")
    ap.add_argument("--out", default="outputs/logs/ssv2_decode.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = json.loads((Path(args.cache_dir) / "metadata.json").read_text())
    wins = meta["windows"]; W = int(meta["args"]["window_frames"])
    rng = np.random.RandomState(0); idx = rng.permutation(len(wins))
    tr_i = idx[: args.n_train]; va_i = idx[args.n_train: args.n_train + args.n_val]

    def latents(ii):
        return torch.stack([torch.load(Path(args.cache_dir) / wins[i]["path"],
                                       map_location="cpu", weights_only=False)["latent"].float()
                            for i in ii])
    Ltr, Lva = latents(tr_i), latents(va_i)
    T = Ltr.shape[2]
    print(f"[data] {len(tr_i)} train / {len(va_i)} val windows, T_lat={T}, W={W}", flush=True)

    res = {}
    vae = None
    fc = Path(args.feat_cache) if args.feat_cache else None
    if fc:
        fc.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        # ---- features ----------------------------------------------------
        cpath = (fc / f"{method}_{args.n_train}_{args.n_val}.pt") if fc else None
        if cpath is not None and cpath.exists() and method != "vae":
            d = torch.load(cpath, map_location="cpu", weights_only=False)
            Ftr, Fva = d["tr"], d["va"]
            print(f"  [{method}] features from cache", flush=True)
        elif method == "vae":
            # Substrate ceiling: decode the TRUE latent, no head at all. Separates how
            # much fidelity the frozen VAE loses from how much the shared decode head
            # loses -- on real video the head, not the VAE, turns out to bind.
            if vae is None:
                vae = load_wan_vae("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch.float16, dev)
            ps = []
            for i in range(0, min(args.n_pixel, len(va_i)), 4):
                ii = va_i[i:i + 4]
                gt = [read_window(Path(args.video_dir) / f"{wins[k]['video_id']}.webm",
                                  int(wins[k]["start_frame"]), W) for k in ii]
                ok = [n for n, g in enumerate(gt) if g is not None]
                if not ok:
                    continue
                g = torch.stack([gt[n] for n in ok]).permute(0, 2, 1, 3, 4).to(dev)
                pix = wan_decode(vae, Lva[i:i + 4][ok].to(dev), dev, torch.float16)
                t = min(g.shape[2], pix.shape[2])
                ps.append(psnr(pix[:, :, :t], g[:, :, :t]).cpu())
            p = float(torch.cat(ps).mean()) if ps else float("nan")
            res[method] = {"dim": int(np.prod(Lva.shape[1:])), "latent_mse": 0.0, "psnr": p}
            print(f"[{method:<13}] dim {int(np.prod(Lva.shape[1:])):>6}  "
                  f"latent_mse 0.000000  PSNR {p:6.2f} dB", flush=True)
            Path(args.out).write_text(json.dumps(res, indent=2))
            continue
        elif method == "wanflat":
            Ftr, Fva = Ltr.flatten(1), Lva.flatten(1)
        elif method == "wanmean":
            Ftr, Fva = Ltr.mean(dim=(2, 3, 4)), Lva.mean(dim=(2, 3, 4))
        elif method == "ours":
            enc, _, a = build(args.ckpt, dev)
            def ours(L):
                out = []
                with torch.no_grad():
                    for i in range(0, len(L), 64):
                        g, z, _ = enc(L[i:i + 64].to(dev).unsqueeze(1))
                        out.append(torch.cat([g[:, 0].flatten(1), z[:, 0].flatten(1)], 1).cpu())
                return torch.cat(out)
            Ftr, Fva = ours(Ltr), ours(Lva)
        else:
            sys.path.insert(0, "ml-videoflextok")
            from scripts.probes.clevrer_baselines_probe import build_extractor
            ext = build_extractor(method, dev)
            def rgb(ii):
                out = []
                for k, i in enumerate(ii):
                    w = wins[i]
                    p = Path(args.video_dir) / f"{w['video_id']}.webm"
                    clip = read_window(p, int(w["start_frame"]), W)
                    if clip is None:
                        out.append(np.zeros(ext.dim, np.float32))
                    else:
                        c = ((clip + 1) * 127.5).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                        out.append(np.asarray(ext.feat(c, [0], W), np.float32))
                    if (k + 1) % 500 == 0:
                        print(f"  [{method}] {k+1}/{len(ii)}", flush=True)
                return torch.from_numpy(np.stack(out))
            Ftr, Fva = rgb(tr_i), rgb(va_i)
            del ext; torch.cuda.empty_cache()

        if cpath is not None and not cpath.exists() and method != "vae":
            torch.save({"tr": Ftr, "va": Fva}, cpath)

        # ---- equal-capacity head ----------------------------------------
        mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-5
        ftr, fva = ((Ftr - mu) / sd).to(dev), ((Fva - mu) / sd).to(dev)
        ytr, yva = Ltr.to(dev), Lva.to(dev)
        head = Head(ftr.shape[1], T).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(args.epochs):
            perm = torch.randperm(len(ftr), device=dev)
            for i in range(0, len(ftr), 64):
                j = perm[i:i + 64]
                loss = F.mse_loss(head(ftr[j]), ytr[j])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = torch.cat([head(fva[i:i + 64]) for i in range(0, len(fva), 64)])
            mse = float(F.mse_loss(pred, yva))

        # ---- pixels -------------------------------------------------------
        if vae is None:
            vae = load_wan_vae("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch.float16, dev)
        ps = []
        for i in range(0, min(args.n_pixel, len(va_i)), 4):
            ii = va_i[i:i + 4]
            gt = [read_window(Path(args.video_dir) / f"{wins[k]['video_id']}.webm",
                              int(wins[k]["start_frame"]), W) for k in ii]
            ok = [n for n, g in enumerate(gt) if g is not None]
            if not ok:
                continue
            g = torch.stack([gt[n] for n in ok]).permute(0, 2, 1, 3, 4).to(dev)
            pix = wan_decode(vae, pred[i:i + 4][ok], dev, torch.float16)
            t = min(g.shape[2], pix.shape[2])
            ps.append(psnr(pix[:, :, :t], g[:, :, :t]).cpu())
        p = float(torch.cat(ps).mean()) if ps else float("nan")
        res[method] = {"dim": int(Ftr.shape[1]), "latent_mse": mse, "psnr": p}
        print(f"[{method:<13}] dim {Ftr.shape[1]:>6}  latent_mse {mse:.6f}  PSNR {p:6.2f} dB",
              flush=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print("SSV2_DECODE_OK")


if __name__ == "__main__":
    main()
