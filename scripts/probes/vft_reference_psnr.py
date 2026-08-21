"""Run the REAL pretrained VideoFlexTok tokenizer on CLEVRER clips and measure
reconstruction PSNR. This is the honest ceiling check for the flow-decoder
premise: their full stack (VidTok VAE + register-token resampler + rectified-flow
DiT decoder), released weights, zero-shot on CLEVRER.

If this is sharp (>=30 dB), the architecture direction is validated and our port
is just underpowered/undertrained. If even their real model is mediocre on
CLEVRER, that's a decisive signal.

Runs in the `videoflextok` conda env (torch 2.8 / diffusers 0.20), NOT river.
"""
import argparse, glob, math, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/Dialga/ml-videoflextok")
from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import read_mp4, denormalize


def psnr(pred, gt):
    # pred, gt in [-1,1], shape (...,); compute over all pixels
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EPFL-VILAB/videoflextok_d18_d18_k600")
    ap.add_argument("--video_glob",
                    default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/"
                            "train_video/video_00000-01000/video_0000*.mp4")
    ap.add_argument("--n_clips", type=int, default=8)
    ap.add_argument("--k_keep", type=int, default=0, help="0 = keep all 256 tokens")
    ap.add_argument("--timesteps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="/storage/home/hcoda1/8/lwang831/workspace/Dialga/"
                                     "outputs/vft_reference")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    model = VideoFlexTokFromHub.from_pretrained(args.model).eval().to(dev)
    pp = dict(model.video_preprocess_args)
    print(f"[load] done. video_preprocess_args={pp}", flush=True)
    # map config keys -> read_mp4 kwargs
    rm_kwargs = {}
    if "size" in pp: rm_kwargs["size"] = pp["size"]
    if "chunk_size" in pp: rm_kwargs["chunk_size"] = pp["chunk_size"]
    if "overlap_size_frames" in pp: rm_kwargs["overlap_size"] = pp["overlap_size_frames"]
    elif "overlap_size" in pp: rm_kwargs["overlap_size"] = pp["overlap_size"]

    vids = sorted(glob.glob(args.video_glob))[:args.n_clips]
    assert vids, f"no videos matched {args.video_glob}"

    scores = []
    for i, vp in enumerate(vids):
        video = read_mp4(vp, fps=args.fps, **rm_kwargs).to(dev)   # (C,T,H,W) in [-1,1]
        # overlap=0 encoder wants T = K*chunk_size (read_mp4 returns 1+K*stride); trim.
        ov = rm_kwargs.get("overlap_size", 0)
        cs = rm_kwargs.get("chunk_size", 17)
        if ov == 0:
            keep = (video.shape[1] // cs) * cs
            video = video[:, :keep]
        with torch.no_grad():
            tokens = model.tokenize(video[None])
            if args.k_keep > 0:
                tokens = [t[..., :args.k_keep] for t in tokens]
            recon = model.detokenize(tokens, timesteps=args.timesteps,
                                     guidance_scale=args.guidance_scale,
                                     perform_norm_guidance=True)
        rec = recon[0] if isinstance(recon, (list, tuple)) else recon      # [1,3,T,H,W]
        rec = rec[0].to(dev)                                               # (3,T,H,W)
        T = min(rec.shape[1], video.shape[1])
        p = psnr(rec[:, :T], video[:, :T])
        scores.append(p)
        print(f"[clip {i}] {Path(vp).name}  T={video.shape[1]}  PSNR={p:.2f} dB", flush=True)

        if i < 3:  # save a few side-by-sides
            try:
                import imageio
                # (C,T,H,W) in [-1,1] -> (T,H,W,C) in [0,255]
                g = ((video[:, :T].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
                r = ((rec[:, :T].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
                sbs = (np.concatenate([g, r], axis=2) * 255).clip(0, 255).astype(np.uint8)
                imageio.mimwrite(str(outdir / f"vft_sbs_{i}.mp4"), sbs, fps=8)
            except Exception as e:
                print(f"[warn] save clip {i} failed: {e}", flush=True)

    print(f"\n[RESULT] VideoFlexTok zero-shot CLEVRER whole-frame PSNR: "
          f"mean={np.mean(scores):.2f} dB  (n={len(scores)}, "
          f"min={np.min(scores):.2f}, max={np.max(scores):.2f})", flush=True)
    print("VFT_REF_DONE", flush=True)


if __name__ == "__main__":
    main()
