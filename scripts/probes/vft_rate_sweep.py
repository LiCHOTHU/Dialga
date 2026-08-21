"""Rate-matched VideoFlexTok reconstruction on CLEVRER.

The earlier vft_reference_psnr.py ran VFT at its FULL rate (all 256 registers ->
1280 tokens / 17-frame chunk, ~20k bits). That is NOT comparable to our DIALGA
budget (964 floats / 33 frames ~= 497 float-slots per 17f-equiv). This probe
sweeps num_keep_tokens (the coarse-to-fine nested-dropout knob, the CORRECT API
per the official notebook: keep FIRST-K registers, decode) so we can read
reconstruction quality AT a matched latent size.

Slot-match: 5 temporal * K spatial ~= 497  ->  K ~= 100 of 256 registers.

Runs in the `videoflextok` conda env.
"""
import argparse, glob, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/Dialga/ml-videoflextok")
from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import read_mp4


def psnr(pred, gt):
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EPFL-VILAB/videoflextok_d18_d18_k600")
    ap.add_argument("--video_glob",
                    default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/"
                            "train_video/video_00000-01000/video_0000*.mp4")
    ap.add_argument("--n_clips", type=int, default=8)
    ap.add_argument("--video_ids", type=int, nargs="+", default=None,
                    help="explicit CLEVRER video ids (held-out val set). Overrides --video_glob.")
    ap.add_argument("--video_dir",
                    default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video")
    ap.add_argument("--keeps", type=int, nargs="+",
                    default=[256, 128, 100, 64, 32, 16])
    ap.add_argument("--timesteps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--save_keeps", type=int, nargs="+", default=[256, 100, 32])
    ap.add_argument("--out", default="/storage/home/hcoda1/8/lwang831/workspace/Dialga/"
                                     "outputs/vft_rate_sweep")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    model = VideoFlexTokFromHub.from_pretrained(args.model).eval().to(dev)
    pp = dict(model.video_preprocess_args)
    print(f"[load] done. preprocess={pp}", flush=True)
    rm_kwargs = {}
    if "size" in pp: rm_kwargs["size"] = pp["size"]
    if "chunk_size" in pp: rm_kwargs["chunk_size"] = pp["chunk_size"]
    ov = pp.get("overlap_size_frames", pp.get("overlap_size", 0))
    rm_kwargs["overlap_size"] = ov
    cs = rm_kwargs.get("chunk_size", 17)

    if args.video_ids:
        # resolve ids -> mp4 paths (CLEVRER is chunked into 1000-video subdirs)
        vids = []
        for vid in args.video_ids:
            lo = (vid // 1000) * 1000
            sub = f"video_{lo:05d}-{lo+1000:05d}"
            p = Path(args.video_dir) / sub / f"video_{vid:05d}.mp4"
            assert p.exists(), f"missing {p}"
            vids.append(str(p))
    else:
        vids = sorted(glob.glob(args.video_glob))[:args.n_clips]
    assert vids, f"no videos matched {args.video_glob}"

    # scores[K] = list of per-clip PSNR
    scores = {k: [] for k in args.keeps}
    for i, vp in enumerate(vids):
        video = read_mp4(vp, fps=args.fps, **rm_kwargs).to(dev)   # (C,T,H,W) in [-1,1]
        if ov == 0:
            keep = (video.shape[1] // cs) * cs
            video = video[:, :keep]
        with torch.no_grad():
            tokens = model.tokenize(video[None])                 # full-rate tokens
            for K in args.keeps:
                recon = model.detokenize(
                    tokens, num_keep_tokens_list=[K],
                    timesteps=args.timesteps, guidance_scale=args.guidance_scale,
                    perform_norm_guidance=True)
                rec = recon[0] if isinstance(recon, (list, tuple)) else recon
                rec = rec[0].to(dev)                             # (3,T,H,W)
                T = min(rec.shape[1], video.shape[1])
                p = psnr(rec[:, :T], video[:, :T])
                scores[K].append(p)
                if i < 3 and K in args.save_keeps:
                    try:
                        import imageio
                        g = ((video[:, :T].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
                        r = ((rec[:, :T].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).numpy()
                        sbs = (np.concatenate([g, r], axis=2) * 255).clip(0, 255).astype(np.uint8)
                        imageio.mimwrite(str(outdir / f"vft_k{K}_clip{i}.mp4"), sbs, fps=8)
                    except Exception as e:
                        print(f"[warn] save k{K} clip{i}: {e}", flush=True)
        line = "  ".join(f"K{K}={np.mean(scores[K][-1:]):.2f}" for K in args.keeps)
        print(f"[clip {i}] {Path(vp).name} T={video.shape[1]}  {line}", flush=True)

    print("\n[RESULT] VideoFlexTok rate sweep, CLEVRER whole-frame PSNR (mean over "
          f"{len(vids)} clips):", flush=True)
    print(f"{'K (registers)':>14}{'tokens/17f':>12}{'PSNR dB':>10}", flush=True)
    for K in args.keeps:
        tag = "  <- slot-match (~ours)" if K == 100 else ""
        print(f"{K:>14}{5*K:>12}{np.mean(scores[K]):>10.2f}{tag}", flush=True)
    print("VFT_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
