"""rFVD table — the metric this literature actually judges tokenizers on.

WHY
---
We had been scoring ourselves on PSNR, and concluded from `rd_table.py` that
H.264 beats us by 10 dB so the reconstruction story is dead. That conclusion was
measured against the wrong yardstick:

  * VideoFlexTok (2604.12887) headlines rFVD / gFVD. It reports NO PSNR anywhere.
  * FlexTok (2502.13967) headlines rFID; PSNR/SSIM/LPIPS appear only in appendix
    Table 8, where their BEST model at full 256-token rate scores PSNR 17.70 dB.
  * PV-VAE (2605.02134) headlines FVD/rFVD.
  * NONE of the three compares against H.264/H.265 at any bitrate.

So per-pixel fidelity is a target this field abandoned on purpose: a rectified-
flow decoder hallucinates sharp detail, which tanks PSNR but wins FVD. Our
deterministic decoder does the reverse — it produces the blur PSNR rewards and
FVD punishes. We therefore have NO evidence on the axis that decides the paper.
This script produces it.

NOTE ON A COMPARISON WE CANNOT MAKE
-----------------------------------
FlexTok's 17.70 dB is on ImageNet 256x256. Ours is CLEVRER 128x128 with a static
camera, constant background and ~5 rigid objects. Our 32.8 dB is NOT "15 dB
better than FlexTok" — the datasets are not remotely comparable in difficulty.
Only the methodological point transfers (which metric the field ranks on).

WHAT THIS REPORTS  (FlexTok appendix Table 8 layout + rFVD as headline)
    floats | bytes | rFVD | PSNR | SSIM | LPIPS
for each rate arm, plus the Wan-VAE roundtrip ceiling and the H.264/H.265 rows
at matched bitrate. rFVD is distributional over the whole eval set, so stats are
accumulated in a streaming fashion rather than held in RAM.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKLWan
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.latent_decoder import SlotDecoder, SpatialGridDecoder
from scripts.probes.rd_table import (build_enc_dec, codec_roundtrip, floats_to_kbps,
                                     psnr, rate_floats)


def to_u8(pix: torch.Tensor) -> np.ndarray:
    """(T,3,H,W) in [-1,1] -> (T,H,W,3) uint8, the layout cdfvd expects."""
    x = ((pix.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=512)
    ap.add_argument("--codec_kbps", type=float, default=23.4)
    ap.add_argument("--fvd_model", default="i3d", choices=["i3d", "videomae"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae_dtype", default="bfloat16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]

    from cdfvd import fvd
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                     normalize=False).to(device)
    evaluator = fvd.cdfvd(args.fvd_model, device=str(device))

    def pixel_metrics(pred, gt):
        p, g = pred.clamp(-1, 1), gt.clamp(-1, 1)
        with torch.no_grad():
            lp = lpips_fn(p.to(device), g.to(device)).item()
            lpips_fn.reset()
            ss = ssim_fn((p.to(device) + 1) / 2, (g.to(device) + 1) / 2, data_range=1.0).item()
        return psnr(p, g), lp, ss

    print(f"[vae] loading frozen Wan VAE ({args.vae_dtype})", flush=True)
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                           subfolder="vae", torch_dtype=vdt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    @torch.no_grad()
    def wan_decode(latent):
        z = latent.unsqueeze(0).to(device=device, dtype=vdt)
        out = vae.decode(z)
        pix = out.sample if hasattr(out, "sample") else out
        return pix.squeeze(0).permute(1, 0, 2, 3).contiguous().float().cpu()

    # one 33-frame observed chunk per distinct val video
    ds = ClevrerChunkPairsWithPixels(args.cache_dir, args.video_dir, image_size=128,
                                     seed=42, max_videos=0, split="val", val_frac=0.2)
    seen, clips = set(), []
    for idx in range(len(ds)):
        s = ds[idx]
        vid = int(s.get("video_id", ds.windows[ds.pairs[idx][0]]["video_id"]))
        if vid in seen:
            continue
        seen.add(vid)
        clips.append((vid, s["chunk_obs"], s["pix_obs"]))
        if len(clips) >= args.n_videos:
            break
    print(f"[data] {len(clips)} distinct val videos, "
          f"{clips[0][2].shape[0]} frames each", flush=True)

    # real stats once
    buf = []
    for _, _, pix in clips:
        buf.append(to_u8(pix))
        if len(buf) >= args.batch:
            evaluator.add_real_stats(np.stack(buf)); buf = []
    if buf:
        evaluator.add_real_stats(np.stack(buf))
    print("[fvd] real stats done", flush=True)

    rows = []

    def score(label, nf, kbps, gen):
        """gen(i) -> reconstructed (T,3,H,W) in [-1,1] for clip i."""
        evaluator.empty_fake_stats()
        buf, pm = [], []
        for i, (_, _, pix) in enumerate(clips):
            rec = gen(i)[: pix.shape[0]]
            pm.append(pixel_metrics(rec, pix))
            buf.append(to_u8(rec))
            if len(buf) >= args.batch:
                evaluator.add_fake_stats(np.stack(buf)); buf = []
        if buf:
            evaluator.add_fake_stats(np.stack(buf))
        r = evaluator.compute_fvd_from_stats()
        m = tuple(sum(x[j] for x in pm) / len(pm) for j in range(3))
        by = int(nf * 4) if nf else None          # fp32 accounting
        rows.append((label, nf, by, kbps, r, *m))
        print(f"  {label:<22} {str(nf or ''):>6}f {kbps:>6.1f}kbps rFVD={r:8.2f} "
              f"PSNR={m[0]:6.2f} LPIPS={m[1]:.4f} SSIM={m[2]:.4f}", flush=True)

    # ceiling: what the frozen Wan-VAE alone achieves (no tokenizer)
    wan_rt = [wan_decode(l) for _, l, _ in clips]
    score("wan_vae_ceiling", None, 0.0, lambda i: wan_rt[i])
    del wan_rt

    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
            enc, dec = build_enc_dec(a, ck["encoder"], ck["decoder"], device)
            nf = rate_floats(a)

            @torch.no_grad()
            def gen(i, enc=enc, dec=dec):
                o = enc(clips[i][1].unsqueeze(0).to(device))
                if isinstance(dec, SlotDecoder):
                    cond = o["z_slots"]
                elif isinstance(dec, SpatialGridDecoder):
                    cond = o["z_static_grid"]
                else:
                    cond = o["z_static"]
                return wan_decode(dec(cond, o["z_dyn"]).squeeze(0).cpu())

            score(label, nf, floats_to_kbps(nf), gen)
            del enc, dec
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {label:<22} FAILED: {type(e).__name__}: {e}", flush=True)

    import tempfile, os
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for codec in ("h264", "h265"):
            try:
                cache = {}

                def gen(i, codec=codec, cache=cache, td=td):
                    r, ab = codec_roundtrip(clips[i][2], codec, args.codec_kbps, td)
                    cache.setdefault("br", []).append(ab)
                    return r

                score(f"{codec}@{args.codec_kbps:.1f}kbps", None,
                      float(np.mean(cache.get("br", [args.codec_kbps]))) if cache else args.codec_kbps,
                      gen)
                # patch in the ACTUAL mean bitrate now that every clip is encoded
                if cache.get("br"):
                    l, nf, by, _, r, p, lp, ss = rows[-1]
                    rows[-1] = (l, nf, by, float(np.mean(cache["br"])), r, p, lp, ss)
                    print(f"    (actual mean bitrate {np.mean(cache['br']):.1f} kbps)",
                          flush=True)
            except Exception as e:
                print(f"  {codec:<22} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'='*100}\nrFVD table — {len(clips)} val videos, {args.fvd_model} features "
          f"(rFVD/LPIPS lower=better)\n{'='*100}")
    print(f"{'run':<24}{'floats':>8}{'bytes':>8}{'kbps':>8}{'rFVD':>10}"
          f"{'PSNR':>8}{'LPIPS':>9}{'SSIM':>8}")
    print("-" * 100)
    for l, nf, by, k, r, p, lp, ss in sorted(rows, key=lambda x: x[4]):
        print(f"{l:<24}{str(nf or ''):>8}{str(by or ''):>8}{k:>8.1f}{r:>10.2f}"
              f"{p:>8.2f}{lp:>9.4f}{ss:>8.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_videos": len(clips), "fvd_model": args.fvd_model,
                   "rows": [{"label": l, "floats": nf, "bytes": by, "kbps": k,
                             "rfvd": r, "psnr": p, "lpips": lp, "ssim": ss}
                            for l, nf, by, k, r, p, lp, ss in rows]},
                  open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")
    print("RFVD_TABLE_DONE")


if __name__ == "__main__":
    main()
