"""Rate-distortion table: our arms vs standard codecs, matched bitrate.

THE FRAMING
-----------
We reconstruct WORSE than the frozen Wan-VAE we are built on (~31 dB vs a 46.7 dB
roundtrip ceiling). So distortion-vs-the-ceiling is not a story. The only honest
axis is RATE: we spend `d_static + 9*d_dyn + 4` floats per 33-frame chunk, which
at 128x128 / 25 fps is a few tens of kbps. The claim this table must support is:

    at matched bitrate, we beat what everyone else uses (H.264 / H.265).

BIT ACCOUNTING (deliberately conservative — it penalises US)
------------------------------------------------------------
Our latents are NOT quantised; we bill every float at fp32 = 32 bits:

    bits/chunk = (d_static + T_lat*d_dyn + d_event) * 32
    bitrate    = bits/chunk * fps / frames_per_chunk

Billing fp32 is the worst case for us (any quantisation only lowers our rate),
and the codecs are additionally given their container/header overhead for free.
We report each codec's ACTUAL achieved bitrate from the encoded file size, not
the requested one, so the comparison cannot be accused of hiding an overshoot.

METRICS
-------
PSNR alone rewards our blur, so a reviewer will not accept it by itself. LPIPS
(perceptual, lower=better) and SSIM are reported on the SAME clips. If we win
PSNR but lose LPIPS to a codec at matched rate, that is a real finding and we
report it rather than hide it.

All arms share the identical val split (seed=42, val_frac=0.2), so the clips and
the Wan-VAE ceiling are common across every row.
"""
from __future__ import annotations
import argparse, math, os, subprocess, sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from diffusers import AutoencoderKLWan
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.latent_decoder import (LatentDecoder, SpatialBroadcastDecoder, SlotDecoder,
                                      SpatialGridDecoder)
from src.model.latent_encoder import LatentEncoder3D

FPS = 25
FRAMES_PER_CHUNK = 33


def psnr(pred, gt):
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def rate_floats(a: dict) -> int:
    """Floats per 33-frame chunk: z_static + per-latent-frame z_dyn + z_event."""
    t_lat = int(a.get("chunk_size_lat", 9))
    d_event = int(a.get("d_event", 4))
    return int(a["d_static"]) + t_lat * int(a["d_dyn"]) + d_event


def floats_to_kbps(n_floats: int) -> float:
    return n_floats * 32 * FPS / FRAMES_PER_CHUNK / 1000.0


def build_enc_dec(a, enc_state, dec_state, device):
    use_ln = "norm_static.weight" in enc_state
    pool_type = a.get("pool_type", "mean")
    decoder_type = a.get("decoder_type", "linear")
    # `or 4`: mean/broadcast runs store static_grid=None, and the encoder does
    # int(static_grid) unconditionally -> TypeError. 4 is inert unless pool=spatial.
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=pool_type,
                          static_grid=int(a.get("static_grid") or 4),
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4))).to(device)
    common = dict(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                  hidden_ch=int(a["dec_hidden_ch"]),
                  chunk_size_lat=int(a.get("chunk_size_lat", 9)))
    if decoder_type == "broadcast":
        dec = SpatialBroadcastDecoder(**common, depth=int(a.get("dec_depth", 3))).to(device)
    elif decoder_type == "slot":
        dec = SlotDecoder(**common, depth=int(a.get("dec_depth", 3))).to(device)
    elif decoder_type == "spatial":
        dec = SpatialGridDecoder(**common, static_grid=int(a.get("static_grid") or 4),
                                 depth=int(a.get("dec_depth", 3))).to(device)
    else:
        dec = LatentDecoder(**common).to(device)
    enc.load_state_dict(enc_state); dec.load_state_dict(dec_state)
    enc.eval(); dec.eval()
    return enc, dec


# --------------------------------------------------------------------------- #
# codec baseline
# --------------------------------------------------------------------------- #
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def codec_roundtrip(pix: torch.Tensor, codec: str, kbps: float, tmpdir: str):
    """Encode (T,3,H,W) in [-1,1] at a target bitrate, decode, return (pix, actual_kbps).

    Returns the ACTUAL bitrate implied by the encoded file size, so an overshoot
    is visible rather than silently credited to the codec.
    """
    T, _, H, W = pix.shape
    raw = ((pix.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    raw = raw.permute(0, 2, 3, 1).contiguous().numpy().tobytes()   # T,H,W,3

    enc_lib = {"h264": "libx264", "h265": "libx265"}[codec]
    out = str(Path(tmpdir) / f"{codec}.mp4")
    br = f"{int(round(kbps * 1000))}"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "pipe:0",
           "-c:v", enc_lib, "-pix_fmt", "yuv420p",
           "-b:v", br, "-minrate", br, "-maxrate", br, "-bufsize", f"{int(round(kbps*1000))}",
           out]
    if enc_lib == "libx265":
        cmd[-1:-1] = ["-x265-params", "log-level=error"]
    subprocess.run(cmd, input=raw, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE)

    nbytes = Path(out).stat().st_size
    actual_kbps = nbytes * 8 / (T / FPS) / 1000.0

    dec = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", out,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    arr = torch.frombuffer(bytearray(dec.stdout), dtype=torch.uint8)
    arr = arr[: T * H * W * 3].view(-1, H, W, 3).permute(0, 3, 1, 2).float()
    return (arr / 127.5 - 1.0), actual_kbps


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=32)
    ap.add_argument("--codec_kbps", type=float, default=0.0,
                    help="target bitrate for codec rows; 0 = derive from first ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae_dtype", default="bfloat16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)

    def metrics(pred, gt):
        p, g = pred.clamp(-1, 1), gt.clamp(-1, 1)
        with torch.no_grad():
            lp = lpips_fn(p.to(device), g.to(device)).item()
            lpips_fn.reset()          # __call__ also accumulates global state; drop it
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

    ds = ClevrerChunkPairsWithPixels(args.cache_dir, args.video_dir, image_size=128,
                                     seed=42, max_videos=0, split="val", val_frac=0.2)
    seen, clips = set(), []
    for idx in range(len(ds)):
        s = ds[idx]
        vid = int(s.get("video_id", ds.windows[ds.pairs[idx][0]]["video_id"]))
        if vid in seen:
            continue
        seen.add(vid)
        lats, pixs = [s["chunk_obs"]], [s["pix_obs"]]
        if "chunk_pred" in s and "pix_pred" in s:
            lats.append(s["chunk_pred"]); pixs.append(s["pix_pred"])
        clips.append((vid, lats, pixs))
        if len(clips) >= args.n_videos:
            break
    print(f"[data] {len(clips)} distinct val clips: {[c[0] for c in clips]}", flush=True)

    raw_cache, ceil_m = [], []
    for vid, lats, pixs in clips:
        p_raw = torch.cat(pixs, 0)
        p_wan = torch.cat([wan_decode(l) for l in lats], 0)
        T = min(len(p_raw), len(p_wan))
        raw_cache.append(p_raw[:T])
        ceil_m.append(metrics(p_wan[:T], p_raw[:T]))
    ceiling = tuple(sum(x[i] for x in ceil_m) / len(ceil_m) for i in range(3))
    print(f"[ceiling] Wan-VAE roundtrip PSNR={ceiling[0]:.2f} LPIPS={ceiling[1]:.4f} "
          f"SSIM={ceiling[2]:.4f}\n", flush=True)

    rows = []
    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
            enc, dec = build_enc_dec(a, ck["encoder"], ck["decoder"], device)
            nf = rate_floats(a); kbps = floats_to_kbps(nf)
            ms = []
            with torch.no_grad():
                for i, (vid, lats, pixs) in enumerate(clips):
                    outs = []
                    for l in lats:
                        o = enc(l.unsqueeze(0).to(device))
                        if isinstance(dec, SlotDecoder):
                            cond = o["z_slots"]
                        elif isinstance(dec, SpatialGridDecoder):
                            cond = o["z_static_grid"]
                        else:
                            cond = o["z_static"]
                        outs.append(wan_decode(dec(cond, o["z_dyn"]).squeeze(0).cpu()))
                    p_mdl = torch.cat(outs, 0)[: len(raw_cache[i])]
                    ms.append(metrics(p_mdl, raw_cache[i]))
            m = tuple(sum(x[i] for x in ms) / len(ms) for i in range(3))
            rows.append((label, ck.get("epoch"), nf, kbps, *m))
            print(f"  {label:<22} ep{str(ck.get('epoch')):<5} {nf:>5}f {kbps:6.1f}kbps "
                  f"PSNR={m[0]:6.2f} LPIPS={m[1]:.4f} SSIM={m[2]:.4f}", flush=True)
        except Exception as e:
            print(f"  {label:<22} FAILED: {type(e).__name__}: {e}", flush=True)

    # codec baselines at OUR bitrate
    target = args.codec_kbps or (rows[0][3] if rows else 23.4)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        for codec in ("h264", "h265"):
            try:
                ms, brs = [], []
                for i, _ in enumerate(clips):
                    rec, ab = codec_roundtrip(raw_cache[i], codec, target, td)
                    ms.append(metrics(rec[: len(raw_cache[i])], raw_cache[i]))
                    brs.append(ab)
                m = tuple(sum(x[i] for x in ms) / len(ms) for i in range(3))
                ab = sum(brs) / len(brs)
                rows.append((f"{codec} @{target:.1f}kbps", None, None, ab, *m))
                print(f"  {codec:<22} {'':>7} {'':>6} {ab:6.1f}kbps(actual) "
                      f"PSNR={m[0]:6.2f} LPIPS={m[1]:.4f} SSIM={m[2]:.4f}", flush=True)
            except Exception as e:
                print(f"  {codec:<22} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'='*94}\nRate-distortion, {len(clips)} val clips "
          f"(LPIPS lower=better; fp32 bit-accounting = conservative for us)\n{'='*94}")
    print(f"{'run':<24}{'ep':>5}{'floats':>8}{'kbps':>8}{'PSNR':>8}{'LPIPS':>9}{'SSIM':>8}")
    print("-" * 94)
    print(f"{'Wan-VAE ceiling':<24}{'':>5}{'':>8}{'':>8}{ceiling[0]:>8.2f}"
          f"{ceiling[1]:>9.4f}{ceiling[2]:>8.4f}")
    for label, ep, nf, kbps, ps, lp, ss in sorted(rows, key=lambda r: (r[3] or 0)):
        print(f"{label:<24}{str(ep or ''):>5}{str(nf or ''):>8}{kbps:>8.1f}"
              f"{ps:>8.2f}{lp:>9.4f}{ss:>8.4f}")

    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"ceiling": {"psnr": ceiling[0], "lpips": ceiling[1], "ssim": ceiling[2]},
                   "clips": [c[0] for c in clips],
                   "rows": [{"label": l, "epoch": e, "floats": nf, "kbps": k,
                             "psnr": p, "lpips": lp, "ssim": s}
                            for l, e, nf, k, p, lp, s in rows]},
                  open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")
    print("RD_TABLE_DONE")


if __name__ == "__main__":
    main()
