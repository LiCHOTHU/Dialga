"""eval_pixel_metrics.py — Pixel-space recon vs Wan2.2 VAE ceiling.

For each val chunk we compute three pixel tensors:

    raw_pix      ground-truth frames in [-1, 1]                  (T_pix, 3, H, W)
    ceil_pix     wan_decode(cached_latent)   VAE round-trip ceiling
    model_pix    wan_decode(our_dec(our_enc(cached_latent)))     our model

and report PSNR / SSIM / LPIPS for three comparisons:

    A) ceiling vs raw         <-- substrate's best case
    B) model vs raw           <-- end-to-end pixel quality
    C) model vs ceiling       <-- marginal cost of our compression

The cached latents were produced by cache_wan_latents.py via vae.encode(raw),
so `wan_decode(cached_latent)` IS the round-trip ceiling — no re-encode needed.

Usage:
  python scripts/eval_pixel_metrics.py \\
    --ckpt outputs/v512_e1_scale_.../v5_best.pt \\
    --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \\
    --video_dir datasets/CLEVRER/train_video \\
    --n_chunks 200 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.clevrer_window import chunk_collate
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import LatentDecoder


# ---- Wan VAE helpers (mirror viz/save_v51_overfit_videos.py) ---------------

def load_wan_vae(model_id, dtype, device):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def wan_decode_batch(vae, latent, device, dtype) -> torch.Tensor:
    """latent: (B, C, T_lat, H, W). Returns (B, T_pix, 3, H_pix, W_pix) in [-1, 1].

    Wan diffusers returns (B, 3, T, H, W); permute to (B, T, 3, H, W) and float-cast.
    """
    z = latent.to(device).to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    pix = pix.permute(0, 2, 1, 3, 4).contiguous().float()
    return pix  # stay on device until metrics are done


# ---- Metric helpers --------------------------------------------------------

def per_clip_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, T, 3, H, W) in [-1, 1].
    Returns per-clip PSNR (B,) — mean over (T, 3, H, W), MAX=2.0.
    """
    se = (pred - target).pow(2).mean(dim=(1, 2, 3, 4))   # (B,)
    psnr = 10.0 * torch.log10(4.0 / se.clamp(min=1e-12))   # MAX=2 -> MAX^2=4
    return psnr


def per_frame_ssim(pred: torch.Tensor, target: torch.Tensor, ssim_fn) -> torch.Tensor:
    """pred, target: (B, T, 3, H, W) in [-1, 1]. Returns per-clip mean SSIM (B,)."""
    B, T = pred.shape[:2]
    p = pred.reshape(B * T, *pred.shape[2:])
    t = target.reshape(B * T, *target.shape[2:])
    # torchmetrics functional SSIM with data_range=2.0 (we're in [-1,1])
    val = ssim_fn(p, t, data_range=2.0, reduction="none")  # (B*T,) or scalar
    if val.dim() == 0:
        return val.unsqueeze(0).expand(B)
    return val.view(B, T).mean(dim=1)


def per_frame_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_module) -> torch.Tensor:
    """pred, target: (B, T, 3, H, W) in [-1, 1]. Returns per-clip mean LPIPS (B,).

    torchmetrics LPIPS averages internally; we want per-clip, so loop per clip.
    """
    B, T = pred.shape[:2]
    out = torch.empty(B, device=pred.device)
    for i in range(B):
        p = pred[i].clamp(-1, 1)         # (T, 3, H, W)
        t = target[i].clamp(-1, 1)
        out[i] = lpips_module(p, t)      # already mean over batch dim T
    return out


# ---- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True,
                    help="CLEVRER train_video root (contains video_AAAAA-BBBBB/ folders)")
    ap.add_argument("--n_chunks", type=int, default=200,
                    help="Number of val chunks to evaluate")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--model_id", type=str,
                    default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    if args.out_json is None:
        args.out_json = str(Path(args.ckpt).parent / "pixel_metrics.json")

    # ---- load v5.1 model ----------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn = int(a.get("d_dyn", 16))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 32))
    dec_hidden_ch = int(a.get("dec_hidden_ch", 64))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))

    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch,
                          use_layer_norm=use_ln).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_hidden_ch,
                        chunk_size_lat=chunk_size_lat).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    print(f"[model] loaded {args.ckpt}  use_layer_norm={use_ln}")
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_h={enc_hidden_ch} dec_h={dec_hidden_ch}")

    # ---- load Wan VAE -------------------------------------------------------
    print(f"[vae] loading {args.model_id}")
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"[vae] loaded; {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")

    # ---- val dataset --------------------------------------------------------
    ds = ClevrerChunkPairsWithPixels(
        cache_dir=args.cache_dir,
        video_dir=args.video_dir,
        image_size=128,
        split="val",
        val_frac=args.val_frac,
        seed=args.seed,
    )
    n_total = len(ds)
    n_use = min(args.n_chunks, n_total)
    # Deterministic subset
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    idxs = sorted(perm[:n_use])
    ds_sub = Subset(ds, idxs)
    loader = DataLoader(
        ds_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=chunk_collate,
    )
    print(f"[data] eval on {n_use}/{n_total} val chunks "
          f"(image_size=128, W_pix={ds.W_pix})")

    # ---- metrics ------------------------------------------------------------
    # We use torchmetrics for SSIM/LPIPS; PSNR computed manually for speed.
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_mod = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=False
    ).to(device).eval()

    psnr_AvR, psnr_MvR, psnr_MvA = [], [], []
    ssim_AvR, ssim_MvR, ssim_MvA = [], [], []
    lpips_AvR, lpips_MvR, lpips_MvA = [], [], []

    t0 = time.time()
    n_done = 0
    for batch in loader:
        chunk_obs = batch["chunk_obs"].to(device)          # (B, C, T_lat, H, W)
        raw_pix   = batch["pix_obs"].to(device)            # (B, T_pix, 3, H, W) in [-1, 1]

        # Ceiling: decode the cached latent. This IS wan_dec(wan_enc(raw)).
        ceil_pix = wan_decode_batch(vae, chunk_obs, device, dtype)

        # Model: encode -> decode -> wan decode
        with torch.no_grad():
            out = enc(chunk_obs)                            # z_static (B,Ds), z_dyn (B,T,Dd)
            recon_latent = dec(out["z_static"], out["z_dyn"])  # (B, C, T_lat, H, W)
        model_pix = wan_decode_batch(vae, recon_latent, device, dtype)

        # Align temporal length: ds gives W_pix=33; wan_decode also returns T_pix.
        # Truncate to the shorter of the two if they differ.
        T_min = min(raw_pix.shape[1], ceil_pix.shape[1], model_pix.shape[1])
        raw_pix   = raw_pix[:, :T_min]
        ceil_pix  = ceil_pix[:, :T_min]
        model_pix = model_pix[:, :T_min]

        with torch.no_grad():
            psnr_AvR.append(per_clip_psnr(ceil_pix, raw_pix).cpu())
            psnr_MvR.append(per_clip_psnr(model_pix, raw_pix).cpu())
            psnr_MvA.append(per_clip_psnr(model_pix, ceil_pix).cpu())

            ssim_AvR.append(per_frame_ssim(ceil_pix, raw_pix, ssim_fn).cpu())
            ssim_MvR.append(per_frame_ssim(model_pix, raw_pix, ssim_fn).cpu())
            ssim_MvA.append(per_frame_ssim(model_pix, ceil_pix, ssim_fn).cpu())

            lpips_AvR.append(per_frame_lpips(ceil_pix, raw_pix, lpips_mod).cpu())
            lpips_MvR.append(per_frame_lpips(model_pix, raw_pix, lpips_mod).cpu())
            lpips_MvA.append(per_frame_lpips(model_pix, ceil_pix, lpips_mod).cpu())

        n_done += raw_pix.shape[0]
        if n_done % (args.batch_size * 4) == 0 or n_done >= n_use:
            dt = time.time() - t0
            print(f"  [{n_done:4d}/{n_use}] {dt:5.1f}s elapsed "
                  f"({dt/max(n_done,1):.2f}s/chunk)")

    # ---- aggregate ----------------------------------------------------------
    def agg(parts):
        t = torch.cat(parts)
        return float(t.mean().item()), float(t.std().item()), int(t.numel())

    rows = {
        "ceiling_vs_raw": {
            "psnr":  agg(psnr_AvR),  "ssim":  agg(ssim_AvR),  "lpips": agg(lpips_AvR),
        },
        "model_vs_raw": {
            "psnr":  agg(psnr_MvR),  "ssim":  agg(ssim_MvR),  "lpips": agg(lpips_MvR),
        },
        "model_vs_ceiling": {
            "psnr":  agg(psnr_MvA),  "ssim":  agg(ssim_MvA),  "lpips": agg(lpips_MvA),
        },
    }
    gap = {
        "psnr_drop_dB":  rows["ceiling_vs_raw"]["psnr"][0]  - rows["model_vs_raw"]["psnr"][0],
        "ssim_drop":     rows["ceiling_vs_raw"]["ssim"][0]  - rows["model_vs_raw"]["ssim"][0],
        "lpips_rise":    rows["model_vs_raw"]["lpips"][0]   - rows["ceiling_vs_raw"]["lpips"][0],
    }

    # ---- print table --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'comparison':<22} {'PSNR (dB) ↑':<18} {'SSIM ↑':<16} {'LPIPS ↓':<16}")
    print("-" * 78)
    for name, r in rows.items():
        ps_m, ps_s, _ = r["psnr"]
        ss_m, ss_s, _ = r["ssim"]
        lp_m, lp_s, _ = r["lpips"]
        print(f"{name:<22} {ps_m:6.2f} ± {ps_s:5.2f}    "
              f"{ss_m:5.3f} ± {ss_s:5.3f}    {lp_m:5.3f} ± {lp_s:5.3f}")
    print("-" * 78)
    print(f"{'gap (ceiling-model)':<22} {gap['psnr_drop_dB']:+6.2f} dB         "
          f"{gap['ssim_drop']:+6.3f}           {gap['lpips_rise']:+6.3f}")
    print("=" * 78)

    out = {
        "ckpt": str(args.ckpt),
        "n_chunks": n_use,
        "image_size": 128,
        "W_pix": int(ds.W_pix),
        "use_layer_norm_loaded": use_ln,
        "rows": rows,
        "gap": gap,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
