"""scripts/eval_rollout_psnr.py — TODO 1: rollout PSNR (A vs B vs C).

Three rollouts of chunk_pred:
    A: fwd.chunk_step(z_dyn_obs[:, -1], T)            — full dynamics
    B: z_dyn_obs[:, -1:].expand(-1, T, -1)            — freeze last observed z_dyn
    C: enc(chunk_pred)["z_dyn"]                       — oracle (upper bound)

Each is fed through the latent decoder + Wan VAE to RGB pixels and PSNR is
computed against the Wan-VAE roundtrip of chunk_pred (the fair "ground truth"
that shares the same VAE bottleneck as the model).

We also report latent-space MSE (matches L_pred from training) for context.

Sampled at default N=100 val videos × 2 chunk pairs each = 200 chunk evals.
~3 VAE decodes per chunk + 1 GT decode = ~800 VAE forward passes total.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


@torch.no_grad()
def wan_decode(vae, wan_latent, dtype):
    """wan_latent: (B, 48, T_lat, 8, 8) -> pixels (B, T_pix, 3, H, W) in [-1, 1]."""
    z = wan_latent.to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.float().permute(0, 2, 1, 3, 4).contiguous()   # (B, T_pix, 3, H, W)


def psnr_pix(pred, gt, max_val: float = 2.0) -> torch.Tensor:
    """Per-batch PSNR. pred, gt in [-1, 1], same shape."""
    mse = (pred - gt).pow(2).mean(dim=tuple(range(1, pred.dim())))
    psnr = 10.0 * torch.log10((max_val ** 2) / mse.clamp_min(1e-12))
    return psnr  # (B,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=100,
                    help="Number of val videos to sample (each emits 2 chunk pairs).")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",   default=None,
                    help="Output JSON path (default <ckpt parent>/rollout_psnr.json)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]

    # ---- load model ----
    print(f"[ckpt] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    d_state  = int(a.get("d_state", 8))
    enc_hid  = int(a.get("enc_hidden_ch", 32))
    dec_hid  = int(a.get("dec_hidden_ch", 64))
    chunk_T  = int(a.get("chunk_size_lat", 9))
    no_proj  = bool(a.get("no_proj", False))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hid, shared_trunk=shared_trunk).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn,
                        hidden_ch=dec_hid, chunk_size_lat=chunk_T).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"])
    for m in (enc, dec, fwd): m.eval()

    print(f"[vae] loading {args.model_id}")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    vae = vae.to(device)
    print(f"[vae] {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")

    # ---- data: sample N val videos ----
    ds = ClevrerChunkPairs(args.cache_dir, split="val",
                            val_frac=args.val_frac, seed=args.seed)
    print(f"[data] full val pairs: {len(ds)}; sampling {args.n_videos * 2} (= {args.n_videos} vids × 2 pairs)")
    # ClevrerChunkPairs emits two pairs per video sequentially -> just take first 2N items.
    take = min(args.n_videos * 2, len(ds))
    sub = Subset(ds, list(range(take)))
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)

    # ---- evaluate ----
    psnrs = {"A_fwd": [], "B_freeze": [], "C_oracle": [], "ceiling_self": []}
    lat_mses = {"A_fwd": [], "B_freeze": [], "C_oracle": []}
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            chunk_obs  = batch["chunk_obs"].to(device)
            chunk_pred = batch["chunk_pred"].to(device)
            enc_obs  = enc(chunk_obs)
            enc_pred = enc(chunk_pred)
            z_static = enc_obs["z_static"]
            z_dyn_obs  = enc_obs["z_dyn"]               # (B, T, D_d)
            z_dyn_oracle = enc_pred["z_dyn"]             # (B, T, D_d)
            z_dyn_last = z_dyn_obs[:, -1]
            T = z_dyn_obs.shape[1]

            z_dyn_A = fwd.chunk_step(z_dyn_last, T)                       # (B, T, D_d)
            z_dyn_B = z_dyn_last.unsqueeze(1).expand(-1, T, -1)             # (B, T, D_d)
            z_dyn_C = z_dyn_oracle                                          # (B, T, D_d)

            # Latent MSE in Wan-latent-channel space (matches L_pred)
            wan_A = dec(z_static, z_dyn_A)                                  # (B, 48, T_lat, H, W)
            wan_B = dec(z_static, z_dyn_B)
            wan_C = dec(z_static, z_dyn_C)
            lat_mses["A_fwd"].append(((wan_A - chunk_pred) ** 2).mean(dim=(1,2,3,4)).cpu())
            lat_mses["B_freeze"].append(((wan_B - chunk_pred) ** 2).mean(dim=(1,2,3,4)).cpu())
            lat_mses["C_oracle"].append(((wan_C - chunk_pred) ** 2).mean(dim=(1,2,3,4)).cpu())

            # Pixel PSNR via Wan VAE — the "GT pixels" are the Wan VAE decode of chunk_pred.
            pix_GT = wan_decode(vae, chunk_pred,  dtype)
            pix_A  = wan_decode(vae, wan_A.detach(),       dtype)
            pix_B  = wan_decode(vae, wan_B.detach(),       dtype)
            pix_C  = wan_decode(vae, wan_C.detach(),       dtype)

            psnrs["A_fwd"].append(psnr_pix(pix_A, pix_GT).cpu())
            psnrs["B_freeze"].append(psnr_pix(pix_B, pix_GT).cpu())
            psnrs["C_oracle"].append(psnr_pix(pix_C, pix_GT).cpu())
            # Self-PSNR of GT (sanity: should be ~inf)
            psnrs["ceiling_self"].append(psnr_pix(pix_GT, pix_GT).cpu())

            n_done += chunk_obs.shape[0]
            if n_done % (4 * args.batch_size) == 0 or n_done >= take:
                el = time.time() - t0
                print(f"  {n_done}/{take}  ({el:.1f}s)")

    results = {}
    for key in psnrs:
        v = torch.cat(psnrs[key])
        results[f"psnr_{key}_mean"] = float(v.mean())
        results[f"psnr_{key}_median"] = float(v.median())
        results[f"psnr_{key}_std"] = float(v.std())
    for key in lat_mses:
        v = torch.cat(lat_mses[key])
        results[f"latent_mse_{key}_mean"] = float(v.mean())
        results[f"latent_mse_{key}_median"] = float(v.median())

    print("\n========== Rollout PSNR & Latent MSE (mean over chunks) ==========")
    print(f"{'variant':<10s} {'pixel PSNR (dB)':>20s} {'latent MSE':>18s}")
    for key in ("A_fwd", "B_freeze", "C_oracle"):
        print(f"{key:<10s} {results[f'psnr_{key}_mean']:>15.3f}     "
              f"{results[f'latent_mse_{key}_mean']:>18.5f}")
    print(f"{'ceiling':<10s} {'inf (self)':>20s}")
    print()
    delta_AB = results['psnr_A_fwd_mean'] - results['psnr_B_freeze_mean']
    delta_AC = results['psnr_C_oracle_mean'] - results['psnr_A_fwd_mean']
    print(f"Δ (A − B) = +{delta_AB:.3f} dB     ← dynamics value over freeze-last")
    print(f"Δ (C − A) = +{delta_AC:.3f} dB     ← residual to oracle")

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "rollout_psnr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_chunks": n_done, "results": results,
                                    "args": vars(args)}, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
