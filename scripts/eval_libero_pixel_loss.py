"""eval_libero_pixel_loss.py — LIBERO version of scripts/eval_pixel_loss.py.

Computes the trainer's-equivalent obs-path pixel loss for a LIBERO checkpoint:

    chunk_obs   = cached Wan latent                              (B, 48, 9, 8, 8)
    recon_obs   = dec(enc(chunk_obs))
    pred_pix    = wan_vae.decode(recon_obs)                      (B, 33, 3, 128, 128)
    gt_pix      = raw mp4 frames[s:s+33], normalized to [-1, 1]
    L_pixel     = MSE(pred_pix, gt_pix)

Also computes the VAE ROUND-TRIP FLOOR: MSE(wan_dec(wan_enc(gt_pix)), gt_pix).
Our model cannot beat that — it's the upper bound on what reconstruction
quality we could ever achieve through this frozen VAE.

The LIBERO model never saw pixel-space supervision (lambda_pixel=0 throughout
training), so this number tells us how decodable our latents happen to be even
without that anchor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.libero_window import LiberoChunkPairs, libero_collate
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import LatentDecoder


def _resolve_wan_path(name: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers") -> str:
    """Prefer local HF snapshot (offline-safe on PACE compute nodes)."""
    import os
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    repo_dir = Path(cache_root) / f"models--{name.replace('/', '--')}"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        snap = repo_dir / "snapshots" / ref_main.read_text().strip()
        if snap.exists(): return str(snap)
    return name


def load_wan_vae(dtype, device):
    from diffusers import AutoencoderKLWan
    path = _resolve_wan_path()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def vae_decode(vae, latent, dtype):
    z = latent.to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()


@torch.no_grad()
def vae_encode(vae, pix, dtype):
    x = pix.permute(0, 2, 1, 3, 4).contiguous().to(dtype)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.float()


def _load_chunk_pixels(libero_root: Path, video_path: str, start_frame: int,
                       W: int = 33) -> torch.Tensor:
    """Load raw mp4 frames[s:s+W] -> (W, 3, 128, 128) float in [-1, 1].

    Caches the last-decoded video so successive chunks of the same episode
    don't re-decode.
    """
    from torchvision.io import read_video
    cache = _load_chunk_pixels  # function-attribute LRU of size 1
    if getattr(cache, "_path", None) != video_path:
        v, _, _ = read_video(str(libero_root / video_path), pts_unit="sec")
        cache._video = v          # (T, H, W, 3) uint8
        cache._path = video_path
    frames = cache._video[start_frame:start_frame + W]               # (W, H, W, 3)
    f = frames.float() / 127.5 - 1.0
    return f.permute(0, 3, 1, 2).contiguous()                        # (W, 3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True,
                    help="LIBERO Wan-latent cache (encode_libero_wan.py output)")
    ap.add_argument("--libero_root", default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90_processed",
                    help="Where the mp4s live (visual_input paths are relative to this).")
    ap.add_argument("--split", default="val", choices=["train", "val", "heldout"])
    ap.add_argument("--n_chunks", type=int, default=200)
    ap.add_argument("--n_holdout_tasks", type=int, default=10)
    ap.add_argument("--task_holdout_seed", type=int, default=0)
    ap.add_argument("--val_every_k_episodes", type=int, default=10)
    ap.add_argument("--require_chunks_per_episode", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Keep 0 — we use the function-LRU video cache, which is "
                         "single-process. Multi-worker re-decodes each chunk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    if args.out_json is None:
        args.out_json = str(Path(args.ckpt).parent / "pixel_loss.json")

    # ---- model ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 64)); d_dyn = int(a.get("d_dyn", 64))
    enc_h = int(a.get("enc_hidden_ch", 128)); dec_h = int(a.get("dec_hidden_ch", 256))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_h,
                          use_layer_norm=use_ln).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_h,
                        chunk_size_lat=chunk_size_lat).to(device)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    print(f"[model] {args.ckpt}")
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_h={enc_h} dec_h={dec_h} use_ln={use_ln}")

    print(f"[vae] loading Wan-2.2 VAE ({args.dtype})")
    vae = load_wan_vae(dtype, device)

    # ---- data ----
    ds = LiberoChunkPairs(
        args.cache_dir, split=args.split,
        n_holdout_tasks=args.n_holdout_tasks,
        task_holdout_seed=args.task_holdout_seed,
        val_every_k=args.val_every_k_episodes,
        pair_seed=args.seed,
        require_chunks_per_video=args.require_chunks_per_episode,
    )
    n_total = len(ds)
    n_use = min(args.n_chunks, n_total)
    g = torch.Generator().manual_seed(args.seed)
    idxs = sorted(torch.randperm(n_total, generator=g).tolist()[:n_use])
    loader = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=libero_collate)
    print(f"[data] {args.split}: eval on {n_use}/{n_total} chunks")

    # We need the raw mp4 path for each pair (= video_path of the chunk_obs).
    # LiberoChunkPairs holds {chunk_keys} -> Wan cache only; we walk the original
    # libero_90.jsonl to recover video paths once.
    libero_root = Path(args.libero_root)
    # Build episode_id -> video_path from the jsonl
    ep_video: dict[int, str] = {}
    ep_start: dict[int, int] = {}  # not needed but kept for clarity
    with (libero_root / "libero_90.jsonl").open() as f:
        for gid, line in enumerate(json.loads(line) if False else None for line in f):
            pass
    # The above generator pattern is awkward — straight pass:
    with (libero_root / "libero_90.jsonl").open() as f:
        for gid, line in enumerate(f):
            r = json.loads(line)
            ep_video[gid] = r["visual_input"]

    # ---- L_pixel + VAE round-trip floor ----
    se_sum_model, se_sum_vae, n_elem = 0.0, 0.0, 0
    per_chunk_model, per_chunk_vae = [], []
    t0 = time.time()
    n_done = 0
    for batch in loader:
        chunk_obs = batch["chunk_obs"].to(device)
        gids = batch["global_episode_id"].tolist()
        sfs  = batch["start_frame"].tolist()
        # Decode per-sample (videos differ across the batch). Pad to (B, 33, 3, 128, 128).
        gt_pix_list = [_load_chunk_pixels(libero_root, ep_video[g], s) for g, s in zip(gids, sfs)]
        gt_pix = torch.stack(gt_pix_list, dim=0).to(device)            # (B, 33, 3, 128, 128)

        with torch.no_grad():
            out = enc(chunk_obs)
            recon = dec(out["z_static"], out["z_dyn"])
            pred_pix = vae_decode(vae, recon, dtype)

            # Round-trip floor through the frozen VAE.
            vae_z = vae_encode(vae, gt_pix, dtype)
            rt_pix = vae_decode(vae, vae_z, dtype)

            T = min(gt_pix.shape[1], pred_pix.shape[1], rt_pix.shape[1])
            diff_m = (pred_pix[:, :T] - gt_pix[:, :T]).pow(2)
            diff_v = (rt_pix[:, :T]   - gt_pix[:, :T]).pow(2)
            per_chunk_model.append(diff_m.mean(dim=(1, 2, 3, 4)).cpu())
            per_chunk_vae.append(  diff_v.mean(dim=(1, 2, 3, 4)).cpu())
            se_sum_model += diff_m.sum().item()
            se_sum_vae   += diff_v.sum().item()
            n_elem += diff_m.numel()
        n_done += chunk_obs.shape[0]
        if n_done % (args.batch_size * 8) == 0 or n_done >= n_use:
            print(f"  [{n_done:4d}/{n_use}] {time.time()-t0:5.1f}s")

    per_chunk_model = torch.cat(per_chunk_model)
    per_chunk_vae   = torch.cat(per_chunk_vae)
    L_model = se_sum_model / n_elem
    L_vae   = se_sum_vae   / n_elem
    gap = L_model - L_vae
    print("\n" + "=" * 64)
    print(f"L_pixel (model)     = {L_model:.6f}   (= MSE between recon and raw pixels)")
    print(f"L_pixel (VAE floor) = {L_vae:.6f}   (= round-trip MSE through Wan VAE)")
    print(f"model - floor       = {gap:.6f}   ({100*gap/L_vae:+.1f}% over the floor)")
    print(f"per-chunk model = {per_chunk_model.mean():.6f} ± {per_chunk_model.std():.6f}")
    print(f"per-chunk vae   = {per_chunk_vae.mean():.6f} ± {per_chunk_vae.std():.6f}")
    print("=" * 64)

    out = {
        "ckpt": str(args.ckpt), "split": args.split, "n_chunks": n_use,
        "L_pixel_model": L_model, "L_pixel_vae_roundtrip": L_vae,
        "gap_over_floor": gap,
        "per_chunk_model_mean": float(per_chunk_model.mean()),
        "per_chunk_model_std":  float(per_chunk_model.std()),
        "per_chunk_vae_mean":   float(per_chunk_vae.mean()),
        "per_chunk_vae_std":    float(per_chunk_vae.std()),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
