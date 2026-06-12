"""Step 0 for TODO 1+2 retrieval evaluations.

Encode all 3 chunks of each held-out val video through a v5.1.2 encoder
checkpoint and save the per-chunk latents to disk. Both downstream retrieval
scripts (scripts/eval_content_retrieval.py, scripts/eval_motion_retrieval.py)
consume this cache, so the encoder runs exactly once per ckpt.

ClevrerChunkPairs yields 2 pairs per video; each pair has chunk_obs + chunk_obs_b
and together they cover all 3 chunks (start_frames {0, 33, 66}). We encode both
and dedupe by (video_id, start_frame).

Output (torch.save):
{
  "ckpt": str,
  "split": {"max_videos", "val_frac", "seed"},
  "chunk_keys": list[(video_id:int, start_frame:int)],   # N_chunks
  "z_static":   (N_chunks, D_s),
  "z_dyn_last": (N_chunks, D_d),
  "z_dyn_mean": (N_chunks, D_d),
  "wan_mean":   (N_chunks, C),                           # Wan latent meaned over T,H,W
  "per_video": dict[video_id -> {"attrs": (K,A), "slot_mask": (K,)}],
  "video_z_static": (N_videos, D_s),                     # mean over chunks of same video
  "video_z_dyn_mean": (N_videos, D_d),
  "video_wan_mean": (N_videos, C),
  "video_ids": list[int],                                # row order of video_* tensors
}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_encoder import LatentEncoder3D


CHUNK_STARTS = (0, 33, 66)


def _obs_b_start(s_obs: int) -> int:
    # ClevrerChunkPairs emits two pairs per video: (s0,s1,s2) and (s1,s2,s0).
    # So obs_b = the chunk that is neither obs nor obs+33.
    return (s_obs + 66) % 99


@torch.no_grad()
def encode_all(enc, loader, device):
    """Encode chunk_obs and chunk_obs_b for each pair, dedupe by (vid, start)."""
    z_static_by_key, z_dyn_last_by_key, z_dyn_mean_by_key = {}, {}, {}
    wan_mean_by_key = {}
    per_video = {}
    for batch in loader:
        obs_lat = batch["chunk_obs"].to(device)           # (B, C, T, H, W)
        b_lat   = batch["chunk_obs_b"].to(device)
        B = obs_lat.shape[0]
        # one forward over the concatenated chunks to halve kernel launches
        stacked = torch.cat([obs_lat, b_lat], dim=0)      # (2B, C, T, H, W)
        out = enc(stacked)
        z_static = out["z_static"].float().cpu()          # (2B, D_s)
        z_dyn    = out["z_dyn"].float().cpu()             # (2B, T, D_d)

        wan_mean = stacked.float().mean(dim=(2, 3, 4)).cpu()   # (2B, C)

        for b in range(B):
            vid = int(batch["video_id"][b])
            s_obs = int(batch["start_frame"][b])
            s_b   = _obs_b_start(s_obs)
            for local_idx, s in ((b, s_obs), (B + b, s_b)):
                key = (vid, s)
                if key in z_static_by_key:
                    continue
                z_static_by_key[key]   = z_static[local_idx]
                z_dyn_last_by_key[key] = z_dyn[local_idx, -1]
                z_dyn_mean_by_key[key] = z_dyn[local_idx].mean(dim=0)
                wan_mean_by_key[key]   = wan_mean[local_idx]
            if vid not in per_video:
                per_video[vid] = {
                    "attrs":     batch["attrs"][b].clone(),
                    "slot_mask": batch["slot_mask"][b].clone().bool(),
                }
    return z_static_by_key, z_dyn_last_by_key, z_dyn_mean_by_key, wan_mean_by_key, per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="Wan W=33 latent cache directory (with metadata.json).")
    ap.add_argument("--ckpt", required=True,
                    help="v5.1.2 checkpoint .pt (uses 'encoder' state and 'args').")
    ap.add_argument("--out", default=None,
                    help="Output .pt path; default <ckpt_dir>/val_embeddings.pt")
    ap.add_argument("--max_videos", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 64))
    d_dyn    = int(a.get("d_dyn",    64))
    enc_hid  = int(a.get("enc_hidden_ch", a.get("hidden_ch", 128)))
    shared_trunk = bool(a.get("shared_trunk", False))
    use_ln = "norm_static.weight" in ckpt["encoder"]

    enc = LatentEncoder3D(
        d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hid,
        shared_trunk=shared_trunk, use_layer_norm=use_ln,
    ).to(device)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    print(f"[model] {args.ckpt}")
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_h={enc_hid} use_ln={use_ln}")

    ds = ClevrerChunkPairs(
        args.cache_dir, split="val",
        val_frac=args.val_frac, seed=args.seed, max_videos=args.max_videos,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=chunk_collate,
    )
    print(f"[data] {len(ds)} val pairs "
          f"(max_videos={args.max_videos}, val_frac={args.val_frac}, seed={args.seed})")

    t0 = time.time()
    zs, zd_l, zd_m, wan, per_video = encode_all(enc, loader, device)
    print(f"[encode] {len(per_video)} videos, {len(zs)} chunks in {time.time()-t0:.1f}s")

    # Per-chunk tensors, in a deterministic order
    chunk_keys = sorted(zs.keys())
    z_static_t   = torch.stack([zs[k]   for k in chunk_keys])
    z_dyn_last_t = torch.stack([zd_l[k] for k in chunk_keys])
    z_dyn_mean_t = torch.stack([zd_m[k] for k in chunk_keys])
    wan_mean_t   = torch.stack([wan[k]  for k in chunk_keys])

    # Sanity: z_static should be ~constant across the chunks of one video
    # (InfoNCE pulled them together during training); print the average
    # within-video std as a generalization diagnostic.
    by_vid_idx: dict[int, list[int]] = {}
    for i, (vid, _s) in enumerate(chunk_keys):
        by_vid_idx.setdefault(vid, []).append(i)
    within_stds = []
    for vid, idxs in by_vid_idx.items():
        if len(idxs) > 1:
            within_stds.append(z_static_t[idxs].std(dim=0).mean().item())
    if within_stds:
        ws = torch.tensor(within_stds)
        print(f"[sanity] within-video std of z_static (mean over dims, mean over videos) = "
              f"{ws.mean():.4f}  (expected small if InfoNCE generalizes; large=does not)")

    # Per-video aggregated features (mean over chunks of same video)
    video_ids = sorted(per_video.keys())
    video_z_static, video_z_dyn_mean, video_wan_mean = [], [], []
    for vid in video_ids:
        idxs = by_vid_idx[vid]
        video_z_static.append(z_static_t[idxs].mean(dim=0))
        video_z_dyn_mean.append(z_dyn_mean_t[idxs].mean(dim=0))
        video_wan_mean.append(wan_mean_t[idxs].mean(dim=0))
    video_z_static_t   = torch.stack(video_z_static)
    video_z_dyn_mean_t = torch.stack(video_z_dyn_mean)
    video_wan_mean_t   = torch.stack(video_wan_mean)

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "val_embeddings.pt"
    torch.save({
        "ckpt": args.ckpt,
        "split": {"max_videos": args.max_videos,
                  "val_frac": args.val_frac, "seed": args.seed},
        "chunk_keys":      chunk_keys,
        "z_static":        z_static_t,
        "z_dyn_last":      z_dyn_last_t,
        "z_dyn_mean":      z_dyn_mean_t,
        "wan_mean":        wan_mean_t,
        "per_video":       per_video,
        "video_ids":       video_ids,
        "video_z_static":  video_z_static_t,
        "video_z_dyn_mean": video_z_dyn_mean_t,
        "video_wan_mean":  video_wan_mean_t,
    }, out_path)
    print(f"[done] wrote {out_path}")
    print(f"        chunks: {z_static_t.shape}  videos: {video_z_static_t.shape}")


if __name__ == "__main__":
    main()
