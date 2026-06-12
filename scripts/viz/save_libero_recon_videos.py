"""Side-by-side reconstruction videos for the LIBERO v5.1.2 checkpoint.

Renders 3 LABELED panels per frame so you can see WHERE the model's pixel
loss comes from (the eval_libero_pixel_loss.py number was 0.0215 — ~10× the
VAE round-trip floor; this viz says whether it's coarse blur, structure
drift, or color/identity issues):

  1. RAW          - GT frames decoded from the mp4
  2. VAE FLOOR    - decode(encode(RAW)) through the FROZEN Wan-2.2 VAE.
                    Best the small model could possibly do — sets the upper
                    bound on reconstruction quality.
  3. MODEL        - decode(small_dec(small_enc(cached_latent))) through the
                    same frozen Wan VAE. Difference vs panel 2 = the small
                    model's drift in latent space.

One mp4 per chunk: <out_dir>/<task>__ep<id>__sf<start>.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from torchvision.io import read_video

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


W = 33


def _resolve_wan_path(name: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers") -> str:
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    repo_dir = Path(cache_root) / f"models--{name.replace('/', '--')}"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        snap = repo_dir / "snapshots" / ref_main.read_text().strip()
        if snap.exists(): return str(snap)
    return name


def _vae_encode(vae, pix, dtype):
    x = pix.permute(0, 2, 1, 3, 4).contiguous().to(dtype)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.float()


def _vae_decode(vae, latent, dtype):
    z = latent.to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()       # (T,H,W,3)


def label_panel(frames_u8, text):
    out = []
    for fr in frames_u8:
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, len(text) * 6 + 4, 12], fill=(0, 0, 0))
        d.text((2, 1), text, fill=(255, 255, 0))
        out.append(np.asarray(im))
    return np.stack(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--libero_root", default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90_processed")
    ap.add_argument("--out_dir", required=True,
                    help="Local outputs/ dir for the rendered mp4s.")
    ap.add_argument("--n_videos", type=int, default=6,
                    help="N chunks/episodes rendered, drawn from distinct task families.")
    ap.add_argument("--full_episode", action="store_true",
                    help="If set, render the FULL episode (concatenate all chunks) "
                         "instead of a single 33-frame chunk. Output is one mp4 per episode.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- our small model ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 64)); d_dyn = int(a.get("d_dyn", 64))
    enc_h = int(a.get("enc_hidden_ch", 128)); dec_h = int(a.get("dec_hidden_ch", 256))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_h,
                          use_layer_norm=use_ln).to(device).eval()
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_h,
                        chunk_size_lat=chunk_size_lat).to(device).eval()
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    print(f"[ckpt] {args.ckpt}  ep={ckpt.get('epoch')}  use_ln={use_ln}")

    # ---- Wan VAE (frozen) ----
    from diffusers import AutoencoderKLWan
    path = _resolve_wan_path()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype).eval().to(device)
    for p in vae.parameters(): p.requires_grad_(False)
    print(f"[vae ] {path}")

    # ---- pick N diverse chunks from val ----
    metadata = json.loads((Path(args.cache_dir) / "metadata.json").read_text())
    windows = metadata["windows"]
    task_id_map = metadata["task_id_map"]
    inv_task = {v: k for k, v in task_id_map.items()}
    # Use the SAME train/val partition the trainer uses (val_every_k=10 mod
    # episode_id; n_holdout_tasks=10 default holdout seed=0).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.libero_window import _make_holdout_task_ids
    holdout = set(_make_holdout_task_ids(len(task_id_map), 10, 0))

    # Walk all val windows once, classify by SCENE family from the task name
    # prefix, then sample n_videos uniformly across scenes so the renders span
    # KITCHEN / LIVING_ROOM / STUDY rather than clustering on one.
    #
    # For full-episode mode we group windows by global_episode_id so a single
    # render covers ALL chunks of one demonstration.
    val_by_scene: dict[str, list[dict]] = {}
    val_eps_by_scene: dict[str, dict[int, list[dict]]] = {}
    for w in windows:
        tid = int(w["task_id"])
        if tid in holdout: continue
        b = torch.load(Path(args.cache_dir) / w["path"], map_location="cpu",
                       weights_only=False)
        if int(b["episode_id"]) % 10 != 0: continue
        scene = inv_task[tid].split("_SCENE")[0]            # KITCHEN, LIVING_ROOM, STUDY
        item = {**w, "blob": b}
        val_by_scene.setdefault(scene, []).append(item)
        gid = int(b["global_episode_id"])
        val_eps_by_scene.setdefault(scene, {}).setdefault(gid, []).append(item)

    chosen = []
    seen_tasks = set()
    scenes = sorted(val_by_scene.keys())

    if args.full_episode:
        # Pick distinct tasks across scenes; for each, gather ALL chunks of one
        # episode sorted by start_frame. chosen items are LISTS of chunk dicts.
        while len(chosen) < args.n_videos:
            added = False
            for sc in scenes:
                # Episodes in this scene, ordered by gid for determinism
                for gid in sorted(val_eps_by_scene[sc].keys()):
                    chunks = val_eps_by_scene[sc][gid]
                    tid = int(chunks[0]["task_id"])
                    if tid in seen_tasks: continue
                    chunks_sorted = sorted(chunks, key=lambda c: int(c["start_frame"]))
                    chosen.append((tid, chunks_sorted))
                    seen_tasks.add(tid)
                    added = True
                    break
                if len(chosen) >= args.n_videos: break
            if not added: break
        print(f"[pick] {len(chosen)} FULL episodes from distinct tasks (val split):")
        for tid, chs in chosen:
            sfs = [int(c["start_frame"]) for c in chs]
            print(f"  task={inv_task[tid][:55]} ep={chs[0]['blob']['episode_id']}"
                  f" n_chunks={len(chs)} sf={sfs}")
    else:
        # Round-robin across scenes, take first chunk of distinct tasks within each.
        while len(chosen) < args.n_videos:
            added = False
            for sc in scenes:
                for cand in val_by_scene[sc]:
                    tid = int(cand["task_id"])
                    if tid in seen_tasks: continue
                    chosen.append((tid, cand))
                    seen_tasks.add(tid)
                    added = True
                    break
                if len(chosen) >= args.n_videos: break
            if not added: break
        print(f"[pick] {len(chosen)} chunks from distinct tasks (val split):")
        for tid, w in chosen:
            print(f"  task={inv_task[tid][:60]} ep={w['blob']['episode_id']} sf={w['start_frame']}")

    # ---- per-chunk render ----
    # Map global_episode_id -> mp4 path via libero_90.jsonl.
    libero_root = Path(args.libero_root)
    ep_video = {}
    with (libero_root / "libero_90.jsonl").open() as f:
        for gid, line in enumerate(f):
            r = json.loads(line)
            ep_video[gid] = r["visual_input"]

    def _render_chunk(blob, mp4_path):
        """Encode/decode one chunk; return (gt, vae_rt, model_pix) each (T,3,H,W) on cpu."""
        sf = int(blob["start_frame"])
        v, _, _ = read_video(str(mp4_path), pts_unit="sec")
        gt = v[sf:sf + W].float() / 127.5 - 1.0                      # (33, H, W, 3)
        gt = gt.permute(0, 3, 1, 2).contiguous()                     # (33, 3, H, W)

        cached_latent = blob["latent"].unsqueeze(0).to(device)
        gt_b = gt.unsqueeze(0).to(device)

        with torch.no_grad():
            vae_lat = _vae_encode(vae, gt_b, dtype)
            vae_rt  = _vae_decode(vae, vae_lat, dtype)[0].cpu()
            out = enc(cached_latent)
            recon = dec(out["z_static"], out["z_dyn"])
            model_pix = _vae_decode(vae, recon, dtype)[0].cpu()
        return gt, vae_rt, model_pix

    if args.full_episode:
        for tid, chs in chosen:
            ep_id = int(chs[0]["blob"]["episode_id"])
            gid   = int(chs[0]["blob"]["global_episode_id"])
            mp4_path = libero_root / ep_video[gid]
            gt_all, vae_all, model_all = [], [], []
            for c in chs:
                gt, vae_rt, model_pix = _render_chunk(c["blob"], mp4_path)
                T = min(gt.shape[0], vae_rt.shape[0], model_pix.shape[0])
                gt_all.append(gt[:T]); vae_all.append(vae_rt[:T]); model_all.append(model_pix[:T])
            gt_full    = torch.cat(gt_all, dim=0)
            vae_full   = torch.cat(vae_all, dim=0)
            model_full = torch.cat(model_all, dim=0)

            p1 = label_panel(to_uint8(gt_full),    "RAW")
            p2 = label_panel(to_uint8(vae_full),   "VAE floor")
            p3 = label_panel(to_uint8(model_full), "MODEL")
            grid = np.concatenate([p1, p2, p3], axis=2)

            safe_task = inv_task[tid][:55].replace("/", "_")
            name = f"{safe_task}__ep{ep_id:02d}__FULL_{len(chs)}chunks.mp4"
            out_path = out_dir / name
            imageio.mimwrite(str(out_path), grid, fps=args.fps, quality=8,
                             macro_block_size=1)
            mse_floor = ((vae_full   - gt_full) ** 2).mean().item()
            mse_model = ((model_full - gt_full) ** 2).mean().item()
            print(f"  wrote {out_path}  ({grid.shape[0]} frames)  "
                  f"| mse_floor={mse_floor:.5f} mse_model={mse_model:.5f}"
                  f" gap={mse_model - mse_floor:+.5f}")
    else:
        for tid, w in chosen:
            blob = w["blob"]
            gid = int(blob["global_episode_id"]); sf = int(blob["start_frame"])
            mp4_path = libero_root / ep_video[gid]
            gt, vae_rt, model_pix = _render_chunk(blob, mp4_path)
            T = min(gt.shape[0], vae_rt.shape[0], model_pix.shape[0])
            p1 = label_panel(to_uint8(gt[:T]), "RAW")
            p2 = label_panel(to_uint8(vae_rt[:T]), "VAE floor")
            p3 = label_panel(to_uint8(model_pix[:T]), "MODEL")
            grid = np.concatenate([p1, p2, p3], axis=2)

            safe_task = inv_task[tid][:55].replace("/", "_")
            name = f"{safe_task}__ep{blob['episode_id']:02d}__sf{sf:03d}.mp4"
            out_path = out_dir / name
            imageio.mimwrite(str(out_path), grid, fps=args.fps, quality=8,
                             macro_block_size=1)
            mse_floor = ((vae_rt[:T] - gt[:T]) ** 2).mean().item()
            mse_model = ((model_pix[:T] - gt[:T]) ** 2).mean().item()
            print(f"  wrote {out_path}  | mse_floor={mse_floor:.5f} mse_model={mse_model:.5f}"
                  f" gap={mse_model - mse_floor:+.5f}")


if __name__ == "__main__":
    main()
