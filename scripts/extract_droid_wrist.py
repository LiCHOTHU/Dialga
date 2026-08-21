"""Cache DROID wrist-camera episodes as Wan-VAE latent chunks (W=33) WITH the
known per-frame camera pose (v5.8 moving-camera experiment).

The wrist camera is rigidly mounted to the robot end-effector, so the robot's
proprioceptive EE pose (`observation/cartesian_position`, 6-DoF xyz+rpy) IS the
known camera trajectory -- no SfM, no depth needed for the pose-conditioning
ablation. (The DROID RLDS release ships left images only: no stereo, no depth,
no separate extrinsics. Depth-based parallax reprojection is a deferred upgrade
via monocular depth; this cache supports the pose-only figure.)

Per episode we slide W=33 non-overlapping windows from frame 0 and drop the tail
shorter than W. Each window -> one Wan latent (48,9,8,8). The 33 per-frame poses
are subsampled to the 9 latent frames (Wan causal temporal stride 4: latent
frame 0 <- pixel frame 0; latent frame j>=1 <- pixel frame 4j). Rotation angles
are unwrapped over the window before subsampling so the frame-0-relative pose the
model consumes is continuous (no +/-pi jumps).

Image: DROID wrist frames are 180x320. We centre-crop to 180x180 (keep natural
proportions -- the gripper sits centred) then resize to 128x128 to match the
Wan/CLEVRER 8x8-latent pipeline.

Blob keys (per chunk):
    latent            (48, 9, 8, 8) float32  Wan-VAE mean
    pose              (9, 6)        float32  cartesian_position at the 9 latent frames
    vel               (9, 6)        float32  cartesian_velocity  at the 9 latent frames
    episode_id        int                    global episode index
    start_frame       int                    pixel-frame offset of this chunk
    num_frames_total  int                    full episode length
    file_path         str                    DROID source recording path
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

W = 33          # Wan temporal stride: T=33 -> T_lat=9
LAT_IDX = [0, 4, 8, 12, 16, 20, 24, 28, 32]   # pixel frame per latent frame
CROP = (70, 250)   # centre 180 columns of the 320-wide wrist frame
RES = 128


def load_wan_vae(model_id: str, dtype: torch.dtype, device: torch.device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def encode_chunk(vae, frames_uint8: torch.Tensor, device, dtype) -> torch.Tensor:
    """frames_uint8: (T, H, W, 3) uint8 -> latent (C, T_lat, H_lat, W_lat) float32."""
    x = frames_uint8.to(device).to(dtype)
    x = (x / 127.5) - 1.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()   # (1,3,T,H,W)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.squeeze(0).cpu().float()


def crop_resize(frames: np.ndarray) -> torch.Tensor:
    """(T,180,320,3) uint8 -> (T,128,128,3) uint8, centre-crop square + resize."""
    x = torch.from_numpy(frames[:, :, CROP[0]:CROP[1], :]).float()   # (T,180,180,3)
    x = x.permute(0, 3, 1, 2)                                        # (T,3,180,180)
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)  # (T,128,128,3)
    return x.contiguous()


def subsample_pose(pose33: np.ndarray) -> np.ndarray:
    """(33,6) -> (9,6): unwrap the 3 rotation dims over the window, then pick the
    9 latent-frame representatives. Unwrapping keeps frame-0-relative pose smooth."""
    p = pose33.copy()
    p[:, 3:6] = np.unwrap(p[:, 3:6], axis=0)   # roll/pitch/yaw continuity
    return p[LAT_IDX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--droid_dir", type=str,
                    default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/droid_full/1.0.1")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--start_episode", type=int, default=0)
    ap.add_argument("--end_episode", type=int, default=200,
                    help="Exclusive. Small-first default = first 200 episodes.")
    ap.add_argument("--min_ee_span", type=float, default=0.05,
                    help="Skip episodes whose EE xyz travels < this (metres): a "
                         "near-static camera has nothing to divide out.")
    ap.add_argument("--save_frames", action="store_true",
                    help="Also store the (33,128,128,3) uint8 GT chunk in each "
                         "blob so the readout can compute true pixel PSNR.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir)
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")   # keep the GPU for Wan only
    import tensorflow_datasets as tfds

    print(f"[data] opening DROID at {args.droid_dir}")
    builder = tfds.builder_from_directory(args.droid_dir)
    split = f"train[{args.start_episode}:{args.end_episode}]"
    ds = builder.as_dataset(split=split, shuffle_files=False)
    print(f"[data] split={split}")

    print(f"[vae ] loading {args.model_id} (~1 min)")
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"[vae ] {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params, {dtype}")

    metadata = []
    t0 = time.time()
    chunk_idx = 0
    n_static_skip = 0
    for ep_i, ep in enumerate(tfds.as_numpy(ds)):
        global_ep = args.start_episode + ep_i
        # steps is a nested iterable dataset (one dict per frame) -> stack.
        imgs_l, cps_l, vel_l = [], [], []
        for st in ep["steps"]:
            imgs_l.append(st["observation"]["wrist_image_left"])
            cps_l.append(st["observation"]["cartesian_position"])       # EE pose = wrist-cam pose
            vel_l.append(st["action_dict"]["cartesian_velocity"])       # velocity lives in action_dict
        imgs = np.stack(imgs_l)                                # (T,180,320,3) uint8
        cps = np.stack(cps_l).astype(np.float32)               # (T,6)
        vel = np.stack(vel_l).astype(np.float32)               # (T,6)
        file_path = ep["episode_metadata"]["file_path"]
        if isinstance(file_path, bytes):
            file_path = file_path.decode()
        n = imgs.shape[0]
        span = float(np.linalg.norm(cps[:, :3].max(0) - cps[:, :3].min(0)))
        if span < args.min_ee_span:
            n_static_skip += 1
            continue
        n_chunks = n // W
        for c in range(n_chunks):
            s = c * W
            out_path = out_dir / "latents" / f"{chunk_idx:06d}.pt"
            if not out_path.exists():
                frames = crop_resize(imgs[s:s + W])                  # (33,128,128,3) u8
                z = encode_chunk(vae, frames, device, dtype)         # (48,9,8,8)
                blob = {
                    "latent":           z,
                    "pose":             torch.from_numpy(subsample_pose(cps[s:s + W])).float(),
                    "vel":              torch.from_numpy(vel[s:s + W][LAT_IDX]).float(),
                    "episode_id":       int(global_ep),
                    "start_frame":      int(s),
                    "num_frames_total": int(n),
                    "file_path":        file_path,
                }
                if args.save_frames:
                    blob["frames"] = frames                          # (33,128,128,3) u8 GT
                torch.save(blob, out_path)
            metadata.append({
                "idx": chunk_idx,
                "path": str(out_path.relative_to(out_dir)),
                "episode_id": int(global_ep),
                "start_frame": int(s),
            })
            chunk_idx += 1
        if (ep_i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  ep {ep_i+1} (gid {global_ep})  chunks={chunk_idx}  "
                  f"static_skip={n_static_skip}  {el:.1f}s "
                  f"({chunk_idx/max(el,1e-6):.2f} chunk/s)")

    with (out_dir / "metadata.json").open("w") as f:
        json.dump({"args": vars(args), "n_chunks": len(metadata),
                   "windows": metadata}, f, indent=2)
    print(f"\n[done] {len(metadata)} chunks from episodes "
          f"[{args.start_episode},{args.end_episode}) "
          f"({n_static_skip} near-static episodes skipped) -> {out_dir}")


if __name__ == "__main__":
    main()
