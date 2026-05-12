"""Rung 7: counterfactual edit → rollout → decode.

Picks a target video (default: video 4242, which has a collision in the window),
identifies a slot to delete (default: the lower-index slot in the first
collision pair), and renders two variants through the wan-flow decoder:

  - factual    : original 5-video state, rolled out, rendered
  - counter    : same state but with the chosen slot's visibility = 0
                 (the dynamics no longer feels its V_pair contributions, and
                 the decoder masks it from cross-attention)

Saves a side-by-side grid: GT / Wan-VAE ceiling / factual rollout / counterfactual rollout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")

import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.accel_net import AccelNet
from src.model.slot_lagrangian import CollisionImpulse
from src.model.wan_flow_decoder import WanLatentFlowDecoder
from scripts.overfit_wan_flow import (
    build_encoder, load_wan_vae, latent_norm_buffers,
    encode_video, decode_latent, velocities_from_positions, save_grid,
)
from scripts.train_slot import (
    _autoregressive_rollout_eval, encode_window, pool_z_static,
)
from scripts.eval_rollout_decode import build_dynamics, build_impulse, euler_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_slot", required=True)
    ap.add_argument("--ckpt_decoder", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_video", type=int, default=4242,
                    help="video_id to apply counterfactual to; falls back to first video with a collision")
    ap.add_argument("--slot_to_delete", type=int, default=-1,
                    help="slot index to delete (default: pick automatically from a collision pair)")
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Slot pipeline
    slot_ckpt = torch.load(args.ckpt_slot, map_location=device, weights_only=False)
    cfg = OmegaConf.create(slot_ckpt["config"])
    pos_norm = float(cfg.dataset.pos_normalize)

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg.dataset.data_dir),
        annotation_dir=str(cfg.dataset.annotation_dir),
        split=str(cfg.dataset.split),
        window_length=int(cfg.training.window_length),
        frames_per_video=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.windows_per_video),
        max_videos=int(cfg.training.max_videos),
        max_objects=int(cfg.dataset.max_objects),
        coordinate_mode=str(cfg.dataset.coordinate_mode),
        image_size=int(cfg.dataset.image_size),
        seed=int(cfg.training.seed),
    )
    attr_dim = dataset.attr_dim
    d_static = int(cfg.model.get("d_static", 16))

    encoder = build_encoder(slot_ckpt, attr_dim, d_static, device)
    dyn, dyn_type = build_dynamics(cfg, attr_dim, device)
    dyn.load_state_dict(slot_ckpt["lagrangian_state_dict"])
    impulse = build_impulse(cfg, device)
    if "impulse_state_dict" in slot_ckpt:
        impulse.load_state_dict(slot_ckpt["impulse_state_dict"])

    vae = load_wan_vae(args.model_id, dtype, device)
    lat_mean, lat_std = latent_norm_buffers(vae, device, torch.float32)

    dec_ckpt = torch.load(args.ckpt_decoder, map_location=device, weights_only=False)
    train_args = dec_ckpt["args"]
    decoder = WanLatentFlowDecoder(
        latent_channels=48, latent_grid=8, latent_T=2,
        d_model=int(train_args.get("d_model", 384)),
        n_heads=int(train_args.get("n_heads", 6)),
        n_blocks=int(train_args.get("n_blocks", 6)),
        t_dim=int(train_args.get("d_model", 384)),
        d_static=(attr_dim if train_args["cond_source"] == "gt_attrs" else d_static),
        max_objects=int(cfg.dataset.max_objects),
        window_length=int(cfg.training.window_length),
        use_i0=bool(train_args.get("use_i0", True)),
    ).to(device)
    state_key = "ema_decoder_state_dict" if "ema_decoder_state_dict" in dec_ckpt else "decoder_state_dict"
    decoder.load_state_dict(dec_ckpt[state_key])
    decoder.eval()

    # --- Pick the target video (must have a collision when possible)
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= 5:
            break
    if not seen:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= 5:
                break
    samples = list(seen.values())

    target_idx = None
    for n, s in enumerate(samples):
        if s["video_id"] == args.target_video:
            target_idx = n
            break
    if target_idx is None:
        # fall back to first video with collisions
        for n, s in enumerate(samples):
            if len(s["collisions"]) > 0:
                target_idx = n
                break
    if target_idx is None:
        target_idx = 0
    target = samples[target_idx]
    print(f"Counterfactual target: video {target['video_id']} (idx {target_idx})")
    print(f"  collisions in window: {target['collisions']}")

    # Build batch with all 5 (so impulse module sees the same batch shape)
    batch = paired_collate(samples)
    frames = batch["frames"].to(device).to(dtype)
    gt_pos = batch["positions"].to(device).float() / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device).float()
    collisions = batch["collisions"]
    N, T = frames.shape[:2]
    K = gt_pos.shape[2]

    # --- Pick a slot to delete from the target video
    if args.slot_to_delete >= 0:
        slot_id = args.slot_to_delete
    elif len(collisions[target_idx]) > 0:
        # take the first collision; delete the smaller-id slot in the pair
        t_c, i_c, j_c = collisions[target_idx][0]
        slot_id = min(i_c, j_c)
        print(f"  collision at t={t_c}: slots ({i_c}, {j_c}) — deleting slot {slot_id}")
    else:
        # smooth video — pick the slot with most visible frames as a non-trivial deletion
        per_slot = visib[target_idx].sum(dim=0)  # (K,)
        slot_id = int(per_slot.argmax().item())
        print(f"  no collision in window — deleting most-visible slot {slot_id}")

    # --- Encode + pool z_static
    with torch.no_grad():
        q_enc, z_static_pf = encode_window(encoder, frames.float(), attrs)
        z_static = pool_z_static(z_static_pf, visib)            # (N, K, d_static)

    # --- Build factual visibility = original
    visib_factual = visib.clone()

    # --- Build counterfactual visibility = delete slot_id from target video at all frames
    visib_counter = visib.clone()
    visib_counter[target_idx, :, slot_id] = 0.0

    # --- Rollout with each visibility (encoder seed for t=0,1; dynamics rolls forward)
    q_factual = _autoregressive_rollout_eval(
        dyn, impulse,
        q_enc, visib_factual, z_static,
        collisions_per_sample=collisions,
        solver_alpha=float(cfg.training.solver_alpha),
        solver_steps=max(int(cfg.training.solver_steps), 2),
        direction_clip=0.5,
        dynamics_type=dyn_type,
    )
    # For the counterfactual, also remove the deleted slot's collisions so the
    # impulse module doesn't try to apply them.
    coll_counter = [list(cc) for cc in collisions]
    coll_counter[target_idx] = [
        (t, i, j) for (t, i, j) in collisions[target_idx]
        if i != slot_id and j != slot_id
    ]
    q_counter = _autoregressive_rollout_eval(
        dyn, impulse,
        q_enc, visib_counter, z_static,
        collisions_per_sample=coll_counter,
        solver_alpha=float(cfg.training.solver_alpha),
        solver_steps=max(int(cfg.training.solver_steps), 2),
        direction_clip=0.5,
        dynamics_type=dyn_type,
    )

    # --- Wan-VAE encode + i0
    with torch.no_grad():
        z_full = encode_video(vae, frames).float()
        ceiling = decode_latent(vae, z_full.to(dtype)).float()
        if bool(train_args.get("use_i0", True)):
            z_i0 = encode_video(vae, frames[:, :1]).float()
            z_i0_norm = (z_i0 - lat_mean) / lat_std
        else:
            z_i0_norm = None

    cond_source = train_args["cond_source"]
    z_cond = attrs if cond_source == "gt_attrs" else z_static

    g = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.randn(N, 48, 2, 8, 8, generator=g, device=device)

    def sample(q, vis):
        v = velocities_from_positions(q)
        with torch.no_grad():
            x = euler_sample(decoder, args.num_steps, x0.clone(),
                             q, v, z_cond, vis, z_i0_norm)
            z_sample = x * lat_std + lat_mean
            return decode_latent(vae, z_sample.to(dtype)).float()

    rec_factual = sample(q_factual, visib_factual)
    rec_counter = sample(q_counter, visib_counter)

    frames_f = frames.float()
    T_out = ceiling.shape[1]
    T_cmp = min(T, T_out)
    ceil_mse = (ceiling[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()

    # --- Per-video pixel MSE (factual = closeness to GT; counterfactual differs by design)
    print(f"\nWan-VAE ceiling MSE: {ceil_mse:.6f}")
    print(f"\n{'Video':>6} | {'factual MSE vs GT':>18} | {'counter MSE vs GT':>18} | {'counter ≠ factual':>18}")
    print("-" * 80)
    for n, s in enumerate(samples):
        m_f = (rec_factual[n, :T_cmp] - frames_f[n, :T_cmp]).pow(2).mean().item()
        m_c = (rec_counter[n, :T_cmp] - frames_f[n, :T_cmp]).pow(2).mean().item()
        m_diff = (rec_counter[n, :T_cmp] - rec_factual[n, :T_cmp]).pow(2).mean().item()
        marker = "  <- target" if n == target_idx else ""
        print(f"{s['video_id']:>6} | {m_f:>18.6f} | {m_c:>18.6f} | {m_diff:>18.6f}{marker}")

    # --- Save grid for the target video: GT / ceiling / factual / counterfactual
    n = target_idx
    rows = []
    rows.extend([frames_f[n, t] for t in range(T_cmp)])
    rows.extend([ceiling[n, t] for t in range(T_cmp)])
    rows.extend([rec_factual[n, t] for t in range(T_cmp)])
    rows.extend([rec_counter[n, t] for t in range(T_cmp)])
    fname = f"counterfactual_video_{target['video_id']}_delete_slot{slot_id}.png"
    save_grid(rows, out_dir / fname, nrow=T_cmp)
    print(f"\nGrid saved: {out_dir / fname}")
    print("Rows: GT / Wan-VAE ceiling / factual rollout / counterfactual rollout (slot {} deleted)".format(slot_id))

    # --- Also save grids for the other 4 videos (to confirm they're untouched)
    for n, s in enumerate(samples):
        if n == target_idx:
            continue
        rows = []
        rows.extend([frames_f[n, t] for t in range(T_cmp)])
        rows.extend([ceiling[n, t] for t in range(T_cmp)])
        rows.extend([rec_factual[n, t] for t in range(T_cmp)])
        rows.extend([rec_counter[n, t] for t in range(T_cmp)])
        save_grid(rows, out_dir / f"untouched_video_{s['video_id']}.png", nrow=T_cmp)


if __name__ == "__main__":
    main()
