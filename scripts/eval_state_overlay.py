"""
State-space evaluation & visualization for the slot-Lagrangian model.

Produces (no learned pixel decoder needed):
  1. Per-video 2D world-coord trajectory plots:
       - GT  (solid lines, faded markers per frame)
       - Rollout (dashed lines, markers)
       - Encoder predictions (small dots, only on the start frames)
  2. Per-video animated top-down scene mp4:
       - colored dots for GT object positions vs rollout positions vs encoder predictions
  3. Aggregate state metrics over a longer rollout window than W=6.

This is the canonical physics-world-model evaluation: state errors over
extended rollouts, no pixel decoder involved.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.data.clevrer_states import COLOR_VOCAB, SHAPE_VOCAB
from src.model.slot_lagrangian import (
    SlotQueryEncoder, LatentSIGRegEncoder, CollisionImpulse,
    apply_collision_impulse_to_positions,
)
from src.model.accel_net import AccelNet, verlet_step
from scripts.train_slot import encode_window


# -- color palette per slot (chosen to be distinguishable) --
SLOT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# -- CLEVRER Blender camera (constants from the public dataset/render scripts) --
_CLEVRER_CAM_LOC = np.array([7.4811, -6.5076, 5.3437], dtype=np.float64)
_CLEVRER_CAM_EULER_XYZ = (1.10878, 0.0, 0.81526)  # radians, Blender intrinsic XYZ
_CLEVRER_FOCAL_MM = 35.0
_CLEVRER_SENSOR_W_MM = 32.0
_CLEVRER_RAW_IMG_WH = (480, 320)


def _euler_xyz_to_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # Blender intrinsic XYZ rotation


def make_clevrer_pixel_proj(image_size_px, ground_z=0.2):
    """
    Build a function world_xy → pixel_xy using CLEVRER's Blender camera.

    image_size_px : the (W, H) of the *displayed* (resized) frame, e.g. (128, 128).
    ground_z      : world Z of the object centers (objects sit on z≈0.2 floor).

    Returns: callable np.ndarray (N, 2) → (N, 2)  [pixel coords in displayed frame]
    """
    R_local_to_world = _euler_xyz_to_matrix(*_CLEVRER_CAM_EULER_XYZ)
    R_world_to_cam = R_local_to_world.T
    T = _CLEVRER_CAM_LOC

    raw_w, raw_h = _CLEVRER_RAW_IMG_WH
    sensor_h_mm = _CLEVRER_SENSOR_W_MM * raw_h / raw_w  # square pixels assumption
    f_pix_x = _CLEVRER_FOCAL_MM / _CLEVRER_SENSOR_W_MM * raw_w
    f_pix_y = _CLEVRER_FOCAL_MM / sensor_h_mm * raw_h     # equal to f_pix_x for square px
    cx, cy = raw_w / 2.0, raw_h / 2.0

    img_w, img_h = image_size_px
    sx = img_w / raw_w
    sy = img_h / raw_h

    def proj(world_xy):
        N = world_xy.shape[0]
        P = np.zeros((N, 3), dtype=np.float64)
        P[:, 0] = world_xy[:, 0]
        P[:, 1] = world_xy[:, 1]
        P[:, 2] = ground_z
        P_cam = (P - T) @ R_world_to_cam.T
        # Blender camera looks along -Z in local, so visible points have z<0
        z = -P_cam[:, 2]
        # avoid div by zero for degenerate cases
        z = np.where(z > 1e-3, z, 1e-3)
        x = P_cam[:, 0] / z
        y = P_cam[:, 1] / z
        # to pixel coords on raw image
        u_raw = x * f_pix_x + cx
        v_raw = -y * f_pix_y + cy   # flip Y for image-origin top-left
        # rescale to displayed frame
        return np.stack([u_raw * sx, v_raw * sy], axis=-1)

    return proj


def build_modules_from_ckpt(ckpt, attr_dim, device):
    cfg = ckpt["config"]
    enc_type = str(cfg["model"]["encoder_type"])
    d_static = int(cfg["model"].get("d_static", 16))
    common = dict(
        image_size=int(cfg["dataset"]["image_size"]),
        patch_size=int(cfg["model"]["patch_size"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        depth=int(cfg["model"]["encoder_depth"]),
        num_heads=int(cfg["model"]["num_heads"]),
        mlp_ratio=float(cfg["model"]["mlp_ratio"]),
        max_objects=int(cfg["dataset"]["max_objects"]),
        attr_dim=int(attr_dim),
        num_state_dims=int(cfg["model"]["num_state_dims"]),
        d_static=d_static,
    )
    if enc_type == "slot":
        encoder = SlotQueryEncoder(**common)
    else:
        encoder = LatentSIGRegEncoder(latent_dim=int(cfg["model"]["latent_dim"]), **common)
    encoder.load_state_dict(ckpt["encoder_state_dict"]); encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    dyn_type = "accel"
    # Dynamics + impulse are conditioned on z_static (dim=d_static), not attrs.
    lagrangian = AccelNet(
        num_state_dims=int(cfg["model"]["num_state_dims"]),
        attr_dim=d_static,
        hidden=int(cfg["model"]["lagrangian_hidden"]),
        use_pair=bool(cfg["model"]["lagrangian_use_pair"]),
    )
    lagrangian.load_state_dict(ckpt["lagrangian_state_dict"]); lagrangian.to(device).eval()
    for p in lagrangian.parameters():
        p.requires_grad_(False)

    impulse = CollisionImpulse(
        attr_dim=d_static,
        hidden=int(cfg["model"].get("impulse_hidden", 64)),
    )
    if "impulse_state_dict" in ckpt:
        impulse.load_state_dict(ckpt["impulse_state_dict"])
    impulse.to(device).eval()
    for p in impulse.parameters():
        p.requires_grad_(False)

    return cfg, encoder, lagrangian, impulse, enc_type, dyn_type


def autoregressive_rollout(dynamics, impulse, q_seq_gt, alpha_seq, cond,
                           collisions_per_sample, solver_alpha, solver_steps,
                           dynamics_type="accel"):
    """
    Roll out the AccelNet+Verlet dynamics with open-system handlers:

      - any frame where alpha=0 (slot invisible) → inject GT.
      - collision frames → apply CollisionImpulse correction.

    `cond` is the per-slot conditioning latent (z_static_video, dim=d_static).
    """
    B, W, K, D = q_seq_gt.shape
    pred = q_seq_gt.clone()
    for t in range(2, W):
        q_prev = pred[:, t - 2].clone()
        q_curr = pred[:, t - 1]
        if impulse is not None:
            pairs_at_prev = [
                [(0, i, j) for (tt, i, j) in collisions_per_sample[b] if tt == t - 1]
                for b in range(B)
            ]
            if any(len(p) > 0 for p in pairs_at_prev):
                with torch.enable_grad():
                    q_prev = apply_collision_impulse_to_positions(
                        impulse, dynamics, q_prev, q_curr, cond, pairs_at_prev,
                    ).detach()
        with torch.no_grad():
            q_next = verlet_step(
                dynamics, q_prev, q_curr, cond,
                alpha_seq[:, t - 2], alpha_seq[:, t - 1],
                accel_clip=0.5,
            )
        # GT injection for any currently-invisible slot ("external" object)
        invisible_now = (alpha_seq[:, t] < 0.5).unsqueeze(-1)
        q_next = torch.where(invisible_now, q_seq_gt[:, t], q_next)
        # NaN/inf guard: if the solver blew up, fall back to last-velocity extrapolation
        bad = ~torch.isfinite(q_next)
        if bad.any():
            extrap = 2 * q_curr - q_prev
            q_next = torch.where(bad, extrap, q_next)
        pred[:, t] = q_next
    return pred


def build_long_window_dataset(cfg, window_length):
    """Pull whole-video samples by overriding window_length on the dataset."""
    return ClevrerPairedDataset(
        data_dir=str(cfg["dataset"]["data_dir"]),
        annotation_dir=str(cfg["dataset"]["annotation_dir"]),
        split=str(cfg["dataset"]["split"]),
        window_length=int(window_length),
        frames_per_video=int(cfg["dataset"]["video_num_frames"]),
        windows_per_video=int(cfg["training"]["windows_per_video"]),
        max_videos=int(cfg["training"]["max_videos"]),
        max_objects=int(cfg["dataset"]["max_objects"]),
        coordinate_mode=str(cfg["dataset"]["coordinate_mode"]),
        image_size=int(cfg["dataset"]["image_size"]),
        seed=int(cfg["training"]["seed"]),
    )


def attrs_to_label(attrs_vec):
    """Decode (A,) one-hot attrs to a label string like 'red metal cube'."""
    nC, nM, nS = len(COLOR_VOCAB), 2, len(SHAPE_VOCAB)
    color = COLOR_VOCAB[int(attrs_vec[:nC].argmax())]
    mat = ["rubber", "metal"][int(attrs_vec[nC:nC + nM].argmax())]
    shape = SHAPE_VOCAB[int(attrs_vec[nC + nM:nC + nM + nS].argmax())]
    return f"{color} {mat} {shape}"


def trajectory_plot(gt_pos, rollout_pos, encoder_pos, visib, attrs, video_id,
                    out_path, world_extent=2.5):
    """
    gt_pos, rollout_pos, encoder_pos: (W, K, 2) numpy arrays, world coords.
    visib: (W, K) bool/float numpy.
    attrs: (K, A) numpy.

    Plots GT trajectory (solid) vs rollout (dashed) per slot, only when visible.
    """
    W, K, _ = gt_pos.shape
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for k in range(K):
        m = visib[:, k] > 0.5
        if m.sum() < 2:
            continue
        c = SLOT_COLORS[k % len(SLOT_COLORS)]
        label = f"slot {k}: {attrs_to_label(attrs[k])}"
        ax.plot(gt_pos[m, k, 0], gt_pos[m, k, 1], "-", color=c, lw=1.5,
                label=label, alpha=0.9)
        ax.plot(rollout_pos[m, k, 0], rollout_pos[m, k, 1], "--", color=c, lw=1.5,
                alpha=0.9)
        # start marker
        i_first = int(np.argmax(m))
        ax.plot(gt_pos[i_first, k, 0], gt_pos[i_first, k, 1], "o",
                color=c, ms=8, mec="black", mew=0.5)
        # end marker for rollout
        i_last = W - 1 - int(np.argmax(m[::-1]))
        ax.plot(rollout_pos[i_last, k, 0], rollout_pos[i_last, k, 1], "x",
                color=c, ms=10, mew=2)
    ax.set_xlim(-world_extent, world_extent)
    ax.set_ylim(-world_extent, world_extent)
    ax.set_aspect("equal")
    ax.set_xlabel("world X")
    ax.set_ylabel("world Y")
    ax.set_title(f"video {video_id} — solid: GT, dashed: rollout, ●: start, ✕: end")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def scene_animation(gt_pos, rollout_pos, encoder_pos, visib, attrs, video_id,
                    out_path, world_extent=2.5, fps=8, frames_seq=None,
                    pixel_proj=None):
    """Three panels: actual CLEVRER video | GT dots top-down | rollout dots top-down.

    frames_seq: optional (W, 3, H, W) tensor in [-1, 1] of GT video frames.
    pixel_proj: optional callable taking (K, 2) world XY → (K, 2) pixel coords
                in the displayed video frame. If supplied, predicted/rollout
                positions are overlaid on the video panel as dots.
    """
    W, K, _ = gt_pos.shape
    have_video = frames_seq is not None
    n_panels = 3 if have_video else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 2:
        axes = [None] + list(axes)
    titles = ["video", "GT (top-down)", "Rollout (top-down)"]

    if have_video:
        H = frames_seq.shape[-2]; W_img = frames_seq.shape[-1]
        img0 = ((frames_seq[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        im_artist = axes[0].imshow(img0)
        axes[0].set_title(f"video {video_id} — frames")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        # video-overlay artists (only used if pixel_proj is given)
        if pixel_proj is not None:
            # GT: filled colored circle with black outline (already visible)
            video_gt_scatter = axes[0].scatter([], [], s=80, c="white",
                                               edgecolors="black", linewidths=1.8)
            # Rollout: high-contrast X — black outline draw first, white inside
            video_rl_outline = axes[0].scatter([], [], s=160, marker="X",
                                               c="black", linewidths=0)
            video_rl_scatter = axes[0].scatter([], [], s=80, marker="X", c="white",
                                               edgecolors="black", linewidths=1.5)
        else:
            video_gt_scatter = video_rl_scatter = video_rl_outline = None
    else:
        im_artist = None
        video_gt_scatter = video_rl_scatter = video_rl_outline = None

    for ax, title in zip(axes[1:], titles[1:]):
        ax.set_xlim(-world_extent, world_extent)
        ax.set_ylim(-world_extent, world_extent)
        ax.set_aspect("equal")
        ax.set_title(f"video {video_id} — {title}")
        ax.set_xlabel("world X"); ax.set_ylabel("world Y")
        ax.grid(True, alpha=0.3)

    gt_scatter = axes[1].scatter([], [], s=80, c="white", edgecolors="black")
    rl_scatter = axes[2].scatter([], [], s=80, c="white", edgecolors="black")

    # legend on GT-dots panel
    handles = []
    from matplotlib.patches import Patch
    seen_attrs = set()
    for k in range(K):
        if not visib[:, k].any():
            continue
        c = SLOT_COLORS[k % len(SLOT_COLORS)]
        lbl = f"slot {k}: {attrs_to_label(attrs[k])}"
        if lbl in seen_attrs:
            continue
        seen_attrs.add(lbl)
        handles.append(Patch(facecolor=c, edgecolor="black", label=lbl))
    axes[1].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)

    title_text = fig.suptitle(f"frame 0/{W-1}", fontsize=12)

    def _frame(t):
        gt_xy, gt_c = [], []
        rl_xy, rl_c = [], []
        for k in range(K):
            if visib[t, k] < 0.5:
                continue
            c = SLOT_COLORS[k % len(SLOT_COLORS)]
            gt_xy.append(gt_pos[t, k]); gt_c.append(c)
            rl_xy.append(rollout_pos[t, k]); rl_c.append(c)

        gt_scatter.set_offsets(np.array(gt_xy) if gt_xy else np.empty((0, 2)))
        if gt_xy:
            gt_scatter.set_color(gt_c); gt_scatter.set_sizes([120] * len(gt_xy))
        rl_scatter.set_offsets(np.array(rl_xy) if rl_xy else np.empty((0, 2)))
        if rl_xy:
            rl_scatter.set_color(rl_c); rl_scatter.set_sizes([120] * len(rl_xy))

        if im_artist is not None:
            img = ((frames_seq[t] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            im_artist.set_data(img)

        if video_gt_scatter is not None and pixel_proj is not None and gt_xy:
            gt_px = pixel_proj(np.array(gt_xy))
            rl_px = pixel_proj(np.array(rl_xy))
            video_gt_scatter.set_offsets(gt_px)
            video_gt_scatter.set_color(gt_c)
            video_rl_scatter.set_offsets(rl_px)
            video_rl_scatter.set_color(rl_c)
            if video_rl_outline is not None:
                video_rl_outline.set_offsets(rl_px)

        title_text.set_text(f"frame {t}/{W-1}")
        out = [gt_scatter, rl_scatter, title_text]
        if im_artist is not None:
            out.append(im_artist)
        return tuple(out)

    anim = FuncAnimation(fig, _frame, frames=W, interval=1000 / fps, blit=False)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer)
    except Exception as e:
        print(f"  ffmpeg writer failed ({e}); saving GIF fallback")
        anim.save(out_path.replace(".mp4", ".gif"), writer="pillow", fps=fps)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to stage1.pt or stage2.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rollout_length", type=int, default=32,
                    help="length of rollout window in frames (max 128)")
    ap.add_argument("--solver_alpha", type=float, default=0.1)
    ap.add_argument("--solver_steps", type=int, default=2)
    ap.add_argument("--num_videos", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    pos_norm = float(cfg["dataset"]["pos_normalize"])

    # build a dataset with the requested longer rollout window
    dataset = build_long_window_dataset(cfg, window_length=args.rollout_length)
    attr_dim = dataset.attr_dim
    cfg_obj, encoder, lagrangian, impulse, enc_type, dyn_type = build_modules_from_ckpt(
        ckpt, attr_dim, device,
    )
    print(f"Dynamics type: {dyn_type}")

    # one window per video, prefer start_frame == 0
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= args.num_videos:
            break
    if not seen:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= args.num_videos:
                break
    samples = list(seen.values())
    print(f"Eval videos: {[s['video_id'] for s in samples]}, "
          f"start_frames: {[s['start_frame'] for s in samples]}")

    batch = paired_collate(samples)
    frames = batch["frames"].to(device)
    gt_pos = batch["positions"].to(device) / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device)
    collisions = batch["collisions"]
    N, W = frames.shape[:2]
    print(f"Rollout length: {W}, num videos: {N}")
    print(f"GT collisions per video: {[len(c) for c in collisions]}")

    with torch.no_grad():
        q_enc, z_static_per_frame = encode_window(encoder, frames, attrs)
        from scripts.train_slot import pool_z_static
        z_static_video = pool_z_static(z_static_per_frame, visib)

    rollout_q = autoregressive_rollout(
        lagrangian, impulse, gt_pos, visib, z_static_video, collisions,
        args.solver_alpha, args.solver_steps,
        dynamics_type=dyn_type,
    )

    # ----- aggregate metrics -----
    if W > 2:
        diff = (rollout_q[:, 2:] - gt_pos[:, 2:]).pow(2).sum(dim=-1) * visib[:, 2:]
        denom = visib[:, 2:].sum().clamp_min(1.0)
        rollout_state_mse = (diff.sum() / denom).item()
        print(f"Rollout state MSE (t≥2, normalized): {rollout_state_mse:.6f}")
        print(f"Rollout state MSE (t≥2, world):      {rollout_state_mse * pos_norm**2:.6f}")
        # also encoder MSE
        e_diff = (q_enc - gt_pos).pow(2).sum(dim=-1) * visib
        e_denom = visib.sum().clamp_min(1.0)
        enc_mse = (e_diff.sum() / e_denom).item()
        print(f"Encoder state MSE (normalized):      {enc_mse:.6f}")
        print(f"Encoder state MSE (world):           {enc_mse * pos_norm**2:.6f}")

        # per-frame rollout error to plot decay
        per_t = []
        for t in range(W):
            d = (rollout_q[:, t] - gt_pos[:, t]).pow(2).sum(dim=-1) * visib[:, t]
            denom_t = visib[:, t].sum().clamp_min(1.0)
            per_t.append((d.sum() / denom_t).item() * pos_norm**2)
        # decay plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(per_t, "-o", lw=1.5, ms=4)
        ax.set_xlabel("frame index")
        ax.set_ylabel("state MSE (world units²)")
        ax.set_title(f"Rollout error decay over {W} frames "
                     f"(avg t≥2 = {rollout_state_mse * pos_norm**2:.3e})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "rollout_error_decay.png", dpi=140)
        plt.close(fig)

    # ----- per-video plots & animations -----
    gt_np = (gt_pos * pos_norm).cpu().numpy()       # (N, W, K, 2) world units
    roll_np = (rollout_q * pos_norm).cpu().numpy()
    enc_np = (q_enc * pos_norm).cpu().numpy()
    visib_np = visib.cpu().numpy()
    attrs_np = attrs.cpu().numpy()

    # determine plot extent dynamically (per video)
    for i, s in enumerate(samples):
        vid = s["video_id"]
        m = visib_np[i] > 0.5
        if not m.any():
            continue
        # extent from GT visible positions
        finite_pos = gt_np[i][m]
        ext = max(2.0, float(np.abs(finite_pos).max()) * 1.1)

        traj_path = out_dir / f"trajectory_video_{vid}.png"
        trajectory_plot(gt_np[i], roll_np[i], enc_np[i],
                        visib_np[i], attrs_np[i], vid, traj_path,
                        world_extent=ext)
        print(f"  trajectory plot: {traj_path}")

        anim_path = out_dir / f"scene_video_{vid}.mp4"
        H = frames.shape[-2]; W_img = frames.shape[-1]
        proj = make_clevrer_pixel_proj((W_img, H))
        scene_animation(gt_np[i], roll_np[i], enc_np[i],
                        visib_np[i], attrs_np[i], vid, str(anim_path),
                        world_extent=ext, fps=8,
                        frames_seq=frames[i], pixel_proj=proj)
        print(f"  animation:       {anim_path}")

    print(f"\nAll outputs saved under {out_dir}/")


if __name__ == "__main__":
    main()
