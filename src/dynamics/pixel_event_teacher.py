"""Pixel-space prediction-error event teachers.

Two variants live here:

  pixel_event_soft_from_frames        (operator = Δ²I = per-pixel 2nd-diff)
      - First attempt at the cogsci-correct teacher. Math shows it is the
        WRONG operator: for constant-velocity smooth motion,
            Δ²I ≈ v² · obj''      (Laplacian of object shape)
        is generally non-zero — proportional to v² at object edges. At a
        perfect elastic collision,
            Δ²I ≈ -2v · obj'      (gradient of object shape)
        which is the same order of magnitude for typical CLEVRER edges and
        velocities. The "smooth motion baseline" doesn't vanish, so the
        teacher mostly detects "where there is motion", not "where motion
        is impulsive". Kept for record only.

  motion_centroid_event_soft          (operator = Δ² c, c = motion centroid)
      - Math-corrected teacher. We aggregate per-pixel motion magnitude
        M(u,v) = |I^{t+1}(u,v) - I^{t-1}(u,v)| inside each slot's Gaussian
        mask, then take the centroid:
            c_i^t = weighted-mean of (u,v) by M(u,v) · mask_i^t
        For constant-velocity motion of an object inside the slot, c_i^t
        is *linear in t* (tracks the object's pixel-space center), so
            Δ² c_i^t  ≈  0        (constant velocity = no curvature)
        At an impulsive collision frame, c_i^t has a sharp kink, so
            Δ² c_i^t  ≈  large
        This is the right shape: smooth-baseline → 0, collisions → spike.
        Encoder cannot smooth this away because M is fixed by the video.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_event_soft_from_frames(
    frames: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    *,
    attention_sigma: float = 0.15,
    z_thresh: float = 1.5,
    sharpness: float = 2.5,
    label_dilation: int = 1,
) -> torch.Tensor:
    """Compute per-slot event soft labels from frame-space 2nd-difference.

    frames : (B, T, 3, H, W)   pixel values in [-1, 1]
    q      : (B, T, K, 2)      slot positions in normalized world coords (≈ [-1, 1])
    alpha  : (B, T, K)         visibility (0/1 or soft)

    The spatial alignment between q (normalized world coords from
    pos_normalize) and the image grid [-1, 1]^2 is treated as approximate
    identity: q_x ↔ image_x, q_y ↔ image_y. This is the same convention
    that the q-relative SlotPixelDecoder uses, and it is sufficient for
    coarse per-slot attention even when the camera perspective is slightly
    off.

    Returns:
        event_soft : (B, T, K) in [0, 1], detached (no gradient flow back
                     through the teacher).
    """
    B, T, C, H, W = frames.shape
    K = q.shape[2]
    if T < 3:
        return q.new_zeros(B, T, K)

    # --- (1) Pixel-space 2nd-difference magnitude.
    # Detach: the teacher should be a *fixed* target for the head, not a
    # function the encoder can pull on through the spatial mask gradients.
    frames_d = frames.detach()
    d = frames_d[:, 2:] - 2 * frames_d[:, 1:-1] + frames_d[:, :-2]  # (B, T-2, 3, H, W)
    d_mag = d.abs().sum(dim=2)                                      # (B, T-2, H, W)

    # --- (2) Per-slot Gaussian spatial mask centered at q.
    # Grid coords in [-1, 1] matching q's normalized scale.
    device = frames.device
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)                            # (H, W, 2)

    q_mid = q[:, 1:-1].detach()                                     # (B, T-2, K, 2)
    # Distance squared from each pixel to each slot's position
    diff = grid.view(1, 1, 1, H, W, 2) - q_mid.view(B, T - 2, K, 1, 1, 2)
    dist2 = diff.pow(2).sum(dim=-1)                                 # (B, T-2, K, H, W)
    mask = torch.exp(-dist2 / (2.0 * attention_sigma ** 2))         # (B, T-2, K, H, W)

    # --- (3) Per-slot signal = average d_mag inside the slot's mask.
    slot_signal_num = (d_mag.unsqueeze(2) * mask).sum(dim=(-1, -2))  # (B, T-2, K)
    mask_norm = mask.sum(dim=(-1, -2)).clamp_min(1.0)                # (B, T-2, K)
    slot_signal = slot_signal_num / mask_norm                        # (B, T-2, K)

    # --- (4) Per-slot z-score over time (gated by visibility triple).
    gate = alpha[:, :-2] * alpha[:, 1:-1] * alpha[:, 2:]             # (B, T-2, K)
    masked = slot_signal * gate
    cnt = gate.sum(dim=1, keepdim=True).clamp_min(1.0)               # (B, 1, K)
    mu = masked.sum(dim=1, keepdim=True) / cnt
    var = ((slot_signal - mu) ** 2 * gate).sum(dim=1, keepdim=True) / cnt
    sigma = var.sqrt().clamp_min(1e-4)
    z = (slot_signal - mu) / sigma                                   # (B, T-2, K)

    event_inner = torch.sigmoid((z - z_thresh) * sharpness) * gate

    # --- (5) Pad back to T frames; boundaries are 0 (residual undefined).
    event = q.new_zeros(B, T, K)
    event[:, 1:-1] = event_inner

    # --- (6) Optional temporal dilation (matches the GT-mode dilation).
    if label_dilation > 0:
        k = 2 * int(label_dilation) + 1
        x = event.transpose(1, 2).reshape(B * K, 1, T)
        x = F.max_pool1d(x, kernel_size=k, stride=1, padding=label_dilation)
        event = x.view(B, K, T).transpose(1, 2)

    return event.detach()


def motion_centroid_event_soft(
    frames: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    *,
    attention_sigma: float = 0.20,
    z_thresh: float = 2.0,
    sharpness: float = 2.5,
    label_dilation: int = 1,
    saliency_floor: float = 1e-3,
    abs_thresh: float = 0.0,
    hard_binarize: bool = False,
    pair_augment_distance: float = 0.0,
) -> torch.Tensor:
    """Math-corrected pixel-space event teacher.

    Principle: the inertial baseline says "objects move at constant velocity."
    Its residual = 2nd-difference of *object position*. We need a way to
    measure object position from pixels alone (encoder-uncontrollable).

    Operator:
      1. Per-frame foreground SALIENCY = |I^t - per-frame mean(I^t)|. Objects
         deviate from the frame's average pixel value; background does not.
         This gives a per-pixel "object-mass" field that is non-zero exactly
         where objects are — including at collision frames (unlike pixel
         motion magnitude, which vanishes when an elastic bounce returns the
         object to the same pixels at t-1 and t+1).
      2. Per-slot saliency-weighted centroid c_i^t = E[(u,v) | saliency · mask_i].
         Tracks the object's pixel-space center for whatever object is inside
         slot i's Gaussian mask. For smooth motion → c is linear in t.
      3. Residual: Δ² c_i^t = c_i^{t+1} - 2 c_i^t + c_i^{t-1}.
         Vanishes for constant velocity, spikes at impulsive moments.
      4. Per-slot z-score over time → sigmoid → soft event label.

    The encoder controls only the mask LOCATION (via q), not the saliency
    inside the mask. With even mildly correct q, the mask captures the
    object and the centroid tracks it. Spike timing is fixed by the video.

    Args
    ----
    frames : (B, T, 3, H, W) in [-1, 1]
    q      : (B, T, K, 2) slot positions (normalized world coords ≈ [-1, 1])
    alpha  : (B, T, K) visibility

    Returns
    -------
    event_soft : (B, T, K) in [0, 1], detached
    """
    B, T, C, H, W = frames.shape
    K = q.shape[2]
    if T < 3:
        return q.new_zeros(B, T, K)

    frames_d = frames.detach()

    # --- (1) Per-frame foreground saliency.
    # Per-frame mean is dominated by background (most pixels). Pixels that
    # deviate from this mean are objects. saliency: (B, T, H, W) ≥ 0.
    fr_mean = frames_d.mean(dim=(-1, -2), keepdim=True)                # (B, T, C, 1, 1)
    saliency = (frames_d - fr_mean).abs().mean(dim=2)                  # (B, T, H, W)

    # --- (2) Spatial grid in normalized [-1, 1] coords (matches q scale).
    device = frames.device
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = gx                                                        # (H, W)
    grid_y = gy                                                        # (H, W)

    # --- (3) Per-slot Gaussian spatial mask centered at q (encoder's
    # current best guess of object location). The encoder controls *where*
    # to look, but not *what is there*.
    q_det = q.detach()
    diff_x = grid_x.view(1, 1, 1, H, W) - q_det[..., 0].view(B, T, K, 1, 1)
    diff_y = grid_y.view(1, 1, 1, H, W) - q_det[..., 1].view(B, T, K, 1, 1)
    dist2 = diff_x.pow(2) + diff_y.pow(2)                              # (B, T, K, H, W)
    mask = torch.exp(-dist2 / (2.0 * attention_sigma ** 2))            # (B, T, K, H, W)

    # --- (4) Saliency-weighted centroid per slot per frame.
    weights = saliency.unsqueeze(2) * mask                             # (B, T, K, H, W)
    W_sum = weights.sum(dim=(-1, -2))                                  # (B, T, K)
    W_safe = W_sum.clamp_min(saliency_floor)
    cx = (weights * grid_x.view(1, 1, 1, H, W)).sum(dim=(-1, -2)) / W_safe   # (B, T, K)
    cy = (weights * grid_y.view(1, 1, 1, H, W)).sum(dim=(-1, -2)) / W_safe   # (B, T, K)

    # --- (5) Temporal 2nd-difference of centroid trajectory.
    cx_a = cx[:, 2:] - 2 * cx[:, 1:-1] + cx[:, :-2]                    # (B, T-2, K)
    cy_a = cy[:, 2:] - 2 * cy[:, 1:-1] + cy[:, :-2]                    # (B, T-2, K)
    a_mag = (cx_a.pow(2) + cy_a.pow(2)).sqrt()                          # (B, T-2, K)

    # --- (6) Gates: slot visible in all 3 frames AND has nontrivial
    # saliency in the mask (otherwise centroid is noisy).
    vis_gate = alpha[:, :-2] * alpha[:, 1:-1] * alpha[:, 2:]            # (B, T-2, K)
    pres_gate = (W_sum[:, 1:-1] > saliency_floor).float()               # (B, T-2, K)
    gate = vis_gate * pres_gate

    # --- (7) Threshold a_mag to produce event labels.
    #
    # Two modes:
    #   abs_thresh > 0  : ABSOLUTE threshold on |Δ²c|. Centroid acceleration
    #                     has a clean physical scale (smooth motion ≈ 0,
    #                     collisions ≈ 0.05-0.4 in normalized [-1,1] coords).
    #                     This is the correct mode at small T (window-z-score
    #                     across T-2=4 frames is statistical noise).
    #   abs_thresh == 0 : per-window z-score, sigmoid (legacy behavior; only
    #                     valid for long windows).
    if abs_thresh > 0:
        # Soft sigmoid around the absolute threshold for graceful BCE.
        event_inner = torch.sigmoid((a_mag - abs_thresh) * sharpness) * gate
    else:
        cnt = gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        mu = (a_mag * gate).sum(dim=1, keepdim=True) / cnt
        var = ((a_mag - mu) ** 2 * gate).sum(dim=1, keepdim=True) / cnt
        sigma = var.sqrt().clamp_min(1e-4)
        z = (a_mag - mu) / sigma                                        # (B, T-2, K)
        event_inner = torch.sigmoid((z - z_thresh) * sharpness) * gate

    if hard_binarize:
        # Convert soft sigmoid to hard {0, 1} labels. With BCE this removes
        # the mid-range mass that confuses pos_weight tuning at sparse-label
        # regimes.
        event_inner = (event_inner > 0.5).float() * gate

    event = q.new_zeros(B, T, K)
    event[:, 1:-1] = event_inner

    # Pair augmentation: when a slot's centroid spikes at frame t, also label
    # nearby slots (within `pair_augment_distance` of slot's q at that frame)
    # as positive. Collisions are inherently pairwise — one partner may have
    # tiny a_mag (heavier object barely reacting) but is still a participant.
    if pair_augment_distance > 0:
        pa_d2 = pair_augment_distance ** 2
        # q is (B, T, K, 2), `event` is (B, T, K) binary-ish.
        # For each (b, t, k) positive, find neighbors j with ||q_b,t,k - q_b,t,j||^2 < pa_d2
        # and j visible. Propagate the positive label to j.
        q_det_aug = q.detach()  # (B, T, K, 2)
        dx = q_det_aug.unsqueeze(2) - q_det_aug.unsqueeze(3)            # (B, T, K, K, 2)
        d2_pair = (dx * dx).sum(dim=-1)                                  # (B, T, K, K)
        eye = torch.eye(K, dtype=torch.bool, device=q.device).view(1, 1, K, K)
        near = (d2_pair < pa_d2) & (~eye)                                # (B, T, K, K)
        # both ends visible
        vis_i = alpha.unsqueeze(3) > 0.5
        vis_j = alpha.unsqueeze(2) > 0.5
        near = near & vis_i & vis_j
        # event_src[b,t,k]=1 → for each j with near[b,t,k,j], add to event[b,t,j]
        propagated = (event.unsqueeze(-1) * near.float()).max(dim=2).values  # (B, T, K)
        event = torch.maximum(event, propagated)

    if label_dilation > 0:
        k = 2 * int(label_dilation) + 1
        x = event.transpose(1, 2).reshape(B * K, 1, T)
        x = F.max_pool1d(x, kernel_size=k, stride=1, padding=label_dilation)
        event = x.view(B, K, T).transpose(1, 2)

    return event.detach()


__all__ = ["pixel_event_soft_from_frames", "motion_centroid_event_soft"]
