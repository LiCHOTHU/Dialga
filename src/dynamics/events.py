"""
Inertial-baseline event extractor.

Discrete events extracted from per-slot position trajectories by detecting
deviations from inertial (constant-velocity) motion. A collision is, by
definition, a violation of Newton's first law — sudden non-zero acceleration —
so the per-slot second difference

    a[t, k] = q[t+1, k] - 2*q[t, k] + q[t-1, k]

is the natural event signal. Unlike a residual against the *trained* dynamics,
this signal does NOT vanish as the dynamics fits, because the inertial
baseline is fixed by definition.

Threshold: per-video robust z-score (median + z * 1.4826 * MAD) on the visible
accelerations, so the threshold adapts to per-video noise without global
tuning.

Spatial filter (optional): a high-acceleration slot must have at least one
*other* visible slot within `contact_distance`, ruling out single-slot noise
spikes (e.g., visibility-edge transients on an off-camera object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch


@dataclass
class Event:
    t: int                       # frame index (caller chooses local vs. global)
    participants: List[int]      # slot indices flagged as event participants
    magnitude: float             # peak acceleration magnitude across participants

    def as_tuple(self) -> Tuple[int, Tuple[int, ...], float]:
        return (self.t, tuple(sorted(self.participants)), self.magnitude)


def _robust_threshold(values: torch.Tensor, mask: torch.Tensor, z_threshold: float,
                       abs_floor: float = 0.0, eps: float = 1e-6) -> float:
    """median + z * 1.4826 * MAD on visible, *non-zero* accelerations, lower-
    bounded by abs_floor.

    Most CLEVRER slot trajectories are at rest in world frame (slot exists but
    object isn't moving). Their accelerations are exactly zero, which would
    pull median and MAD to zero and collapse the threshold. We restrict the
    statistic to slots that actually moved (|a| > eps); if none did, we fall
    back to abs_floor.
    """
    sel = values[mask & (values > eps)]
    if sel.numel() == 0:
        return abs_floor
    med = sel.median()
    mad = (sel - med).abs().median()
    thr = float(med + z_threshold * 1.4826 * mad)
    return max(thr, abs_floor)


def extract_inertial_events_single(
    q: torch.Tensor,                # (W, K, D)
    alpha: torch.Tensor,            # (W, K), bool or float
    *,
    z_threshold: float = 3.0,
    contact_distance: float = 0.8,  # world units; CLEVRER spheres have radius ~0.4
    require_neighbor: bool = True,
    min_temporal_extent: int = 1,
    nms_window: int = 3,            # suppress repeated flags within this many frames per slot
    min_participants: int = 2,      # by Newton's 3rd law, collisions involve >=2 slots
    abs_floor: float = 1e-3,        # absolute lower bound on threshold (world units)
    smooth_kernel: int = 1,         # triangular smoothing of q before 2nd diff (odd; 1 = off)
) -> List[Event]:
    """Extract events for one video.

    Args:
        q:                 (W, K, D) position trajectory.
        alpha:             (W, K) visibility (bool or float, threshold at 0.5).
        z_threshold:       MAD-based z-score threshold on the per-video accel
                           magnitude distribution.
        contact_distance:  Spatial-filter radius. A flagged slot must have
                           another visible slot within this distance to be
                           kept (in the same units as q).
        require_neighbor:  If False, skip the spatial filter (use the raw
                           magnitude threshold only).
        min_temporal_extent: Require a slot to be flagged for this many
                           consecutive frames; 1 disables the filter.
        nms_window:        Per-slot NMS — once a slot is flagged at frame t,
                           suppress further flags on the same slot for
                           [t+1, t+nms_window). Prevents one collision being
                           counted as several events on the same slot.
        min_participants:  Drop events whose participant count is below this.
                           Default 2 enforces Newton's 3rd law (real pairwise
                           collisions show as simultaneous high acceleration
                           on both slots). Set to 1 to allow single-slot
                           events (e.g., entry/exit, wall impacts).

    Returns:
        list[Event] sorted by t. Each event groups all slots flagged at the
        same central frame.
    """
    W, K, D = q.shape
    if W < 3:
        return []
    device = q.device
    alpha_f = alpha.float()

    # 0) optional temporal smoothing on q before second difference. High-
    #    frequency encoder jitter inflates the 2nd-diff noise floor; smoothing
    #    with a small triangular kernel suppresses noise while preserving
    #    collision kinks (which span 2-3 frames in CLEVRER).
    if smooth_kernel and smooth_kernel > 1:
        ks = int(smooth_kernel)
        if ks % 2 == 0:
            ks += 1
        half = ks // 2
        weights = torch.tensor([min(i + 1, ks - i) for i in range(ks)],
                               dtype=q.dtype, device=device)
        weights = weights / weights.sum()
        # (W, K, D) -> (K*D, 1, W) for conv1d
        q_t = q.permute(1, 2, 0).reshape(K * D, 1, W)
        q_pad = torch.nn.functional.pad(q_t, (half, half), mode="replicate")
        q_smooth = torch.nn.functional.conv1d(q_pad, weights.view(1, 1, ks))
        q = q_smooth.reshape(K, D, W).permute(2, 0, 1).contiguous()

    # 1) acceleration magnitude (second difference). accel[s] is the central
    #    frame t = s + 1.
    accel = q[2:] - 2 * q[1:-1] + q[:-2]                                  # (W-2, K, D)
    accel_mag = accel.norm(dim=-1)                                        # (W-2, K)

    # 2) validity: need all three frames in the triple to be visible
    valid = (alpha_f[1:-1] > 0.5) & (alpha_f[:-2] > 0.5) & (alpha_f[2:] > 0.5)

    # 3) per-video robust threshold (restricted to non-zero accels; floored)
    thr = _robust_threshold(accel_mag, valid, z_threshold, abs_floor=abs_floor)
    flagged = (accel_mag > thr) & valid                                   # (W-2, K)

    # 4) temporal-extent filter
    if min_temporal_extent > 1:
        keep = flagged.clone()
        for shift in range(1, min_temporal_extent):
            keep[:-shift] = keep[:-shift] & flagged[shift:]
            keep[-shift:] = False
        flagged = keep

    # 5) spatial coincidence: another visible slot within contact_distance
    if require_neighbor:
        q_mid = q[1:-1]                                                   # (W-2, K, D)
        d_mat = (q_mid.unsqueeze(2) - q_mid.unsqueeze(1)).norm(dim=-1)    # (W-2, K, K)
        eye = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)
        alpha_mid = alpha_f[1:-1] > 0.5                                   # (W-2, K)
        valid_pair = alpha_mid.unsqueeze(2) & alpha_mid.unsqueeze(1) & ~eye
        has_neighbor = ((d_mat < contact_distance) & valid_pair).any(dim=-1)  # (W-2, K)
        flagged = flagged & has_neighbor

    # 6) per-slot NMS to suppress repeated flags around a single event
    if nms_window > 1:
        flagged_np = flagged.clone()
        S = flagged.shape[0]
        for k in range(K):
            s = 0
            while s < S:
                if flagged_np[s, k]:
                    end = min(s + nms_window, S)
                    flagged_np[s + 1:end, k] = False
                    s = end
                else:
                    s += 1
        flagged = flagged_np

    # 7) merge co-occurring slots into events at the same central frame
    raw: List[Event] = []
    for s in range(flagged.shape[0]):
        slots = flagged[s].nonzero(as_tuple=False).flatten().tolist()
        if len(slots) < min_participants:
            continue
        mag = float(accel_mag[s, slots].max().item())
        raw.append(Event(t=s + 1, participants=slots, magnitude=mag))

    # 8) event-level NMS: suppress weaker events within `event_nms_window` of
    #    a stronger one. Handles the "post-collision oscillation" failure mode
    #    where the second derivative spikes once at the collision and again a
    #    couple of frames later as the post-collision velocity settles.
    if not raw:
        return raw
    by_mag = sorted(range(len(raw)), key=lambda i: -raw[i].magnitude)
    keep = [True] * len(raw)
    for i in by_mag:
        if not keep[i]:
            continue
        for j in range(len(raw)):
            if i == j or not keep[j]:
                continue
            if abs(raw[j].t - raw[i].t) <= nms_window and raw[j].magnitude < raw[i].magnitude:
                # check overlap in participants — suppress only if same physical event
                overlap = set(raw[i].participants) & set(raw[j].participants)
                if overlap:
                    keep[j] = False
    events = [raw[i] for i in range(len(raw)) if keep[i]]
    events.sort(key=lambda e: e.t)
    return events


def extract_inertial_events_batch(
    q: torch.Tensor,                # (B, W, K, D)
    alpha: torch.Tensor,            # (B, W, K)
    **kwargs,
) -> List[List[Event]]:
    """Per-video event extraction. Returns list[B] of list[Event]."""
    return [
        extract_inertial_events_single(q[b], alpha[b], **kwargs)
        for b in range(q.shape[0])
    ]


# ---------------------------------------------------------------- comparison


def compare_events_to_gt(
    extracted: Sequence[Event],
    gt_events: Sequence[Tuple[int, int, int]],     # (t, i, j) per CLEVRER convention
    *,
    time_tolerance: int = 2,
    require_pair_overlap: bool = True,
) -> dict:
    """Compare extracted events to CLEVRER GT collision events.

    A GT event (t_gt, i, j) is matched if there exists an extracted event with
        |t - t_gt| <= time_tolerance
    AND (if require_pair_overlap) {i, j} ⊆ participants.

    Each extracted event can match at most one GT event (greedy nearest-time).

    Returns:
        dict with tp, fp, fn, precision, recall, f1, n_extracted, n_gt.
    """
    matched_extracted = [False] * len(extracted)
    matched_gt = [False] * len(gt_events)

    for k, (t_gt, i, j) in enumerate(gt_events):
        best, best_dt = None, None
        for r, ev in enumerate(extracted):
            if matched_extracted[r]:
                continue
            dt = abs(ev.t - t_gt)
            if dt > time_tolerance:
                continue
            if require_pair_overlap and (i not in ev.participants or j not in ev.participants):
                continue
            if best is None or dt < best_dt:
                best, best_dt = r, dt
        if best is not None:
            matched_extracted[best] = True
            matched_gt[k] = True

    tp = sum(matched_gt)
    fn = len(gt_events) - tp
    fp = len(extracted) - sum(matched_extracted)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_extracted=len(extracted), n_gt=len(gt_events))


def extract_velocity_jump_events_single(
    q: torch.Tensor,                 # (W, K, D)
    alpha: torch.Tensor,             # (W, K)
    *,
    window: int = 3,                 # frames on each side for median-velocity
    z_threshold: float = 3.0,
    contact_distance: float = 0.8,
    require_neighbor: bool = True,
    nms_window: int = 3,
    min_participants: int = 2,
    abs_floor: float = 5e-3,         # in world units (jump scale, not accel)
) -> List[Event]:
    """Velocity-jump event detector — more robust to encoder noise than the
    second-difference detector.

    For each slot at each frame t, compute:
        v_pre  = median over [t - window, ..., t - 1] of  (q[s+1] - q[s])
        v_post = median over [t, ..., t + window - 1]  of (q[s+1] - q[s])
        jump   = || v_post - v_pre ||

    A collision causes a large velocity jump (~ 2 * |v|), while encoder noise
    in the velocity has σ ≈ √2 σ_q. The median over `window` further reduces
    noise by ~ √window, so SNR improves vs the per-frame second difference.

    Same per-video robust threshold + spatial coincidence + min_participants
    + NMS pipeline as the second-difference extractor.
    """
    W, K, D = q.shape
    if W < 2 * window + 2:
        return []
    device = q.device
    alpha_f = alpha.float()

    v = q[1:] - q[:-1]                                                   # (W-1, K, D)
    # for each central frame t in [window, W - window], median over pre/post v windows
    valid_t_lo, valid_t_hi = window, W - window                          # central frames
    jumps = []
    for t in range(valid_t_lo, valid_t_hi):
        # v_pre indices: [t - window, ..., t - 1] inclusive  (length window)
        # v_post indices: [t, ..., t + window - 1] inclusive (length window)
        v_pre = v[t - window:t]                                          # (window, K, D)
        v_post = v[t:t + window]
        v_pre_med = v_pre.median(dim=0).values
        v_post_med = v_post.median(dim=0).values
        j = (v_post_med - v_pre_med).norm(dim=-1)                        # (K,)
        jumps.append(j)
    jump_mag = torch.stack(jumps, dim=0)                                  # (T_central, K)

    # validity: require all frames in [t-window, t+window] visible per slot
    central_count = jump_mag.shape[0]
    valid = torch.ones(central_count, K, dtype=torch.bool, device=device)
    for t in range(valid_t_lo, valid_t_hi):
        rel = t - valid_t_lo
        for s in range(t - window, t + window + 1):
            if 0 <= s < W:
                valid[rel] = valid[rel] & (alpha_f[s] > 0.5)

    # robust threshold (skip exact zeros = at-rest slots)
    thr = _robust_threshold(jump_mag, valid, z_threshold, abs_floor=abs_floor)
    flagged = (jump_mag > thr) & valid

    # spatial filter
    if require_neighbor:
        for rel in range(central_count):
            t = rel + valid_t_lo
            if not flagged[rel].any():
                continue
            q_at_t = q[t]                                                 # (K, D)
            d_mat = (q_at_t.unsqueeze(1) - q_at_t.unsqueeze(0)).norm(dim=-1)
            eye = torch.eye(K, dtype=torch.bool, device=device)
            alpha_t = (alpha_f[t] > 0.5)
            valid_pair = alpha_t.unsqueeze(1) & alpha_t.unsqueeze(0) & ~eye
            has_neighbor = ((d_mat < contact_distance) & valid_pair).any(dim=-1)
            flagged[rel] = flagged[rel] & has_neighbor

    # NMS per slot
    if nms_window > 1:
        for k in range(K):
            s = 0
            while s < central_count:
                if flagged[s, k]:
                    flagged[s + 1:min(s + nms_window, central_count), k] = False
                    s += nms_window
                else:
                    s += 1

    # build events
    raw: List[Event] = []
    for rel in range(central_count):
        slots = flagged[rel].nonzero(as_tuple=False).flatten().tolist()
        if len(slots) < min_participants:
            continue
        mag = float(jump_mag[rel, slots].max().item())
        raw.append(Event(t=rel + valid_t_lo, participants=slots, magnitude=mag))

    if not raw:
        return raw
    by_mag = sorted(range(len(raw)), key=lambda i: -raw[i].magnitude)
    keep = [True] * len(raw)
    for i in by_mag:
        if not keep[i]:
            continue
        for j in range(len(raw)):
            if i == j or not keep[j]:
                continue
            if abs(raw[j].t - raw[i].t) <= nms_window and raw[j].magnitude < raw[i].magnitude:
                if set(raw[i].participants) & set(raw[j].participants):
                    keep[j] = False
    events = [raw[i] for i in range(len(raw)) if keep[i]]
    events.sort(key=lambda e: e.t)
    return events


__all__ = [
    "Event",
    "extract_inertial_events_single",
    "extract_inertial_events_batch",
    "extract_velocity_jump_events_single",
    "compare_events_to_gt",
]
