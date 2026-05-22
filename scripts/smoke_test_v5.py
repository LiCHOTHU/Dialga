"""v5.1 smoke test — runs ~50 training steps across stages 1 and 2 and asserts:

  (1) L_recon decreases between step 0 and the end.
  (2) z_static_std > 0.1 by the end of stage 2 (no collapse).
  (3) L_pred / L_recon < 5 by the end (rollout is hard but not catastrophic).
  (4) L_fwd is finite and decreases between mid-stage-2 and end.

If any assertion fails, the script exits with code 1 — do NOT proceed to the
100-vid run.

Defaults to synthetic data (works without the W=33 cache). Pass
`--cache_dir <path>` once the cache lands to smoke against 5 real videos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D
from scripts.train_v5 import compute_losses, stage_at_epoch


def synth_batches(n: int, B: int = 4, T: int = 9, n_videos: int = 5):
    """Synthetic batches with deterministic per-video z_static signal so
    InfoNCE has *something* to latch onto and z_static_std can rise."""
    rng = torch.Generator().manual_seed(0)
    # Fixed per-video bias added to chunks — makes chunks of the same video
    # share a small signal, so InfoNCE can do non-trivial work.
    vid_signals = torch.randn(n_videos, 48, 1, 1, 1, generator=rng) * 0.3
    out = []
    for _ in range(n):
        vid_ids = torch.randint(0, n_videos, (B,), generator=rng)
        chunk_obs   = torch.randn(B, 48, T, 8, 8, generator=rng) + vid_signals[vid_ids]
        chunk_pred  = torch.randn(B, 48, T, 8, 8, generator=rng) + vid_signals[vid_ids]
        chunk_obs_b = torch.randn(B, 48, T, 8, 8, generator=rng) + vid_signals[vid_ids]
        gate_GT     = (torch.rand(B, generator=rng) < 0.4).float()
        out.append({"chunk_obs": chunk_obs, "chunk_pred": chunk_pred,
                    "chunk_obs_b": chunk_obs_b, "gate_GT": gate_GT})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="If set, load 5 real videos from this cache; else synthetic.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--stage1_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    # ---- data ----
    if args.cache_dir is not None:
        ds = ClevrerChunkPairs(args.cache_dir, max_videos=5, seed=42)
        print(f"[smoke] real cache: {len(ds)} pairs (5 videos)")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=chunk_collate, drop_last=False)
        # iterate enough times
        batches = []
        it = iter(loader)
        for _ in range(args.steps):
            try:
                batches.append(next(it))
            except StopIteration:
                it = iter(loader)
                batches.append(next(it))
    else:
        batches = synth_batches(args.steps, B=args.batch_size)
        print(f"[smoke] synthetic data: {len(batches)} batches, B={args.batch_size}")

    # ---- models ----
    enc = LatentEncoder3D(d_static=32, d_dyn=16, hidden_ch=32).to(device)
    dec = LatentDecoder(d_static=32, d_dyn=16, hidden_ch=64, chunk_size_lat=9).to(device)
    fwd = ForwardDynamics(d_dyn=16, d_state=8).to(device)
    eh = EventHead(d_dyn=16, d_event=4).to(device)
    ge = GEvent(d_event=4, d_dyn=16).to(device)
    gp = GatePredictor(d_dyn=16).to(device)
    models = (enc, dec, fwd, eh, ge, gp)
    for m in models: m.train()

    cfg = SimpleNamespace(
        d_dyn=16, lambda_pred=1.0, lambda_fwd=0.1, lambda_consist=1.0,
        lambda_event_aux=0.1, lambda_gate=0.1, infonce_temperature=0.1,
    )

    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)

    # ---- run ----
    history = []
    print(f"\n{'step':>4s} {'stage':>5s} {'recon':>8s} {'pred':>8s} {'fwd':>8s} "
          f"{'consist':>8s} {'event_aux':>9s} {'gate':>6s} | "
          f"{'z_s_std':>7s} {'z_d_norm':>8s} {'z_d_roll':>8s} {'z_ev':>6s}")
    for step, batch in enumerate(batches):
        stage = 1 if step < args.stage1_steps else 2  # we test stages 1+2 only; 3 needs full setup
        losses = compute_losses(batch, models, cfg, stage, device)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        rec = {k: float(v.detach()) for k, v in losses.items()}
        rec["stage"] = stage
        rec["step"] = step
        history.append(rec)
        if step % 5 == 0 or step == len(batches) - 1:
            print(f"{step:>4d} {stage:>5d} "
                  f"{rec['recon']:>8.4f} {rec['pred']:>8.4f} {rec['fwd']:>8.4f} "
                  f"{rec['consist']:>8.3f} {rec['event_aux']:>9.4f} {rec['gate']:>6.3f} | "
                  f"{rec['z_static_std']:>7.3f} {rec['z_dyn_obs_norm']:>8.3f} "
                  f"{rec['z_dyn_roll_norm']:>8.3f} {rec['z_event_norm']:>6.3f}")

    # ---- assertions ----
    print("\n========== Assertions ==========")
    failures = []

    rec0 = history[0]["recon"]
    recN = history[-1]["recon"]
    print(f"(1) L_recon  step 0 = {rec0:.4f}  ->  step {len(history)-1} = {recN:.4f}  "
          f"{'PASS' if recN < rec0 else 'FAIL'}")
    if recN >= rec0: failures.append("L_recon did not decrease")

    z_std_end = history[-1]["z_static_std"]
    print(f"(2) z_static_std at end = {z_std_end:.3f}  (threshold > 0.1)  "
          f"{'PASS' if z_std_end > 0.1 else 'FAIL'}")
    if z_std_end <= 0.1: failures.append(f"z_static_std collapsed to {z_std_end:.3f}")

    pred_end = history[-1]["pred"]
    ratio = pred_end / max(recN, 1e-9)
    print(f"(3) L_pred/L_recon at end = {pred_end:.4f}/{recN:.4f} = {ratio:.2f}  "
          f"(threshold < 5)  {'PASS' if ratio < 5 else 'FAIL'}")
    if ratio >= 5: failures.append(f"L_pred/L_recon = {ratio:.2f} >= 5")

    # L_fwd: finite & decreased between mid-stage-2 and end
    mid_idx = (args.stage1_steps + len(history)) // 2
    fwd_mid = history[mid_idx]["fwd"] if mid_idx < len(history) else None
    fwd_end = history[-1]["fwd"]
    fwd_finite = (fwd_end == fwd_end) and (fwd_end != float("inf"))
    fwd_dec = (fwd_mid is not None) and (fwd_end < fwd_mid)
    print(f"(4) L_fwd  mid = {fwd_mid:.4f}  ->  end = {fwd_end:.4f}  "
          f"finite={fwd_finite}  decreased={fwd_dec}  "
          f"{'PASS' if fwd_finite and fwd_dec else 'FAIL'}")
    if not fwd_finite: failures.append(f"L_fwd not finite ({fwd_end})")
    if not fwd_dec: failures.append(f"L_fwd did not decrease in stage 2 ({fwd_mid:.4f} -> {fwd_end:.4f})")

    if failures:
        print(f"\n[smoke] FAILED ({len(failures)} issues):")
        for f in failures: print(f"  - {f}")
        sys.exit(1)
    print("\n[smoke] ALL ASSERTIONS PASSED. Proceed to 100-vid run.")


if __name__ == "__main__":
    main()
