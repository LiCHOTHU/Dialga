"""Train DIALGA's factorization over WHOLE videos, with a persistent static memory.

Single-variable comparison: every arm shares trunks, heads, decoder, losses, data
and schedule; only --mem_mode changes (none | median | ema | gru | attn).

Evaluated on the four things this direction is supposed to fix:
  recon      val latent MSE, reported per chunk index (does late-video decay?)
  drift      how much the static code moves between chunks of ONE video
             (the measured problem: 0.083 / 0.154 / 0.187 at lag 1/2/3 on raw latents)
  semantics  linear probe, attribute presence from the static code (mAP)
  split      the intuition test -- attributes of STATIONARY vs MOVING objects,
             read from z_static vs z_dyn. A static SCENE memory should hold the
             stationary objects and push the movers into z_dyn.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_sequence import ClevrerSequence
from src.loss.info_nce import info_nce
from src.loss.vicreg import cross_decorr, vicreg_var_cov
from src.model.base_delta_decoder import BaseDeltaDecoder
from src.model.camera_pose import synthetic_pan_sequence
from src.model.latent_decoder import SpatialGridDecoder
from src.model.memory_encoder import MemoryEncoder

N_COLOR, N_MATERIAL, N_SHAPE = 8, 2, 3


def prepare(batch, args, device, gen=None):
    """-> (seq, pose). With --synth_pan one continuous camera pans across the whole
    video, so each chunk observes a different window of the scene and a cross-chunk
    memory has something real to accumulate. The warped latent is BOTH input and
    reconstruction target: the model must explain what the camera actually saw."""
    seq = batch["latents"].to(device, non_blocking=True)
    pose = None
    if args.synth_pan:
        seq, pose = synthetic_pan_sequence(seq, generator=gen)
        if args.d_pose <= 0:
            pose = None                      # pan the camera but WITHHOLD the pose
    return seq, pose


# --------------------------------------------------------------------- losses
def losses(enc, dec, seq, pose, args):
    B, K = seq.shape[:2]
    grids, zdyn, _ = enc(seq, pose)                       # (B,K,c,g,g), (B,K,T,d_dyn)
    flat = seq.reshape(B * K, *seq.shape[2:])
    recon = dec(grids.reshape(B * K, *grids.shape[2:]),
                zdyn.reshape(B * K, *zdyn.shape[2:]))
    L_recon = F.mse_loss(recon, flat)

    zs = grids.flatten(2)                            # (B,K,d_static)
    zd_pool = zdyn.mean(dim=2)                       # (B,K,d_dyn)
    zs_f = zs.reshape(B * K, -1)
    L_indep = cross_decorr(zs_f, zd_pool.reshape(B * K, -1))
    v1, c1 = vicreg_var_cov(zs_f)
    v2, c2 = vicreg_var_cov(zdyn.reshape(-1, zdyn.shape[-1]))
    # InfoNCE across two chunks of the same video (the existing consistency term);
    # with a memory it is near-trivially satisfied, which is the point.
    L_nce = info_nce(zs[:, 0], zs[:, -1], temperature=0.1) if B > 1 else zs.sum() * 0

    total = (L_recon
             + args.lambda_indep * L_indep
             + args.lambda_consist * L_nce
             + 0.05 * (v1 + v2) + 0.05 * (c1 + c2))
    return total, {"recon": float(L_recon), "indep": float(L_indep),
                   "nce": float(L_nce), "total": float(total)}


# ----------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(enc, dec, loader, device, args):
    enc.eval(); dec.eval()
    per_chunk, drift, retain, abl = {}, {}, {}, {}
    ZS, ZD, ATT, MASK, SPD = [], [], [], [], []
    gen = torch.Generator(device=device).manual_seed(1234)   # same views every eval
    for b in loader:
        seq, pose = prepare(b, args, device, gen)
        B, K = seq.shape[:2]
        grids, zdyn, _ = enc(seq, pose)
        for k in range(K):
            r = dec(grids[:, k], zdyn[:, k])
            per_chunk.setdefault(k, []).append(float(F.mse_loss(r, seq[:, k])))
            # RETENTION: decode chunk k from the FINAL memory (after the whole
            # video). A real static-scene memory should still explain early
            # chunks; a per-chunk code should degrade the further back you go.
            r2 = dec(grids[:, -1], zdyn[:, k])
            retain.setdefault(k, []).append(float(F.mse_loss(r2, seq[:, k])))
        # CONTRIBUTION: how much does each code actually matter to the decoder?
        # zero one slot, and swap z_static between different VIDEOS (roll the
        # batch). If zeroing/rolling z_static barely moves recon, the static code
        # is decorative and every other static metric is measuring noise.
        k = 0
        abl.setdefault("full", []).append(
            float(F.mse_loss(dec(grids[:, k], zdyn[:, k]), seq[:, k])))
        abl.setdefault("no_static", []).append(
            float(F.mse_loss(dec(torch.zeros_like(grids[:, k]), zdyn[:, k]), seq[:, k])))
        abl.setdefault("no_dyn", []).append(
            float(F.mse_loss(dec(grids[:, k], torch.zeros_like(zdyn[:, k])), seq[:, k])))
        abl.setdefault("static_from_other_video", []).append(
            float(F.mse_loss(dec(grids[:, k].roll(1, 0), zdyn[:, k]), seq[:, k])))
        zs = grids.flatten(2)                                   # (B,K,d_static)
        base = zs[:, 0]
        for k in range(1, K):
            d = ((zs[:, k] - base) ** 2).sum(-1) / (base ** 2).sum(-1).clamp_min(1e-8)
            drift.setdefault(k, []).append(float(d.mean()))
        ZS.append(zs[:, -1].cpu())            # final memory = the video's static code
        ZD.append(zdyn.mean(dim=(1, 2)).cpu())
        ATT.append(b["attrs"]); MASK.append(b["slot_mask"]); SPD.append(b["speeds"])
    enc.train(); dec.train()
    return (
        {f"chunk{k}": float(np.mean(v)) for k, v in sorted(per_chunk.items())},
        {f"lag{k}": float(np.mean(v)) for k, v in sorted(drift.items())},
        {f"chunk{k}": float(np.mean(v)) for k, v in sorted(retain.items())},
        {k: float(np.mean(v)) for k, v in abl.items()},
        torch.cat(ZS).numpy(), torch.cat(ZD).numpy(),
        torch.cat(ATT).numpy(), torch.cat(MASK).numpy(), torch.cat(SPD).numpy(),
    )


def _presence(attrs, mask, sel=None):
    """Multi-hot: which attribute classes appear among the selected objects."""
    keep = mask if sel is None else (mask & sel)
    out = (attrs * keep[..., None]).max(axis=1)          # (N, A)
    return (out > 0.5).astype(np.float32), keep.sum(axis=1)


def probe(X, Y, valid, n_splits=2):
    """Linear probe -> mean average precision over classes with both labels present."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    X, Y = X[valid], Y[valid]
    if len(X) < 40:
        return float("nan")
    n = len(X) // 2
    Xtr, Xte, Ytr, Yte = X[:n], X[n:], Y[:n], Y[n:]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    aps = []
    for c in range(Y.shape[1]):
        if len(np.unique(Ytr[:, c])) < 2 or len(np.unique(Yte[:, c])) < 2:
            continue
        m = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, Ytr[:, c])
        aps.append(average_precision_score(Yte[:, c], m.predict_proba(Xte)[:, 1]))
    return float(np.mean(aps)) if aps else float("nan")


def semantic_report(ZS, ZD, ATT, MASK, SPD):
    """Overall attribute mAP, plus the stationary/moving split."""
    out = {}
    allp, n = _presence(ATT, MASK)
    ok = n > 0
    out["static_code_mAP"] = probe(ZS, allp, ok)
    out["dyn_code_mAP"] = probe(ZD, allp, ok)

    # split objects by ground-truth speed (within-dataset percentiles)
    sp = SPD[MASK]
    lo, hi = np.percentile(sp, 40), np.percentile(sp, 60)
    still_sel, move_sel = (SPD <= lo) & MASK, (SPD >= hi) & MASK
    out["speed_thresholds"] = {"still<=": float(lo), "moving>=": float(hi)}
    for name, sel in (("stationary", still_sel), ("moving", move_sel)):
        p, cnt = _presence(ATT, MASK, sel)
        ok = cnt > 0
        out[f"{name}_from_static"] = probe(ZS, p, ok)
        out[f"{name}_from_dyn"] = probe(ZD, p, ok)
        out[f"{name}_n_videos"] = int(ok.sum())
    return out


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mem_update", default="none")
    ap.add_argument("--mem_collapse", default="mean")
    ap.add_argument("--synth_pan", action="store_true",
                    help="pan one continuous camera across the whole video")
    ap.add_argument("--d_pose", type=int, default=0)
    ap.add_argument("--attn_gate_bias", type=float, default=-2.0)
    ap.add_argument("--decoder", default="grid", choices=["grid", "basedelta"])
    ap.add_argument("--zero_mean_dyn", action="store_true",
                    help="project z_dyn onto the zero-temporal-mean subspace")
    ap.add_argument("--n_chunks", type=int, default=4)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_static", type=int, default=96)
    ap.add_argument("--static_grid", type=int, default=4)
    ap.add_argument("--d_dyn", type=int, default=256)
    ap.add_argument("--dyn_grid", type=int, default=8)
    ap.add_argument("--enc_hidden_ch", type=int, default=192)
    ap.add_argument("--dec_hidden_ch", type=int, default=384)
    ap.add_argument("--lambda_indep", type=float, default=1.0)
    ap.add_argument("--lambda_consist", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)

    tr = ClevrerSequence(args.cache_dir, args.n_chunks, args.max_videos, "train",
                         preload=args.preload)
    va = ClevrerSequence(args.cache_dir, args.n_chunks, 0, "val",
                         preload=args.preload)
    print(f"[data] train {len(tr)} videos | val {len(va)} videos "
          f"| {args.n_chunks} chunks each", flush=True)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=0 if args.preload else args.num_workers,
                    pin_memory=True,
                    persistent_workers=not args.preload and args.num_workers > 0)
    dlv = DataLoader(va, batch_size=args.batch_size,
                     num_workers=0 if args.preload else 2)

    enc = MemoryEncoder(hidden_ch=args.enc_hidden_ch, d_static=args.d_static,
                        static_grid=args.static_grid, d_dyn=args.d_dyn,
                        dyn_grid=args.dyn_grid, mem_update=args.mem_update,
                        mem_collapse=args.mem_collapse, d_pose=args.d_pose,
                        zero_mean_dyn=args.zero_mean_dyn,
                        attn_gate_bias=args.attn_gate_bias).to(dev)
    if args.decoder == "basedelta":
        dec = BaseDeltaDecoder(d_static=args.d_static, static_grid=args.static_grid,
                               d_dyn=args.d_dyn, dyn_grid=args.dyn_grid,
                               hidden_ch=args.dec_hidden_ch).to(dev)
    else:
        dec = SpatialGridDecoder(d_static=args.d_static, static_grid=args.static_grid,
                                 d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                 dyn_spatial=True, dyn_grid=args.dyn_grid).to(dev)
    npar = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"[model] upd={args.mem_update} col={args.mem_collapse} pan={args.synth_pan} dec={args.decoder} "
      f"zmd={args.zero_mean_dyn} params {npar/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-3)

    ck, ep0, hist = out / "ckpt.pt", 0, []
    if ck.exists():                                    # resume (restart wrapper)
        s = torch.load(ck, map_location="cpu", weights_only=False)
        enc.load_state_dict(s["enc"]); dec.load_state_dict(s["dec"])
        opt.load_state_dict(s["opt"]); ep0 = s["epoch"] + 1; hist = s.get("hist", [])
        print(f"[resume] from epoch {ep0}", flush=True)

    for ep in range(ep0, args.epochs):
        t0, agg, nb = time.time(), {}, 0
        for b in dl:
            seq, pose = prepare(b, args, dev)
            total, log = losses(enc, dec, seq, pose, args)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
            for k, v in log.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        agg = {k: v / max(1, nb) for k, v in agg.items()}
        row = {"epoch": ep, "sec": round(time.time() - t0, 1), **agg}

        if ep % 5 == 4 or ep == args.epochs - 1:
            pc, dr, rt, ab, ZS, ZD, ATT, MASK, SPD = evaluate(enc, dec, dlv, dev, args)
            row["val_recon"] = pc
            row["drift"] = dr
            row["retention"] = rt
            row["ablation"] = ab
            row["semantics"] = semantic_report(ZS, ZD, ATT, MASK, SPD)
        hist.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                    "opt": opt.state_dict(), "epoch": ep, "hist": hist,
                    "args": vars(args)}, ck)
        (out / "history.json").write_text(json.dumps(hist, indent=2))

    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
