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
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_sequence import ClevrerSequence
from src.data.ssv2_sequence import SSv2Sequence
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


def static_target(seq, how):
    """seq (B,K,C,T,H,W) -> (B,C,H,W) the computed 'static scene' image."""
    B, K, C, T, H, W = seq.shape
    if how == "chunk_mean":
        return seq.mean(dim=(1, 3))
    X = seq.permute(0, 2, 1, 3, 4, 5).reshape(B, C, K * T, H, W)
    return X.median(dim=2).values if how == "video_median" else X.mean(dim=2)


# --------------------------------------------------------------------- losses
def losses(enc, dec, seq, pose, args, aux=None, dino=None, dhead=None,
           dino_med=None, dino_res=None, dyhead=None):
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

    # --- complementarity: neither code may be sufficient on its own ---
    L_comp = torch.zeros((), device=seq.device)
    if args.lambda_comp > 0:
        gf = grids.reshape(B * K, *grids.shape[2:])
        zf = zdyn.reshape(B * K, *zdyn.shape[2:])
        solo_s = F.mse_loss(dec(gf, torch.zeros_like(zf)), flat)
        solo_d = F.mse_loss(dec(torch.zeros_like(gf), zf), flat)
        m = args.comp_margin
        # each solo error must exceed the joint error by a factor of (1+m)
        L_comp = (F.relu(m - (solo_s / L_recon.detach().clamp_min(1e-8) - 1.0))
                  + F.relu(m - (solo_d / L_recon.detach().clamp_min(1e-8) - 1.0)))

    L_tgt = torch.zeros((), device=seq.device)
    if aux is not None and args.lambda_static_tgt > 0:
        tgt = static_target(seq, args.static_target)                  # (B,C,H,W)
        pred = aux(grids.reshape(B * K, *grids.shape[2:]))            # (B*K,C,H,W)
        L_tgt = F.mse_loss(pred, tgt.unsqueeze(1).expand(-1, K, -1, -1, -1)
                                   .reshape(B * K, *tgt.shape[1:]))

    L_dino = torch.zeros((), device=seq.device)
    if dhead is not None and dino is not None and args.lambda_dino > 0:
        gin = grids.reshape(B * K, *grids.shape[2:])
        if args.dino_to == "both":
            zd_m = zdyn.mean(2).reshape(B * K, -1, 1, 1).expand(-1, -1, *gin.shape[-2:])
            gin = torch.cat([gin, zd_m], 1)
        pred = dhead(gin)                                   # (B*K, D, H, W)
        tgt = dino.reshape(B * K, *dino.shape[2:]).permute(0, 3, 1, 2)
        L_dino = (1.0 - F.cosine_similarity(pred, tgt, dim=1)).mean()

    # decomposed teacher: each code gets the half of the semantic signal the other
    # is not being taught
    L_dyn_teach = torch.zeros((), device=seq.device)
    if dyhead is not None and dino_res is not None and args.lambda_dyn_teach > 0:
        zf = zdyn.reshape(B * K, *zdyn.shape[2:])            # (B*K, T, d_dyn)
        pr = dyhead(zf.mean(1).unsqueeze(-1).unsqueeze(-1)
                    .expand(-1, -1, 8, 8))                   # (B*K, D, H, W)
        tg = dino_res.reshape(B * K, *dino_res.shape[2:]).permute(0, 3, 1, 2)
        L_dyn_teach = (1.0 - F.cosine_similarity(pr, tg, dim=1)).mean()

    total = (L_recon
             + args.lambda_dyn_teach * L_dyn_teach
             + args.lambda_comp * L_comp
             + args.lambda_dino * L_dino
             + args.lambda_static_tgt * L_tgt
             + args.lambda_indep * L_indep
             + args.lambda_consist * L_nce
             + 0.05 * (v1 + v2) + 0.05 * (c1 + c2))
    return total, {"recon": float(L_recon), "indep": float(L_indep),
                   "nce": float(L_nce), "stat_tgt": float(L_tgt), "dino": float(L_dino), "comp": float(L_comp), "dyn_teach": float(L_dyn_teach),
                   "total": float(total)}


# ----------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(enc, dec, loader, device, args):
    enc.eval(); dec.eval()
    per_chunk, drift, retain, abl, idem = {}, {}, {}, {}, []
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
            # COSINE drift. The rel-MSE version is not scale-invariant, and several
            # memories change the code's MAGNITUDE across chunks rather than its
            # direction -- patch memory adds a residual each step and read 14.2 / 31.3
            # on SSv2, a ConvGRU read 1.40 at lag1 and 0.005 at lag3. Those numbers
            # measured norm growth and the recurrence's own trajectory, not scene
            # consistency, so they are not comparable across arms. 1 - cos is bounded
            # in [0,2] and answers the question actually being asked: has the static
            # code turned into a different code?
            d = 1.0 - F.cosine_similarity(zs[:, k], base, dim=-1)
            drift.setdefault(k, []).append(float(d.mean()))
        if not idem:
            # IDEMPOTENCE: feed the SAME chunk K times. A content-driven memory
            # converges (cos -> 1); oscillation means the recurrence learned a
            # position-dependent trajectory and "drift" is measuring that, not the
            # scene. This is why drift alone must never be reported as consistency.
            rep = seq[:, :1].repeat(1, K, 1, 1, 1, 1)
            rp = None if pose is None else pose[:, :1].repeat(1, K, 1, 1)
            gr, _, _ = enc(rep, rp)
            zr = gr.flatten(2)
            idem.extend([float(F.cosine_similarity(zr[:, k], zr[:, 0], dim=-1).mean())
                         for k in range(K)])
        ZS.append(zs[:, -1].cpu())            # final memory = the video's static code
        ZD.append(zdyn.mean(dim=(1, 2)).cpu())
        if "attrs" in b:
            ATT.append(b["attrs"]); MASK.append(b["slot_mask"]); SPD.append(b["speeds"])
        else:                                    # SSv2: one action class per clip
            ATT.append(b["label_id"]); MASK.append(b["label_id"]); SPD.append(b["label_id"])
    enc.train(); dec.train()
    return (
        {f"chunk{k}": float(np.mean(v)) for k, v in sorted(per_chunk.items())},
        {f"lag{k}": float(np.mean(v)) for k, v in sorted(drift.items())},
        {f"chunk{k}": float(np.mean(v)) for k, v in sorted(retain.items())},
        {k: float(np.mean(v)) for k, v in abl.items()},
        [round(x, 4) for x in idem],
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


def action_report(ZS, ZD, Y):
    """SSv2: frozen-feature action-class top-1 from each code."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = {"n_classes": int(len(np.unique(Y))), "n": int(len(Y))}
    n = len(Y) // 2
    for tag, X in (("static_code", ZS), ("dyn_code", ZD)):
        sc = StandardScaler().fit(X[:n])
        m = LogisticRegression(max_iter=400, C=1.0).fit(sc.transform(X[:n]), Y[:n])
        out[f"{tag}_top1"] = float((m.predict(sc.transform(X[n:])) == Y[n:]).mean())
    # concatenating both is the honest "how much is in the pair" reference
    XB = np.concatenate([ZS, ZD], 1)
    sc = StandardScaler().fit(XB[:n])
    m = LogisticRegression(max_iter=400, C=1.0).fit(sc.transform(XB[:n]), Y[:n])
    out["both_top1"] = float((m.predict(sc.transform(XB[n:])) == Y[n:]).mean())
    out["chance"] = float(np.bincount(Y).max() / len(Y))
    return out


def semantic_report(ZS, ZD, ATT, MASK, SPD):
    """Overall attribute mAP, plus the stationary/moving split."""
    if ATT.ndim == 1:                            # SSv2 carries a class id, not attrs
        return action_report(ZS, ZD, ATT.astype(int))
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
    ap.add_argument("--dataset", default="clevrer", choices=["clevrer", "ssv2"])
    ap.add_argument("--chunk_size_lat", type=int, default=9)
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
    ap.add_argument("--rand_chunks", action="store_true",
                    help="per batch, use a random prefix length K in [2, n_chunks]. "
                         "With a FIXED K every video has the same length, so a "
                         "recurrent memory can learn a position-dependent trajectory "
                         "instead of a content-driven one -- measured: a ConvGRU fed "
                         "the SAME chunk 4x returns cos(z_k,z_0)=[1,.63,.67,.999], "
                         "i.e. it memorised chunk index, not scene content.")
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"],
                    help="lr has been fixed at 3e-4 with no schedule for every run so "
                         "far; cosine decay is the standard free win on reconstruction "
                         "and had never been tried here.")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--d_static", type=int, default=96)
    ap.add_argument("--static_grid", type=int, default=4)
    ap.add_argument("--d_dyn", type=int, default=256)
    ap.add_argument("--dyn_grid", type=int, default=8)
    ap.add_argument("--enc_hidden_ch", type=int, default=192)
    ap.add_argument("--dec_hidden_ch", type=int, default=384)
    ap.add_argument("--static_target", default="none",
                    choices=["none", "video_mean", "video_median", "chunk_mean"],
                    help="EXPLICIT memory as a TEACHER: pool the video's own latent "
                         "frames into a static scene image and make z_static predict "
                         "it. Computed from the input, no labels. video_median "
                         "rejects movers (measured: disagrees with the mean 3.8x more "
                         "at high-motion cells) so the target really is the "
                         "non-moving scene. Measured ceiling: a video-level image "
                         "still explains ~86% of every chunk, vs 91% for a per-chunk "
                         "one -- a good teacher, not a cap.")
    ap.add_argument("--lambda_static_tgt", type=float, default=0.0)
    ap.add_argument("--dino_cache_dir", default=None)
    ap.add_argument("--lambda_dino", type=float, default=0.0)
    ap.add_argument("--lambda_dyn_teach", type=float, default=0.0,
                    help="DECOMPOSED teacher. z_static is taught median_t f_t (what "
                         "persists) and z_dyn is taught |f_t - median_t f_t| (the "
                         "per-frame deviation). The two targets are disjoint by "
                         "construction, so this is the only mechanism tried that acts "
                         "on OVERLAP -- measured RBF-CKA between the codes sits at "
                         "0.32-0.51 for every config so far, and lambda_indep is blind "
                         "to it (reads ~0.003 for all of them).")
    ap.add_argument("--dino_to", default="static", choices=["static", "both"],
                    help="'both' is what train_v5's AuxSemanticDecoder does today: it "
                         "feeds z_static AND z_dyn, so the semantic signal can be "
                         "satisfied through z_dyn (24x the rate) and never lands on "
                         "z_static -- while lambda_indep is simultaneously pushing "
                         "identity OUT of z_dyn. 'static' routes it to z_static alone.")
    ap.add_argument("--lambda_comp", type=float, default=0.0,
                    help="COMPLEMENTARITY hinge. Orthogonality (lambda_indep) is the "
                         "field-standard tool and it does not work here: measured "
                         "L_indep=0.0037 (near-perfect decorrelation) in a model whose "
                         "z_static holds nothing (solo recon 24.90 dB, vs 24.89 dB with "
                         "8x the rate). Decorrelation is satisfied by a NOISE z_static. "
                         "This instead requires each code ALONE to reconstruct at least "
                         "`comp_margin` relatively worse than the pair -- i.e. it "
                         "directly optimises the zs_cost/zd_cost we actually measure, "
                         "which is the PID notion of unique information + synergy "
                         "rather than mere redundancy.")
    ap.add_argument("--comp_margin", type=float, default=1.0)
    ap.add_argument("--shared_trunk", action="store_true")
    ap.add_argument("--lambda_indep", type=float, default=1.0)
    ap.add_argument("--lambda_consist", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DS = ClevrerSequence if args.dataset == "clevrer" else SSv2Sequence
    dkw = ({"dino_cache_dir": args.dino_cache_dir}
           if (args.dino_cache_dir and args.dataset == "clevrer") else {})
    tr = DS(args.cache_dir, args.n_chunks, args.max_videos, "train",
            preload=args.preload, **dkw)
    # cap val with the train budget: the action probe (sklearn, ~170 classes)
    # dominates eval time on the full val split.
    n_val = max(200, args.max_videos // 4) if args.max_videos else 0
    va = DS(args.cache_dir, args.n_chunks, n_val, "val", preload=args.preload, **dkw)
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
                        chunk_size_lat=args.chunk_size_lat,
                        zero_mean_dyn=args.zero_mean_dyn,
                        attn_gate_bias=args.attn_gate_bias,
                        shared_trunk=args.shared_trunk).to(dev)
    if args.decoder == "basedelta":
        dec = BaseDeltaDecoder(d_static=args.d_static, static_grid=args.static_grid,
                               d_dyn=args.d_dyn, dyn_grid=args.dyn_grid,
                               hidden_ch=args.dec_hidden_ch).to(dev)
    else:
        dec = SpatialGridDecoder(d_static=args.d_static, static_grid=args.static_grid,
                                 d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                 chunk_size_lat=args.chunk_size_lat,
                             dyn_spatial=True, dyn_grid=args.dyn_grid).to(dev)
    npar = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"[model] upd={args.mem_update} col={args.mem_collapse} pan={args.synth_pan} dec={args.decoder} "
      f"zmd={args.zero_mean_dyn} params {npar/1e6:.2f}M", flush=True)
    aux = None
    if args.lambda_static_tgt > 0:
        c_s = args.d_static // (args.static_grid ** 2)
        aux = torch.nn.Sequential(
            torch.nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False),
            torch.nn.Conv2d(c_s, 128, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(128, 48, 1)).to(dev)
        print(f"[aux] static target = {args.static_target} "
              f"lambda={args.lambda_static_tgt}", flush=True)
    dyhead = None
    if args.lambda_dyn_teach > 0:
        d_feat = int(tr[0]["dino_res"].shape[-1])
        dyhead = torch.nn.Sequential(
            torch.nn.Conv2d(args.d_dyn, 256, 1), torch.nn.SiLU(),
            torch.nn.Conv2d(256, d_feat, 1)).to(dev)
        print(f"[teach] decomposed: z_static<-median, z_dyn<-residual "
              f"(lambda={args.lambda_dyn_teach})", flush=True)
    dhead = None
    if args.lambda_dino > 0:
        c_s = args.d_static // (args.static_grid ** 2)
        din = c_s + (args.d_dyn if args.dino_to == "both" else 0)
        d_feat = int(tr[0]["dino"].shape[-1])
        dhead = torch.nn.Sequential(
            torch.nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False),
            torch.nn.Conv2d(din, 256, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(256, d_feat, 1)).to(dev)
        print(f"[dino] teacher ON -> {args.dino_to}, d_feat={d_feat}, "
              f"lambda={args.lambda_dino}", flush=True)
    params = list(enc.parameters()) + list(dec.parameters())
    if dhead is not None:
        params += list(dhead.parameters())
    if dyhead is not None:
        params += list(dyhead.parameters())
    if aux is not None:
        params += list(aux.parameters())
    opt = torch.optim.AdamW(params,
                            lr=args.lr, weight_decay=1e-3)

    sched = None
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (
            (e + 1) / max(1, args.warmup) if e < args.warmup else
            0.5 * (1 + math.cos(math.pi * (e - args.warmup)
                                / max(1, args.epochs - args.warmup)))))
    ck, ep0, hist = out / "ckpt.pt", 0, []
    if ck.exists():                                    # resume (restart wrapper)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        # Only resume into an IDENTICAL model. The restart wrapper relaunches on any
        # exit, so a checkpoint left by a different config would otherwise crash the
        # arm forever (or, worse, load partially) instead of just starting over.
        SHAPE = ("d_static", "static_grid", "d_dyn", "dyn_grid", "mem_update",
                 "mem_collapse", "d_pose", "decoder", "enc_hidden_ch", "dec_hidden_ch",
                 "chunk_size_lat", "dataset")
        prev = st.get("args", {})
        mismatch = [k for k in SHAPE if prev.get(k) != getattr(args, k, None)]
        if mismatch:
            print(f"[resume] IGNORING checkpoint: config differs on {mismatch}; "
                  f"starting fresh", flush=True)
        else:
            enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
            opt.load_state_dict(st["opt"])
            ep0 = st["epoch"] + 1; hist = st.get("hist", [])
            print(f"[resume] from epoch {ep0}", flush=True)

    for ep in range(ep0, args.epochs):
        t0, agg, nb = time.time(), {}, 0
        for b in dl:
            seq, pose = prepare(b, args, dev)
            if args.rand_chunks:
                k = int(torch.randint(2, seq.shape[1] + 1, (1,)).item())
                seq = seq[:, :k]
                pose = None if pose is None else pose[:, :k]
            dn = b["dino_med"].to(dev) if "dino_med" in b else (
                 b["dino"].to(dev) if "dino" in b else None)
            dres = b["dino_res"].to(dev) if "dino_res" in b else None
            total, log = losses(enc, dec, seq, pose, args, aux, dn, dhead,
                                dn, dres, dyhead)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for k, v in log.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        if sched is not None:
            sched.step()
        agg = {k: v / max(1, nb) for k, v in agg.items()}
        row = {"epoch": ep, "sec": round(time.time() - t0, 1), **agg}

        if ep % args.eval_every == args.eval_every - 1 or ep == args.epochs - 1:
            pc, dr, rt, ab, idem, ZS, ZD, ATT, MASK, SPD = evaluate(enc, dec, dlv, dev, args)
            row["val_recon"] = pc
            row["drift"] = dr
            row["retention"] = rt
            row["ablation"] = ab
            row["idempotence"] = idem
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
