"""Isolation experiment (#50): is the BLUR caused by a too-weak decoder, or by
information already destroyed in the frozen latent code?

Procedure: load a trained checkpoint, FREEZE the encoder (+ its global pool +
the 964-float bottleneck), and train ONLY a much stronger decoder on the exact
same objective the model already optimizes: L_recon = MSE(decoder(z), wan_latent).

Verdict:
  * train/val recon MSE drops WELL BELOW the frozen model's ~0.0124 floor
      -> the decoder was the bottleneck; sharpness is recoverable cheaply.
  * recon MSE stalls at ~the same floor
      -> the information is gone by the code; the fix must be upstream
         (the global pool / the rate), not the decoder.

Primary metric is latent recon MSE (fast, no VAE). The script saves a checkpoint
in the SAME format as train_v5.py (encoder + new decoder + args), so
scripts/viz/save_v512_single_recon.py can render PSNR from it directly.
"""
import argparse, json, sys, time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder
from src.model.latent_encoder import LatentEncoder3D


def build_encoder(a, enc_state, device):
    use_ln = "norm_static.weight" in enc_state
    enc = LatentEncoder3D(
        d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
        hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
        pool_type=a.get("pool_type", "mean"),
        n_queries=int(a.get("pool_queries", 8)),
        n_heads=int(a.get("pool_heads", 4))).to(device)
    enc.load_state_dict(enc_state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_decoder(a, hidden_ch, depth, device):
    """A deliberately stronger decoder than the trained one (deeper conv stack
    via SpatialBroadcastDecoder), trained from scratch on the frozen code."""
    return SpatialBroadcastDecoder(
        d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
        hidden_ch=hidden_ch,
        chunk_size_lat=int(a.get("chunk_size_lat", 9)),
        depth=depth).to(device)


@torch.no_grad()
def eval_val(enc, dec, dl, device, max_batches):
    dec.eval()
    tot, n = 0.0, 0
    for bi, batch in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        x = batch["chunk_obs"].to(device)
        z = enc(x)
        recon = dec(z["z_static"], z["z_dyn"])
        tot += F.mse_loss(recon, x).item() * x.size(0)
        n += x.size(0)
    dec.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="frozen-encoder source checkpoint")
    ap.add_argument("--out", required=True, help="output .pt (viz-compatible)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dec_hidden_ch", type=int, default=512)
    ap.add_argument("--dec_depth", type=int, default=6)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--val_batches", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    if not isinstance(a, dict):
        a = vars(a)

    enc = build_encoder(a, ck["encoder"], device)
    dec = build_decoder(a, args.dec_hidden_ch, args.dec_depth, device)
    n_old = sum(p.numel() for p in LatentDecoder(
        d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
        hidden_ch=int(a["dec_hidden_ch"]),
        chunk_size_lat=int(a.get("chunk_size_lat", 9))).parameters())
    n_new = sum(p.numel() for p in dec.parameters())
    print(f"[ckpt] {args.ckpt} ep={ck.get('epoch')} pool={a.get('pool_type','mean')}")
    print(f"[decoder] old(linear)={n_old/1e6:.2f}M -> new(broadcast d{args.dec_depth} "
          f"h{args.dec_hidden_ch})={n_new/1e6:.2f}M params; encoder FROZEN")

    common = dict(val_frac=float(a.get("val_frac", 0.2)),
                  seed=int(a.get("seed", 42)),
                  max_videos=int(a.get("max_videos", 0)))
    tr = ClevrerChunkPairs(a["cache_dir"], split="train", **common)
    va = ClevrerChunkPairs(a["cache_dir"], split="val", **common)
    tdl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                     collate_fn=chunk_collate, drop_last=True)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2,
                     collate_fn=chunk_collate)
    print(f"[data] train={len(tr)} val={len(va)} pairs")

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr)
    hist = []
    base_floor = 0.0124  # the frozen model's recon plateau, for reference
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        dec.train()
        t0 = time.time()
        run, nb = 0.0, 0
        for batch in tdl:
            x = batch["chunk_obs"].to(device)
            with torch.no_grad():
                z = enc(x)
            recon = dec(z["z_static"], z["z_dyn"])
            loss = F.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item(); nb += 1
        tr_mse = run / max(nb, 1)
        line = f"ep{ep:>3} train_recon={tr_mse:.5f}"
        if ep % args.val_every == 0 or ep == args.epochs:
            v = eval_val(enc, dec, vdl, device, args.val_batches)
            line += (f"  val_recon={v:.5f}  (frozen floor {base_floor:.4f}; "
                     f"{'BELOW -> decoder was limiting' if v < base_floor*0.85 else 'at floor -> code-limited'})")
            hist.append({"epoch": ep, "train_recon": tr_mse, "val_recon": v})
            if v < best_val:
                best_val = v
                save_args = dict(a)
                save_args.update(decoder_type="broadcast",
                                 dec_hidden_ch=args.dec_hidden_ch,
                                 dec_depth=args.dec_depth)
                torch.save({"encoder": ck["encoder"], "decoder": dec.state_dict(),
                            "args": save_args, "epoch": ep,
                            "src_ckpt": args.ckpt, "val_recon": v,
                            "history": hist}, args.out)
        line += f"  ({time.time()-t0:.0f}s)"
        print(line, flush=True)

    print(f"\n[done] best val_recon={best_val:.5f}  vs frozen floor {base_floor:.4f}")
    print(f"[done] saved viz-compatible ckpt -> {args.out}")
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
