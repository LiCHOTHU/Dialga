"""Convert the AttrsHead linear-probe CE into interpretable top-1 accuracy.

For each run checkpoint: rebuild LatentEncoder3D + AttrsHead from the stored
args, run the deterministic val split, and report per-attribute (color 8-way /
material 2-way / shape 3-way) top-1 accuracy of the MODAL-class prediction from
the pooled global z_static, alongside the majority-class baseline.

Runs on CPU; only needs the cached Wan latents (no VAE decode).
"""
import argparse, json, sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.attrs_head import AttrsHead
from src.model.latent_encoder import LatentEncoder3D
from torch.utils.data import DataLoader

N_COLOR, N_MATERIAL, N_SHAPE = len(COLOR_VOCAB), len(MATERIAL_VOCAB), len(SHAPE_VOCAB)


def modal_labels(attrs, slot_mask, lo, hi):
    block = attrs[..., lo:hi]
    cls = block.argmax(dim=-1)
    onehot = F.one_hot(cls, hi - lo).float()
    masked = onehot * slot_mask.float().unsqueeze(-1)
    counts = masked.sum(dim=1)
    labels = counts.argmax(dim=-1)
    valid = slot_mask.bool().any(dim=-1)
    return torch.where(valid, labels, torch.full_like(labels, -100))


@torch.no_grad()
def eval_ckpt(ckpt_path, max_batches, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = Namespace(**ck["args"])
    enc = LatentEncoder3D(d_static=a.d_static, d_dyn=a.d_dyn,
                          hidden_ch=a.enc_hidden_ch,
                          shared_trunk=getattr(a, "shared_trunk", False),
                          pool_type=getattr(a, "pool_type", "mean"),
                          n_queries=getattr(a, "pool_queries", 8),
                          n_heads=getattr(a, "pool_heads", 4)).to(device).eval()
    ah = AttrsHead(d_static=a.d_static, n_color=N_COLOR, n_material=N_MATERIAL,
                   n_shape=N_SHAPE, hidden=getattr(a, "attrs_hidden", 0)).to(device).eval()
    enc.load_state_dict(ck["encoder"]); ah.load_state_dict(ck["attrs_head"])

    ds = ClevrerChunkPairs(a.cache_dir, split="val", val_frac=a.val_frac,
                           seed=a.seed, max_videos=a.max_videos)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                    collate_fn=chunk_collate)

    corr = {"color": 0, "material": 0, "shape": 0}
    ntot = 0
    ycat = {"color": [], "material": [], "shape": []}
    for bi, batch in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        z = enc(batch["chunk_obs"].to(device))["z_static"]
        logits = ah(z)
        attrs = batch["attrs"].to(device); sm = batch["slot_mask"].to(device)
        ys = {"color": modal_labels(attrs, sm, 0, N_COLOR),
              "material": modal_labels(attrs, sm, N_COLOR, N_COLOR + N_MATERIAL),
              "shape": modal_labels(attrs, sm, N_COLOR + N_MATERIAL,
                                    N_COLOR + N_MATERIAL + N_SHAPE)}
        valid = ys["color"] != -100
        ntot += int(valid.sum())
        for k in corr:
            pred = logits[k].argmax(-1)
            corr[k] += int(((pred == ys[k]) & valid).sum())
            ycat[k].append(ys[k][valid])
    epoch = ck.get("epoch", "?")
    acc = {k: corr[k] / max(ntot, 1) for k in corr}
    # majority-class baseline per attribute on this val set
    base = {}
    for k in corr:
        y = torch.cat(ycat[k])
        base[k] = float(torch.bincount(y).max()) / max(len(y), 1)
    return epoch, ntot, acc, base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--max_batches", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdr = f"{'run':<26}{'ep':>5}{'n':>7} | {'color':>7}{'(maj)':>7} {'matl':>7}{'(maj)':>7} {'shape':>7}{'(maj)':>7} {'mean':>7}"
    print(hdr); print("-" * len(hdr))
    for c in args.ckpts:
        tag = Path(c).parent.name
        try:
            ep, n, acc, base = eval_ckpt(c, args.max_batches, device)
            mean = sum(acc.values()) / 3
            print(f"{tag:<26}{str(ep):>5}{n:>7} | "
                  f"{acc['color']*100:6.1f}%{base['color']*100:6.1f}% "
                  f"{acc['material']*100:6.1f}%{base['material']*100:6.1f}% "
                  f"{acc['shape']*100:6.1f}%{base['shape']*100:6.1f}% "
                  f"{mean*100:6.1f}%")
        except Exception as e:
            print(f"{tag:<26} ERROR {type(e).__name__}: {e}")
