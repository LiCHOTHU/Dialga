"""Cache SigLIP patch features aligned to the Wan-latent chunk grid.

Fork of scripts/cache_dino_patch.py. SigLIP is a DUAL encoder (vision + text), so
`AutoModel` returns a SiglipModel whose config has no `hidden_size` -- the vision half
has to be pulled out explicitly, which is why the DINO path fails on it.

Why SigLIP rather than DINOv2 as the teacher: it is LANGUAGE-ALIGNED, so its features
encode nameable identity -- category, colour, material -- where DINOv2's are trained
for visual/textural similarity. z_static's job is "what is in the scene", and our
attribute mAP has been pinned at ~0.75 through every lever tried with the DINOv2
teacher.

For every window in an existing Wan-latent cache (e.g. wan_10000vid_W33,
windows of 33 RGB frames -> latent (48, 9, 8, 8)), compute frozen DINOv2
patch features for the 9 RGB frames that anchor the 9 latent frames
(frame start+4*i, i=0..8; Wan temporal compression is first-frame + groups
of 4), and average-pool the 16x16 DINO patch grid down to the latent's 8x8.

Output is ONE memmap file (no per-window blobs -> no inode pressure):
    <out_dir>/features.f16.bin   float16, shape (n_windows, 9, 8, 8, 384)
    <out_dir>/index.json         {n_windows, shape, dtype, wan_cache_dir,
                                  model, windows: [{idx, video_id, start_frame}]}
    <out_dir>/progress.json      {next_idx}  (resume marker)

Row i corresponds exactly to window idx i of the Wan cache metadata.

Usage (smoke):
    python scripts/cache_dino_patch.py --max_windows 60 \
        --out_dir /storage/scratch1/8/lwang831/cache/dino_patch_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def light_collate(batch):
    return {
        "frames": torch.stack([b["frames"] for b in batch]),       # (B, 33, 3, H, W) in [-1,1]
        "video_id": torch.tensor([int(b["video_id"]) for b in batch]),
        "start_frame": torch.tensor([int(b["start_frame"]) for b in batch]),
    }


@torch.no_grad()
def dino_patch_grid(model, frames, device, pool_hw=8):
    """frames: (N, 3, 224, 224) in [-1,1] -> (N, pool_hw, pool_hw, D) float32."""
    x = frames.to(device)
    # SigLIP normalises with mean=std=0.5, i.e. it expects [-1,1] -- which is exactly
    # what `frames` already is. Applying DINOv2's ImageNet renormalisation here would
    # feed it out-of-distribution inputs and silently produce garbage features.
    if getattr(model.config, "model_type", "") != "siglip_vision_model":
        x = (x * 0.5 + 0.5 - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    out = model(pixel_values=x)
    # SigLIP's vision tower emits PURE patch tokens -- no CLS, no registers -- so the
    # DINOv2 strip would discard a real patch and misalign the grid.
    if getattr(model.config, "model_type", "") == "siglip_vision_model":
        tokens = out.last_hidden_state
    else:
        tokens = out.last_hidden_state[:, 1 + getattr(model.config, "num_register_tokens", 0):]
    n, l, d = tokens.shape
    g = int(l ** 0.5)
    assert g * g == l, f"non-square patch grid: {l}"
    grid = tokens.permute(0, 2, 1).reshape(n, d, g, g)
    grid = F.adaptive_avg_pool2d(grid, pool_hw)                    # (N, D, 8, 8)
    return grid.permute(0, 2, 3, 1).float().cpu()                  # (N, 8, 8, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan_cache_dir", default="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33")
    ap.add_argument("--out_dir", default="/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--pool_hw", type=int, default=8)
    ap.add_argument("--batch_windows", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_windows", type=int, default=0, help="0 = all (smoke: small N)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wan_meta = json.load(open(Path(args.wan_cache_dir) / "metadata.json"))
    wargs = wan_meta["args"]
    windows = wan_meta["windows"]
    n_windows = len(windows) if args.max_windows == 0 else min(args.max_windows, len(windows))
    # NOTE: max_videos must equal the wan-cache value — any smaller value makes
    # ClevrerPairedDataset rng.sample() a DIFFERENT random video subset and the
    # window ordering no longer matches the cache. Smoke runs just iterate fewer
    # windows of the identically-built dataset.
    ds = ClevrerPairedDataset(
        data_dir=wargs["data_dir"],
        annotation_dir=wargs["annotation_dir"],
        split=wargs["split"],
        window_length=wargs["window_length"],
        frames_per_video=wargs["frames_per_video"],
        windows_per_video=wargs["windows_per_video"],
        max_videos=wargs["max_videos"],
        max_objects=wargs["max_objects"],
        coordinate_mode="world_xy",
        image_size=args.image_size,
        seed=wargs["seed"],
        deterministic_starts=[int(s) for s in wargs["deterministic_starts"].split(",")],
    )
    assert len(ds) >= n_windows, f"dataset has {len(ds)} windows, wan cache expects {n_windows}"
    t_lat = windows[0]["latent_shape"][1]                          # 9
    stride = (wargs["window_length"] - 1) // (t_lat - 1)           # 4

    print(f"[dino] loading {args.model}")
    if "siglip" in args.model.lower():
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(args.model).to(device).eval()
        d_feat = model.config.hidden_size
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model).to(device).eval()
        d_feat = model.config.hidden_size
    for p in model.parameters():
        p.requires_grad_(False)

    shape = (n_windows, t_lat, args.pool_hw, args.pool_hw, d_feat)
    bin_path = out_dir / "features.f16.bin"
    mode = "r+" if bin_path.exists() else "w+"
    mm = np.memmap(bin_path, dtype=np.float16, mode=mode, shape=shape)

    prog_path = out_dir / "progress.json"
    next_idx = json.load(open(prog_path))["next_idx"] if prog_path.exists() else 0
    print(f"[cache] {n_windows} windows, T_lat={t_lat}, stride={stride}, "
          f"feat={d_feat}, resume at idx {next_idx}")

    loader = DataLoader(
        torch.utils.data.Subset(ds, range(next_idx, n_windows)),
        batch_size=args.batch_windows, shuffle=False,
        num_workers=args.num_workers, collate_fn=light_collate,
    )

    t0, done = time.time(), next_idx
    frame_sel = torch.arange(t_lat) * stride                       # 0,4,...,32
    for batch in loader:
        bsz = batch["frames"].shape[0]
        # alignment assertion against the wan metadata
        for b in range(bsz):
            w = windows[done + b]
            assert int(batch["video_id"][b]) == int(w["video_id"]) and \
                   int(batch["start_frame"][b]) == int(w["start_frame"]), \
                f"window {done+b}: dataset ({int(batch['video_id'][b])},{int(batch['start_frame'][b])}) " \
                f"!= wan cache ({w['video_id']},{w['start_frame']})"
        frames = batch["frames"][:, frame_sel]                     # (B, 9, 3, 224, 224)
        flat = frames.reshape(-1, *frames.shape[2:])
        feats = dino_patch_grid(model, flat, device, args.pool_hw) # (B*9, 8, 8, D)
        feats = feats.reshape(bsz, t_lat, args.pool_hw, args.pool_hw, d_feat)
        assert torch.isfinite(feats).all(), f"non-finite DINO feats at idx {done}"
        mm[done:done + bsz] = feats.numpy().astype(np.float16)
        done += bsz
        if (done // args.batch_windows) % 25 == 0 or done >= n_windows:
            mm.flush()
            json.dump({"next_idx": done}, open(prog_path, "w"))
            el = time.time() - t0
            print(f"  {done}/{n_windows}  {el:.0f}s  ({(done-next_idx)/max(el,1e-9):.2f} win/s)")

    mm.flush()
    json.dump({"next_idx": done}, open(prog_path, "w"))
    json.dump({
        "n_windows": n_windows, "shape": list(shape), "dtype": "float16",
        "wan_cache_dir": args.wan_cache_dir, "model": args.model,
        "frame_stride": int(stride), "pool_hw": args.pool_hw,
        "windows": [{"idx": w["idx"], "video_id": w["video_id"],
                     "start_frame": w["start_frame"]} for w in windows[:n_windows]],
    }, open(out_dir / "index.json", "w"))
    print(f"[done] {done} windows -> {bin_path} ({bin_path.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
