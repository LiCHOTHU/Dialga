"""Cache DINOv2-S features per-window for DINOSAUR-style auxiliary supervision.

For each window in an existing Wan-latent cache:
  - Decode the original CLEVRER frames (T_pix frames per window)
  - Run DINOv2-S/14 (frozen) on each frame
  - Save: (T_pix, 384) CLS features and (T_pix, n_patches, 384) patch features

Output goes into <wan_cache_dir>/dino/<idx>.pt — co-located with the Wan latents.
This makes the trajectory training script able to load both in lockstep.

Usage:
    python scripts/cache_dino_features.py \\
        --cache_dir outputs/cache/wan_100vid_W12 \\
        --device cuda --dtype float16
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset


def load_dinov2(device, dtype):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small", torch_dtype=dtype)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m.to(device)


@torch.no_grad()
def encode_frames(model, frames_tensor, device, dtype):
    """frames_tensor: (T, 3, H, W) in [-1, 1] — CLEVRER 128x128.
    Resize to 224x224 (DINOv2 patch=14 → 16x16 grid), run, return CLS + patch features.
    """
    T = frames_tensor.shape[0]
    # Map [-1, 1] → [0, 1] then ImageNet-normalize
    x = (frames_tensor + 1) / 2.0
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x.to(device).to(dtype) - mean.to(dtype)) / std.to(dtype)
    out = model(pixel_values=x)
    last_hidden = out.last_hidden_state             # (T, 1+N, 384) -- CLS at [:, 0]
    cls = last_hidden[:, 0]                          # (T, 384)
    patches = last_hidden[:, 1:]                     # (T, 256, 384)
    return cls.cpu().float(), patches.cpu().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="Existing Wan-latent cache (we co-locate DINO features inside it).")
    ap.add_argument("--data_dir", default="datasets/CLEVRER/train_video")
    ap.add_argument("--annotation_dir", default="datasets/CLEVRER/annotations")
    ap.add_argument("--split", default="train")
    ap.add_argument("--frames_per_video", type=int, default=128)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_patches", action="store_true",
                    help="If set, save (T, 256, 384) patch features too (larger).")
    ap.add_argument("--max_windows", type=int, default=0,
                    help="If > 0, only encode this many windows (smoke testing).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cache_dir = Path(args.cache_dir)
    dino_dir = cache_dir / "dino"; dino_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((cache_dir / "metadata.json").read_text())
    cache_args = meta["args"]
    window_length = int(cache_args["window_length"])
    print(f"[data] cache has {meta['n_windows']} windows; W={window_length} frames")

    # Build the same paired dataset that the Wan cache used (to read the original frames)
    ds = ClevrerPairedDataset(
        data_dir=args.data_dir, annotation_dir=args.annotation_dir,
        split=args.split, window_length=window_length,
        frames_per_video=args.frames_per_video,
        windows_per_video=int(cache_args["windows_per_video"]),
        max_videos=int(cache_args["max_videos"]),
        max_objects=args.max_objects,
        coordinate_mode="world_xy", image_size=args.image_size,
        seed=int(cache_args["seed"]),
    )
    assert len(ds) == meta["n_windows"], f"Dataset length mismatch: {len(ds)} vs cache {meta['n_windows']}"

    print(f"[dino] loading DINOv2-S ({device}, {dtype})...")
    model = load_dinov2(device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dino] loaded ({n_params/1e6:.1f}M params)")

    n_total = len(ds) if args.max_windows <= 0 else min(args.max_windows, len(ds))
    t0 = time.time()
    n_skipped = 0
    for idx in range(n_total):
        out_path = dino_dir / f"{idx:06d}.pt"
        if out_path.exists():
            n_skipped += 1
            if n_skipped == 1:
                print(f"[resume] skipping existing dino blobs…")
            if (idx + 1) % 1000 == 0:
                print(f"[resume] skipped {idx+1} so far")
            continue
        s = ds[idx]
        frames = s["frames"]                                  # (T_pix, 3, H, W)
        cls, patches = encode_frames(model, frames, device, dtype)
        out = {"cls": cls}
        if args.save_patches:
            out["patches"] = patches
        torch.save(out, out_path)
        if (idx + 1) % 25 == 0 or idx == n_total - 1:
            done_now = (idx + 1) - n_skipped
            elapsed = time.time() - t0
            rate = done_now / max(elapsed, 1e-3)
            print(f"  cached {idx+1}/{n_total}  ({n_skipped} skipped)  "
                  f"{elapsed:.1f}s  ({rate:.2f} win/s)")
    print(f"[done] DINOv2 features cached to {dino_dir} ({n_total} windows, {n_skipped} skipped)")


if __name__ == "__main__":
    main()
