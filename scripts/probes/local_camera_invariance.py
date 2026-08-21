"""LOCAL go/no-go: does KNOWN camera-trajectory conditioning let the encoder
produce a CAMERA-INVARIANT z_static under a moving camera?

Controlled setup (depth-free, exact pose -> the honest first gate):
  * content  = real DROID Wan-latents (48,9,8,8).
  * camera   = synthetic_pan (constant-velocity 2D pan+zoom), pose is EXACT.
  * two arms, identical data/seed/steps: ON (encoder gets the true pose) vs
    BLIND (pose zeroed). BLIND is the "pool+concat can't see the camera" control.

Trained objective mirrors train_v5 synth path: L_recon (render the observed
warped view) + L_cam_inv (z_static of warped+pose must match z_static of the
un-warped canonical reference).

Verdict metric = invariance ratio on HELD-OUT content:
    for each content, warp it with K different cameras, encode each ->
    ratio = (within-content spread of z_static across cameras) /
            (across-content spread of z_static).
  ratio -> 0  means z_static depends on CONTENT, not CAMERA (what we want).
  ON should be << BLIND if the conditioning works.
"""
import argparse, glob, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/Dialga")
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import SpatialGridDecoder
from src.model.camera_pose import CameraConditioner, synthetic_pan


def load_content(n, device):
    files = sorted(glob.glob("/storage/project/r-agarg35-0/lwang831/tmp/"
                             "droid_fast_cache/latents/*.pt"))[:n]
    lats = [torch.load(f, map_location="cpu")["latent"].float() for f in files]
    return torch.stack(lats, 0).to(device)          # (N,48,9,8,8)


def build(device, d_pose):
    enc = LatentEncoder3D(latent_ch=48, hidden_ch=128, d_static=96, d_dyn=96,
                          pool_type="spatial", static_grid=4,
                          chunk_size_lat=9, d_pose=d_pose).to(device)
    dec = SpatialGridDecoder(latent_ch=48, d_static=96, static_grid=4, d_dyn=96,
                             hidden_ch=256, chunk_size_lat=9,
                             d_pose=d_pose).to(device)
    cc = CameraConditioner(pose_dim=3, d_pose=d_pose, mode="concat",
                           n_trans=2).to(device)
    return enc, dec, cc


def train_arm(blind, content, steps, bs, device, seed, ms=0.4, mz=0.15):
    torch.manual_seed(seed)                          # identical data/init per arm
    enc, dec, cc = build(device, d_pose=32)
    params = (list(enc.parameters()) + list(dec.parameters())
              + list(cc.parameters()))
    opt = torch.optim.Adam(params, lr=1e-4)          # lower LR: runaway control
    N = content.shape[0]
    zero_pose = torch.zeros(bs, 9, 3, device=device)
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), device=device)
        base = content[idx]                          # (bs,48,9,8,8)
        warped, pose = synthetic_pan(base, ms, mz)
        pemb = cc.embed(cc.relative(pose))
        pemb_id = cc.embed(cc.relative(zero_pose))
        if blind:
            pemb = torch.zeros_like(pemb); pemb_id = torch.zeros_like(pemb_id)
        ref = enc(base, pose_emb=pemb_id)            # canonical reference
        out = enc(warped, pose_emb=pemb)
        recon = dec(out["z_static_grid"], out["z_dyn"], pose_emb=pemb)
        L_recon = F.mse_loss(recon, warped)
        # SCALE-FREE invariance: unit-normalize z_static so the objective cannot
        # be gamed by inflating magnitude (known runaway failure mode, task #94).
        zw = F.normalize(out["z_static"], dim=-1)
        zr = F.normalize(ref["z_static"].detach(), dim=-1)
        L_caminv = F.mse_loss(zw, zr)
        loss = L_recon + 1.0 * L_caminv
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)  # runaway control
        opt.step()
        if it % 150 == 0 or it == steps - 1:
            znorm = out["z_static"].norm(dim=-1).mean().item()
            print(f"  [{'BLIND' if blind else 'ON  '}] step {it:4d} "
                  f"recon {L_recon.item():.4f}  cam_inv {L_caminv.item():.4f}  "
                  f"|z_static| {znorm:.2f}")
    return enc, dec, cc


@torch.no_grad()
def invariance_ratio(enc, cc, content, blind, device, K=6, seed=1234):
    """Per content, warp with K cameras; measure z_static within/across spread."""
    g = torch.Generator(device=device).manual_seed(seed)
    zs = []                                          # (K, M, d_static)
    for k in range(K):
        wl, pose = synthetic_pan(content, 0.4, 0.15, generator=g)
        pemb = cc.embed(cc.relative(pose))
        if blind:
            pemb = torch.zeros_like(pemb)
        zs.append(F.normalize(enc(wl, pose_emb=pemb)["z_static"], dim=-1))
    zs = torch.stack(zs, 0)                          # (K, M, D) unit-normalized
    within = zs.std(dim=0).mean().item()             # spread across cameras
    across = zs.mean(dim=0).std(dim=0).mean().item() # spread across content
    return within, across, within / max(across, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=96)
    ap.add_argument("--n_eval", type=int, default=48)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    content = load_content(args.n_train + args.n_eval, device)
    train, evalc = content[:args.n_train], content[args.n_train:]
    print(f"content {tuple(content.shape)}  train {train.shape[0]} eval {evalc.shape[0]}")

    results = {}
    for blind in (False, True):
        tag = "BLIND" if blind else "ON"
        print(f"==== train arm: {tag} ====")
        enc, dec, cc = train_arm(blind, train, args.steps, args.bs, device, seed=0)
        enc.eval()
        w, a, r = invariance_ratio(enc, cc, evalc, blind, device)
        results[tag] = (w, a, r)
        print(f"  -> {tag}: within(cam)={w:.4f} across(content)={a:.4f} "
              f"RATIO={r:.4f}")

    on_r, bl_r = results["ON"][2], results["BLIND"][2]
    print("\n================ VERDICT ================")
    print(f"invariance ratio  ON={on_r:.4f}   BLIND={bl_r:.4f}")
    print(f"ON/BLIND = {on_r / max(bl_r,1e-8):.3f}  "
          f"(<1 => pose conditioning makes z_static MORE camera-invariant)")
    if on_r < 0.75 * bl_r:
        print("PASS: known-pose conditioning removes camera dependence from z_static.")
    else:
        print("FAIL/WEAK: pose conditioning did NOT meaningfully de-camera z_static.")


if __name__ == "__main__":
    main()
