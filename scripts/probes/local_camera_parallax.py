"""LOCAL go/no-go, PARALLAX regime: the pan test proved a blind encoder is
already invariant to a depth-free 2D pan (pose redundant). Pose conditioning can
only matter under PARALLAX -- camera TRANSLATION with content at different depths,
where near objects shift more than far ones. A pooled/blind encoder provably
cannot separate that from object motion; a KNOWN-pose multi-frame encoder can.
This is the real DROID wrist-camera regime.

Controlled synthetic latent scene (fully known geometry):
  * N objects, each = a Gaussian blob at (H,W) with a random C-dim appearance and
    an inverse-depth d in [0.3,1.0].
  * camera translates over the T frames; object apparent shift = -cam(t)*d
    (disparity proportional to inverse-depth -> genuine parallax).
  * reference = zero-camera render (objects static at canonical position).

Two arms (identical data/seed): ON (encoder gets the true per-frame translation)
vs BLIND (pose zeroed). Verdict = invariance ratio of z_static across DIFFERENT
camera trajectories on HELD-OUT scenes. ON should be << BLIND here, unlike pans.
"""
import argparse, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/storage/home/hcoda1/8/lwang831/workspace/Dialga")
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import SpatialGridDecoder
from src.model.camera_pose import CameraConditioner

C, T, H, W = 48, 9, 8, 8
N_OBJ = 4
SIGMA = 1.1


def make_scenes(M, device, g):
    """Random scenes: appearance (M,N,C), canonical pos (M,N,2), inv-depth (M,N)."""
    app = torch.randn(M, N_OBJ, C, generator=g, device=device)
    pos0 = torch.rand(M, N_OBJ, 2, generator=g, device=device) * (H - 3) + 1.5
    depth = torch.rand(M, N_OBJ, generator=g, device=device) * 0.7 + 0.3
    return app, pos0, depth


def render(app, pos0, depth, cam):
    """cam : (M,T,2) per-frame camera translation (latent grid units).
    Returns latent (M,C,T,H,W). Object apparent shift = -cam*depth (parallax)."""
    M = app.shape[0]
    dev = app.device
    shift = -cam[:, :, None, :] * depth[:, None, :, None]        # (M,T,N,2)
    pos = pos0[:, None, :, :] + shift                            # (M,T,N,2)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    dx = xx[None, None, None] - pos[..., 0:1].unsqueeze(-1)      # (M,T,N,1,W)->bcast
    dy = yy[None, None, None] - pos[..., 1:2].unsqueeze(-1)
    blob = torch.exp(-((dx ** 2 + dy ** 2) / (2 * SIGMA ** 2)))  # (M,T,N,H,W)
    frame = torch.einsum("mtnhw,mnc->mtchw", blob, app)          # (M,T,C,H,W)
    return frame.permute(0, 2, 1, 3, 4).contiguous()            # (M,C,T,H,W)


def rand_cam(M, device, g, vmax=2.5):
    """Constant-velocity translation trajectory, (M,T,2) grid units."""
    vel = (torch.rand(M, 2, generator=g, device=device) * 2 - 1) * vmax
    frac = torch.linspace(0, 1, T, device=device)
    return vel[:, None, :] * frac[None, :, None]                # (M,T,2)


def cam_to_pose(cam):
    """(M,T,2) -> (M,T,3) pose vector (tx,ty,0) matching pose_dim=3."""
    return torch.cat([cam, torch.zeros_like(cam[..., :1])], dim=-1)


def build(device):
    enc = LatentEncoder3D(latent_ch=C, hidden_ch=128, d_static=96, d_dyn=96,
                          pool_type="spatial", static_grid=4,
                          chunk_size_lat=T, d_pose=32).to(device)
    dec = SpatialGridDecoder(latent_ch=C, d_static=96, static_grid=4, d_dyn=96,
                             hidden_ch=256, chunk_size_lat=T, d_pose=32).to(device)
    cc = CameraConditioner(pose_dim=3, d_pose=32, mode="concat", n_trans=2).to(device)
    return enc, dec, cc


def train_arm(blind, scenes, steps, bs, device, seed):
    torch.manual_seed(seed)
    app, pos0, depth = scenes
    M = app.shape[0]
    enc, dec, cc = build(device)
    params = list(enc.parameters()) + list(dec.parameters()) + list(cc.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    zero_cam = torch.zeros(bs, T, 2, device=device)
    for it in range(steps):
        idx = torch.randint(0, M, (bs,), device=device)
        a, p, d = app[idx], pos0[idx], depth[idx]
        cam = rand_cam(bs, device, g)
        warped = render(a, p, d, cam)
        ref_frames = render(a, p, d, zero_cam)
        pemb = cc.embed(cc.relative(cam_to_pose(cam)))
        pemb_id = cc.embed(cc.relative(cam_to_pose(zero_cam)))
        if blind:
            pemb = torch.zeros_like(pemb); pemb_id = torch.zeros_like(pemb_id)
        ref = enc(ref_frames, pose_emb=pemb_id)
        out = enc(warped, pose_emb=pemb)
        recon = dec(out["z_static_grid"], out["z_dyn"], pose_emb=pemb)
        L_recon = F.mse_loss(recon, warped)
        zw = F.normalize(out["z_static"], dim=-1)
        zr = F.normalize(ref["z_static"].detach(), dim=-1)
        L_caminv = F.mse_loss(zw, zr)
        loss = L_recon + 1.0 * L_caminv
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % 200 == 0 or it == steps - 1:
            print(f"  [{'BLIND' if blind else 'ON  '}] step {it:4d} "
                  f"recon {L_recon.item():.4f}  cam_inv {L_caminv.item():.4f}")
    return enc, cc


@torch.no_grad()
def invariance_ratio(enc, cc, scenes, blind, device, K=6, seed=777):
    app, pos0, depth = scenes
    g = torch.Generator(device=device).manual_seed(seed)
    zs = []
    for k in range(K):
        cam = rand_cam(app.shape[0], device, g)
        warped = render(app, pos0, depth, cam)
        pemb = cc.embed(cc.relative(cam_to_pose(cam)))
        if blind:
            pemb = torch.zeros_like(pemb)
        zs.append(F.normalize(enc(warped, pose_emb=pemb)["z_static"], dim=-1))
    zs = torch.stack(zs, 0)                                     # (K,M,D)
    within = zs.std(dim=0).mean().item()
    across = zs.mean(dim=0).std(dim=0).mean().item()
    return within, across, within / max(across, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  PARALLAX regime  seed={args.seed}")
    g = torch.Generator(device=device).manual_seed(args.seed)
    tr = make_scenes(args.n_train, device, g)
    ev = make_scenes(args.n_eval, device, g)

    results = {}
    for blind in (False, True):
        tag = "BLIND" if blind else "ON"
        print(f"==== train arm: {tag} ====")
        enc, cc = train_arm(blind, tr, args.steps, args.bs, device, seed=args.seed)
        enc.eval()
        w, a, r = invariance_ratio(enc, cc, ev, blind, device)
        results[tag] = r
        print(f"  -> {tag}: within(cam)={w:.4f} across(content)={a:.4f} RATIO={r:.4f}")

    on_r, bl_r = results["ON"], results["BLIND"]
    print("\n================ VERDICT (PARALLAX) ================")
    print(f"invariance ratio  ON={on_r:.4f}   BLIND={bl_r:.4f}")
    print(f"ON/BLIND = {on_r / max(bl_r,1e-8):.3f}")
    if on_r < 0.75 * bl_r:
        print("PASS: under parallax, known-pose conditioning de-cameras z_static "
              "where the blind encoder CANNOT.")
    else:
        print("FAIL/WEAK: even under parallax, pose conditioning did not help.")


if __name__ == "__main__":
    main()
