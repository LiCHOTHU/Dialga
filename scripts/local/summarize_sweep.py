"""Print the memory sweep as one comparable table (reads each arm's history.json)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_eval(hist):
    for row in reversed(hist):
        if "ablation" in row:
            return row
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="outputs/mem_sweep")
    args = ap.parse_args()

    rows = []
    for d in sorted(Path(args.sweep_dir).iterdir()):
        h = d / "history.json"
        if not h.is_file():
            continue
        hist = json.loads(h.read_text())
        r = last_eval(hist)
        if r is None:
            continue
        ab, sem = r["ablation"], r["semantics"]
        ssv2 = "static_code_top1" in sem
        rows.append({
            "idem": (min(r.get("idempotence") or [1.0])),
            "arm": d.name, "ep": r["epoch"],
            "recon": sum(r["val_recon"].values()) / len(r["val_recon"]),
            "drift1": r["drift"]["lag1"], "drift3": r["drift"].get("lag3", float("nan")),
            # Retention as a RATIO, not an absolute. Decoding chunk 0 from the
            # final memory can only hurt in proportion to how much z_static is used
            # at all, so the raw penalty rewards arms whose static code is ignored.
            # Normalising by the wrong-VIDEO penalty asks the scale-free question:
            # of the damage a completely wrong static code would do, what fraction
            # does a right-video-but-wrong-time code do? Lower = the memory still
            # explains early video.
            "retain0": ((r["retention"]["chunk0"] / max(1e-9, r["val_recon"]["chunk0"]) - 1)
                        / max(1e-9, ab["static_from_other_video"] / ab["full"] - 1)),
            "zs_cost": ab["no_static"] / ab["full"] - 1,
            "zd_cost": ab["no_dyn"] / ab["full"] - 1,
            "swap_cost": ab["static_from_other_video"] / ab["full"] - 1,
            "sem_zs": sem["static_code_top1"] if ssv2 else sem["static_code_mAP"],
            "sem_zd": sem["dyn_code_top1"] if ssv2 else sem["dyn_code_mAP"],
            "still_zs": sem.get("both_top1", sem.get("stationary_from_static", float("nan"))),
            "still_zd": sem.get("chance", sem.get("stationary_from_dyn", float("nan"))),
            "move_zs": sem.get("moving_from_static", float("nan")),
            "move_zd": sem.get("moving_from_dyn", float("nan")),
        })

    if not rows:
        print("no completed evals yet")
        return
    hdr = (f"{'arm':<18}{'ep':>4}{'recon':>9}{'drift1':>8}{'drift3':>8}{'ret/swap':>9}"
           f"{'zs_cost':>9}{'zd_cost':>9}{'swap':>8}{'sem_zs':>8}{'sem_zd':>8}"
           f"{'sem_both':>9}{'chance':>9}{'move_zs':>9}{'move_zd':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<18}{r['ep']:>4}{r['recon']:>9.4f}{r['drift1']:>8.3f}"
              f"{r['drift3']:>8.3f}{r['retain0']*100:>8.1f}%{r['zs_cost']*100:>8.1f}%"
              f"{r['zd_cost']*100:>8.1f}%{r['swap_cost']*100:>7.1f}%"
              f"{r['sem_zs']:>8.3f}{r['sem_zd']:>8.3f}"
              f"{r['still_zs']:>9.3f}{r['still_zd']:>9.3f}"
              f"{r['move_zs']:>9.3f}{r['move_zd']:>9.3f}")
    print("\nrecon    val latent MSE, mean over chunks (lower better)")
    print("drift1/3 how far the static code moves by chunk lag 1 / 3 (lower = more "
          "video-length consistent)")
    print("ret/swap extra error decoding chunk 0 from the FINAL memory, as a "
          "FRACTION of\n         the wrong-video penalty (scale-free; lower = the "
          "memory still explains\n         early video). The raw penalty is "
          "confounded: it can only be large if\n         z_static is used at all.")
    print("zs_cost  recon penalty for DELETING z_static  (higher = z_static matters)")
    print("zd_cost  recon penalty for DELETING z_dyn")
    print("swap     recon penalty for z_static from another video (higher = it is "
          "video-specific)")
    print("still_/move_  attribute mAP for STATIONARY / MOVING objects, read from "
          "z_static vs z_dyn")


if __name__ == "__main__":
    main()
