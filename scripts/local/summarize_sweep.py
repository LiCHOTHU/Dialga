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
        rows.append({
            "arm": d.name, "ep": r["epoch"],
            "recon": sum(r["val_recon"].values()) / len(r["val_recon"]),
            "drift1": r["drift"]["lag1"], "drift3": r["drift"].get("lag3", float("nan")),
            # retention loss: decoding chunk 0 from the FINAL memory vs its own code
            "retain0": r["retention"]["chunk0"] / max(1e-9, r["val_recon"]["chunk0"]) - 1,
            "zs_cost": ab["no_static"] / ab["full"] - 1,
            "zd_cost": ab["no_dyn"] / ab["full"] - 1,
            "swap_cost": ab["static_from_other_video"] / ab["full"] - 1,
            "sem_zs": sem["static_code_mAP"], "sem_zd": sem["dyn_code_mAP"],
            "still_zs": sem["stationary_from_static"], "still_zd": sem["stationary_from_dyn"],
            "move_zs": sem["moving_from_static"], "move_zd": sem["moving_from_dyn"],
        })

    if not rows:
        print("no completed evals yet")
        return
    hdr = (f"{'arm':<18}{'ep':>4}{'recon':>9}{'drift1':>8}{'drift3':>8}{'retain0':>9}"
           f"{'zs_cost':>9}{'zd_cost':>9}{'swap':>8}{'sem_zs':>8}{'sem_zd':>8}"
           f"{'still_zs':>9}{'still_zd':>9}{'move_zs':>9}{'move_zd':>9}")
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
    print("retain0  extra error decoding chunk 0 from the FINAL memory (lower = the "
          "memory still explains early video)")
    print("zs_cost  recon penalty for DELETING z_static  (higher = z_static matters)")
    print("zd_cost  recon penalty for DELETING z_dyn")
    print("swap     recon penalty for z_static from another video (higher = it is "
          "video-specific)")
    print("still_/move_  attribute mAP for STATIONARY / MOVING objects, read from "
          "z_static vs z_dyn")


if __name__ == "__main__":
    main()
