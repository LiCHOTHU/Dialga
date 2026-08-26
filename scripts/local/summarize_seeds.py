"""Mean +- std across seeds, and whether the recon gap survives the noise.

A single-seed gap is an observation, not a result. This reports the spread and puts
the between-arm difference next to the within-arm one, which is the only comparison
that decides anything.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def last_eval(hist):
    for r in reversed(hist):
        if "ablation" in r:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="outputs/seed_sweep")
    ap.add_argument("--baseline", default="H0")
    args = ap.parse_args()

    runs = defaultdict(dict)          # arm -> seed -> metrics
    for d in sorted(Path(args.sweep_dir).iterdir()):
        h = d / "history.json"
        if not h.is_file():
            continue
        m = re.match(r"(.+)_s(\d+)$", d.name)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        r = last_eval(json.loads(h.read_text()))
        if r is None:
            continue
        ab = r["ablation"]
        runs[arm][seed] = {
            "recon": sum(r["val_recon"].values()) / len(r["val_recon"]),
            "zs_cost": ab["no_static"] / ab["full"] - 1,
            "swap": ab["static_from_other_video"] / ab["full"] - 1,
        }

    if not runs:
        print("no runs yet")
        return

    keys = ["recon", "zs_cost", "swap"]
    stats = {a: {k: np.array([s[k] for s in sd.values()]) for k in keys}
             for a, sd in runs.items()}

    print(f"{'arm':<8}{'n':>3}   " + "".join(f"{k:>22}" for k in keys))
    print("-" * 78)
    for a in sorted(stats):
        row = f"{a:<8}{len(runs[a]):>3}   "
        for k in keys:
            v = stats[a][k]
            row += f"{v.mean():>13.4f} +-{v.std(ddof=1) if len(v) > 1 else 0:>7.4f}"
        print(row)

    base = args.baseline
    if base in stats:
        print(f"\nvs {base} (recon; negative = better):")
        b = stats[base]["recon"]
        for a in sorted(stats):
            if a == base:
                continue
            v = stats[a]["recon"]
            delta = (v.mean() / b.mean() - 1) * 100
            # pooled spread: does the gap clear the seed-to-seed noise?
            sd = np.sqrt((v.var(ddof=1) + b.var(ddof=1)) / 2) if len(v) > 1 else 0
            nsd = abs(v.mean() - b.mean()) / sd if sd > 0 else float("inf")
            verdict = ("SURVIVES" if nsd >= 2 else
                       "marginal" if nsd >= 1 else "WITHIN NOISE")
            print(f"  {a:<8}{delta:+7.2f}%   gap/pooled-sd = {nsd:5.2f}   {verdict}")
        print("\ngap/pooled-sd is how many seed-standard-deviations the arms differ by."
              "\n>=2 is a real effect at this sample size; <1 means the seeds overlap"
              "\nand the single-seed number was noise.")


if __name__ == "__main__":
    main()
