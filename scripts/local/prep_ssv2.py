"""Turn the downloaded SSv2 archive + jsonl labels into what cache_wan_ssv2.py wants.

The official release ships as a split tar.gz (two parts that must be concatenated),
and the labels here come from the lmms-lab-eval jsonl mirror rather than the original
label json, so both need converting:

    labels.json  {template-without-brackets: "class id"}
    split.json   [{"id": video_id, "template": ...}, ...]

Extraction stops as soon as `--max_videos` clips are out, so we decompress a small
prefix of the 19.4 GB archive instead of all of it.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl_dir", default="outputs/ssv2_dl")
    ap.add_argument("--video_out", default="datasets/ssv2/videos")
    ap.add_argument("--meta_out", default="datasets/ssv2")
    ap.add_argument("--max_videos", type=int, default=6000)
    args = ap.parse_args()

    dl = Path(args.dl_dir)
    vout = Path(args.video_out); vout.mkdir(parents=True, exist_ok=True)
    mout = Path(args.meta_out); mout.mkdir(parents=True, exist_ok=True)

    # ---- labels ----------------------------------------------------------
    rows = [json.loads(l) for l in (dl / "train.jsonl").read_text().splitlines() if l]
    val = [json.loads(l) for l in (dl / "validation.jsonl").read_text().splitlines() if l]
    def clean(t):
        return t.replace("[", "").replace("]", "")
    templates = sorted({clean(r["template"]) for r in rows + val})
    label_map = {t: str(i) for i, t in enumerate(templates)}
    (mout / "labels.json").write_text(json.dumps(label_map, indent=0))
    print(f"[labels] {len(templates)} action templates")

    by_id = {str(r["video_id"]): clean(r["template"]) for r in rows}
    by_id_val = {str(r["video_id"]): clean(r["template"]) for r in val}
    print(f"[labels] {len(by_id)} train clips, {len(by_id_val)} val clips")

    # ---- extract a prefix of the split archive ---------------------------
    parts = sorted(dl.glob("20bn-something-something-v2-*"))
    if not parts:
        raise SystemExit("archive parts not found")
    print(f"[tar] streaming {len(parts)} parts, stopping after {args.max_videos} clips")

    class Chained(io.RawIOBase):
        """Concatenate the split parts into one stream (the archive is split, not
        independently extractable)."""
        def __init__(self, paths):
            self.paths, self.i = paths, 0
            self.f = open(self.paths[0], "rb")
        def readable(self):
            return True
        def readinto(self, b):
            while True:
                n = self.f.readinto(b)
                if n:
                    return n
                self.f.close(); self.i += 1
                if self.i >= len(self.paths):
                    return 0
                self.f = open(self.paths[self.i], "rb")

    got, kept = 0, []
    stream = io.BufferedReader(Chained([str(p) for p in parts]), buffer_size=1 << 22)
    with tarfile.open(fileobj=stream, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".webm"):
                continue
            vid = Path(m.name).stem
            if vid not in by_id and vid not in by_id_val:
                continue
            dst = vout / f"{vid}.webm"
            if not dst.exists():
                f = tf.extractfile(m)
                if f is None:
                    continue
                dst.write_bytes(f.read())
            kept.append(vid)
            got += 1
            if got % 500 == 0:
                print(f"[tar] {got}/{args.max_videos}", flush=True)
            if got >= args.max_videos:
                break

    split = [{"id": int(v), "template": (by_id.get(v) or by_id_val[v])} for v in kept]
    (mout / "split.json").write_text(json.dumps(split))
    print(f"[done] {len(split)} clips -> {vout}")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
