"""LIBERO-90 -> the {labels,split}.json + <id>.webm layout cache_wan_ssv2 expects.

Videos are symlinked (not copied) from single_view_video/. The "class" is the TASK
(90 of them), which is what the held-out-task action probe splits on. Per-frame 7-DoF
actions are copied alongside so the probe can regress them.
"""
import json, os
from pathlib import Path

src = Path('/home/licho/libero_90_processed')
out = Path('datasets/libero'); (out / 'videos').mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (src / 'libero_90_single_view.jsonl').open()] \
    if (src / 'libero_90_single_view.jsonl').exists() else \
    [json.loads(l) for l in (src / 'libero_90.jsonl').open()]

tasks = sorted({r['task'] for r in rows})
labels = {t: str(i) for i, t in enumerate(tasks)}
split, kept = [], 0
for i, r in enumerate(rows):
    vid = src / 'single_view_video' / Path(r['visual_input']).name
    if not vid.exists():
        vid = src / r['visual_input']
    if not vid.exists():
        continue
    dst = out / 'videos' / f"{i}.webm"          # cache script globs <id>.webm
    if not dst.exists():
        os.symlink(vid.resolve(), dst)
    split.append({"id": i, "template": r['task'], "action": r['action'],
                  "demo_id": r['demo_id']})
    kept += 1
(out / 'labels.json').write_text(json.dumps(labels))
(out / 'split.json').write_text(json.dumps(split))
print(f"[prep] {kept} demos, {len(tasks)} tasks -> {out}")
