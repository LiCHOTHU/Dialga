#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
cd /home/licho/workspace/Dialga
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, pathlib
out = pathlib.Path("outputs/ssv2_dl"); out.mkdir(parents=True, exist_ok=True)
# labels first (small, and they gate everything downstream)
for repo, fn in [("lmms-lab-eval/ssv2","train.jsonl"),
                 ("lmms-lab-eval/ssv2","validation.jsonl")]:
    p = hf_hub_download(repo, fn, repo_type="dataset", resume_download=True)
    shutil.copy(p, out/fn); print("[labels]", fn, flush=True)
# the two split-tar parts of the official archive (resumable)
for fn in ["20bn-something-something-v2-00","20bn-something-something-v2-01"]:
    p = hf_hub_download("Qnancy/ssv2", fn, repo_type="dataset", resume_download=True)
    tgt = out/fn
    if not tgt.exists():
        tgt.symlink_to(p)
    print("[part]", fn, pathlib.Path(p).stat().st_size/1e9, "GB", flush=True)
print("SSV2_DL_DONE", flush=True)
PY
