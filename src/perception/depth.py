import os
from pathlib import Path


def depth_checkpoint_root():
    return Path(os.environ.get("SCRATCH", "/tmp")) / "Dialga" / "checkpoints" / "depth"
