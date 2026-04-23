import os
from pathlib import Path


class SAM2Tracker:
    def __init__(self, checkpoint_root=None, model_id=None):
        scratch_root = Path(os.environ.get("SCRATCH", "/tmp")) / "Dialga"
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else scratch_root / "checkpoints" / "sam2"
        self.model_id = model_id

    def track(self, video_frames, initial_clicks=None):
        raise NotImplementedError(
            "SAM2 integration is not wired in this pass. Install the SAM2 package, "
            "download the checkpoints under $SCRATCH/Dialga/checkpoints/sam2, and "
            "then bind its video predictor here."
        )
