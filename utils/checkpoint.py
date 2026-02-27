import os
import re
from typing import List, Optional

_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def extract_step_from_filename(fname: str) -> Optional[int]:
    """Return the training step encoded in a SB3 checkpoint filename, or None."""
    m = _STEP_RE.search(fname)
    return int(m.group(1)) if m else None


def list_checkpoint_files(checkpoint_dir: str) -> List[str]:
    """Return absolute paths to all .zip checkpoints in *checkpoint_dir*, sorted by step."""
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]

    def sort_key(f: str) -> tuple:
        step = extract_step_from_filename(f)
        return (0, step) if step is not None else (1, f)

    files.sort(key=sort_key)
    return [os.path.join(checkpoint_dir, f) for f in files]
