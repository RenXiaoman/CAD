"""Run BMA-Net FullPICAI inference with the project evaluation pipeline."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_SCRIPT = PROJECT_ROOT / "test_prostate_val.py"

DEFAULT_ARGUMENTS = [
    "--model_path",
    "checkpoints_FullPICAI/SegGland_BMA_FullPICAI/best_dice_model.pth",
    "--data_path",
    "dataset/PI-CAI/nnUNet_raw/Dataset141_FullPICAI",
    "--gpu_ids",
    "0",
    "--num_workers",
    "10",
]


if __name__ == "__main__":
    user_arguments = sys.argv[1:]
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    sys.argv = [str(EVALUATION_SCRIPT), *DEFAULT_ARGUMENTS, *user_arguments]
    runpy.run_path(str(EVALUATION_SCRIPT), run_name="__main__")
