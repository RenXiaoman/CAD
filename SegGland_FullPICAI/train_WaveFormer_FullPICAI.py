"""Train the WaveFormer baseline on Dataset141_FullPICAI."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Options.Options_Waveformer import Options_Waveformer_FullPICAI


def load_training_module():
    script = PROJECT_ROOT / "Step-1-SegGland" / "train_Waveformer_PICAI.py"
    spec = importlib.util.spec_from_file_location("train_Waveformer_PICAI", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load training module: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    load_training_module().main(Options_Waveformer_FullPICAI)
