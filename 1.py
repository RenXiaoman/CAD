from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("dataset/PI-CAI/PI-CAI")
SOURCE_DIR = ROOT / "whole_gland" / "Guerbet23"
TARGET_DIRS = {
    "imagesTr": ROOT / "GlandLabelsTr",
    "imagesTs": ROOT / "GlandLabelsTs",
}


def collect_case_ids(images_dir: Path) -> list[str]:
    return sorted({path.name[:-12] for path in images_dir.glob("*_0000.nii.gz")})


def copy_gland_labels(split_name: str, target_dir: Path) -> None:
    case_ids = collect_case_ids(ROOT / split_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for case_id in case_ids:
        src = SOURCE_DIR / f"{case_id}.nii.gz"
        dst = target_dir / src.name

        if not src.exists():
            raise FileNotFoundError(f"Missing gland label: {src}")

        shutil.copy2(src, dst)
        copied += 1

    print(f"{split_name} -> {target_dir.name}: copied {copied} files")


def main() -> None:
    print(f"Source gland label dir: {SOURCE_DIR}")
    copy_gland_labels("imagesTr", TARGET_DIRS["imagesTr"])
    copy_gland_labels("imagesTs", TARGET_DIRS["imagesTs"])


if __name__ == "__main__":
    main()
