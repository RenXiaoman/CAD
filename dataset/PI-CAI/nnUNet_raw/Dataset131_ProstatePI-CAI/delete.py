#!/usr/bin/env python3

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
TARGET_DIRS = ("imagesTr", "imagesTs")
TARGET_SUFFIXES = ("0001.nii.gz", "0002.nii.gz")


def should_delete(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in TARGET_SUFFIXES)


def main():
    deleted_count = 0

    for dir_name in TARGET_DIRS:
        target_dir = DATASET_DIR / dir_name
        if not target_dir.is_dir():
            print(f"Skip missing directory: {target_dir}")
            continue

        for file_path in sorted(target_dir.glob("*.nii.gz")):
            if should_delete(file_path):
                file_path.unlink()
                deleted_count += 1
                print(f"Deleted: {file_path}")

    print(f"Done. Deleted {deleted_count} files.")


if __name__ == "__main__":
    main()
