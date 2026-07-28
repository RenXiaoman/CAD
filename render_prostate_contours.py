#!/usr/bin/env python3
"""Generate prostate validation contour images with a process pool."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage import measure
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate contour images from saved validation predictions")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max_slices", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N cases")
    return parser.parse_args()


def load_volume(path: Path, dtype) -> np.ndarray:
    volume = np.asarray(nib.load(path).dataobj, dtype=dtype)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}: {path}")
    return np.transpose(volume, (2, 1, 0))


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(volume, (0.5, 99.5))
    volume = np.clip(volume, lower, upper)
    nonzero = volume > 0
    values = volume[nonzero] if np.any(nonzero) else volume
    return (volume - values.mean()) / (values.std() + 1e-8)


def render_case(job):
    patient_name, image_path, label_path, prediction_path, output_dir, max_slices = job
    try:
        mri = normalize_mri(load_volume(Path(image_path), np.float32))
        label = load_volume(Path(label_path), np.float32) > 0
        prediction = load_volume(Path(prediction_path), np.uint8) > 0

        if mri.shape != label.shape or mri.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch for {patient_name}: MRI {mri.shape}, "
                f"GT {label.shape}, prediction {prediction.shape}"
            )

        patient_dir = Path(output_dir) / patient_name
        patient_dir.mkdir(parents=True, exist_ok=True)
        slice_count = min(max_slices, mri.shape[0])

        for z in range(slice_count):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(mri[z], cmap="gray")

            for index, contour in enumerate(measure.find_contours(label[z], 0.5)):
                ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=2, label="GT" if index == 0 else "")
            for index, contour in enumerate(measure.find_contours(prediction[z], 0.5)):
                ax.plot(contour[:, 1], contour[:, 0], "y-", linewidth=2, label="Pred" if index == 0 else "")

            ax.set_title(f"T2W - Slice {z}")
            ax.axis("off")
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles, labels, loc="lower right", fontsize=10)

            fig.savefig(patient_dir / f"slice_{z:02d}.png", bbox_inches="tight", pad_inches=0.1, dpi=100)
            plt.close(fig)

        return patient_name, slice_count, None
    except Exception as exc:
        return patient_name, 0, str(exc)


def build_jobs(args):
    labels_root = args.data_path / args.labels_dir
    images_root = args.data_path / args.images_dir
    jobs = []

    for label_path in sorted(labels_root.glob("*.nii.gz")):
        patient_name = label_path.name.removesuffix(".nii.gz")
        prediction_path = args.predictions_dir / f"{patient_name}_pred.nii.gz"
        if not prediction_path.exists():
            continue
        jobs.append(
            (
                patient_name,
                str(images_root / f"{patient_name}_0000.nii.gz"),
                str(label_path),
                str(prediction_path),
                str(args.output_dir),
                args.max_slices,
            )
        )

    if args.limit is not None:
        jobs = jobs[:args.limit]
    return jobs


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    jobs = build_jobs(args)
    if not jobs:
        raise FileNotFoundError(f"No matching predictions found in {args.predictions_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    rendered_slices = 0
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        results = executor.map(render_case, jobs, chunksize=1)
        for patient_name, slice_count, error in tqdm(results, total=len(jobs), desc="Rendering contours"):
            rendered_slices += slice_count
            if error is not None:
                errors.append((patient_name, error))

    print(f"Rendered {rendered_slices} slices for {len(jobs) - len(errors)}/{len(jobs)} cases")
    if errors:
        for patient_name, error in errors:
            print(f"Warning: {patient_name}: {error}")
        raise RuntimeError(f"Contour rendering failed for {len(errors)} cases")


if __name__ == "__main__":
    main()
