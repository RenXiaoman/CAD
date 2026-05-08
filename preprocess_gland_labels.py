from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk


@dataclass
class PreprocessingSettings:
    matrix_size: Optional[Iterable[int]] = None  # z, y, x
    spacing: Optional[Iterable[float]] = None    # z, y, x


SETTINGS = PreprocessingSettings(
    matrix_size=[16, 256, 256],
    spacing=[3.5, 0.5, 0.5],
)

# ROOT = Path("dataset/PI-CAI/PI-CAI")
# SPLITS = {
#     "imagesTr": ROOT / "GlandLabelsTr",
#     "imagesTs": ROOT / "GlandLabelsTs",
# }


ROOT = Path("dataset/AHCDU/二分类图像/nnUNet_val_gland_origin")
SPLITS = {
    "imagesTr": ROOT / "labelsTr",
    # "imagesTs": ROOT / "labelsTr",
}

# ROOT = Path("dataset/PI-CAI/PI-CAI")
# SPLITS = {
#     "imagesTr": ROOT / "GlandLabelsTr",
#     "imagesTs": ROOT / "GlandLabelsTs",
# }


def collect_case_ids(images_dir: Path) -> list[str]:
    return sorted(path.name[:-12] for path in images_dir.glob("*_0000.nii.gz"))


def verify_reference_image(reference_image: sitk.Image, case_id: str) -> None:
    size_zyx = list(reference_image.GetSize())[::-1]
    spacing_zyx = list(reference_image.GetSpacing())[::-1]

    if SETTINGS.matrix_size is not None and list(SETTINGS.matrix_size) != size_zyx:
        raise ValueError(
            f"{case_id}: unexpected reference size {size_zyx}, expected {SETTINGS.matrix_size}"
        )

    if SETTINGS.spacing is not None and not np.allclose(spacing_zyx, SETTINGS.spacing, atol=1e-6):
        raise ValueError(
            f"{case_id}: unexpected reference spacing {spacing_zyx}, expected {SETTINGS.spacing}"
        )


def resample_label_to_reference(label_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(label_image)


def binarize_label(label_image: sitk.Image) -> sitk.Image:
    label_array = sitk.GetArrayFromImage(label_image)
    label_array = (label_array > 0).astype(np.uint8)
    output = sitk.GetImageFromArray(label_array)
    output.CopyInformation(label_image)
    return output


def align_metadata(image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())
    return image


def process_split(images_split: str, labels_dir: Path) -> None:
    images_dir = ROOT / images_split
    case_ids = collect_case_ids(images_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for case_id in case_ids:
        reference_path = images_dir / f"{case_id}_0000.nii.gz"
        label_path = labels_dir / f"{case_id}.nii.gz"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing gland label: {label_path}")

        reference_image = sitk.ReadImage(str(reference_path))
        verify_reference_image(reference_image, case_id)

        label_image = sitk.ReadImage(str(label_path))
        processed_label = resample_label_to_reference(label_image, reference_image)
        processed_label = binarize_label(processed_label)
        processed_label = align_metadata(processed_label, reference_image)

        sitk.WriteImage(processed_label, str(label_path))
        processed += 1

    print(f"{images_split} -> {labels_dir.name}: processed {processed} labels in place")


def main() -> None:
    print("Mode: inplace processing of GlandLabelsTr/GlandLabelsTs")
    print(f"Target matrix_size (z,y,x): {SETTINGS.matrix_size}")
    print(f"Target spacing (z,y,x): {SETTINGS.spacing}")

    for images_split, labels_dir in SPLITS.items():
        process_split(images_split, labels_dir)


if __name__ == "__main__":
    main()
