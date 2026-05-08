import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


@dataclass
class PreprocessingSettings:
    matrix_size: Optional[Iterable[int]] = None  # z, y, x
    spacing: Optional[Iterable[float]] = None    # z, y, x


# BASE = Path("dataset/AHCDU/二分类图像")
# NNUNET_SRC_DIR = BASE / "nnUNet_train_gland_origin"
# NNUNET_DEST_DIR = BASE / "nnUNet_train_gland"

BASE = Path("dataset/AHCDU/二分类图像")
NNUNET_SRC_DIR = BASE / "nnUNet_val_gland_origin"
NNUNET_DEST_DIR = BASE / "nnUNet_val_gland"

SETTINGS = PreprocessingSettings(
    matrix_size=[16, 256, 256],
    spacing=[3.5, 0.5, 0.5],
)


def resample_image(
    image: sitk.Image,
    out_spacing: Iterable[float],
    interpolation: int,
    pad_value: float = 0.0,
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_spacing_xyz = list(out_spacing)[::-1]

    out_size = [
        int(np.round(size * (spacing_in / spacing_out)))
        for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing_xyz)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing_xyz)
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(interpolation)
    return resample.Execute(image)


def crop_or_pad(image: sitk.Image, size_zyx: Iterable[int], pad_value: float = 0.0) -> sitk.Image:
    target_size_xyz = list(size_zyx)[::-1]
    shape_xyz = image.GetSize()

    padding = [[0, 0] for _ in range(3)]
    slicer = [slice(None) for _ in range(3)]

    for i in range(3):
        if shape_xyz[i] < target_size_xyz[i]:
            padding[i][0] = (target_size_xyz[i] - shape_xyz[i]) // 2
            padding[i][1] = target_size_xyz[i] - shape_xyz[i] - padding[i][0]
        else:
            start = int(np.floor((shape_xyz[i] - target_size_xyz[i]) / 2.0))
            end = start + target_size_xyz[i]
            slicer[i] = slice(start, end)

    cropped = image[tuple(slicer)]
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound([pad[0] for pad in padding])
    pad_filter.SetPadUpperBound([pad[1] for pad in padding])
    pad_filter.SetConstant(pad_value)
    return pad_filter.Execute(cropped)


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


def preprocess_t2w(t2w_image: sitk.Image) -> sitk.Image:
    t2w_image = sitk.Cast(t2w_image, sitk.sitkFloat32)
    t2w_image = resample_image(t2w_image, SETTINGS.spacing, sitk.sitkBSpline, pad_value=0.0)
    t2w_image = crop_or_pad(t2w_image, SETTINGS.matrix_size, pad_value=0.0)
    return t2w_image


def process_case(patient_id: str):
    src_images_dir = NNUNET_SRC_DIR / "imagesTr"
    src_labels_dir = NNUNET_SRC_DIR / "labelsTr"
    dst_images_dir = NNUNET_DEST_DIR / "imagesTr"
    dst_labels_dir = NNUNET_DEST_DIR / "labelsTr"

    t2w_src = src_images_dir / f"{patient_id}_0000.nii.gz"
    label_src = src_labels_dir / f"{patient_id}.nii.gz"

    if not (t2w_src.exists() and label_src.exists()):
        print(f"[ERROR] Patient {patient_id} missing T2W or label.")
        return

    try:
        t2w_image = sitk.ReadImage(str(t2w_src), sitk.sitkFloat32)
        label_image = sitk.ReadImage(str(label_src), sitk.sitkUInt8)

        processed_t2w = preprocess_t2w(t2w_image)
        processed_label = resample_label_to_reference(label_image, processed_t2w)
        processed_label = binarize_label(processed_label)
        processed_label = align_metadata(processed_label, processed_t2w)

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(processed_t2w, str(dst_images_dir / f"{patient_id}_0000.nii.gz"))
        sitk.WriteImage(processed_label, str(dst_labels_dir / f"{patient_id}.nii.gz"))
        print(f"Processed gland case: {patient_id}")
    except Exception as e:
        print(f"[ERROR] {patient_id} preprocessing failed: {e}")


def get_patient_ids():
    labels_dir = NNUNET_SRC_DIR / "labelsTr"
    if not labels_dir.exists():
        print("[ERROR] labelsTr directory not found!")
        return []

    return sorted(file.stem.replace(".nii", "") for file in labels_dir.glob("*.nii.gz"))


if __name__ == "__main__":
    patient_ids = get_patient_ids()
    if not patient_ids:
        print("No gland patient files found in nnUNet format!")
    else:
        print(f"Found {len(patient_ids)} gland patients to process")
        print(f"Target matrix_size (z,y,x): {SETTINGS.matrix_size}")
        print(f"Target spacing (z,y,x): {SETTINGS.spacing}")
        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_case, patient_ids), total=len(patient_ids)))
