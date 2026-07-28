from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from gland_dataset_nnunet import Lits_DataSet


def make_3d_boundary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask > 0.5, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=radius)
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=radius, border_value=0)
    boundary = np.logical_and(dilated, np.logical_not(eroded))
    return boundary.astype(np.float32)


class BoundaryLitsDataSet:
    def __init__(
        self,
        TASK: Path,
        images_dir,
        labels_dir,
        size=(16, 256, 256),
        augmentation_params=None,
        enable_augmentation=True,
        boundary_radius: int = 1,
    ):
        self.dataset = Lits_DataSet(
            TASK=TASK,
            images_dir=images_dir,
            labels_dir=labels_dir,
            size=size,
            augmentation_params=augmentation_params,
            enable_augmentation=enable_augmentation,
        )
        self.boundary_radius = boundary_radius

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, patient_name = self.dataset[idx]
        boundary = make_3d_boundary(label[0], radius=self.boundary_radius)
        boundary = boundary[np.newaxis, ...]
        return image, label, boundary, patient_name


Boundary_DataSet = BoundaryLitsDataSet
