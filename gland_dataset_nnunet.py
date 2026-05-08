from pathlib import Path
import random
from typing import Union

import numpy as np
import SimpleITK as sitk
from monai.transforms import ClipIntensityPercentiles
from torch.utils.data import Dataset

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform

try:
    import scipy.ndimage as ndi
except ImportError:
    ndi = None


def check_exists(*paths):
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")


default_3D_augmentation_params = {
    "do_elastic": False,
    "p_eldef": 0.2,
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.8, 1.4),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "border_mode_data": "constant",
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}


class GlandDataset(Dataset):
    def __init__(
        self,
        TASK: Path,
        images_dir: Union[Path, str],
        labels_dir: Union[Path, str],
        size=(16, 256, 256),
        augmentation_params=None,
        enable_augmentation=True,
    ):
        self.TASK = TASK
        self.size = size
        self.enable_augmentation = enable_augmentation

        self.images_dir = TASK / images_dir
        self.labels_dir = TASK / labels_dir

        if (not self.images_dir.exists()) or (not self.labels_dir.exists()):
            raise FileNotFoundError(
                f"one of labels and images dir not exist: {self.images_dir}, {self.labels_dir}"
            )

        self.labels_list = sorted(self.labels_dir.glob("*.nii.gz"))
        self.clip = ClipIntensityPercentiles(lower=0.5, upper=99.5, channel_wise=False)

        if augmentation_params is None:
            self.augmentation_params = default_3D_augmentation_params
        else:
            self.augmentation_params = augmentation_params

        self.transforms = self._create_augmentation_transforms() if enable_augmentation else None

    def __len__(self):
        return len(self.labels_list)

    def _create_augmentation_transforms(self):
        tr_transforms = []

        tr_transforms.append(
            SpatialTransform(
                patch_size=self.size,
                patch_center_dist_from_border=None,
                do_elastic_deform=self.augmentation_params.get("do_elastic"),
                alpha=self.augmentation_params.get("elastic_deform_alpha"),
                sigma=self.augmentation_params.get("elastic_deform_sigma"),
                do_rotation=self.augmentation_params.get("do_rotation"),
                angle_x=self.augmentation_params.get("rotation_x"),
                angle_y=self.augmentation_params.get("rotation_y"),
                angle_z=self.augmentation_params.get("rotation_z"),
                p_rot_per_axis=self.augmentation_params.get("rotation_p_per_axis"),
                do_scale=self.augmentation_params.get("do_scaling"),
                scale=self.augmentation_params.get("scale_range"),
                border_mode_data=self.augmentation_params.get("border_mode_data"),
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=self.augmentation_params.get("random_crop"),
                p_el_per_sample=self.augmentation_params.get("p_eldef"),
                p_scale_per_sample=self.augmentation_params.get("p_scale"),
                p_rot_per_sample=self.augmentation_params.get("p_rot"),
                independent_scale_for_each_axis=self.augmentation_params.get(
                    "independent_scale_factor_for_each_axis"
                ),
            )
        )

        if self.augmentation_params.get("do_mirror") or self.augmentation_params.get("mirror"):
            tr_transforms.append(MirrorTransform(self.augmentation_params.get("mirror_axes")))

        return Compose(tr_transforms)

    def _apply_intensity_transforms(self, img):
        if random.random() < 0.1:
            noise = np.random.normal(0, 0.1, img.shape)
            img = img + noise

        if random.random() < 0.2:
            if ndi is not None:
                sigma = random.uniform(0.5, 1.0)
                img = ndi.gaussian_filter(img, sigma=sigma)

        if random.random() < 0.15:
            multiplier = random.uniform(0.75, 1.25)
            img = img * multiplier

        if random.random() < 0.15:
            mean_val = np.mean(img)
            contrast_factor = random.uniform(0.75, 1.25)
            img = mean_val + contrast_factor * (img - mean_val)

        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.5)
            min_val, max_val = np.min(img), np.max(img)
            if max_val > min_val:
                img_normalized = (img - min_val) / (max_val - min_val)
                img_gamma = np.power(img_normalized, gamma)
                img = img_gamma * (max_val - min_val) + min_val

        return img

    def __getitem__(self, idx):
        label_path = self.labels_list[idx]
        patient_name = label_path.name.split(".")[0]

        t2w_path = self.images_dir / f"{patient_name}_0000.nii.gz"
        gt_path = label_path

        check_exists(t2w_path, gt_path)

        t2w_raw = self.load(t2w_path).astype(np.float32)
        gt = self.load(gt_path).astype(np.float32)

        gt[gt > 0] = 1.0

        t2w = self.z_score_normalization(self.clip(t2w_raw))
        
        # print("data is numpy:", isinstance(t2w, np.ndarray))
        
        if self.enable_augmentation:
            t2w = self._apply_intensity_transforms(t2w)
            
        t2w = np.asarray(t2w, dtype=np.float32)
        t2w = t2w[np.newaxis, :]  # [1, D, H, W]
        gt = gt[np.newaxis, :]    # [1, D, H, W]
        
        

        if self.enable_augmentation and self.transforms is not None:
            data_dict = {
                "data": t2w[np.newaxis, ...],
                "seg": gt[np.newaxis, ...],
            }
            data_dict = self.transforms(**data_dict)
            
            t2w = data_dict["data"][0]
            gt = data_dict["seg"][0]

        return t2w, gt, str(patient_name)

    def load(self, file: Union[Path, str]) -> np.ndarray:
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image

    def z_score_normalization(self, img):
        non_zero_mask = img > 0
        if np.any(non_zero_mask):
            mean_val = np.mean(img[non_zero_mask])
            std_val = np.std(img[non_zero_mask])
        else:
            mean_val = np.mean(img)
            std_val = np.std(img)
        out = (img - mean_val) / (std_val + 1e-8)
        return out


Lits_DataSet = GlandDataset


if __name__ == "__main__":
    task = Path("./dataset/PI-CAI/PI-CAI")
    train_dataset = GlandDataset(task, "imagesTr", "GlandLabelsTr", enable_augmentation=False)
    t2w, gt, patient_name = train_dataset[0]
    print("dataset length:", len(train_dataset))
    print("patient:", patient_name)
    print("T2W shape:", t2w.shape, "GT shape:", gt.shape)
