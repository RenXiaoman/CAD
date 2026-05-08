import os
import shutil


def copy_first_matching_file(src_dir, predicate, dst_path, description):
    for file_name in sorted(os.listdir(src_dir)):
        if predicate(file_name):
            src_path = os.path.join(src_dir, file_name)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {description}: {file_name} -> {os.path.basename(dst_path)}")
            return True
    return False


def count_nii_gz_files(dir_path):
    return sum(1 for f in os.listdir(dir_path) if f.endswith(".nii.gz"))


def find_dwi_dir(patient_dir):
    """
    Find DWI directory using the rule:
    - if only one subdir starts with 'f_tra', use it
    - if multiple exist, use the one with the most .nii.gz files
    """
    candidates = []
    for name in os.listdir(patient_dir):
        full_path = os.path.join(patient_dir, name)
        if os.path.isdir(full_path) and name.startswith("f_tra"):
            candidates.append(full_path)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda p: count_nii_gz_files(p), reverse=True)
    return candidates[0]


def organize_nnunet_train_data(train_dir, lesion_output_dir, gland_output_dir):
    """
    Organize train data into two nnUNet-style datasets:
    - lesion dataset: labels come from T2W/*.nii.gz starting with 'T2'
    - gland dataset:  labels come from T2W/*.nii.gz starting with 'Prostate'

    Both datasets share the same imagesTr content.
    """
    lesion_images_tr = os.path.join(lesion_output_dir, "imagesTr")
    lesion_labels_tr = os.path.join(lesion_output_dir, "labelsTr")
    gland_images_tr = os.path.join(gland_output_dir, "imagesTr")
    gland_labels_tr = os.path.join(gland_output_dir, "labelsTr")

    for path in [lesion_images_tr, lesion_labels_tr, gland_images_tr, gland_labels_tr]:
        os.makedirs(path, exist_ok=True)

    patients = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )

    for patient in patients:
        patient_dir = os.path.join(train_dir, patient)
        t2w_dir = os.path.join(patient_dir, "T2W")
        adc_dir = os.path.join(patient_dir, "ADC")
        dwi_dir = find_dwi_dir(patient_dir)

        print(f"Processing patient: {patient}")

        if os.path.exists(t2w_dir):
            t2w_dst_lesion = os.path.join(lesion_images_tr, f"{patient}_0000.nii.gz")
            t2w_dst_gland = os.path.join(gland_images_tr, f"{patient}_0000.nii.gz")

            found_t2w = copy_first_matching_file(
                t2w_dir,
                lambda f: f.startswith("1") and f.endswith(".nii.gz"),
                t2w_dst_lesion,
                "T2W",
            )
            if found_t2w:
                shutil.copy2(t2w_dst_lesion, t2w_dst_gland)

            copy_first_matching_file(
                t2w_dir,
                lambda f: f.startswith("T2") and f.endswith(".nii.gz"),
                os.path.join(lesion_labels_tr, f"{patient}.nii.gz"),
                "lesion label",
            )

            copy_first_matching_file(
                t2w_dir,
                lambda f: f.startswith("Prostate") and f.endswith(".nii.gz"),
                os.path.join(gland_labels_tr, f"{patient}.nii.gz"),
                "gland label",
            )

        if os.path.exists(adc_dir):
            adc_dst_lesion = os.path.join(lesion_images_tr, f"{patient}_0001.nii.gz")
            adc_dst_gland = os.path.join(gland_images_tr, f"{patient}_0001.nii.gz")

            found_adc = copy_first_matching_file(
                adc_dir,
                lambda f: f.startswith("1") and f.endswith(".nii.gz"),
                adc_dst_lesion,
                "ADC",
            )
            if found_adc:
                shutil.copy2(adc_dst_lesion, adc_dst_gland)

        if dwi_dir and os.path.exists(dwi_dir):
            dwi_dst_lesion = os.path.join(lesion_images_tr, f"{patient}_0002.nii.gz")
            dwi_dst_gland = os.path.join(gland_images_tr, f"{patient}_0002.nii.gz")

            found_dwi = copy_first_matching_file(
                dwi_dir,
                lambda f: f.startswith("1") and f.endswith(".nii.gz"),
                dwi_dst_lesion,
                "DWI",
            )
            if found_dwi:
                shutil.copy2(dwi_dst_lesion, dwi_dst_gland)


def delete_gland_non_t2w_sequences(gland_output_dir):
    """Delete ADC/DWI files from gland imagesTr, keeping only T2W (_0000)."""
    gland_images_tr = os.path.join(gland_output_dir, "imagesTr")
    if not os.path.isdir(gland_images_tr):
        return

    deleted_count = 0
    for file_name in os.listdir(gland_images_tr):
        if file_name.endswith("_0001.nii.gz") or file_name.endswith("_0002.nii.gz"):
            file_path = os.path.join(gland_images_tr, file_name)
            os.remove(file_path)
            deleted_count += 1
            print(f"Deleted gland non-T2W image: {file_name}")

    print(f"Deleted {deleted_count} ADC/DWI files from {gland_images_tr}")


if __name__ == "__main__":
    # train_dir = "dataset/AHCDU/二分类图像/val"
    # lesion_output_dir = "dataset/AHCDU/二分类图像/nnUNet_val_lesion_origin"
    # gland_output_dir = "dataset/AHCDU/二分类图像/nnUNet_val_gland_origin"
    
    train_dir = "dataset/AHCDU/二分类图像/train"
    lesion_output_dir = "dataset/AHCDU/二分类图像/nnUNet_train_lesion_origin"
    gland_output_dir = "dataset/AHCDU/二分类图像/nnUNet_train_gland_origin"

    organize_nnunet_train_data(train_dir, lesion_output_dir, gland_output_dir)
    delete_gland_non_t2w_sequences(gland_output_dir)
    print("nnUNet train data organization completed!")
