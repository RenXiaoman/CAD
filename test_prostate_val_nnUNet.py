#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from skimage import measure
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Evaluate nnUNet validation predictions")
parser.add_argument(
    "--model_path",
    type=str,
    default="checkpoints/SegGland_nnUNet_PICAI",
    help="Path to the nnUNet prediction root directory containing fold0-fold4",
)
parser.add_argument(
    "--data_dir",
    type=str,
    # default="dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU",
    default="dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI",
    help="Path to the nnUNet dataset directory containing imagesTs/labelsTs",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="val_results",
    help="Output directory for processed predictions and summaries",
)
parser.add_argument(
    "--slices_dir",
    type=str,
    default="slice_contours",
    help="Directory for slice images with contours",
)
parser.add_argument(
    "--keep_largest_cc",
    action="store_true",
    default=False,
    help="Keep only the largest connected component in predictions",
)
parser.add_argument(
    "--mode",
    type=str,
    default="val",
    choices=["val", "test"],
    help="Mode used only for output directory naming",
)

args = parser.parse_args()


def calculate_dice(preds, targets):
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    pred_sum = preds_flat.sum()
    target_sum = targets_flat.sum()

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()


def calculate_miou(preds, targets):
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def get_surface_points(binary_mask):
    struct = generate_binary_structure(3, 3)
    eroded = binary_erosion(binary_mask, struct)
    surface = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)
    return np.argwhere(surface > 0)


def calculate_hd95(pred, target, percentile=95):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)

    try:
        pred_surface = get_surface_points(pred)
        target_surface = get_surface_points(target)

        if len(pred_surface) == 0 or len(target_surface) == 0:
            return 0.0

        target_tree = cKDTree(target_surface)
        distances_pred_to_target, _ = target_tree.query(pred_surface, k=1)

        pred_tree = cKDTree(pred_surface)
        distances_target_to_pred, _ = pred_tree.query(target_surface, k=1)

        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        hd95 = np.percentile(all_distances, percentile)

        if np.isnan(hd95) or np.isinf(hd95):
            return 0.0

        return float(hd95)
    except Exception as e:
        print(f"Warning: Error calculating HD95: {e}")
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)


def calculate_dice_intervals(dice_scores):
    intervals = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 1.0),
    ]

    interval_counts = {}
    for i, (lower, upper) in enumerate(intervals):
        if i == len(intervals) - 1:
            count = sum(lower <= score <= upper for score in dice_scores)
        else:
            count = sum(lower <= score < upper for score in dice_scores)
        interval_counts[f"{lower:.1f}-{upper:.1f}"] = count

    return interval_counts


def keep_largest_connected_component_3d(pred_data):
    import SimpleITK as sitk

    mask_img = sitk.GetImageFromArray(pred_data)
    cc = sitk.ConnectedComponent(mask_img, True)
    cc_sorted = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest_cc = sitk.Equal(cc_sorted, 1)
    return sitk.GetArrayFromImage(largest_cc).astype(np.uint8)


def save_t2w_slices_with_contours(t2w_data, gt_data, pred_data, patient_name, output_dir):
    patient_dir = output_dir / patient_name
    patient_dir.mkdir(parents=True, exist_ok=True)

    num_slices = t2w_data.shape[0]
    for z in range(num_slices):
        if z >= 16:
            break

        t2w_slice = t2w_data[z]
        gt_slice = gt_data[z].astype(bool)
        pred_slice = pred_data[z].astype(bool)

        plt.figure(figsize=(6, 6))
        plt.imshow(t2w_slice, cmap="gray")

        contours = measure.find_contours(gt_slice, 0.5)
        for contour in contours:
            plt.plot(
                contour[:, 1],
                contour[:, 0],
                "r-",
                linewidth=2,
                label="GT" if "GT" not in plt.gca().get_legend_handles_labels()[1] else "",
            )

        contours = measure.find_contours(pred_slice, 0.5)
        for contour in contours:
            plt.plot(
                contour[:, 1],
                contour[:, 0],
                "y-",
                linewidth=2,
                label="Pred" if "Pred" not in plt.gca().get_legend_handles_labels()[1] else "",
            )

        plt.title(f"T2W - Slice {z}")
        plt.axis("off")
        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:
            plt.legend(handles, ["GT", "Pred"], loc="lower right", fontsize=10)

        output_path = patient_dir / f"slice_{z:02d}.png"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
        plt.close()


def strip_nii_suffix(path):
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def load_nifti_as_dhw(path):
    data = nib.load(path).get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {path}")
    return np.transpose(data, (2, 1, 0))


def save_prediction_like_reference(pred_data, reference_path, output_path):
    reference_img = nib.load(reference_path)
    pred_xyz = np.transpose(pred_data, (2, 1, 0)).astype(np.uint8)
    pred_img = nib.Nifti1Image(pred_xyz, reference_img.affine, reference_img.header)
    nib.save(pred_img, output_path)


def resolve_fold_dirs(pred_root):
    fold_dirs = []
    for i in range(5):
        fold_dir = pred_root / f"fold{i}"
        if fold_dir.is_dir():
            fold_dirs.append(fold_dir)
    if fold_dirs:
        return fold_dirs
    return sorted(
        [path for path in pred_root.iterdir() if path.is_dir() and path.name.startswith("fold")]
    )


def resolve_data_subdirs(data_dir):
    if args.mode == "val":
        image_dir = data_dir / "imagesTs"
        label_dir = data_dir / "labelsTs"
    else:
        image_dir = data_dir / "imagesTs"
        label_dir = data_dir / "labelsTs"

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(f"Label directory does not exist: {label_dir}")

    return image_dir, label_dir


def evaluate_fold(fold_dir, image_dir, gt_dir, output_dir):
    pred_files = sorted(fold_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {fold_dir}")

    fold_parent_dir = output_dir / fold_dir.name
    fold_parent_dir.mkdir(parents=True, exist_ok=True)
    fold_output_dir = fold_parent_dir / args.output_dir
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    fold_slices_dir = fold_parent_dir / args.slices_dir
    fold_slices_dir.mkdir(parents=True, exist_ok=True)

    all_dice_scores = []
    all_miou_scores = []
    all_hd95_scores = []
    case_metrics = []

    for pred_path in tqdm(pred_files, desc=f"Testing {fold_dir.name}"):
        patient_name = strip_nii_suffix(pred_path)
        gt_path = gt_dir / pred_path.name
        image_path = image_dir / f"{patient_name}_0000.nii.gz"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found for {patient_name}: {gt_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {patient_name}: {image_path}")

        pred_numpy = (load_nifti_as_dhw(pred_path) > 0).astype(np.uint8)
        gt_numpy = (load_nifti_as_dhw(gt_path) > 0).astype(np.uint8)
        image_numpy = load_nifti_as_dhw(image_path)

        if args.keep_largest_cc:
            pred_numpy_to_save = keep_largest_connected_component_3d(pred_numpy)
        else:
            pred_numpy_to_save = pred_numpy

        preds_tensor = torch.from_numpy(pred_numpy_to_save).unsqueeze(0).unsqueeze(0)
        labels_tensor = torch.from_numpy(gt_numpy).unsqueeze(0).unsqueeze(0)

        dice_score = calculate_dice(preds_tensor, labels_tensor)
        miou_score = calculate_miou(preds_tensor, labels_tensor)
        hd95_score = calculate_hd95(pred_numpy_to_save, gt_numpy, percentile=95)

        all_dice_scores.append(dice_score)
        all_miou_scores.append(miou_score)
        all_hd95_scores.append(hd95_score)
        case_metrics.append(
            {
                "patient_name": patient_name,
                "dice_score": round(dice_score, 4),
                "miou_score": round(miou_score, 4),
                "hd95_score": round(hd95_score, 4),
            }
        )

        pred_output_path = fold_output_dir / f"{patient_name}_pred.nii.gz"
        save_prediction_like_reference(pred_numpy_to_save, gt_path, pred_output_path)
        save_t2w_slices_with_contours(
            image_numpy,
            gt_numpy,
            pred_numpy_to_save,
            patient_name,
            fold_slices_dir,
        )
        print(
            f"[{fold_dir.name}] Patient: {patient_name}, "
            f"Dice: {dice_score*100:.2f}, mIoU: {miou_score*100:.2f}, HD95: {hd95_score:.4f}"
        )

    avg_dice = round(np.mean(all_dice_scores), 4)
    avg_miou = round(np.mean(all_miou_scores), 4)
    avg_hd95 = round(np.mean(all_hd95_scores), 4)
    std_dice = round(np.std(all_dice_scores), 4)
    std_miou = round(np.std(all_miou_scores), 4)
    std_hd95 = round(np.std(all_hd95_scores), 4)
    dice_interval_stats = calculate_dice_intervals([case["dice_score"] for case in case_metrics])

    summary_data = {
        "prediction_root": str(fold_dir.parent),
        "fold": fold_dir.name,
        "data_dir": str(image_dir.parent),
        "image_dir": str(image_dir),
        "gt_dir": str(gt_dir),
        "keep_largest_cc": args.keep_largest_cc,
        "num_cases": len(case_metrics),
        "cases": case_metrics,
        "overall_metrics": {
            "average_dice": avg_dice,
            "average_miou": avg_miou,
            "average_hd95": avg_hd95,
            "std_dice": std_dice,
            "std_miou": std_miou,
            "std_hd95": std_hd95,
        },
        "dice_interval_statistics": dice_interval_stats,
    }

    summary_file = fold_parent_dir / f"{fold_dir.name}.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n=== {fold_dir.name} Results ===")
    print(f"Connected component filtering: {'Enabled' if args.keep_largest_cc else 'Disabled'}")
    print(f"Average Dice: {avg_dice*100:.2f} ± {std_dice*100:.2f}")
    print(f"Average mIoU: {avg_miou*100:.2f} ± {std_miou*100:.2f}")
    print(f"Average HD95: {avg_hd95:.4f} ± {std_hd95:.4f}")
    print(f"Predictions saved to: {fold_output_dir}")
    print(f"Slice images directory: {fold_slices_dir}")
    print(f"Case summary saved to: {summary_file}")


def main():
    pred_root = Path(args.model_path)
    data_dir = Path(args.data_dir)

    if not pred_root.is_dir():
        raise NotADirectoryError(f"Prediction root does not exist: {pred_root}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {data_dir}")

    image_dir, gt_dir = resolve_data_subdirs(data_dir)

    experiment_name = pred_root.name
    suffix = "_infer" if args.mode == "val" else "_test"
    parent_dir = Path(f"infer/{experiment_name}{suffix}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    output_dir = parent_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = resolve_fold_dirs(pred_root)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {pred_root}")

    print(f"Prediction root: {pred_root}")
    print(f"Data dir: {data_dir}")
    print(f"Image dir: {image_dir}")
    print(f"Ground truth dir: {gt_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Detected folds: {[fold_dir.name for fold_dir in fold_dirs]}")

    for fold_dir in fold_dirs:
        evaluate_fold(fold_dir, image_dir, gt_dir, output_dir)


if __name__ == "__main__":
    main()
