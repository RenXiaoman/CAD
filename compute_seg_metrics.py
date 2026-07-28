#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree


DATASET_LABEL_DIRS = {
    "AHCDU": Path("dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU/labelsTs"),
    "PICAI": Path("dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI/labelsTs"),
}


def load_gt_sitk_order(path: Path) -> np.ndarray:
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return (data > 0).astype(np.uint8)


def load_pred_sitk_order(path: Path, gt_shape: tuple[int, ...]) -> np.ndarray:
    data = nib.load(str(path)).get_fdata()
    data = (data > 0).astype(np.uint8)

    if data.shape == gt_shape:
        return data

    transposed = np.transpose(data, (2, 1, 0))
    if transposed.shape == gt_shape:
        return transposed

    raise ValueError(
        f"Prediction shape mismatch for {path}: pred={data.shape}, "
        f"transposed={transposed.shape}, gt={gt_shape}"
    )


def surface_points(binary_mask: np.ndarray) -> np.ndarray:
    struct = generate_binary_structure(3, 3)
    eroded = binary_erosion(binary_mask.astype(bool), struct)
    surface = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)
    return np.argwhere(surface > 0)


def hd95(pred: np.ndarray, gt: np.ndarray, percentile: int = 95) -> float:
    pred_surface = surface_points(pred)
    gt_surface = surface_points(gt)

    if len(pred_surface) == 0 and len(gt_surface) == 0:
        return 0.0
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 100.0

    pred_to_gt = cKDTree(gt_surface).query(pred_surface)[0]
    gt_to_pred = cKDTree(pred_surface).query(gt_surface)[0]
    value = np.percentile(np.concatenate([pred_to_gt, gt_to_pred]), percentile)

    if np.isnan(value) or np.isinf(value):
        return 100.0
    return float(value)


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    smooth = 1e-6

    tp = np.logical_and(pred_bool, gt_bool).sum(dtype=np.float64)
    fp = np.logical_and(pred_bool, ~gt_bool).sum(dtype=np.float64)
    fn = np.logical_and(~pred_bool, gt_bool).sum(dtype=np.float64)

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = pred_sum + gt_sum - tp

    return {
        "dice_score": float((2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)),
        "miou_score": float((tp + smooth) / (union + smooth)),
        "hd95_score": hd95(pred, gt),
        "sensitivity_score": float((tp + smooth) / (tp + fn + smooth)),
        "precision_score": float((tp + smooth) / (tp + fp + smooth)),
    }


def dataset_from_task(task_name: str) -> str:
    upper_name = task_name.upper()
    matches = [name for name in DATASET_LABEL_DIRS if name in upper_name]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer dataset from task name '{task_name}'. "
            "Task name must contain exactly one of: AHCDU, PICAI."
        )
    return matches[0]


def resolve_task_dir(task_name: str, infer_root: Path) -> Path:
    task_path = Path(task_name)
    candidates = []

    if task_path.exists():
        candidates.append(task_path)
    candidates.append(infer_root / task_name)
    if not task_name.endswith("_infer"):
        candidates.append(infer_root / f"{task_name}_infer")

    for candidate in candidates:
        if (candidate / "val_results").exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find val_results for '{task_name}'. Tried: {tried}")


def patient_names_from_summary(summary_path: Path) -> list[str]:
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text())
    return [case["patient_name"] for case in summary.get("cases", [])]


def patient_names_from_preds(val_results_dir: Path) -> list[str]:
    names = []
    for pred_path in sorted(val_results_dir.glob("*_pred.nii.gz")):
        names.append(pred_path.name.removesuffix("_pred.nii.gz"))
    return names


def mean_std(cases: list[dict[str, float]], key: str) -> tuple[float, float]:
    values = np.array([case[key] for case in cases], dtype=np.float64)
    return round(float(values.mean()), 4), round(float(values.std()), 4)


def format_mean_std(mean_value: float, std_value: float, scale: float = 1.0) -> str:
    return f"{mean_value * scale:.2f} ± {std_value * scale:.2f}"


def compute_task_metrics(
    task_name: str,
    infer_root: Path = Path("infer"),
    output_name: str = "A_Summary_5metrics.json",
    update_summary: bool = False,
) -> dict:
    dataset_name = dataset_from_task(task_name)
    labels_dir = DATASET_LABEL_DIRS[dataset_name]
    if not labels_dir.exists():
        raise FileNotFoundError(f"GT label directory not found: {labels_dir}")

    task_dir = resolve_task_dir(task_name, infer_root)
    val_results_dir = task_dir / "val_results"
    summary_path = val_results_dir / "A_Summary.json"

    patient_names = patient_names_from_summary(summary_path)
    if not patient_names:
        patient_names = patient_names_from_preds(val_results_dir)
    if not patient_names:
        raise FileNotFoundError(f"No prediction files found in {val_results_dir}")

    case_metrics = []
    for patient_name in patient_names:
        pred_path = val_results_dir / f"{patient_name}_pred.nii.gz"
        gt_path = labels_dir / f"{patient_name}.nii.gz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction not found: {pred_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"GT label not found: {gt_path}")

        gt = load_gt_sitk_order(gt_path)
        pred = load_pred_sitk_order(pred_path, gt.shape)
        metrics = calculate_metrics(pred, gt)
        case_metrics.append(
            {
                "patient_name": patient_name,
                **{key: round(value, 4) for key, value in metrics.items()},
            }
        )

    raw_overall_metrics = {}
    for key in (
        "dice_score",
        "miou_score",
        "hd95_score",
        "sensitivity_score",
        "precision_score",
    ):
        metric_name = key.removesuffix("_score")
        avg, std = mean_std(case_metrics, key)
        raw_overall_metrics[metric_name] = (avg, std)

    overall_metrics = {
        "dice": format_mean_std(*raw_overall_metrics["dice"], scale=100.0),
        "miou": format_mean_std(*raw_overall_metrics["miou"], scale=100.0),
        "hd95": format_mean_std(*raw_overall_metrics["hd95"]),
        "sensitivity": format_mean_std(*raw_overall_metrics["sensitivity"], scale=100.0),
        "precision": format_mean_std(*raw_overall_metrics["precision"], scale=100.0),
    }

    result = {
        "task_name": task_dir.name,
        "dataset": dataset_name,
        "pred_dir": str(val_results_dir),
        "gt_dir": str(labels_dir),
        "num_cases": len(case_metrics),
        "overall_metrics": overall_metrics,
        "cases": case_metrics,
    }

    out_path = summary_path if update_summary else val_results_dir / output_name
    out_path.write_text(json.dumps(result, indent=4, ensure_ascii=False) + "\n")
    result["output_path"] = str(out_path)
    return result


def compute_seg_metrics(task_names: Iterable[str]) -> list[dict]:
    """Compute 5 segmentation metrics for a list of infer task names."""
    return [compute_task_metrics(task_name) for task_name in task_names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Dice, mIoU, HD95, Sensitivity, and Precision from saved predictions."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Infer task names, e.g. SegGland_UNET_AHCDU_infer SegGland_UNETR_PICAI_infer",
    )
    parser.add_argument("--infer_root", type=Path, default=Path("infer"))
    parser.add_argument("--output_name", default="A_Summary_5metrics.json")
    parser.add_argument(
        "--update_summary",
        action="store_true",
        help="Overwrite val_results/A_Summary.json instead of writing a new JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task_name in args.tasks:
        result = compute_task_metrics(
            task_name=task_name,
            infer_root=args.infer_root,
            output_name=args.output_name,
            update_summary=args.update_summary,
        )
        metrics = result["overall_metrics"]
        print(f"\n{result['task_name']} ({result['dataset']}, n={result['num_cases']})")
        print(f"  Dice:        {metrics['dice']}")
        print(f"  mIoU:        {metrics['miou']}")
        print(f"  HD95:        {metrics['hd95']}")
        print(f"  Sensitivity: {metrics['sensitivity']}")
        print(f"  Precision:   {metrics['precision']}")
        print(f"  Saved:       {result['output_path']}")


if __name__ == "__main__":
    main()
