"""Run ResGNet FullPICAI validation with ResGNet's single-channel output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gland_dataset_nnunet import Lits_DataSet
from models.ResGNet import ResGNet, VNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ResGNet on Dataset141_FullPICAI.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("checkpoints_FullPICAI/SegGland_ResGNet_FullPICAI/best_dice_model.pth"),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("dataset/PI-CAI/nnUNet_raw/Dataset141_FullPICAI"),
    )
    parser.add_argument("--gpu_ids", type=str, default="3")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--keep_largest_cc",
        action="store_true",
        help="Keep only the largest 3D connected component before metrics and saving.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("infer_FullPICAI/SegGland_ResGNet_FullPICAI_infer/val_results"),
    )
    return parser.parse_args()


def calculate_dice_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    smooth = 1e-6
    outputs_flat = outputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (outputs_flat * targets_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        outputs_flat.sum() + targets_flat.sum() + smooth
    )


def calculate_dice(preds: torch.Tensor, targets: torch.Tensor) -> float:
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    pred_sum = preds_flat.sum()
    target_sum = targets_flat.sum()
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()


def calculate_miou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def format_mean_std(mean_value: float, std_value: float, scale: float = 1.0) -> str:
    return f"{mean_value * scale:.2f} ± {std_value * scale:.2f}"


def save_prediction(pred: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference = sitk.ReadImage(str(reference_path))
    pred_image = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_image.CopyInformation(reference)
    sitk.WriteImage(pred_image, str(output_path))


def keep_largest_connected_component_3d(pred: np.ndarray) -> np.ndarray:
    labeled, num_features = label(pred > 0)
    if num_features == 0:
        return pred.astype(np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    return (labeled == largest_label).astype(np.uint8)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.gpu_ids != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Keep largest connected component: {args.keep_largest_cc}")

    model = VNet(ResGNet).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_dataset = Lits_DataSet(
        args.data_path,
        "imagesTs",
        "labelsTs",
        enable_augmentation=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Predictions: {args.output_dir}")

    losses: list[float] = []
    dice_scores: list[float] = []
    miou_scores: list[float] = []
    case_metrics = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating"):
            inputs, labels, patient_names = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = calculate_dice_loss(outputs, labels)
            preds = (outputs > args.threshold).float()
            losses.append(loss.item())

            for i, patient_name in enumerate(patient_names):
                single_label = labels[i : i + 1]

                pred_array = preds[i].squeeze().cpu().numpy().astype(np.uint8)
                if args.keep_largest_cc:
                    pred_array = keep_largest_connected_component_3d(pred_array)
                single_pred = torch.from_numpy(pred_array).unsqueeze(0).unsqueeze(0).to(device)

                dice_score = calculate_dice(single_pred, single_label)
                miou_score = calculate_miou(single_pred, single_label)
                dice_scores.append(dice_score)
                miou_scores.append(miou_score)

                reference_path = args.data_path / "imagesTs" / f"{patient_name}_0000.nii.gz"
                pred_path = args.output_dir / f"{patient_name}_pred.nii.gz"
                save_prediction(pred_array, reference_path, pred_path)

                case_metrics.append(
                    {
                        "patient_name": patient_name,
                        "dice_score": round(dice_score, 4),
                        "miou_score": round(miou_score, 4),
                        "pred_path": str(pred_path),
                    }
                )

    avg_loss, std_loss = mean_std(losses)
    avg_dice, std_dice = mean_std(dice_scores)
    avg_miou, std_miou = mean_std(miou_scores)

    summary = {
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "threshold": args.threshold,
        "keep_largest_cc": args.keep_largest_cc,
        "num_cases": len(case_metrics),
        "overall_metrics": {
            "val_loss": format_mean_std(avg_loss, std_loss),
            "dice": format_mean_std(avg_dice, std_dice, scale=100.0),
            "miou": format_mean_std(avg_miou, std_miou, scale=100.0),
        },
        "cases": case_metrics,
    }

    summary_path = args.output_dir / "A_Summary.json"
    summary_path.write_text(json.dumps(summary, indent=4) + "\n")

    print("\n=== Overall Results ===")
    print(f"Average Val Loss: {avg_loss:.6f} ± {std_loss:.6f}")
    print(f"Average Dice: {avg_dice * 100:.2f} ± {std_dice * 100:.2f}")
    print(f"Average mIoU: {avg_miou * 100:.2f} ± {std_miou * 100:.2f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
