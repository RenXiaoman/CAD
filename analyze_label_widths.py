from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DATASETS = {
    "AHCDU": Path("dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU"),
    "PICAI": Path("dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI"),
}


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    percentiles = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    return {
        "min": float(arr.min()),
        "p05": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def format_summary(name: str, values: list[float], unit: str) -> str:
    stats = summarize(values)
    if not stats:
        return f"{name}: no valid values\n"
    return (
        f"{name} ({unit}): "
        f"min={stats['min']:.2f}, p05={stats['p05']:.2f}, p10={stats['p10']:.2f}, "
        f"p25={stats['p25']:.2f}, median={stats['median']:.2f}, p75={stats['p75']:.2f}, "
        f"p90={stats['p90']:.2f}, p95={stats['p95']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}\n"
    )


def analyze_dataset(dataset_name: str, dataset_root: Path, split: str) -> str:
    label_dir = dataset_root / split
    label_paths = sorted(label_dir.glob("*.nii.gz"))
    if not label_paths:
        return f"Dataset {dataset_name}: no labels found in {label_dir}\n"

    bbox_z_vox: list[float] = []
    bbox_y_vox: list[float] = []
    bbox_x_vox: list[float] = []
    bbox_z_mm: list[float] = []
    bbox_y_mm: list[float] = []
    bbox_x_mm: list[float] = []
    min_bbox_vox: list[float] = []
    min_bbox_mm: list[float] = []
    max_inside_dist_vox: list[float] = []
    p10_inside_dist_vox: list[float] = []
    median_inside_dist_vox: list[float] = []
    foreground_voxels: list[float] = []
    empty_cases: list[str] = []

    for path in label_paths:
        image = sitk.ReadImage(str(path))
        spacing_xyz = np.asarray(image.GetSpacing(), dtype=np.float64)
        arr = sitk.GetArrayFromImage(image) > 0  # z, y, x
        if not arr.any():
            empty_cases.append(path.name)
            continue

        coords = np.argwhere(arr)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        widths_vox = np.asarray(
            [z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1],
            dtype=np.float64,
        )
        spacing_zyx = spacing_xyz[::-1]
        widths_mm = widths_vox * spacing_zyx

        bbox_z_vox.append(widths_vox[0])
        bbox_y_vox.append(widths_vox[1])
        bbox_x_vox.append(widths_vox[2])
        bbox_z_mm.append(widths_mm[0])
        bbox_y_mm.append(widths_mm[1])
        bbox_x_mm.append(widths_mm[2])
        min_bbox_vox.append(float(widths_vox.min()))
        min_bbox_mm.append(float(widths_mm.min()))
        foreground_voxels.append(float(arr.sum()))

        dist = ndimage.distance_transform_edt(arr)
        inside_dist = dist[arr]
        max_inside_dist_vox.append(float(inside_dist.max()))
        p10_inside_dist_vox.append(float(np.percentile(inside_dist, 10)))
        median_inside_dist_vox.append(float(np.median(inside_dist)))

    lines = [
        f"Dataset: {dataset_name}\n",
        f"Root: {dataset_root}\n",
        f"Split: {split}\n",
        f"Labels: {len(label_paths)}, non-empty: {len(label_paths) - len(empty_cases)}, empty: {len(empty_cases)}\n",
        "\nBounding-box width by axis\n",
        format_summary("z/depth", bbox_z_vox, "vox"),
        format_summary("y/height", bbox_y_vox, "vox"),
        format_summary("x/width", bbox_x_vox, "vox"),
        format_summary("z/depth", bbox_z_mm, "mm"),
        format_summary("y/height", bbox_y_mm, "mm"),
        format_summary("x/width", bbox_x_mm, "mm"),
        format_summary("minimum bbox axis", min_bbox_vox, "vox"),
        format_summary("minimum bbox axis", min_bbox_mm, "mm"),
        "\nForeground and inside-distance thickness proxy\n",
        format_summary("foreground volume", foreground_voxels, "voxels"),
        format_summary("max inside distance", max_inside_dist_vox, "vox"),
        format_summary("p10 inside distance", p10_inside_dist_vox, "vox"),
        format_summary("median inside distance", median_inside_dist_vox, "vox"),
    ]
    if empty_cases:
        lines.append("\nEmpty cases:\n")
        lines.extend(f"- {case}\n" for case in empty_cases[:20])
        if len(empty_cases) > 20:
            lines.append(f"... and {len(empty_cases) - 20} more\n")
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="labelsTr")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/model_parameter_list/label_width_stats.txt"))
    args = parser.parse_args()

    reports = [analyze_dataset(name, root, args.split) for name, root in DATASETS.items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(reports), encoding="utf-8")
    print(args.out)
    print("\n".join(reports))


if __name__ == "__main__":
    main()
