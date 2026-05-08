from __future__ import annotations

import json
from pathlib import Path


# 中文：这个脚本用于读取分割评估结果 JSON，提取 overall_metrics，并按“均值±标准差”格式输出。
# English: This script reads a segmentation results JSON, extracts overall_metrics, and prints them as "mean±std".
# 用法 / Usage: python read_summary_metrics.py --json_dir path/to/your.json
def format_metrics(summary_path: str | Path) -> str:
    """Read `overall_metrics` from a summary JSON and format it."""
    summary_path = Path(summary_path)
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["overall_metrics"]

    def fmt(value: float, scale: float) -> str:
        return f"{value * scale:.2f}"

    dice = f"{fmt(metrics['average_dice'], 100)}±{fmt(metrics['std_dice'], 100)}"
    miou = f"{fmt(metrics['average_miou'], 100)}±{fmt(metrics['std_miou'], 100)}"
    hd95 = f"{fmt(metrics['average_hd95'], 1)}±{fmt(metrics['std_hd95'], 1)}"

    return f"Dice:  {dice} \nmIoU:  {miou} \nHD95:  {hd95}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", nargs="?", type=str, help="Path to the summary JSON")
    parser.add_argument("--json_dir", type=str, help="Path to the summary JSON")
    args = parser.parse_args()
    json_path = args.json_dir or args.summary_json
    if not json_path:
        raise SystemExit("Please provide `--json_dir path/to/your.json` or a positional summary_json.")
    print(format_metrics(json_path))
