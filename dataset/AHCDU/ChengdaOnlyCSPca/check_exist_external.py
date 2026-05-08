import os
import re
from pathlib import Path
from typing import Dict

def format_case_name(name: str) -> str:
    """把 'CHEN ZHI MING' -> 'ChenZhiMing'，去空格、每词首字母大写。"""
    parts = re.split(r"\s+", name.strip())
    return "".join(p.capitalize() for p in parts if p)

def norm_key(s: str) -> str:
    """归一化键：去空格并大写，便于 images 与 labels 配对。"""
    return re.sub(r"\s+", "", s).upper()

def build_label_map(labels_dir: Path) -> Dict[str, str]:
    """
    从 labelsTs 构建映射：归一化键 -> 标准 case 名（已是 CamelCase）
    例如: 'CHENZHIMING' -> 'ChenZhiMing'
    """
    mp = {}
    for f in labels_dir.glob("*.nii.gz"):
        case = f.stem  # 已经是标准名
        mp[norm_key(case)] = case
    return mp

def rename_images_to_match_labels(images_dir: str, labels_dir: str, dry_run: bool = False):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    assert images_dir.exists(), f"imagesTs 不存在：{images_dir}"
    assert labels_dir.exists(), f"labelsTs 不存在：{labels_dir}"

    # 以 labelsTs 为权威
    label_map = build_label_map(labels_dir)

    # 匹配形如 'XXX_0000.nii.gz'
    pat = re.compile(r"^(.+)_([0-9]{4})\.nii\.gz$", re.IGNORECASE)

    changed, skipped, warned = 0, 0, 0

    for f in sorted(images_dir.glob("*.nii.gz")):
        m = pat.match(f.name)
        if not m:
            print(f"[WARN] 非预期命名，跳过：{f.name}")
            warned += 1
            continue

        old_case, suffix = m.group(1), m.group(2)  # old_case='CHEN ZHI MING', suffix='0001'

        # 先按 label 对齐；没有就用规则格式化
        key = norm_key(old_case)
        if key in label_map:
            new_case = label_map[key]
        else:
            new_case = format_case_name(old_case)

        new_name = f"{new_case}_{suffix}.nii.gz"
        dst = f.with_name(new_name)

        if f.name == new_name:
            # 已经是正确命名
            skipped += 1
            continue

        if dst.exists():
            # 目标已存在且不是同一文件，避免覆盖
            print(f"[CONFLICT] 目标已存在，跳过：{f.name} -> {new_name}")
            warned += 1
            continue

        print(f"[RENAME] {f.name} -> {new_name}")
        if not dry_run:
            f.rename(dst)
        changed += 1

    print("\n=== 完成 ===")
    print(f"改名：{changed} 个，已正确：{skipped} 个，警告/跳过：{warned} 个")

if __name__ == "__main__":
    images_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_external/imagesTs"
    labels_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_external/labelsTs"

    # 先试运行（只打印不改动）：dry_run=True
    # 确认输出无误后把 dry_run 改成 False 真正执行
    rename_images_to_match_labels(images_dir, labels_dir, dry_run=False)