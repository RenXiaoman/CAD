import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def sizes_equal(size, target=(256, 256, 16)):
    """Strict size equality (X,Y,Z)."""
    return tuple(size) == tuple(target)

def spacings_close(sp, target=(0.5, 0.5, 3.5), atol=1e-3, rtol=1e-3):
    """Float compare for spacing (X,Y,Z)."""
    sp = np.array(sp, dtype=float)
    target = np.array(target, dtype=float)
    return np.allclose(sp, target, atol=atol, rtol=rtol)

def check_one_image(path, size_target, spacing_target):
    """Return (ok, size, spacing, err_msg)."""
    try:
        img = sitk.ReadImage(str(path))
        size = img.GetSize()        # (X,Y,Z)
        spacing = img.GetSpacing()  # (X,Y,Z)
    except Exception as e:
        return False, None, None, f"read_error: {e}"
    if not sizes_equal(size, size_target):
        return False, size, spacing, f"bad_size: {size}"
    if not spacings_close(spacing, spacing_target):
        return False, size, spacing, f"bad_spacing: {spacing}"
    return True, size, spacing, ""

def delete_files(paths):
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            print(f"  [WARN] 删除失败 {p.name}: {e}")

def clean_nnunet_cases(images_dir, labels_dir,
                       size_target=(256,256,16),
                       spacing_target=(0.5,0.5,3.5),
                       dry_run=False):
    """
    检查并清理 nnUNet 组织的数据：
      - labelsTs/<case>.nii.gz
      - imagesTs/<case>_0000.nii.gz (T2W)
      - imagesTs/<case>_0001.nii.gz (ADC)
      - imagesTs/<case>_0002.nii.gz (DWI)
    若四者任一不满足 Size/Spacing 要求，删除该 case 的所有已存在文件。
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    label_files = sorted(labels_dir.glob("*.nii.gz"))
    total_cases = len(label_files)
    deleted_cases = 0
    deleted_list = []
    kept_cases = 0

    print(f"共发现 {total_cases} 个 case（以 labelsTs 为基准）")
    for lab in label_files:
        case = lab.stem  # 不含扩展名
        img_t2w = images_dir / f"{case}_0000.nii.gz"
        img_adc = images_dir / f"{case}_0001.nii.gz"
        img_dwi = images_dir / f"{case}_0002.nii.gz"

        paths = [lab, img_t2w, img_adc, img_dwi]
        exist_map = {p.name: p.exists() for p in paths}

        # 先检查是否缺文件
        if not all(exist_map.values()):
            reason = "missing_files: " + ", ".join([n for n, ok in exist_map.items() if not ok])
            action = "删除" if not dry_run else "将删除"
            print(f"[DROP] {case}: {reason} -> {action}")
            if not dry_run:
                delete_files([p for p in paths if p.exists()])
            deleted_cases += 1
            deleted_list.append((case, reason))
            continue

        # 读并校验四个文件
        ok_all = True
        errs = []
        for p in paths:
            ok, size, sp, msg = check_one_image(p, size_target, spacing_target)
            if not ok:
                ok_all = False
                errs.append(f"{p.name}:{msg}")

        if not ok_all:
            reason = "; ".join(errs)
            action = "删除" if not dry_run else "将删除"
            print(f"[DROP] {case}: {reason} -> {action}")
            if not dry_run:
                delete_files(paths)
            deleted_cases += 1
            deleted_list.append((case, reason))
        else:
            kept_cases += 1

    print("\n=== 汇总 ===")
    print(f"应有 case：{total_cases}")
    print(f"删除 case：{deleted_cases}")
    print(f"保留 case：{kept_cases}")
    if deleted_cases > 0:
        print("\n删除列表（case: 原因）:")
        for case, reason in deleted_list[:50]:
            print(f"  - {case}: {reason}")
        if deleted_cases > 50:
            print("  ...(仅显示前 50 条)")

if __name__ == "__main__":
    imagesTs = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_train_origin/imagesTs"
    labelsTs = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_train_origin/labelsTs"

    # 真实删除：dry_run=False
    clean_nnunet_cases(imagesTs, labelsTs,
                       size_target=(256,256,16),
                       spacing_target=(0.5,0.5,3.5),
                       dry_run=False)