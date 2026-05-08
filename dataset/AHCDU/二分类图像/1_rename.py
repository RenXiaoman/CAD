import os
from pathlib import Path
import pandas as pd


def normalize_patient_name(name: str) -> str:
    """统一患者姓名格式: 按空格切分后, 每段首字母大写并直接拼接."""
    if pd.isna(name):
        return name
    parts = str(name).strip().split()
    return "".join(p.capitalize() for p in parts)


def rename_patient_dirs(root_dir):
    """
    遍历 root_dir 下的子目录，把目录名从 `chen er hua` 改成 `ChenErHua` 形式
    """
    for name in os.listdir(root_dir):
        old_path = os.path.join(root_dir, name)
        if os.path.isdir(old_path):
            new_name = normalize_patient_name(name)
            new_path = os.path.join(root_dir, new_name)
            
            # 如果新名字和旧名字不同，就重命名
            if old_path != new_path:
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)


def format_name(name: str) -> str:
    """格式化名字：每个词首字母大写，其余小写，并去掉空格"""
    return normalize_patient_name(name)

def rename_patient_names_with_pandas(file_path: str, output_path: str = None, sheet_name: str = 0):
    """
    用 pandas 修改 Excel 中 PatientName 列
    :param file_path: 输入的 Excel 文件路径
    :param output_path: 输出文件路径（默认覆盖原文件）
    :param sheet_name: 表单名或索引（默认第一个 sheet）
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if "PatientName" not in df.columns:
        raise ValueError("Excel 文件中没有找到 'PatientName' 列")

    # 修改 PatientName 列
    df["PatientName"] = df["PatientName"].apply(format_name)

    # 默认保存为新文件，避免覆盖原始文件
    if output_path is None:
        output_path = file_path.replace(".xlsx", "_modified.xlsx")

    df.to_excel(output_path, index=False)
    print(f"已保存修改后的文件到 {output_path}")
    
    
    
def verify_dirs_against_excel(dir_root: str, df: pd.DataFrame) -> None:
    """
    校验：目录名集合 是否都包含在 df['PatientName'] 集合中。
    打印缺失列表（双向可选）。
    """
    dir_root = Path(dir_root)

    # 取重命名后的目录名集合（只看一层子目录）
    dir_names = {p.name for p in dir_root.iterdir() if p.is_dir()}
    excel_names = set(df["PatientName"].dropna().astype(str))

    missing_in_excel = sorted(dir_names - excel_names)
    missing_on_disk = sorted(excel_names - dir_names)

    print("\n========== 校验结果 ==========")
    print(f"目录下患者数：{len(dir_names)}")
    print(f"Excel中患者数：{len(excel_names)}")

    if missing_in_excel:
        print(f"\n[WARN] 下列目录名未在 Excel 的 PatientName 中找到（{len(missing_in_excel)}）:")
        for n in missing_in_excel[:50]:
            print("  -", n)
        if len(missing_in_excel) > 50:
            print("  ...(仅显示前50项)")
    else:
        print("\n[OK] 所有目录名在 Excel 中均能找到。")

    # 如需反向也检查（可保留作为信息）
    if missing_on_disk:
        print(f"\n[INFO] 下列 Excel 的 PatientName 在磁盘目录中不存在（{len(missing_on_disk)}）:")
        for n in missing_on_disk[:50]:
            print("  -", n)
        if len(missing_on_disk) > 50:
            print("  ...(仅显示前50项)")


root_dir = "dataset/AHCDU/二分类图像/val"
excel_path = "dataset/AHCDU/二分类图像/val.xlsx"

# Rename dataset directories
# rename_patient_dirs(root_dir)

# Rename Excel patient names
# rename_patient_names_with_pandas(excel_path)



df_modified = excel_path.replace(".xlsx", "_modified.xlsx")  # 生成修改后的文件名
verify_dirs_against_excel(root_dir, pd.read_excel(df_modified))
