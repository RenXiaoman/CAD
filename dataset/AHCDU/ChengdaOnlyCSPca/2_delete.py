import os
import pandas as pd
from pathlib import Path

def delete_non_patient_dirs(root_dir, excel_file):
    """
    根据Excel文件中的分类信息删除非病人目录
    :param root_dir: 包含病人目录的根目录
    :param excel_file: Excel文件路径，包含PatientName和分类列
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)
    
    # 检查必要的列是否存在
    if "PatientName" not in df.columns or "分类" not in df.columns:
        raise ValueError("Excel文件中必须包含'PatientName'和'分类'列")
    
    # 获取分类为0（非病人）的患者名称
    non_patients = df[df["分类"] == 0]["PatientName"].dropna().astype(str).tolist()
    
    print(f"找到 {len(non_patients)} 个非病人目录需要删除")
    
    # 遍历目录并删除非病人目录
    root_path = Path(root_dir)
    deleted_count = 0
    
    for patient_name in non_patients:
        patient_dir = root_path / patient_name
        if patient_dir.exists() and patient_dir.is_dir():
            try:
                # 删除目录及其内容
                import shutil
                shutil.rmtree(patient_dir)
                print(f"已删除非病人目录: {patient_dir}")
                deleted_count += 1
            except Exception as e:
                print(f"删除目录 {patient_dir} 时出错: {e}")
        else:
            print(f"目录不存在: {patient_dir}")
    
    print(f"总共删除了 {deleted_count} 个非病人目录")

if __name__ == "__main__":
    root_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/val"
    excel_file = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/二分类模型内部测试.xlsx"
    
    # 执行删除操作
    delete_non_patient_dirs(root_dir, excel_file)