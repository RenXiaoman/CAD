import os
from pathlib import Path

def rename_sequence_dirs(root_dir):
    """
    重命名序列目录：
    - 包含 "ADC" 的目录改名为 "ADC"
    - 包含 "t2" 的目录改名为 "T2W"
    :param root_dir: 包含病人case目录的根目录
    """
    root_path = Path(root_dir)
    
    # 遍历所有病人case目录
    for patient_dir in root_path.iterdir():
        if patient_dir.is_dir():
            print(f"处理病人目录: {patient_dir.name}")
            
            # 遍历病人目录下的所有序列目录
            for sequence_dir in patient_dir.iterdir():
                if sequence_dir.is_dir():
                    old_name = sequence_dir.name
                    new_name = None
                    
                    # 检查目录名并确定新名称
                    if "ADC" in old_name:
                        new_name = "ADC"
                    elif "t2" in old_name.lower():  # 不区分大小写匹配t2
                        new_name = "T2W"
                    
                    # 如果需要重命名
                    if new_name and new_name != old_name:
                        new_path = sequence_dir.parent / new_name
                        
                        # 如果目标目录已存在，跳过
                        if new_path.exists():
                            print(f"  跳过: {old_name} -> {new_name} (目标已存在)")
                            continue
                        
                        try:
                            sequence_dir.rename(new_path)
                            print(f"  重命名: {old_name} -> {new_name}")
                        except Exception as e:
                            print(f"  重命名失败: {old_name} -> {new_name}, 错误: {e}")

if __name__ == "__main__":
    root_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/val"
    
    print("开始重命名序列目录...")
    rename_sequence_dirs(root_dir)
    print("重命名完成!")