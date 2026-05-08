import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from deal_with import Case, PreprocessingSettings
import SimpleITK as sitk

BASE = Path('dataset/AHCDU/二分类图像')
NNUNET_SRC_DIR = BASE / "nnUNet_train_lesion_origin"
NNUNET_DEST_DIR = BASE / "nnUNet_train_lesion"

settings = PreprocessingSettings(
    matrix_size=[16, 256, 256],
    spacing=[3.5, 0.5, 0.5],
)

def process_case(patient_id: str):
    """处理单个病例的nnUNet格式数据并移动到目标目录"""
    # 构建源文件路径
    src_imagesTs_dir = NNUNET_SRC_DIR / "imagesTr"
    src_labelsTs_dir = NNUNET_SRC_DIR / "labelsTr"
    
    t2w_src = src_imagesTs_dir / f"{patient_id}_0000.nii.gz"
    adc_src = src_imagesTs_dir / f"{patient_id}_0001.nii.gz"
    dwi_src = src_imagesTs_dir / f"{patient_id}_0002.nii.gz"
    lbl_src = src_labelsTs_dir / f"{patient_id}.nii.gz"
    
    # 检查文件是否存在
    if not (t2w_src.exists() and adc_src.exists() and dwi_src.exists() and lbl_src.exists()):
        print(f"[ERROR] Patient {patient_id} missing files in nnUNet format.")
        return

    try:
        # 读取数据
        t2w_data = sitk.ReadImage(str(t2w_src), sitk.sitkFloat32)
        adc_data = sitk.ReadImage(str(adc_src), sitk.sitkFloat32)
        dwi_data = sitk.ReadImage(str(dwi_src), sitk.sitkFloat32)
        lbl_data = sitk.ReadImage(str(lbl_src), sitk.sitkUInt8)

        # 预处理
        case = Case(t2w_data, adc_data, dwi_data, lbl_data, settings, name=patient_id)
        case.preprocess()

        # 获取预处理结果
        t2w_data = case.get_t2w()
        adc_data = case.get_adc()
        dwi_data = case.get_dwi()
        lbl_data = case.get_lbl()

        # 构建目标文件路径
        dest_imagesTs_dir = NNUNET_DEST_DIR / "imagesTr"
        dest_labelsTs_dir = NNUNET_DEST_DIR / "labelsTr"
        
        # 创建目标目录（如果不存在）
        dest_imagesTs_dir.mkdir(parents=True, exist_ok=True)
        dest_labelsTs_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入到目标目录
        sitk.WriteImage(t2w_data, str(dest_imagesTs_dir / f"{patient_id}_0000.nii.gz"))
        sitk.WriteImage(adc_data, str(dest_imagesTs_dir / f"{patient_id}_0001.nii.gz"))
        sitk.WriteImage(dwi_data, str(dest_imagesTs_dir / f"{patient_id}_0002.nii.gz"))
        sitk.WriteImage(lbl_data, str(dest_labelsTs_dir / f"{patient_id}.nii.gz"))
        
        print(f"Processed: {patient_id}")
        
    except Exception as e:
        print(f"[ERROR] {patient_id} preprocessing failed: {e}")

def get_patient_ids():
    """从源labelsTs目录获取所有患者ID"""
    labelsTs_dir = NNUNET_SRC_DIR / "labelsTr"
    if not labelsTs_dir.exists():
        print("[ERROR] labelsTr directory not found!")
        return []
    
    patient_ids = []
    for file in labelsTs_dir.glob("*.nii.gz"):
        patient_id = file.stem.replace(".nii", "")  # 移除.nii.gz后缀
        patient_ids.append(patient_id)
    
    return patient_ids

# 并行运行任务
if __name__ == '__main__':
    patient_ids = get_patient_ids()
    if not patient_ids:
        print("No patient files found in nnUNet format!")
    else:
        print(f"Found {len(patient_ids)} patients to process")
        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_case, patient_ids), total=len(patient_ids)))