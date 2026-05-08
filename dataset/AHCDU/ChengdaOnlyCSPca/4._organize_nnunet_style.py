import os
import shutil

def organize_nnunet_val_data(val_dir, output_dir):
    """
    Organize validation data into nnUNet format
    
    Args:
        val_dir: Path to the val directory containing patient cases
        output_dir: Path where nnUNet formatted data will be created
    """
    # Create output directories
    imagesTs_dir = os.path.join(output_dir, 'imagesTs')
    labelsTs_dir = os.path.join(output_dir, 'labelsTs')
    
    os.makedirs(imagesTs_dir, exist_ok=True)
    os.makedirs(labelsTs_dir, exist_ok=True)
    
    # Get all patient directories
    patients = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    for patient in patients:
        patient_dir = os.path.join(val_dir, patient)
        
        # Process T2W directory for label and t2w image
        t2w_dir = os.path.join(patient_dir, 'T2W')
        if os.path.exists(t2w_dir):
            for file in os.listdir(t2w_dir):
                file_path = os.path.join(t2w_dir, file)
                
                if file.startswith('1') and file.endswith('.nii.gz'):
                    # T2W image file
                    dest_name = f"{patient}_0000.nii.gz"
                    shutil.copy2(file_path, os.path.join(imagesTs_dir, dest_name))
                    print(f"Copied T2W: {file} -> {dest_name}")
                elif file.startswith('T2') and file.endswith('.nii.gz'):
                    # Label file
                    dest_name = f"{patient}.nii.gz"
                    shutil.copy2(file_path, os.path.join(labelsTs_dir, dest_name))
                    print(f"Copied label: {file} -> {dest_name}")
        
        # Process ADC directory for adc image
        adc_dir = os.path.join(patient_dir, 'ADC')
        if os.path.exists(adc_dir):
            for file in os.listdir(adc_dir):
                file_path = os.path.join(adc_dir, file)
                
                if file.startswith('1') and file.endswith('.nii.gz'):
                    # ADC image file
                    dest_name = f"{patient}_0001.nii.gz"
                    shutil.copy2(file_path, os.path.join(imagesTs_dir, dest_name))
                    print(f"Copied ADC: {file} -> {dest_name}")
        
        # Process DWI directory for dwi image
        dwi_dir = os.path.join(patient_dir, 'DWI')
        if os.path.exists(dwi_dir):
            for file in os.listdir(dwi_dir):
                file_path = os.path.join(dwi_dir, file)
                
                if file.startswith('1') and file.endswith('.nii.gz'):
                    # DWI image file
                    dest_name = f"{patient}_0002.nii.gz"
                    shutil.copy2(file_path, os.path.join(imagesTs_dir, dest_name))
                    print(f"Copied DWI: {file} -> {dest_name}")

if __name__ == "__main__":
    val_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/train"
    output_dir = "/media/lib/D/renxiaoman/PyCharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_train_origin"
    
    organize_nnunet_val_data(val_dir, output_dir)
    print("nnUNet data organization completed!")
