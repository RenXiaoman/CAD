from typing import Tuple, Union, List
import os

from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          citation: Union[List[str], str] = None,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None,
                          reference: str = None,
                          release: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None,
                          license: str = 'Whoever converted this dataset was lazy and didn\'t look it up!',
                          converted_by: str = "Please enter your name, especially when sharing datasets with others in a common infrastructure!",
                          **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."

    # nnFormer uses the old nnU-Net dataset.json schema:
    # modality: {"0": "T2"}
    # labels: {"0": "background", "1": "gland"}
    modality = {}
    for key, value in channel_names.items():
        modality[str(key)] = value

    labels_old_schema = {}
    for label_name, label_value in labels.items():
        if isinstance(label_value, (tuple, list)):
            value = tuple(int(i) for i in label_value)
            labels_old_schema[str(value)] = label_name
        else:
            labels_old_schema[str(int(label_value))] = label_name

    # 构建training列表 - 基于实际的labelsTr文件
    training_list = []
    labels_tr_dir = os.path.join(output_folder, "labelsTr")
    if os.path.exists(labels_tr_dir):
        for filename in sorted(os.listdir(labels_tr_dir)):
            if filename.endswith(file_ending) and '_' not in filename.replace(file_ending, ''):
                # 获取基础文件名（去掉.nii.gz）
                base_name = filename.replace(file_ending, '')
                training_list.append({
                    'image': f"./imagesTr/{base_name}_0000{file_ending}",
                    'label': f"./labelsTr/{filename}"
                })

    # 构建test列表 - 基于imagesTs目录中的文件
    test_cases = set()
    test_dir = os.path.join(output_folder, "imagesTs")
    if os.path.exists(test_dir):
        for filename in sorted(os.listdir(test_dir)):
            if filename.endswith(file_ending):
                base_name = filename.replace(file_ending, '')
                if '_' in base_name:
                    case_name = base_name.rsplit('_', 1)[0]
                else:
                    case_name = base_name
                test_cases.add(case_name)

    test_list = [f"./imagesTs/{case_name}{file_ending}" for case_name in sorted(test_cases)]

    dataset_json = {
        'modality': modality,
        'labels': labels_old_schema,
        'numTraining': len(training_list) if training_list else num_training_cases,
        'numTest': len(test_list),
        'file_ending': file_ending,
        'licence': license,
        'converted_by': converted_by,
        'training': training_list,
        'test': test_list
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if citation is not None:
        dataset_json['citation'] = citation
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


generate_dataset_json(
    output_folder="DATASET/nnFormer_raw/nnFormer_raw_data/Task130_ProstateAHCDU",
    channel_names={0:'T2'},
    labels={'background':0, 'gland':1},
    num_training_cases=357,
    file_ending='.nii.gz',
    dataset_name='AHCDU',
    description='Prostate MRI…'
)
