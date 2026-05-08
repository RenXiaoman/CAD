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
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # Convert labels from {"background": 0, "lesion": 1} to {"0": "background", "1": "lesion"}
    new_labels = {}
    for label_name, label_value in labels.items():
        if isinstance(label_value, (tuple, list)):
            # For region-based training, convert each value in the tuple
            new_labels[str(label_value[0])] = label_name
        else:
            new_labels[str(int(label_value))] = label_name
    labels = new_labels

    # 构建training列表 - 基于实际的labelsTr文件
    training_list = []
    labels_tr_dir = os.path.join(output_folder, "labelsTr")
    if os.path.exists(labels_tr_dir):
        for filename in sorted(os.listdir(labels_tr_dir)):
            if filename.endswith(file_ending):
                base_name = filename.replace(file_ending, '')
                image_file = f"./imagesTr/{base_name}_0000{file_ending}"
                full_path = os.path.join(output_folder, image_file.lstrip('./'))

                if os.path.exists(full_path):
                    training_list.append({
                        'image': f"./imagesTr/{base_name}{file_ending}",
                        'label': f"./labelsTr/{filename}"
                    })

    # 构建test列表 - 基于labelsTs目录中的文件
    test_list = []
    labels_ts_dir = os.path.join(output_folder, "labelsTs")
    if os.path.exists(labels_ts_dir):
        for filename in sorted(os.listdir(labels_ts_dir)):
            if filename.endswith(file_ending):
                base_name = filename.replace(file_ending, '')
                image_file = f"./imagesTs/{base_name}_0000{file_ending}"
                full_path = os.path.join(output_folder, image_file.lstrip('./'))

                if os.path.exists(full_path):
                    test_list.append({
                        'image': f"./imagesTs/{base_name}{file_ending}",
                        'label': f"./labelsTs/{filename}"
                    })

    dataset_json = {
        'modality': channel_names,
        'labels': labels,
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
    output_folder="/home/ikun_server/clib/PycharmProjects/nnFormer/DATASET/nnFormer_raw/nnFormer_raw_data/Task131_ProstatePI-CAI",
    channel_names={0: 'T2'},
    labels={'background': 0, 'gland': 1},
    num_training_cases=357,
    file_ending='.nii.gz',
    dataset_name='PICAI',
    description='Prostate MRI…'
)
