import SimpleITK as sitk
import shutil
import os

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, save_json, nifti_files

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def generate_random_1d_signals(length):
     #generate random signal of sine waves and noise
    x = np.linspace(0, 1, length)
    signal = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 10 * x) + np.sin(2 * np.pi * 20 * x) + np.random.normal(0, 0.1, length)

    seg = np.zeros(length, dtype=int)
    #add a few random segments of ones
    for _ in range(5):
        start = np.random.randint(0, length)
        end = np.random.randint(start, length)
        signal[start:end] = 1
        seg[start:end] = 1

    return signal, seg

def sparsify_segmentation(seg: np.ndarray, label_manager: LabelManager, percent_of_slices: float) -> np.ndarray:
        assert label_manager.has_ignore_label, "This preprocessor only works with datasets that have an ignore label!"
        seg_new = np.ones_like(seg) * label_manager.ignore_label
        x, y, z = seg.shape
        # x
        num_slices = max(1, round(x * percent_of_slices))
        selected_slices = np.random.choice(x, num_slices, replace=False)
        seg_new[selected_slices] = seg[selected_slices]
        # y
        num_slices = max(1, round(y * percent_of_slices))
        selected_slices = np.random.choice(y, num_slices, replace=False)
        seg_new[:, selected_slices] = seg[:, selected_slices]
        # z
        num_slices = max(1, round(z * percent_of_slices))
        selected_slices = np.random.choice(z, num_slices, replace=False)
        seg_new[:, :, selected_slices] = seg[:, :, selected_slices]
        return seg_new


if __name__ == '__main__':
    dataset_name = 'IntegrationTest_1d'
    dataset_id = 995
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"

    try:
        existing_dataset_name = maybe_convert_to_dataset_name(dataset_id)
        if existing_dataset_name != dataset_name:
            raise FileExistsError(f"A different dataset with id {dataset_id} already exists :-(: {existing_dataset_name}. If "
                               f"you intent to delete it, remember to also remove it in nnUNet_preprocessed and "
                               f"nnUNet_results!")
    except RuntimeError:
        pass

    if isdir(join(nnUNet_raw, dataset_name)):
        shutil.rmtree(join(nnUNet_raw, dataset_name))

    #create a new folder for the dataset
    os.makedirs(os.path.join(join(nnUNet_raw, dataset_name), 'imagesTr'))
    os.makedirs(os.path.join(join(nnUNet_raw, dataset_name), 'labelsTr'))
    os.makedirs(os.path.join(join(nnUNet_raw, dataset_name), 'imagesTs'))

    n_train = 300
    n_test = 100

    # additionally optimize entire hippocampus region, remove Posterior
    dj = {}
    dj["channel_names"] = {"0": "signal"}
    dj['labels'] = {
        'background': 0,
        'foreground': 1
    }
    dj['numTraining'] = n_train
    dj['file_ending'] = '.npy'
    dj['regions_class_order'] = None
    save_json(dj, join(nnUNet_raw, dataset_name, 'dataset.json'), sort_keys=False)

    # now add ignore label to segmentation images
    np.random.seed(1234)
    lm = LabelManager(label_dict=dj['labels'], regions_class_order=dj['regions_class_order'])

    for s in range(n_train):
        signal, seg = generate_random_1d_signals(1024)
        recname = f'{s:03d}'
        np.save(os.path.join(nnUNet_raw, dataset_name, 'imagesTr', f'case_{recname}_0000.npy'), signal)
        np.save(os.path.join(nnUNet_raw, dataset_name, 'labelsTr', f'case_{recname}.npy'), seg)

    for s in range(n_test):
        signal, _ = generate_random_1d_signals(1024)
        recname = f'{s:03d}'
        np.save(os.path.join(nnUNet_raw, dataset_name, 'imagesTs', f'case_{recname}_0000.npy'), signal)

