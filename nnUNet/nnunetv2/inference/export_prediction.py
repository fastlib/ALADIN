import os
from copy import deepcopy
from typing import Union, List
import time

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    st = time.time()
    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['data_shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    # predicted_logits = configuration_manager.resampling_fn_probabilities(
    #     predicted_logits,
    #     properties_dict['data_shape_after_cropping_and_before_resampling'],
    #     current_spacing,
    #     [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # )
    resample_time = time.time() - st

    st = time.time()
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    nonlin_time = time.time() - st
        
    # Ensure segmentation is numpy array
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    data_shape_before_cropping = properties_dict['data_shape_before_cropping']
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    transpose_backward = plans_manager.transpose_backward   
    # add axis at the beginning to adjust for the different label channels

    if label_manager.binary_classification:
        data_shape_before_cropping = (segmentation.shape[0], *data_shape_before_cropping) 
        slicer = (slice(None),) + slicer
        transpose_backward = [0] + [i + 1 for i in plans_manager.transpose_backward]
        

    segmentation_reverted_cropping = np.zeros(data_shape_before_cropping,
                                            dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)

    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    st = time.time()
    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(transpose_backward)
    transpose_time = time.time() - st

    if return_probabilities:
        # revert cropping
        st = time.time()
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'data_shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        revert_time = time.time() - st
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities, [resample_time, nonlin_time, transpose_time, revert_time]
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, [resample_time, nonlin_time, transpose_time, 0]

def convert_multiple_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):

    torch.set_num_threads(num_threads_torch)
    avg_logits = torch.mean(predicted_logits, dim=0)

    rets = []
    times = [0,0,0,0]
    for logit in predicted_logits:
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(logit, plans_manager, configuration_manager, label_manager, properties_dict, return_probabilities, num_threads_torch)
        rets.append(ret)
        times[0] += ret[-1][0]  # resample time
        times[1] += ret[-1][1]  # nonlin time
        times[2] += ret[-1][2]  # transpose time
        times[3] += ret[-1][3]  # revert time if probabilities are

    avg_ret = convert_predicted_logits_to_segmentation_with_correct_shape(avg_logits, plans_manager, configuration_manager, label_manager, properties_dict, return_probabilities, num_threads_torch)

    return avg_ret, rets, times
                 

def export_multiple_predictions_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    
    #export average prediction
    avg_logits = torch.mean(predicted_array_or_file, dim=0)
    export_prediction_from_logits(avg_logits, properties_dict, configuration_manager, plans_manager, dataset_json_dict_or_file, output_file_truncated, save_probabilities)

    for i, prediction in enumerate(predicted_array_or_file):
        ofile = output_file_truncated + "_fold" + str(i)
        export_prediction_from_logits(prediction, properties_dict, configuration_manager, plans_manager, dataset_json_dict_or_file, ofile, save_probabilities)

    


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)



def export_prediction_and_classification_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], classification_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    cls_final = classification_array_or_file.argmax(0)
    cls_final = cls_final.cpu().numpy()
    del classification_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + "_seg" + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    rw.write_seg(cls_final, output_file_truncated + '_cls' + dataset_json_dict_or_file['file_ending'], properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes) \
        -> None:
    # # needed for cascade
    # if isinstance(predicted, str):
    #     assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
    #                               "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(predicted)
    #     predicted = np.load(predicted)
    #     os.remove(del_file)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['seg_shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['seg_shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
    torch.set_num_threads(old_threads)
