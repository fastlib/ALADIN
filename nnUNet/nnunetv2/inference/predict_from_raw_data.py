import inspect
import itertools
import multiprocessing
import os
import psutil
from copy import deepcopy
from time import sleep
import time
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, PreprocessAdapterFromNpyWithClassification, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, export_multiple_predictions_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape, convert_multiple_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.vram_available = 0
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)
        else:
            print('Not using torch.compile. Set nnUNet_compile=True to enable it.')

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export, num_images=len(image_or_list_of_images))

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes,
                                   num_images: int = None):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            with tqdm(total=num_images, desc="Processing files") as pbar:
                for preprocessed in tqdm(data_iterator, desc="Processing files"):
                    data = preprocessed['data']
                    if isinstance(data, str):
                        delfile = data
                        data = torch.from_numpy(np.load(data))
                        os.remove(delfile)

                    ofile = preprocessed['ofile']
                    # if ofile is not None:
                    #     print(f'\nPredicting {os.path.basename(ofile)}:')
                    # else:
                    #     print(f'\nPredicting image of shape {data.shape}:')

                    # print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                    properties = preprocessed['data_properties']

                    # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                    # npy files
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                    prediction = self.predict_logits_from_preprocessed_data(data, average=False).cpu()

                    if ofile is not None:
                        # this needs to go into background processes
                        # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                        #                               self.dataset_json, ofile, save_probabilities)
                        #print('sending off prediction to background worker for resampling and export')
                        if prediction.ndim == 5:
                            r.append(
                                export_pool.starmap_async(
                                    export_multiple_predictions_from_logits,
                                    ((prediction, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, ofile, save_probabilities),)
                                )
                            )
                        else:
                            assert prediction.ndim == 4 # c z y x
                            r.append(
                                export_pool.starmap_async(
                                    export_prediction_from_logits,
                                    ((prediction, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, ofile, save_probabilities),)
                                )
                            )
                    else:
                        # convert_predicted_logits_to_segmentation_with_correct_shape(
                        #             prediction, self.plans_manager,
                        #              self.configuration_manager, self.label_manager,
                        #              properties,
                        #              save_probabilities)

                        #print('sending off prediction to background worker for resampling')
                        if prediction.ndim == 5:
                            r.append(
                                export_pool.starmap_async(
                                    convert_multiple_predicted_logits_to_segmentation_with_correct_shape, (
                                        (prediction, self.plans_manager,
                                        self.configuration_manager, self.label_manager,
                                        properties,
                                        save_probabilities),)
                                )
                            )
                        else:
                            assert prediction.ndim == 4 # c z y x
                            r.append(
                                export_pool.starmap_async(
                                    convert_predicted_logits_to_segmentation_with_correct_shape, (
                                        (prediction, self.plans_manager,
                                        self.configuration_manager, self.label_manager,
                                        properties,
                                        save_probabilities),)
                                )
                            )
                    # if ofile is not None:
                    #     print(f'done with {os.path.basename(ofile)}')
                    # else:
                    #     print(f'\nDone with image of shape {data.shape}:')

                    # Update the tqdm progress bar after each file processed
                    pbar.update(1)

            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data'], average=False).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            if predicted_logits.ndim == 5:
                export_multiple_predictions_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                            self.plans_manager, self.dataset_json, output_file_truncated,
                                            save_or_return_probabilities)
            else:
                assert predicted_logits.ndim == 4 # c z y x
                export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                            self.plans_manager, self.dataset_json, output_file_truncated,
                                            save_or_return_probabilities)
        else:
            if predicted_logits.ndim == 5:
                ret = convert_multiple_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            else:
                assert predicted_logits.ndim == 4 # c z y x
                ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, average=True) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None if average else []

        for idx, params in enumerate(self.list_of_parameters):

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if average:
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction.append(self.predict_sliding_window_return_logits(data).to('cpu'))

        if len(self.list_of_parameters) > 1 and average:
            prediction /= len(self.list_of_parameters)
        else:
            prediction = torch.stack(prediction)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        dim = len(self.configuration_manager.patch_size)

        if dim == 1:
            #print(image_size[2:], self.tile_step_size, self.configuration_manager.patch_size)
            steps = compute_steps_for_sliding_window(image_size[2:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)

            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')

            for d in range(image_size[0]):
                for sx in steps[0]:
                    slicers.append(
                        tuple([slice(None), d, 0, slice(sx, sx + self.configuration_manager.patch_size[0])]))

        elif dim == 2:
        #if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        elif dim == 3:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))

        else:
            raise NotImplementedError('This function only supports 1D, 2D and 3D images')

        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

class nnUNetWithClassificationPredictor(nnUNetPredictor):

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction_seg, prediction_cls = self.network(x)

        if self.use_mirroring:
            prediction_seg += self.network(-x)[0]
            prediction_seg /= 2

        for i in range(len(prediction_cls)):
            prediction_cls[i] = prediction_cls[i].repeat_interleave(prediction_seg.shape[2] // prediction_cls[i].shape[0], dim=0)
            #print(prediction_cls[i].shape)
            prediction_cls[i] = prediction_cls[i].permute(1,0)
            prediction_cls[i] = prediction_cls[i].reshape(1, -1, *prediction_seg.shape[2:])
            #print(prediction_cls[i].shape)
            #prediction_cls[i] = prediction_cls[i].unsqueeze(-1)
            #prediction_cls[i] = prediction_cls[i].repeat(1, 1, *prediction_seg.shape[2:])
            #print("Cls shape", prediction_cls[i].shape)

        return prediction_seg, prediction_cls

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_seg_logits = predicted_cls_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_seg_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            
            num_classes_classification_branches = self.configuration_manager.configuration['architecture']['arch_kwargs']['num_classes_classification_branch']
            #num_classification_branches = self.configuration_manager
            predicted_cls_logits = []
            for nc in num_classes_classification_branches:
                predicted_cls_logits.append(torch.zeros((nc, *data.shape[1:]), dtype=torch.half, device=results_device))

            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                res = self._internal_maybe_mirror_and_predict(workon)
                prediction_seg = res[0][0].to(results_device)
                prediction_cls = [r[0].to(results_device) for r in res[1]]

                if self.use_gaussian:
                    prediction_seg *= gaussian
                    for i in range(len(prediction_cls)):
                        prediction_cls[i] *= gaussian

                predicted_seg_logits[sl] += prediction_seg
                for i in range(len(prediction_cls)):
                    #print(predicted_cls_logits[i].shape, prediction_cls[i].shape)
                    predicted_cls_logits[i][sl] += prediction_cls[i]
                n_predictions[sl[1:]] += gaussian

            predicted_seg_logits /= n_predictions
            for i in range(len(predicted_cls_logits)):
                predicted_cls_logits[i] /= n_predictions

            # check for infs
            if torch.any(torch.isinf(predicted_seg_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_seg_logits, predicted_cls_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_seg_logits, predicted_cls_logits


    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
        
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                        'constant', {'value': 0}, True,
                                                        None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_seg_logits, predicted_cls_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                            self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_seg_logits, predicted_cls_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_seg_logits, predicted_cls_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                        self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_seg_logits = predicted_seg_logits[(slice(None), *slicer_revert_padding[1:])]
            for i in range(len(predicted_cls_logits)):
                predicted_cls_logits[i] = predicted_cls_logits[i][(slice(None), *slicer_revert_padding[1:])]


        return predicted_seg_logits, predicted_cls_logits

    def predict_sliding_window_on_batched_data(self, data: torch.Tensor, obj_data, do_on_device: bool = True):

        results_device = self.device if do_on_device else torch.device('cpu')

        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)
        else:
            gaussian = 1

        # empty_cache(self.device)
            
        with torch.no_grad():

            empty_cache(self.device)

            assert isinstance(data, torch.Tensor)
            data_size = torch.cuda.memory_reserved() / (1024**2) if self.device.type == 'cuda' else 0
            self.network = self.network.to(self.device)
            self.network.eval()

            #print("After model load: ", torch.cuda.memory_reserved() / (1024**2))

            num_cb = len(self.configuration_manager.configuration['architecture']['arch_kwargs']['num_classes_classification_branch'])
            predicted_batch_seg_logits = torch.empty((len(data), self.label_manager.num_segmentation_heads, data.shape[-1]), dtype=torch.half, device=self.device)
            predicted_batch_cls_logits = [torch.empty((len(data), num_cb), dtype= torch.half, device=self.device) for _ in range(num_cb)]

            #print("After results reserved: ", torch.cuda.memory_reserved() / (1024**2))

            reserved = torch.cuda.memory_reserved() / (1024**2) if self.device.type == 'cuda' else 0
            #mem_for_results = (len(data) *data.shape[-1] * 10 * 4) / (1024**2)  # 5mb per sample + 20% overhead
            #reserved += mem_for_results

            working_mem_per_sample = (data.shape[-1] * 64 * 4 * 11) / (1024**2)  # 5mb per sample
            mem_per_sample = working_mem_per_sample + 0.2 * working_mem_per_sample  # 20% overhead
            max_batch_size = min(len(data), int(np.floor(((self.vram_available - reserved) / mem_per_sample)/100)*100)) #ong 5mb per sample + 20% overhead
            #max_batch_size = 1000
            if max_batch_size <= 0:
                # Estimate says VRAM is very tight - start small instead of disabling
                # batching altogether (the previous fallback to len(data) did the opposite
                # of what's needed exactly when memory is most constrained).
                max_batch_size = min(len(data), 8)

            #print("Max batch size: ", max_batch_size)

            start = 0
            batch_idx = 0
            while start < len(data):
                current_batch_size = min(max_batch_size, len(data) - start)

                while True:
                    if self.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()

                    if self.verbose:
                        print(f'Processing batch {batch_idx+1} (size {current_batch_size}, offset {start}/{len(data)})')

                    try:
                        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                            predicted_batch_seg_logits_b, predicted_batch_cls_logits_b = self.network(data[start:start+current_batch_size])
                        break
                    except RuntimeError as e:
                        if 'out of memory' not in str(e).lower() or current_batch_size <= 1:
                            raise
                        empty_cache(self.device)
                        if self.verbose:
                            print(f'CUDA OOM at batch size {current_batch_size} - retrying with {max(1, current_batch_size // 2)}')
                        current_batch_size = max(1, current_batch_size // 2)
                        max_batch_size = current_batch_size  # avoid repeating the same failure on later batches

                predicted_batch_seg_logits[start:start+current_batch_size] = predicted_batch_seg_logits_b
                for i in range(len(predicted_batch_cls_logits_b)):
                    predicted_batch_cls_logits[i][start:start+current_batch_size] = predicted_batch_cls_logits_b[i]

                empty_cache(self.device)
                start += current_batch_size
                batch_idx += 1

                #print("Peak during batch: ", torch.cuda.max_memory_reserved() / (1024**2), "MB")


            #print("After running reserved: ", torch.cuda.memory_reserved() / (1024**2))
            for i in range(len(predicted_batch_cls_logits)):
                predicted_batch_cls_logits[i] = predicted_batch_cls_logits[i].unsqueeze(-1).repeat(1, 1, predicted_batch_seg_logits.shape[2])

            record_seg_logits = []
            record_cls_logits = []

            n_predictions = None
            predicted_seg_logits = None
            predicted_cls_logits = None

            predicted_batch_seg_logits = predicted_batch_seg_logits.to(results_device)
            predicted_batch_cls_logits = [r.to(results_device) for r in predicted_batch_cls_logits]
            prediction_num_cls = [r.shape[1] for r in predicted_batch_cls_logits]

            t = time.time()
            for i in range(len(obj_data)):
                
                #define new arrays if we are at the start of a new record
                if n_predictions is None:
                    original_shape = obj_data[i]["shape"]

                
                    n_predictions = torch.zeros(original_shape, dtype=torch.half, device=results_device)
                
                    predicted_seg_logits = torch.zeros((self.label_manager.num_segmentation_heads, *original_shape), dtype=torch.half, device=results_device)

                    
                    num_classes_classification_branches = self.configuration_manager.configuration['architecture']['arch_kwargs']['num_classes_classification_branch']
                    #num_classification_branches = self.configuration_manager
                    predicted_cls_logits = []
                    for nc in num_classes_classification_branches:
                        predicted_cls_logits.append(torch.zeros((nc, *original_shape), dtype=torch.half, device=results_device))   
    

                
                #get results from the GPU output
                # print("Getting results from GPU output")
                # prediction_seg = predicted_batch_seg_logits[i].to(results_device)
                # prediction_cls = [r[i].to(results_device) for r in predicted_batch_cls_logits]
                # print("Done getting results from GPU output")

                #multiply with gaussian if we are using it
                if self.use_gaussian:
                    predicted_batch_seg_logits[i] *= gaussian
                    for j in range(len(prediction_num_cls)):
                        predicted_batch_cls_logits[j][i] *= gaussian

                #add segmentation and classification outputs to the arrays
                predicted_seg_logits[obj_data[i]["slice"]] += predicted_batch_seg_logits[i]
                for j in range(len(prediction_num_cls)):
                    predicted_cls_logits[j][obj_data[i]["slice"]] += predicted_batch_cls_logits[j][i]


                #add gaussian to the n_predictions array
                n_predictions[obj_data[i]["slice"][1:]] += gaussian

                #check if we are at the end of a record, if so, we add the arrays to the record list
                if i == len(obj_data)-1 or obj_data[i+1]["record"] != obj_data[i]["record"]:
                    predicted_seg_logits /= n_predictions
                    for j in range(len(prediction_num_cls)):
                        predicted_cls_logits[j] /= n_predictions
                        
                    # revert padding
                    slicer_revert_padding = obj_data[i]["slicer_revert_padding"]
                    predicted_seg_logits = predicted_seg_logits[(slice(None), *slicer_revert_padding[1:])]
                    for j in range(len(prediction_num_cls)):
                        predicted_cls_logits[j] = predicted_cls_logits[j][(slice(None), *slicer_revert_padding[1:])]

                    #print(predicted_seg_logits.shape, predicted_cls_logits[0].shape)
                    record_seg_logits.append(predicted_seg_logits)
                    record_cls_logits.append(predicted_cls_logits)

                    n_predictions = None
                    predicted_seg_logits = None
                    predicted_cls_logits = None
                
                #print("After results reserved: ", torch.cuda.memory_reserved() / (1024**2))            
            
                empty_cache(self.device)
            

        return record_seg_logits, record_cls_logits

    def embed_sliding_window_on_batched_data(self, data: torch.Tensor, obj_data, do_on_device: bool = True):

        results_device = self.device if do_on_device else torch.device('cpu')

        if self.use_gaussian:
            latent_layer = obj_data[0]['latent_layer']
            print(self.configuration_manager.patch_size)
            new_patch_size = list(self.configuration_manager.patch_size)
            new_patch_size[-1] = int(self.configuration_manager.patch_size[-1]/np.power(2, 4-latent_layer))
            gaussian = compute_gaussian(tuple(new_patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)
        else:
            gaussian = 1

        # empty_cache(self.device)
            
        with torch.no_grad():

            empty_cache(self.device)

            assert isinstance(data, torch.Tensor)
            data_size = torch.cuda.memory_reserved() / (1024**2) if self.device.type == 'cuda' else 0
            self.network = self.network.to(self.device)
            self.network.eval()

            #print("After model load: ", torch.cuda.memory_reserved() / (1024**2))
            embedded_batch = torch.empty((len(data), obj_data[0]['latent_dim'],  obj_data[0]['shape'][-1]), dtype=torch.half, device=self.device)

            #print("After results reserved: ", torch.cuda.memory_reserved() / (1024**2))

            reserved = torch.cuda.memory_reserved() / (1024**2) if self.device.type == 'cuda' else 0

            working_mem_per_sample = (data.shape[-1] * 64 * 4 * 11) / (1024**2)  # 5mb per sample
            mem_per_sample = working_mem_per_sample + 0.2 * working_mem_per_sample  # 20% overhead
            max_batch_size = min(len(data), int(np.floor(((self.vram_available - reserved) / mem_per_sample)/100)*100)) #ong 5mb per sample + 20% overhead
            #max_batch_size = 1000
            if max_batch_size <= 0:
                # Estimate says VRAM is very tight - start small instead of disabling
                # batching altogether (the previous fallback to len(data) did the opposite
                # of what's needed exactly when memory is most constrained).
                max_batch_size = min(len(data), 8)

            start = 0
            batch_idx = 0
            while start < len(data):
                current_batch_size = min(max_batch_size, len(data) - start)

                while True:
                    if self.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()

                    if self.verbose:
                        print(f'Processing batch {batch_idx+1} (size {current_batch_size}, offset {start}/{len(data)})')

                    try:
                        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                            embedded_batch_b = self.network.embed(data[start:start+current_batch_size])[obj_data[0]['latent_layer']]
                        break
                    except RuntimeError as e:
                        if 'out of memory' not in str(e).lower() or current_batch_size <= 1:
                            raise
                        empty_cache(self.device)
                        if self.verbose:
                            print(f'CUDA OOM at batch size {current_batch_size} - retrying with {max(1, current_batch_size // 2)}')
                        current_batch_size = max(1, current_batch_size // 2)
                        max_batch_size = current_batch_size  # avoid repeating the same failure on later batches

                embedded_batch[start:start+current_batch_size] = embedded_batch_b
                empty_cache(self.device)
                start += current_batch_size
                batch_idx += 1

                #print("Peak during batch: ", torch.cuda.max_memory_reserved() / (1024**2), "MB")

            record_embeddings = []
            n_predictions = None
            embeddings = None
            embedded_batch = embedded_batch.to(results_device)

            t = time.time()
            for i in range(len(obj_data)):
                
                #define new arrays if we are at the start of a new record
                if n_predictions is None:
                    original_shape = obj_data[i]["shape"]
                    latent_dim = obj_data[i]['latent_dim']

                    n_predictions = torch.zeros(original_shape, dtype=torch.half, device=results_device)
                    embeddings = torch.zeros((latent_dim, *original_shape), dtype=torch.half, device=results_device)

                
                #get results from the GPU output
                # print("Getting results from GPU output")
                # prediction_seg = predicted_batch_seg_logits[i].to(results_device)
                # prediction_cls = [r[i].to(results_device) for r in predicted_batch_cls_logits]
                # print("Done getting results from GPU output")

                #multiply with gaussian if we are using it
                if self.use_gaussian:
                    embedded_batch[i] *= gaussian

                #add segmentation and classification outputs to the arrays
                embeddings[obj_data[i]["slice"]] += embedded_batch[i]

                #add gaussian to the n_predictions array
                n_predictions[obj_data[i]["slice"][1:]] += gaussian

                #check if we are at the end of a record, if so, we add the arrays to the record list
                if i == len(obj_data)-1 or obj_data[i+1]["record"] != obj_data[i]["record"]:
                    embeddings /= n_predictions
                        
                    # revert padding
                    slicer_revert_padding = obj_data[i]["slicer_revert_padding"]
                    #embeddings = embeddings[(slice(None), *slicer_revert_padding[1:])]

                    #print(predicted_seg_logits.shape, predicted_cls_logits[0].shape)
                    record_embeddings.append(embeddings)

                    n_predictions = None
                    embeddings = None
                
                #print("After results reserved: ", torch.cuda.memory_reserved() / (1024**2))            
            
                empty_cache(self.device)
            
        return record_embeddings

    def predict_logits_from_batched_data(self, data: torch.Tensor, obj_data, do_on_device: bool = True, average=True):
        n_threads = torch.get_num_threads()
        n = len(np.unique([i["record"] for i in obj_data]))
        prediction_seg = None if average else [[] for _ in range(n)]
        prediction_cls = None if average else [[] for _ in range(n)]

        for idx, params in tqdm(enumerate(self.list_of_parameters), total=len(self.list_of_parameters), desc="Folds"):
        #for idx, params in enumerate(self.list_of_parameters):

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
            #if self.verbose:
            #print(f"Currently reserved: {reserved:.2f} MB")

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            t = time.time()
            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if average:
                res = self.predict_sliding_window_on_batched_data(data, obj_data)
                if self.verbose: print(f"Prediction done in {time.time() - t:.2f} seconds")
                if prediction_seg is None:
                    prediction_seg = res[0].detach().cpu()
                    prediction_cls = [res[1][i].detach().cpu() for i in range(len(res[1]))]
                else:
                    prediction_seg += res[0].detach().cpu()
                    for i in range(len(res[1])):
                        prediction_cls[i] += res[1][i].detach().cpu()

            else:
                res = self.predict_sliding_window_on_batched_data(data, obj_data)
                n = len(res[0])
                if self.verbose: print(f"Prediction done in {time.time() - t:.2f} seconds")

                for i in range(len(res[0])):
                    prediction_seg[i].append(res[0][i].detach().cpu())
                    if len(prediction_cls[i]) == 0:
                        for j in range(len(res[1][i])):
                            prediction_cls[i].append([res[1][i][j].detach().cpu()])
                    else:
                        for j in range(len(res[1][i])):
                            prediction_cls[i][j].append(res[1][i][j].detach().cpu())

        if len(self.list_of_parameters) > 1 and average:
            prediction_seg /= len(self.list_of_parameters)
            for i in range(len(prediction_cls)):
                prediction_cls[i] /= len(self.list_of_parameters)
        else:
            for i in range(len(prediction_seg)):
                prediction_seg[i] = torch.stack(prediction_seg[i])
                for j in range(len(prediction_cls[i])):
                    prediction_cls[i][j] = torch.stack(prediction_cls[i][j])

        if self.verbose: print('Prediction done')
        return prediction_seg, prediction_cls

    def embed_logits_from_batched_data(self, data: torch.Tensor, obj_data, do_on_device: bool = True, average=True):
        n_threads = torch.get_num_threads()
        n = len(np.unique([i["record"] for i in obj_data]))
        embedding = [[] for _ in range(n)]

        for idx, params in tqdm(enumerate(self.list_of_parameters), total=len(self.list_of_parameters), desc="Folds"):
        #for idx, params in enumerate(self.list_of_parameters):

            if self.device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
            #if self.verbose:
            #print(f"Currently reserved: {reserved:.2f} MB")

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            t = time.time()
            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if average:
                res = self.embed_sliding_window_on_batched_data(data, obj_data)
                if self.verbose: print(f"Prediction done in {time.time() - t:.2f} seconds")
                if len(embedding[0]) == 0:
                    for i in range(len(res)):
                        embedding[i] = (res[i].detach().cpu())
                else:
                    for i in range(len(res)):
                        embedding[i] += (res[i].detach().cpu())

            else:
                res = self.embed_sliding_window_on_batched_data(data, obj_data)
                if self.verbose: print(f"Prediction done in {time.time() - t:.2f} seconds")

                for i in range(len(res)):
                    embedding[i].append(res[i].detach().cpu())

        if len(self.list_of_parameters) > 1 and average:
            for i in range(len(embedding)):
                embedding[i] /= len(self.list_of_parameters)
        else:
            for i in range(len(embedding)):
                embedding[i] = torch.stack(embedding[i])

        if self.verbose: print('Prediction done')
        return embedding


    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, average=True) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction_seg = None if average else []
        prediction_cls = None if average else []

        for idx, params in enumerate(self.list_of_parameters):
            
            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if average:
                if prediction_seg is None:
                    res = self.predict_sliding_window_return_logits(data)
                    prediction_seg = res[0].to('cpu')
                    prediction_cls = [res[1][i].to('cpu') for i in range(len(res[1]))]
                else:
                    res = self.predict_sliding_window_return_logits(data)
                    prediction_seg += res[0].to('cpu')
                    for i in range(len(res[1])):
                        prediction_cls[i] += res[1][i].to('cpu')
            else:
                res = self.predict_sliding_window_return_logits(data)
                prediction_seg.append(res[0].to('cpu'))
                if len(prediction_cls) == 0:
                    for i in range(len(res[1])):
                        prediction_cls.append([res[1][i].to('cpu')])
                else:
                    for i in range(len(res[1])):
                        prediction_cls[i].append(res[1][i].to('cpu'))

        if len(self.list_of_parameters) > 1 and average:
            prediction_seg /= len(self.list_of_parameters)
            for i in range(len(prediction_cls)):
                prediction_cls[i] /= len(self.list_of_parameters)
        else:
            prediction_seg = torch.stack(prediction_seg)
            for i in range(len(prediction_cls)):
                prediction_cls[i] = torch.stack(prediction_cls[i])

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction_seg, prediction_cls

    def embeddings_from_preprocessed_data(self, data: torch.Tensor, average=True, latent_layer=0) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        embedding = None if average else []
        data = data.to(self.device)

        for idx, params in enumerate(self.list_of_parameters):
            
            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            
            with torch.no_grad():
                self.network = self.network.to(self.device)
                self.network.eval()

                empty_cache(self.device)

                # why not leave prediction on device if perform_everything_on_device? Because this may cause the
                # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
                # this actually saves computation time
                if average:
                    if embedding is None:
                        res = self.network.embed(data[0])[latent_layer]
                        embedding = res.to('cpu')
                    else:
                        res = self.network.embed(data[0])[latent_layer]
                        embedding += res.to('cpu')
                else:
                    res = self.network.embed(data)[latent_layer]
                    embedding.append(res.to('cpu'))

        embedding /= len(self.list_of_parameters)
        embedding = embedding[0]
        #embedding = torch.mean(embedding, dim=2)


        if self.verbose: print('Embedding done')
        torch.set_num_threads(n_threads)
        return embedding

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_or_return_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        # iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
        #                                                     segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        #                                                     properties_or_list_of_properties,
        #                                                     truncated_ofname,
        #                                                     num_processes)
        # return self.predict_from_data_iterator(iterator, save_or_return_probabilities, num_processes_segmentation_export, num_images=len(image_or_list_of_images))
        
        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.deterministic = True

        do_on_device = True
        predicted_seg_logits = predicted_cls_logits = n_predictions = prediction = gaussian = workon = None

        # Check if CUDA is available
        if torch.cuda.is_available():
            # torch.cuda.mem_get_info() reads free memory for the active device directly
            # from the CUDA driver, so it can't be misaligned the way indexing nvidia-smi's
            # device list by torch's local device index could be when CUDA_VISIBLE_DEVICES
            # restricts/reorders visible GPUs.
            free_bytes, _ = torch.cuda.mem_get_info()
            self.vram_available = free_bytes / (1024 ** 2)  # MB, matches previous nvidia-smi units
        else:
            self.vram_available = psutil.virtual_memory().available
            
        #if self.verbose:
        if self.verbose: print(f"Available RAM: {self.vram_available} GB")
        
        objs = []
        props = []
        sliced_data = []

        for i in range(len(image_or_list_of_images)):
            #pad data if it is smaller than tile_size
            input_image = image_or_list_of_images[i]
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size, 'constant', {'constant_values': 0}, True, None)
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            properties = properties_or_list_of_properties[i]
            data_shape_before_cropping = input_image.shape[1:]
            seg_shape_before_cropping = input_image.shape[1:]
            properties['data_shape_before_cropping'] = data_shape_before_cropping
            properties['seg_shape_before_cropping'] = seg_shape_before_cropping
            # this command will generate a segmentation. This is important because of the nonzero mask which we may need
            _, _, bbox = crop_to_nonzero(data)
            properties['bbox_used_for_cropping'] = bbox
            properties['data_shape_after_cropping_and_before_resampling'] = input_image.shape[1:]
            properties['seg_shape_after_cropping_and_before_resampling'] = input_image.shape[1:]
            props.append(properties)

            for slicer in slicers:
                objs.append({
                    "record": i,
                    "shape": data.shape[1:],
                    "data_index": len(sliced_data),
                    "slice": slicer,
                    "slicer_revert_padding": slicer_revert_padding
                })
                sliced_data.append(data[slicer])

        sliced_data = np.stack(sliced_data, axis=0, dtype=np.float32)
        maxn = sliced_data.shape[0]
        sliced_data = sliced_data[:maxn, ...]
        objs = objs[:maxn]
        props = props[:maxn]

        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.synchronize()  # ensures all ops are done

        # allocated = torch.cuda.memory_allocated() / (1024**2)
        # print(f"Currently allocated: {allocated:.2f} MB")

        #param = self.network.compute_conv_feature_map_size(sliced_data.shape[2:])
        #print((param*sliced_data.shape[0]*4)/1024/1024)

        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            if self.verbose: print(f"Currently allocated: {allocated:.2f} MB")

        #get maximum vram available

        st = time.time()
        if self.verbose: print("sliced_data shape:", sliced_data.shape)
        data = torch.from_numpy(sliced_data)
        data = data.to(self.device)
        if self.verbose: print(f"Data loaded in {time.time() - st:.2f} seconds")

        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_reserved() / (1024**2)
            if self.verbose: print(f"After data load allocated: {allocated:.2f} MB, still free: {self.vram_available - allocated:.2f} MB")

        st = time.time()
        res = self.predict_logits_from_batched_data(data, objs, do_on_device=True, average=False)
        predicted_seg_logits, predicted_cls_logits = res[0], res[1]
        if self.verbose: print(f"Prediction done in {time.time() - st:.2f} seconds")
        n = len(predicted_seg_logits)
        out = []

        if self.verbose: print('resampling to original shape')

        times_aggregated = [0,0,0,0]
        for i in range(len(predicted_seg_logits)):
            rec_seg_logits = predicted_seg_logits[i]
            rec_cls_logits = predicted_cls_logits[i]

            if rec_seg_logits.ndim == 5:
                ret = convert_multiple_predicted_logits_to_segmentation_with_correct_shape(rec_seg_logits, self.plans_manager,
                                                                                self.configuration_manager,
                                                                                self.label_manager,
                                                                                props[i],
                                                                                return_probabilities=
                                                                                save_or_return_probabilities)
            else:
                assert rec_seg_logits.ndim == 4 # c z y x
                ret = convert_predicted_logits_to_segmentation_with_correct_shape(rec_seg_logits, self.plans_manager,
                                                                                self.configuration_manager,
                                                                                self.label_manager,
                                                                                props[i],
                                                                                return_probabilities=
                                                                                save_or_return_probabilities)

            if save_or_return_probabilities:
                out.append({"seg":[ret[0], ret[1]], "cls": rec_cls_logits})
            else:
                out.append({"seg":ret[0], "cls": rec_cls_logits})

            times_aggregated[0] += ret[-1][0]
            times_aggregated[1] += ret[-1][1]
            times_aggregated[2] += ret[-1][2]
            times_aggregated[3] += ret[-1][3]    

        if self.verbose:
            print(f"Resampling done in {times_aggregated[0]:.2f} seconds, "
                f"nonlinear done in {times_aggregated[1]:.2f} seconds, "
                f"transpose done in {times_aggregated[2]:.2f} seconds, "
                f"reverttime done in {times_aggregated[3]:.2f} seconds")

        # After inference
        empty_cache(self.device)
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)

            if self.verbose:
                print(f"Currently allocated: {allocated:.2f} MB")
                print(f"Currently reserved: {reserved:.2f} MB")
                print(f"Peak allocated during inference: {peak:.2f} MB")
                
        return out

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes,
                                   num_images: int = None):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r_seg = []
            r_cls = []
            with tqdm(total=num_images, desc="Processing files") as pbar:
                for preprocessed in tqdm(data_iterator, desc="Processing files"):
                    t = time.time()
                    data = preprocessed['data']
                    if isinstance(data, str):
                        delfile = data
                        data = torch.from_numpy(np.load(data))
                        os.remove(delfile)

                    ofile = preprocessed['ofile']
                    # if ofile is not None:
                    #     print(f'\nPredicting {os.path.basename(ofile)}:')
                    # else:
                    #     print(f'\nPredicting image of shape {data.shape}:')

                    # print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                    properties = preprocessed['data_properties']

                    # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                    # npy files
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r_seg, allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(export_pool, worker_list, r_seg, allowed_num_queued=2)

                    res = self.predict_logits_from_preprocessed_data(data, average=False)
                    predicted_seg_logits, predicted_cls_logits = res[0], res[1]
                    r_cls.append(predicted_cls_logits)

                    if ofile is not None:
                        if predicted_seg_logits.ndim == 5:
                            r_seg.append(
                                export_pool.starmap_async(
                                    export_multiple_predictions_from_logits,
                                    ((predicted_seg_logits, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, ofile, save_probabilities),)
                                )
                            )
                        else:
                            assert predicted_seg_logits.ndim == 4 # c z y x
                            r_seg.append(
                                export_pool.starmap_async(
                                    export_prediction_from_logits,
                                    ((predicted_seg_logits, properties, self.configuration_manager, self.plans_manager,
                                    self.dataset_json, ofile, save_probabilities),)
                                )
                            )
                    else:
                        if predicted_seg_logits.ndim == 5:
                            r_seg.append(
                                export_pool.starmap_async(
                                    convert_multiple_predicted_logits_to_segmentation_with_correct_shape, (
                                        (predicted_seg_logits, self.plans_manager,
                                        self.configuration_manager, self.label_manager,
                                        properties,
                                        save_probabilities),)
                                )
                            )
                        else:
                            assert predicted_seg_logits.ndim == 4 # c z y x
                            r_seg.append(
                                export_pool.starmap_async(
                                    convert_predicted_logits_to_segmentation_with_correct_shape, (
                                        (predicted_seg_logits, self.plans_manager,
                                        self.configuration_manager, self.label_manager,
                                        properties,
                                        save_probabilities),)
                                )
                            )
                    pbar.update(1)
                    print(f'Prediction took {time.time() - t:.2f} seconds')
                

            ret = [i.get()[0] for i in r_seg]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

        if save_probabilities:
            return [{"seg":[r[0], r[1]], "cls": cls_logits} for r, cls_logits in zip(ret, r_cls)]
        else:
            return [{"seg":r, "cls": cls_logits} for r, cls_logits in zip(ret, r_cls)]

        #             if ofile is not None:
        #                 # this needs to go into background processes
        #                 # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
        #                 #                               self.dataset_json, ofile, save_probabilities)
        #                 #print('sending off prediction to background worker for resampling and export')
        #                 if prediction.ndim == 5:
        #                     r.append(
        #                         export_pool.starmap_async(
        #                             export_multiple_predictions_from_logits,
        #                             ((prediction, properties, self.configuration_manager, self.plans_manager,
        #                             self.dataset_json, ofile, save_probabilities),)
        #                         )
        #                     )
        #                 else:
        #                     assert prediction.ndim == 4 # c z y x
        #                     r.append(
        #                         export_pool.starmap_async(
        #                             export_prediction_from_logits,
        #                             ((prediction, properties, self.configuration_manager, self.plans_manager,
        #                             self.dataset_json, ofile, save_probabilities),)
        #                         )
        #                     )
        #             else:
        #                 # convert_predicted_logits_to_segmentation_with_correct_shape(
        #                 #             prediction, self.plans_manager,
        #                 #              self.configuration_manager, self.label_manager,
        #                 #              properties,
        #                 #              save_probabilities)

        #                 #print('sending off prediction to background worker for resampling')
        #                 if prediction.ndim == 5:
        #                     r.append(
        #                         export_pool.starmap_async(
        #                             convert_multiple_predicted_logits_to_segmentation_with_correct_shape, (
        #                                 (prediction, self.plans_manager,
        #                                 self.configuration_manager, self.label_manager,
        #                                 properties,
        #                                 save_probabilities),)
        #                         )
        #                     )
        #                 else:
        #                     assert prediction.ndim == 4 # c z y x
        #                     r.append(
        #                         export_pool.starmap_async(
        #                             convert_predicted_logits_to_segmentation_with_correct_shape, (
        #                                 (prediction, self.plans_manager,
        #                                 self.configuration_manager, self.label_manager,
        #                                 properties,
        #                                 save_probabilities),)
        #                         )
        #                     )
        #             # if ofile is not None:
        #             #     print(f'done with {os.path.basename(ofile)}')
        #             # else:
        #             #     print(f'\nDone with image of shape {data.shape}:')

        #             # Update the tqdm progress bar after each file processed
        #             pbar.update(1)

        #     ret = [i.get()[0] for i in r]

        # if isinstance(data_iterator, MultiThreadedAugmenter):
        #     data_iterator._finish()

        # # clear lru cache
        # compute_gaussian.cache_clear()
        # # clear device cache
        # empty_cache(self.device)
        # return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """

        ppa = PreprocessAdapterFromNpyWithClassification([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        res = self.predict_logits_from_preprocessed_data(dct['data'], average=False)
        predicted_seg_logits, predicted_cls_logits = res[0], res[1]

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            if predicted_seg_logits.ndim == 5:
                export_multiple_predictions_from_logits(predicted_seg_logits, dct['data_properties'], self.configuration_manager,
                                            self.plans_manager, self.dataset_json, output_file_truncated,
                                            save_or_return_probabilities)
            else:
                assert predicted_seg_logits.ndim == 4 # c z y x
                export_prediction_from_logits(predicted_seg_logits, dct['data_properties'], self.configuration_manager,
                                            self.plans_manager, self.dataset_json, output_file_truncated,
                                            save_or_return_probabilities)
        else:
            if predicted_seg_logits.ndim == 5:
                ret = convert_multiple_predicted_logits_to_segmentation_with_correct_shape(predicted_seg_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            else:
                assert predicted_seg_logits.ndim == 4 # c z y x
                ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_seg_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return {"seg":[ret[0], ret[1]], "cls": predicted_cls_logits}
            else:
                return {"seg":ret, "cls": predicted_cls_logits}

    def embed_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_or_return_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes,
                                        latent_layer=0):
        # iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
        #                                                     segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        #                                                     properties_or_list_of_properties,
        #                                                     truncated_ofname,
        #                                                     num_processes)
        # return self.predict_from_data_iterator(iterator, save_or_return_probabilities, num_processes_segmentation_export, num_images=len(image_or_list_of_images))
        
        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.deterministic = True

        do_on_device = True
        embeddings = n_predictions = prediction = gaussian = workon = None

        # Check if CUDA is available
        if torch.cuda.is_available():
            # torch.cuda.mem_get_info() reads free memory for the active device directly
            # from the CUDA driver, so it can't be misaligned the way indexing nvidia-smi's
            # device list by torch's local device index could be when CUDA_VISIBLE_DEVICES
            # restricts/reorders visible GPUs.
            free_bytes, _ = torch.cuda.mem_get_info()
            self.vram_available = free_bytes / (1024 ** 2)  # MB, matches previous nvidia-smi units
        else:
            self.vram_available = psutil.virtual_memory().available
            
        #if self.verbose:
        if self.verbose: print(f"Available RAM: {self.vram_available} GB")
        
        objs = []
        props = []
        sliced_data = []

        for i in range(len(image_or_list_of_images)):
            #pad data if it is smaller than tile_size
            input_image = image_or_list_of_images[i]
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size, 'constant', {'constant_values': 0}, True, None)
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            properties = properties_or_list_of_properties[i]
            data_shape_before_cropping = input_image.shape[1:]
            seg_shape_before_cropping = input_image.shape[1:]
            properties['data_shape_before_cropping'] = data_shape_before_cropping
            properties['seg_shape_before_cropping'] = seg_shape_before_cropping
            # this command will generate a segmentation. This is important because of the nonzero mask which we may need
            _, _, bbox = crop_to_nonzero(data)
            properties['bbox_used_for_cropping'] = bbox
            properties['data_shape_after_cropping_and_before_resampling'] = input_image.shape[1:]
            properties['seg_shape_after_cropping_and_before_resampling'] = input_image.shape[1:]
            props.append(properties)
            latent_dim = (64*np.pow(2, 4-latent_layer)).astype(int)
            latent_shape = data.shape[1:]
            latent_shape = int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2]/np.power(2, 4-latent_layer))
            print(latent_shape)

            for slicer in slicers:
                objs.append({
                    "record": i,
                    "shape": latent_shape,
                    "latent_dim": latent_dim,
                    "latent_layer": latent_layer,
                    "data_index": len(sliced_data),
                    "slice": slicer,
                    "slicer_revert_padding": slicer_revert_padding
                })
                sliced_data.append(data[slicer])

        sliced_data = np.stack(sliced_data, axis=0, dtype=np.float32)
        maxn = sliced_data.shape[0]
        sliced_data = sliced_data[:maxn, ...]
        objs = objs[:maxn]
        props = props[:maxn]

        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.synchronize()  # ensures all ops are done

        # allocated = torch.cuda.memory_allocated() / (1024**2)
        # print(f"Currently allocated: {allocated:.2f} MB")

        #param = self.network.compute_conv_feature_map_size(sliced_data.shape[2:])
        #print((param*sliced_data.shape[0]*4)/1024/1024)

        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            if self.verbose: print(f"Currently allocated: {allocated:.2f} MB")

        #get maximum vram available

        st = time.time()
        if self.verbose: print("sliced_data shape:", sliced_data.shape)
        data = torch.from_numpy(sliced_data)
        data = data.to(self.device)
        if self.verbose: print(f"Data loaded in {time.time() - st:.2f} seconds")

        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_reserved() / (1024**2)
            if self.verbose: print(f"After data load allocated: {allocated:.2f} MB, still free: {self.vram_available - allocated:.2f} MB")

        st = time.time()
        embeddings = self.embed_logits_from_batched_data(data, objs, do_on_device=True, average=True)
        if self.verbose: print(f"Prediction done in {time.time() - st:.2f} seconds")
        n = len(embeddings)
        out = []

        if self.verbose: print('resampling to original shape')

        times_aggregated = [0,0,0,0]
        for i in range(len(embeddings)):
            out.append({"embedding":embeddings[i]})

        # After inference
        empty_cache(self.device)
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)

            if self.verbose:
                print(f"Currently allocated: {allocated:.2f} MB")
                print(f"Currently reserved: {reserved:.2f} MB")
                print(f"Peak allocated during inference: {peak:.2f} MB")
                
        return out

    def embed_single_npy_array(self, input_image: np.ndarray, image_properties: dict, latent_layer=0):

        ppa = PreprocessAdapterFromNpyWithClassification([input_image], [None], [image_properties],
                                       [None],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        embedding = self.embeddings_from_preprocessed_data(dct['data'], average=True, latent_layer=latent_layer)

        return embedding

class nnUNetLSTMWithClassificationPredictor(nnUNetWithClassificationPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        
        super().__init__(tile_step_size, use_gaussian, use_mirroring, perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        
        self.use_gaussian = False

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)

        self.configuration_manager.patch_size = [6144]  # this is the patch size for the LSTM model

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        patch_size = [6144]
        dim = len(patch_size)

        if dim == 1:
            steps = compute_steps_for_sliding_window(image_size[2:], patch_size,
                                                     self.tile_step_size)

            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0])}, image size is'
                                   f' {image_size}, tile_size {patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')

            for d in range(image_size[0]):
                for sx in steps[0]:
                    slicers.append(
                        tuple([slice(None), d, 0, slice(sx, sx + patch_size[0])]))
                        
        # if dim == 1:

        #     if self.verbose: print(f'n_steps {image_size[0] * len(steps[0])}, image size is'
        #                            f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
        #                            f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')

        #     for d in range(image_size[0]):
        #         slicers.append(
        #             tuple([slice(None), d, 0, slice(0,None)]))

        else:
            raise NotImplementedError('This function only supports 1D, 2D and 3D images')

        return slicers


def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    # predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
                                 join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # predict a numpy array
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
    ret = predictor.predict_single_npy_array(img, props, None, None, False)

    iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    ret = predictor.predict_from_data_iterator(iterator, False, 1)

    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_device=True,
    #     device=torch.device('cuda', 0),
    #     verbose=False,
    #     allow_tqdm=True
    #     )
    # predictor.initialize_from_trained_model_folder(
    #     join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_cascade_fullres'),
    #     use_folds=(0,),
    #     checkpoint_name='checkpoint_final.pth',
    # )
    # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predCascade'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                              folder_with_segs_from_prev_stage='/media/isensee/data/nnUNet_raw/Dataset003_Liver/imagesTs_predlowres',
    #                              num_parts=1, part_id=0)