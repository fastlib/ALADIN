from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import autocast, nn
from torch.optim import Adam, AdamW
import multiprocessing
import numpy as np
import torch
import warnings
from time import time, sleep

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.logging.nnunet_logger import nnUNetWithClassificationLogger

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.dataloading.data_loader_1d import nnUNetDataLoader1D
from nnunetv2.training.dataloading.data_loader_1d_classification import nnUNetWithClassificationDataLoader1D
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.nnunet_classification_dataset import nnUNetWithClassificationDataset
from nnunetv2.inference.export_prediction import export_prediction_from_logits, export_prediction_and_classification_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, nnUNetWithClassificationPredictor
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, compute_metrics_with_classification_on_folder

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class InverseTransform(BasicTransform):
    def __init__(self):
        super().__init__()

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return -img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        return segmentation

    def _apply_to_classification(self, classification, **params) -> torch.Tensor:
        classification[1] = 1
        return classification

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return -regression_target

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

        if data_dict.get('classification') is not None:
            data_dict['classification'] = self._apply_to_classification(data_dict['classification'], **params)

        if data_dict.get('regression_target') is not None:
            data_dict['regression_target'] = self._apply_to_segmentation(data_dict['regression_target'], **params)

        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)

        if data_dict.get('keypoints') is not None:
            data_dict['keypoints'] = self._apply_to_keypoints(data_dict['keypoints'], **params)

        if data_dict.get('bbox') is not None:
            data_dict['bbox'] = self._apply_to_bbox(data_dict['bbox'], **params)

        return data_dict



class nnUNetWithClassificationTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.alpha = 1
        self.beta = 0
        
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 20
        self.num_epochs = 64
        self.current_epoch = 0
        self.enable_deep_supervision = True

        self.logger = nnUNetWithClassificationLogger()
    
    def _build_loss(self):
        
        print(self.plans_manager.__dict__)

        clf_loss = nn.CrossEntropyLoss()

        if self.label_manager.has_regions or self.label_manager.binary_classification:
            seg_loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            seg_loss.dc = torch.compile(seg_loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            seg_loss = DeepSupervisionWrapper(seg_loss, weights)

        return seg_loss, clf_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetWithClassificationDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetWithClassificationDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val
    
    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 1:
            dl_tr = nnUNetWithClassificationDataLoader1D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetWithClassificationDataLoader1D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            raise RuntimeError("unsupported dimensionality")

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None


        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.5),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 3.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))

        transforms.append(
            RandomTransform(
                InverseTransform()
            , apply_probability=0.1)
        )

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []

        transforms.append(
            RandomTransform(
                InverseTransform()
            , apply_probability=0.1)
        )
        
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['seg_target']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)

        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
            cls_target = [i.to(self.device, non_blocking=True) for i in cls_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
            cls_target = cls_target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        #with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        seg, cls = self.network(data)
        # del data

        l = self.loss[0](seg, seg_target)
        for i in range(len(cls)):
            cls_l = self.loss[1](cls[i], cls_target[i])
            l += self.beta*cls_l

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            self.logger.log('train_losses', loss_here, self.current_epoch)
        else:
            loss_here = np.mean(outputs['loss'])
            self.logger.log('train_losses', loss_here, self.current_epoch)
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        seg_target = batch['seg_target']
        cls_target = batch['cls_target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
            cls_target = [i.to(self.device, non_blocking=True) for i in cls_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)
            cls_target = cls_target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.

        #with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        seg, cls = self.network(data)
        del data

        l = self.loss[0](seg, seg_target)
        for i in range(len(cls)):
            cls_l = self.loss[1](cls[i], cls_target[i])
            l += self.beta*cls_l

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            seg = seg[0]
            seg_target = seg_target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, seg.ndim))

        if self.label_manager.has_regions or self.label_manager.binary_classification:
            predicted_segmentation_onehot = (torch.sigmoid(seg) > 0.5).long()
        else:
            # no need for softmax
            output_seg = seg.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg.shape, device=seg.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions and not self.label_manager.binary_classification:
                mask = (seg_target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                seg_target[seg_target == self.label_manager.ignore_label] = 0
            else:
                if seg_target.dtype == torch.bool:
                    mask = ~seg_target[:, -1:]
                else:
                    mask = 1 - seg_target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                seg_target = seg_target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, seg_target, axes=axes, mask=mask)

        tp_seg = tp.detach().cpu().numpy()
        fp_seg = fp.detach().cpu().numpy()
        fn_seg = fn.detach().cpu().numpy()
        
        seg_metrics = [tp_seg, fp_seg, fn_seg]
        cls_metrics = []
        for i in range(len(cls)):
            pred_class = cls[i].argmax(1)
            tp = ((pred_class == 1) & (cls_target[i] == 1)).sum()
            fp = ((pred_class == 1) & (cls_target[i] == 0)).sum()
            fn = ((pred_class == 0) & (cls_target[i] == 1)).sum()
            tp = tp.detach().cpu().numpy()
            fp = fp.detach().cpu().numpy()
            fn = fn.detach().cpu().numpy()
            cls_metrics.append([tp, fp, fn])

        # we now handle the removal of the background dice in the labelmanager

        # if not self.label_manager.has_regions:
        #     # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        #     # (softmax training) there needs tobe one output for the background. We are not interested in the
        #     # background Dice
        #     # [1:] in order to remove background
        #     tp_hard = tp_hard[1:]
        #     fp_hard = fp_hard[1:]
        #     fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'cls_loss': cls_l.detach().cpu().numpy(), 'seg_metrics': np.array(seg_metrics), 'cls_metrics': np.array(cls_metrics)}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        print(outputs_collated['cls_metrics'].shape)
        seg_metrics = np.sum(outputs_collated['seg_metrics'], 0)
        cls_metrics = np.sum(outputs_collated['cls_metrics'], 0)
        tp_seg = seg_metrics[0,:]
        fp_seg = seg_metrics[1,:]
        fn_seg = seg_metrics[2,:]
        tp_cls = cls_metrics[:,0]
        fp_cls = cls_metrics[:,1]
        fn_cls = cls_metrics[:,2]

        print("TP:", tp_cls, "FP:", fp_cls, "FN:", fn_cls)
        print(tp_seg.shape)
        tp_seg = tp_seg[self.label_manager._get_indices_to_calc_dice()]
        fp_seg = fp_seg[self.label_manager._get_indices_to_calc_dice()]
        fn_seg = fn_seg[self.label_manager._get_indices_to_calc_dice()]

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp_seg)
            tp_seg = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp_seg)
            fp_seg = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn_seg)
            fn_seg = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp_seg, fp_seg, fn_seg)]]
        se = [tp_cls / (tp_cls + fn_cls) if tp_cls + fn_cls > 0 else 0 for tp_cls, fn_cls in zip(tp_cls, fn_cls)]
        pr = [tp_cls / (tp_cls + fp_cls) if tp_cls + fp_cls > 0 else 0 for tp_cls, fp_cls in zip(tp_cls, fp_cls)]
        global_classification_f1 = [2 * p * s / (p + s) if p + s > 0 else 0 for p, s in zip(pr, se)]
        print("Global F1 per branch", global_classification_f1)

        mean_fg_dice = np.nanmean(global_dc_per_class)
        global_classification_f1 = np.nanmean(global_classification_f1)

        self.logger.log('classification_f1', global_classification_f1, self.current_epoch)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))

        self.print_to_log_file('classification F1', np.round(self.logger.my_fantastic_logging['classification_f1'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnUNetWithClassificationPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = nnUNetWithClassificationDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, cls, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction_seg, prediction_cls = predictor.predict_sliding_window_return_logits(data)
                prediction_seg = prediction_seg.cpu()
                prediction_cls = prediction_cls.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_and_classification_from_logits, (
                            (prediction_seg, prediction_cls, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # results.append(
                #     segmentation_export_pool.starmap_async(
                #         export_prediction_from_logits, (
                #             (prediction_seg, properties, self.configuration_manager, self.plans_manager,
                #              self.dataset_json, output_filename_truncated, save_probabilities),
                #         )
                #     )
                # )
                # for
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_with_classification_on_folder(
                                                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                join(self.preprocessed_dataset_folder_base, 'gt_cls'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            # metrics = compute_metrics_on_folder(
            #                                     join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
            #                                     validation_output_folder,
            #                                     join(validation_output_folder, 'summary.json'),
            #                                     self.plans_manager.image_reader_writer_class(),
            #                                     self.dataset_json["file_ending"],
            #                                     self.label_manager.foreground_regions if self.label_manager.has_regions else
            #                                     self.label_manager.foreground_labels,
            #                                     self.label_manager.ignore_label, chill=True,
            #                                     num_processes=default_num_processes * dist.get_world_size() if
            #                                     self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

