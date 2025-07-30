import numpy as np
import inspect
from typing import Union, List
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from tqdm import tqdm

import argparse
import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from load import MongoDBChunkGenerator, MongoDataLoader
from model import HannunNet, ECGFounderNet




def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated



class Collator:
    def __call__(self, batch):
        assert all('xs' in x for x in batch)
        assert all('ys' in x for x in batch)
        assert all('records' in x for x in batch)
        assert all('dbs' in x for x in batch)
        
        return {
            'xs': torch.tensor(np.array([x['xs'] for x in batch])),
            'ys': torch.tensor(np.array([x['ys'] for x in batch])),
            'records': np.array([x['records'] for x in batch]),
            'dbs': np.array([x['dbs'] for x in batch]),
        }


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class Logger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_f1': list(),
            'ema_f1': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_f1':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_f1'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_f1']) > 0 else value
            self.log('ema_f1', new_ema_pseudo_dice, epoch)


    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_f1'][:epoch + 1], color='g', ls='dotted', label="F1",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_f1'][:epoch + 1], color='g', ls='-', label="F1 (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint

class Trainer(object):
    def __init__(self, device: torch.device = None):

        self.local_rank = 0
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if self.device.type == 'cuda':
            # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
            self.device = torch.device(type='cuda', index=0)
        print(f"Using device: {self.device}")

        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        self.output_folder = "./new_weights"

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-3
        self.weight_decay = 1e-5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 100
        self.current_epoch = 0
        self.batch_size = 128

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = Logger()
        self.collator = Collator()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start
        self.labels = None  # see initialize

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        self.was_initialized = False
    
    def initialize(self):
        if not self.was_initialized:
            self.labels = ["NSR","AFIB/AFL","AVB_TYPE2","SUDDEN_BRADY","BIGEMINY","TRIGEMINY","EAR","IVR","JUNCTIONAL","SVT","VT","WENCKEBACH","NOISE"]
            self.num_input_channels = 1
            self.num_segmentation_heads = len(self.labels)

            self.network = self.get_network(
                self.num_input_channels,
                self.num_segmentation_heads
            ).to(self.device)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.loss = self._build_loss()

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def get_network(self, num_input_channels, num_classes):
        network = HannunNet(num_classes, num_input_channels, 1)

        return network

    def get_dataloaders(self):

        diagnoses = [
            "SR","SB","ST","SA",
            "AFIB","AFL","AFIB/AFL",
            "HAY","AVB II",
            "CHB",
            "BIG",
            "TRI",
            "AT","AR","AER",
            "IVR","AIVR",
            "JR","AJR","AVJR",
            "SVT","SVA","AVRT","AVNRT",
            "VT",
            "WENCK",
            "NOISE"
        ]

        train = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="train_only_labels", batch_size=self.batch_size)
        val = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="val_only_labels", batch_size=self.batch_size)

        train_loader = MongoDataLoader(train, self.batch_size,
                                        do_sampling=True,
                                        do_balanced=False,
                                        multiple_labels=False,
                                       transforms=None)
        val_loader = MongoDataLoader(val, self.batch_size,
                                        do_sampling=False,
                                        do_balanced=False,
                                        multiple_labels=False,
                                       transforms=None)

        _ = next(train_loader)
        _ = next(val_loader)

        return train_loader, val_loader

    def _build_loss(self):
        loss = nn.CrossEntropyLoss()
        loss = loss.to(self.device)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        #lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=0.1, verbose=True)
        return optimizer, lr_scheduler


    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def on_train_start(self):
        # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
        # which may not be present  when doing inference
        if not self.was_initialized:
            self.initialize()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        maybe_mkdir_p(self.output_folder)
        empty_cache(self.device)

        print(f"batch size: {self.batch_size}")

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, self.network.name+"_checkpoint_latest.pth")):
            os.remove(join(self.output_folder, self.network.name+"_checkpoint_latest.pth"))

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['xs']
        target = batch['ys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        del data
        
        output = output.view(-1, output.size(-1))  # Shape: [batch_size * num_predictions, num_classes]
        target = target.view(-1)  # Shape: [batch_size * num_predictions]

        l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 13)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 13)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}

    def on_epoch_start(self):
        pass

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        loss = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def find_optimal_thresholds(self, gt, pred):
        optimal_thresholds = []
        for i in range(gt.shape[1]):
            #fpr, tpr, thresholds = roc_curve(gt[:, i], pred[:, i])
            #optimal_idx = np.argmax(tpr - fpr)  
            #optimal_thresholds.append(thresholds[optimal_idx])
            #get f1 score
            precision, recall, thresholds = precision_recall_curve(gt[:, i], pred[:, i])
            f1 = 2 * (precision * recall) / (precision + recall)
            optimal_idx = np.argmax(f1)
            print(f"Optimal threshold for {self.labels[i]}: {thresholds[optimal_idx]}, F1: {f1[optimal_idx]}")
            optimal_thresholds.append(thresholds[optimal_idx])
        
        return np.array(optimal_thresholds)

    def validation_step(self, batch: dict) -> dict:
        data = batch['xs']
        target = batch['ys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.

        output = self.network(data)
        del data

        output = output.view(-1, output.size(-1))  # Shape: [batch_size * num_predictions, num_classes]
        target = target.view(-1)  # Shape: [batch_size * num_predictions]
        output_res = output.argmax(1)
        #print(output.shape, target.shape)
        l = self.loss(output, target)

        #print(output.shape)

        # no need for softmax
        output_res = output_res.to('cpu').numpy()
        target = target.to('cpu').numpy()

        tp = np.zeros(self.num_segmentation_heads)
        fp = np.zeros(self.num_segmentation_heads)
        fn = np.zeros(self.num_segmentation_heads)
        tn = np.zeros(self.num_segmentation_heads)

        for i in range(self.num_segmentation_heads):
            tp[i] = np.sum((output_res == i) & (target == i))
            fp[i] = np.sum((output_res == i) & (target != i))
            fn[i] = np.sum((output_res != i) & (target == i))
            tn[i] = np.sum((output_res != i) & (target != i))

        return {'loss': l.detach().cpu().numpy(), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp'], 0)
        fp = np.sum(outputs_collated['fp'], 0)
        fn = np.sum(outputs_collated['fn'], 0)
        tn = np.sum(outputs_collated['tn'], 0)

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        print(f1)
        
        loss = np.nanmean(outputs_collated['loss'])

        self.logger.log('val_losses', loss, self.current_epoch)
        self.logger.log('mean_f1', np.nanmean(f1), self.current_epoch)

    def only_validation(self):
        if not self.was_initialized:
            self.initialize()

        self.load_checkpoint(join(self.output_folder, self.network.name+'_checkpoint_best.pth'))

        self.perform_actual_validation()

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.network.eval()

        val = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", split=True, train_ratio=0.8, type="val_only_labels", batch_size=self.batch_size)
        keys = val.get_ids()

        tp = np.zeros(self.num_segmentation_heads)
        fp = np.zeros(self.num_segmentation_heads)
        fn = np.zeros(self.num_segmentation_heads)
        tn = np.zeros(self.num_segmentation_heads)

        gt = np.zeros((len(keys), self.num_segmentation_heads))
        pred = np.zeros((len(keys), self.num_segmentation_heads))

        for k in tqdm(keys):
            data, target, properties = val.load_case(k)
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target)


            data = data.to(self.device, non_blocking=True)
            if isinstance(target, list):
                target = [i.to(self.device, non_blocking=True) for i in target]
            else:
                target = target.to(self.device, non_blocking=True)
            
            data = data[None, :]

            output = self.network(data)[0]
            output = output.argmax(1)
            del data

            #detach and move to cpu
            output_res = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            for i in range(self.num_segmentation_heads):
                tp[i] += np.sum((output_res == i) & (target == i))
                fp[i] += np.sum((output_res == i) & (target != i))
                fn[i] += np.sum((output_res != i) & (target == i))
                tn[i] += np.sum((output_res != i) & (target != i))

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        for i in range(self.num_segmentation_heads):
            print(f"{self.labels[i]}: {f1[i]*100}%")

        print(f"Mean F1: {np.mean(f1)*100}%")

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, self.network.name+'_checkpoint_latest.pth'))

        #handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_f1'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_f1'][-1]
            self.print_to_log_file(f"Yayy! New best EMA F1: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, self.network.name+'_checkpoint_best.pth'))

        # if self.local_rank == 0:
        #     self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def on_train_end(self):
        pass

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            new_state_dict[k] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']

        # messing with state dict naming schemes. Facepalm.
        if isinstance(self.network, OptimizedModule):
            self.network._orig_mod.load_state_dict(new_state_dict)
        else:
            self.network.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

        self.perform_actual_validation()


class ECGFounderTrainer(Trainer):
    def __init__(self, device: torch.device = None):
        super(ECGFounderTrainer, self).__init__(device=device)

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 30
        self.current_epoch = 0
        self.batch_size = 128

    def get_network(self, num_input_channels, num_classes):
        network = ECGFounderNet(num_classes, num_input_channels)
        return network
    
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        #lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
        return optimizer, lr_scheduler
    
    def get_dataloaders(self):

        diagnoses = [
            "SR","SB","ST","SA",
            "AFIB","AFL","AFIB/AFL",
            "HAY","AVB II",
            "CHB",
            "BIG",
            "TRI",
            "AT","AR","AER",
            "IVR","AIVR",
            "JR","AJR","AVJR",
            "SVT","SVA","AVRT","AVNRT",
            "VT",
            "WENCK",
            "NOISE"
        ]

        train = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="train_only_labels", batch_size=self.batch_size, multiple_labels=True, size=1000)
        val = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="val_only_labels", batch_size=self.batch_size, multiple_labels=True, size=1000)

        train_loader = MongoDataLoader(train, self.batch_size,
                                        do_sampling=True,
                                        do_balanced=False,
                                        multiple_labels=True,
                                       transforms=None)
        val_loader = MongoDataLoader(val, self.batch_size,
                                        do_sampling=False,
                                        do_balanced=False,
                                        multiple_labels=True,
                                       transforms=None)

        _ = next(train_loader)
        _ = next(val_loader)

        return train_loader, val_loader

    def train_step(self, batch: dict) -> dict:
        data = batch['xs']
        target = batch['ys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        del data

        l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 13)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 13)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['xs']
        target = batch['ys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.

        output = self.network(data)
        del data

        l = self.loss(output, target)
        output = torch.sigmoid(output)
        output_res = torch.round(output)

        #print(output.shape)

        # no need for softmax
        output_res = output_res.to('cpu').numpy()
        target = target.to('cpu').numpy()

        tp = np.zeros(self.num_segmentation_heads)
        fp = np.zeros(self.num_segmentation_heads)
        fn = np.zeros(self.num_segmentation_heads)
        tn = np.zeros(self.num_segmentation_heads)

        # Convert to boolean arrays
        y_true = target.astype(bool)
        y_pred = output_res.astype(bool)

        # Calculate true positives, false positives, false negatives, and true negatives
        for i in range(self.num_segmentation_heads):
            tp[i] = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            fp[i] = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
            fn[i] = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
            tn[i] = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0))

        return {'loss': l.detach().cpu().numpy(), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.network.eval()

        val = MongoDBChunkGenerator(cachenames=[self.network.name], folder="", split=True, train_ratio=0.8, type="val_only_labels", batch_size=self.batch_size, multiple_labels=True, size=1000)
        keys = val.get_ids()

        tp = np.zeros(self.num_segmentation_heads)
        fp = np.zeros(self.num_segmentation_heads)
        fn = np.zeros(self.num_segmentation_heads)
        tn = np.zeros(self.num_segmentation_heads)

        gt = np.zeros((len(keys), self.num_segmentation_heads))
        pred = np.zeros((len(keys), self.num_segmentation_heads))

        for k in tqdm(keys):
            data, target, properties = val.load_case(k)
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target)


            data = data.to(self.device, non_blocking=True)
            if isinstance(target, list):
                target = [i.to(self.device, non_blocking=True) for i in target]
            else:
                target = target.to(self.device, non_blocking=True)
            
            data = data[None, :]

            output = self.network(data)[0]
            output = torch.sigmoid(output)
            output = torch.round(output)

            #print(output.shape, target.shape)

            pred[k, :] = output.detach().cpu().numpy()
            gt[k, :] = target.detach().cpu().numpy()

            del data

            #detach and move to cpu
            output_res = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            y_pred = output_res.astype(bool)
            y_true = target.astype(bool)

            for i in range(self.num_segmentation_heads):
                tp[i] += np.sum((y_pred[i] == 1) & (y_true[i] == 1))
                fp[i] += np.sum((y_pred[i] == 1) & (y_true[i] == 0))
                fn[i] += np.sum((y_pred[i] == 0) & (y_true[i] == 1))
                tn[i] += np.sum((y_pred[i] == 0) & (y_true[i] == 0))

        print(self.find_optimal_thresholds(gt, pred))

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        for i in range(self.num_segmentation_heads):
            print(f"{self.labels[i]}: {f1[i]*100}%")

        print(f"Mean F1: {np.mean(f1)*100}%")

    def _build_loss(self):
        loss = nn.BCEWithLogitsLoss()
        loss = loss.to(self.device)

        return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--method', type=str, help='RESNET or ECGFOUNDER', required=True)
    args = parser.parse_args()

    if args.method == "RESNET":
        print("Using RESNET")
        trainer = Trainer()
        trainer.run_training()
    elif args.method == "ECGFOUNDER":
        print("Using ECGFOUNDER")
        trainer = ECGFounderTrainer()
        trainer.run_training()
    else:
        raise ValueError("Method not recognized")