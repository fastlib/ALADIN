import numpy as np
import inspect
from typing import Tuple, Union, List
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from tqdm import tqdm

import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from torch.utils.data import DataLoader

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from load import MongoDBChunkGenerator, MongoDataLoader, FromFolderGenerator
from train import Trainer, ECGFounderTrainer
from model import HannunNet, ECGFounderNet


class Predictor(object):
    def __init__(self, model_path):
        
        self.trainer = Trainer()
        self.trainer.initialize()
        self.network = self.trainer.get_network(1, len(self.trainer.labels))
        self.model_path = model_path
        self.device = self.trainer.device

    def predict_on_folder(self, folder):

        # Load the model

        if isinstance(self.model_path, str):
            checkpoint = torch.load(self.model_path, map_location=self.device)

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            new_state_dict[k] = value

        self.network.load_state_dict(new_state_dict)
        self.network = self.network.to(self.device)
        self.network.eval()

        data = FromFolderGenerator(source_folder=folder)
        all_predictions = []

        # Predict
        self.network.eval()
        with torch.no_grad():
            for i, rec in enumerate(data):
                inp = torch.from_numpy(rec["xs"]).to(self.device)
                inp = inp.unsqueeze(0)
                output = self.network(inp)[0]
                preds = torch.argmax(output, dim=1)
                preds = preds.cpu().numpy()
                all_predictions.append(preds)

        all_predictions = np.concatenate(all_predictions)

        return all_predictions



class ECGFounderPredictor(Predictor):
    def __init__(self, model_path):
        super().__init__(model_path)
        
        self.trainer = ECGFounderTrainer()
        self.trainer.initialize()
        self.network = self.trainer.get_network(1, len(self.trainer.labels))
        self.model_path = model_path
        self.device = self.trainer.device

    def predict_on_folder(self, folder):

        # Load the model

        if isinstance(self.model_path, str):
            checkpoint = torch.load(self.model_path, map_location=self.device)

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            new_state_dict[k] = value

        self.network.load_state_dict(new_state_dict)
        self.network = self.network.to(self.device)
        self.network.eval()

        data = FromFolderGenerator(source_folder=folder)
        all_predictions = []

        # Predict
        self.network.eval()
        with torch.no_grad():
            for i, rec in enumerate(data):
                inp = torch.from_numpy(rec["xs"]).to(self.device)
                inp = inp.unsqueeze(0)
                output = self.network(inp)[0]
                output = torch.sigmoid(output)
                output = output.cpu().numpy()
                all_predictions.append(output)

        all_predictions = np.concatenate(all_predictions)

        return all_predictions