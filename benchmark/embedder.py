import numpy as np
import inspect
from typing import Tuple, Union, List
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from tqdm import tqdm
import json
import pickle
import argparse
import os

import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from torch.utils.data import DataLoader

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from aladin import ALADIN
from aladin.utils.helpers import Record
from load import MongoDBChunkGenerator, FromFolderGenerator

from inference import Predictor

class Embedder(Predictor):
    def __init__(self, model_path):
        super().__init__(model_path)

    def embed_folder(self, datatype="train"):

        if isinstance(self.model_path, str):
            checkpoint = torch.load(self.model_path, map_location=self.device)

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            new_state_dict[k] = value

        self.network.load_state_dict(new_state_dict)
        self.network = self.network.to(self.device)
        self.network.eval()

        diagnoses = [
            "SR","SB","ST","SA",
            "AFIB","AFL","AFIB/AFL",
            "HAY","CHB","AVB II",
            "BIG",
            "TRI",
            "AT","AR","AER",
            "IVR","AIVR",
            "JR","AJR","AVJR",
            "SVT","SVA","AVRT","AVNRT",
            "VT",
            "WENCK"
        ]

        if datatype == "train":
            data = MongoDBChunkGenerator(cachenames=["mongo"], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="train_only_labels", batch_size=32)
            df = data.get_balanced_data()
        elif datatype == "id_test":
            data = MongoDBChunkGenerator(cachenames=["mongo"], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="val_only_labels", batch_size=32)
            df = data.get_balanced_data()
        elif datatype == "ood_test":
            basefolder = os.environ.get('benchmark_data')
            data = FromFolderGenerator(source_folder=basefolder+"/STANFORD")
            df = data.get_df()
            
        embeddings = []
        meta = []
        
        # Predict
        self.network.eval()
        with torch.no_grad():
            for i in tqdm(range(len(df))):
                xs = df.iloc[i]["xs"].astype(np.float32)[None,:]
                labels = df.iloc[i]["label"] if "label" in df.columns else df.iloc[i]["label_256"]
                print(labels)
                dbs = df.iloc[i]["dbs"]
                inp = torch.from_numpy(xs).to(self.device)
                print(inp.shape)
                output = self.network.embed(inp)[0]
                print(output.shape)
                output = output.permute(1,0)
                output = output[4::7,:]
                labels = labels[4::7]
                #output = torch.mean(output, dim=1)
                embeddings.append(output.cpu().numpy())
                meta.extend([{"db": dbs, "label": labels[i]} for i in range(output.shape[0])])

        embeddings = np.concatenate(embeddings)

        return embeddings, meta


class ALADINEmbedder():
    def __init__(self, modelpaths):

        self.aladin = ALADIN(modelpaths=modelpaths, debug={"segmenter": True})

    def embed_folder(self, datatype="train"):

        
        diagnoses = [
            "SR","SB","ST","SA",
            "AFIB","AFL","AFIB/AFL",
            "HAY","CHB","AVB II",
            "BIG",
            "TRI",
            "AT","AR","AER",
            "IVR","AIVR",
            "JR","AJR","AVJR",
            "SVT","SVA","AVRT","AVNRT",
            "VT",
            "WENCK"
        ]

        if datatype == "train":
            data = MongoDBChunkGenerator(cachenames=["mongo"], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="train_only_labels", batch_size=32)
            df = data.get_balanced_data()
        elif datatype == "id_test":
            data = MongoDBChunkGenerator(cachenames=["mongo"], folder="", diagnoses=diagnoses, split=True, train_ratio=0.8, type="val_only_labels", batch_size=32)
            df = data.get_balanced_data()
        elif datatype == "ood_test":
            basefolder = os.environ.get('benchmark_data')
            data = FromFolderGenerator(source_folder=basefolder+"/STANFORD")
            df = data.get_df()

        allembeddings = []
        allmetas = []

        randomindices = np.random.choice(range(len(df)), size=20, replace=False)
        
        #for i in tqdm(range(len(df))):
        for i in tqdm(randomindices):
        #for i in tqdm(range(len(data))):
            ecg = df.iloc[i]["xs"][0]
            ids = df.iloc[i]["ids"][0]
            if "label" in df.columns:
                label = df.iloc[i]["label"]
            else:
                label = None
            record = Record(ecg, None, None, 200, df.iloc[i]["dbs"], df.iloc[i]["ids"])
            embedding = self.aladin.embed(record)
            embedding = list(embedding.values())

            allembeddings.append(embedding)
            allmetas.append({"dbs": df.iloc[i]["dbs"], "label": label, "ecg": ecg})

        return allembeddings, allmetas

def embed_reference(datatype="train"):

    embedder = Embedder(
        model_path='./weights/HannunNet_checkpoint_best.pth',
    )
    all_embeddings, allmetas = embedder.embed_folder(datatype=datatype)
    
    with open(f"featurespaces/embeddings_REF_{datatype}.pkl", "wb") as f:
       pickle.dump({"embeddings":all_embeddings, "meta":allmetas}, f)

def embed_aladin(datatype="train", modelpaths=[]):

    embedder = ALADINEmbedder(modelpaths=modelpaths)
    all_embeddings, allmetas = embedder.embed_folder(datatype=datatype)

    with open(f"featurespaces/embeddings_ALADIN_{datatype}.pkl", "wb") as f:
       pickle.dump({"embeddings":all_embeddings, "meta":allmetas}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="reference")
    parser.add_argument('--datatype', type=str, default="train")
    parser.add_argument('--modelpaths', nargs='+', help='Paths to the checkpoint used by ALADIN', required=False, default=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"])
    args = parser.parse_args()
    datatype = args.datatype
    modelpaths = args.modelpaths

    if not os.path.exists("featurespaces"):
        os.makedirs("featurespaces")

    if args.method == "reference":
        embed_reference(datatype=datatype)
    else:
        embed_aladin(datatype=datatype, modelpaths=modelpaths)