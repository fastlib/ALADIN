import numpy as np
import pandas as pd
import pickle
import torch
from typing import Tuple, Union, List
import glob
import os
import wfdb
import json

import multiprocessing
from tqdm import tqdm
import scipy.signal as sps
from scipy.signal import butter, filtfilt

from dataloader import MongoDBDatasetGenerator

from batchgenerators.dataloading.data_loader import DataLoader


class MongoDBChunkGenerator(MongoDBDatasetGenerator):
    def __init__(self, detector="gt", cachenames=[""], split=True, type="train", expid=None, size=256, batch_size=16, augment=False, multiple_labels=False, diagnoses=[], patientsplit=False, trainsplit=None, testsplit=None, valsplit=None, train_ratio=0.5, message="Preprocessing...", folder="./data/"):
        
        self.label_mapper = [
            ["SR","SB","ST","SA"], #NSR
            ["AFIB","AFL","AFIB/AFL"], #AFIB/AFL
            ["HAY","AVB II"], #AVB2/CHB
            ["CHB"],
            ["BIG"], #BIGEMINY
            ["TRI"], #TRIGEMINY
            ["AT","AR","AER"], #EAR
            ["IVR","AIVR"], #IVR
            ["JR","AJR","AVJR"], #JR
            ["SVT","SVA","AVRT","AVNRT"], #SVT
            ["VT"], #VT
            ["WENCK"], #Wenckebach
            ["NOISE"] #Noise
        ]
        #NSR,AFIB,CHB,BIG,TRI,EAR,IVR,JR,SVT,VT,WENCK,NOISE
        self.multiple_labels = multiple_labels
        self.sampling_probablities = [
            1,
            1,
            20,
            20,
            5,
            10,
            10,
            20,
            20,
            5,
            20,
            40,
            20
        ]
        # self.sampling_probablities = [
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1,
        #     1
        # ]
        
        super().__init__(detector=detector, cachenames=cachenames, split=False, type=type, expid=expid, size=size, batch_size=batch_size, augment=augment, diagnoses=diagnoses, patientsplit=patientsplit, trainsplit=trainsplit, testsplit=testsplit, valsplit=valsplit, train_ratio=train_ratio, message=message, folder=folder)


        if split and not os.path.isfile(self.folder+self.cachename+"_processed.pkl"):
            self.random_balanced_split(train_ratio=train_ratio)

            with open((self.folder+self.cachename+"_processed.pkl"), "wb") as f:
                pickle.dump(self.df, f, protocol=4)


        if(self.type=="train" or self.type=="pool" or self.type=="train_only_labels"):
            self.df = self.df[self.df["istrain"]==1]
        if(self.type=="test"):
            self.df = self.df[self.df["istrain"]==0]
        if(self.type=="val" or self.type=="val_only_labels"):
            self.df = self.df[self.df["isval"]==1]

        print(self.type+" set size: ", len(self.df))

        #tally up the number of each label
        self.label_counts = [0]*len(self.label_mapper)
        for i in range(len(self.df)):
            labels = self.df.iloc[i]["label"]
            if self.multiple_labels:
                unique_labels = np.where(labels == 1)[0]
                for label in unique_labels:
                    self.label_counts[label] += 1
            else:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    self.label_counts[label] += 1

        #print the number of each label
        for i in range(len(self.label_mapper)):
            print(self.label_mapper[i], self.label_counts[i])

        #reset index
        self.df = self.df.reset_index(drop=True)


    def preprocess_ecg(self, ecg, fs):

        ecg = self.resize_signal(ecg, 2000)
        
        if np.sum(ecg) == 0:
            return ecg
        ecg = ecg - np.mean(ecg)
        ecg = (ecg) / (np.std(ecg))

        return ecg
    
    def preprocess_ecg_ecgfounder(self, ecg, fs):

        ecg = self.resize_signal(ecg, 5000)
        fs = 500

        #highpass filter 1hz
        b, a = butter(4, 1/(fs/2), btype='highpass')
        ecg = filtfilt(b, a, ecg)

        #lowpass filter 30hz
        b, a = butter(2, 30/(fs/2), btype='lowpass')
        ecg = filtfilt(b, a, ecg)

        #notch filter 50hz
        b, a = sps.iirnotch(50, 30, fs)
        ecg = filtfilt(b, a, ecg)
        #notch filter 60hz
        b, a = sps.iirnotch(60, 30, fs)
        ecg = filtfilt(b, a, ecg)
        
        if np.sum(ecg) == 0:
            return ecg
        
        ecg = ecg - np.mean(ecg)
        ecg = (ecg) / (np.std(ecg))

        return ecg

    def to_one_hot(self, label):
        one_hot = np.zeros((len(self.label_mapper),))
        one_hot[label] = 1
        return one_hot

    def worker(self, data):
        fs = 200 

        #monitor_memory()
        rawII = np.array(data["ecg"])[:,0]
        if self.multiple_labels:
            rawII = self.preprocess_ecg_ecgfounder(rawII, fs) #ecgfounder model
        else:
            rawII = self.preprocess_ecg(rawII, fs)

        trimmedlen = np.floor(len(rawII)/self.size).astype(int)*self.size
        rawII = rawII[:trimmedlen]
        signal = np.array([rawII])

        length = len(rawII)

        label = data["diagnoses"]
        labels = []
        for i in range(len(label)):
            for k in range(len(self.label_mapper)):
                if label[i] in self.label_mapper[k]:
                    labels.append(k)

        if len(labels) == 0:
            print("Warning: No label detected.", label)
            return None

        labels = list(set(labels))

        if not self.multiple_labels:
            if len(labels) > 1 and 0 in labels:
                labels.remove(0)

            if len(labels) > 1:
                print("Warning: Multiple labels detected: ", labels)
                return None

            labels = [labels[0]]*7
            labels = np.array(labels)

        if "predictions" not in data:
            return None

        predictionnames = [channel["name"] for channel in data["predictions"]]
        if "Prediction Single" in predictionnames:
            index = predictionnames.index("Prediction Single")
        elif "Prediction New" in predictionnames:
            index = predictionnames.index("Prediction New")
        elif "Prediction Old" in predictionnames:
            index = predictionnames.index("Prediction Old")
        else:
            print(predictionnames)
            return None
        predictions = data["predictions"][index]["predictions"]
        noisepreds = [pred for pred in predictions if pred["type"] == 3]
        noise = np.zeros(len(rawII))
        for pred in noisepreds:
            onset = int(pred["start"]*fs)
            offset = int(pred["end"]*fs)
            noise[onset:offset] = 1

        # if np.sum(noise) > len(rawII)/3:
        #     labels = [11]
        #     print("Warning: Noise detected in signal.")
        noise_per_256 = np.array([int(np.mean(noise[i:i+self.size])>0.5) for i in range(0, len(noise), self.size)])

        if not self.multiple_labels:
            noise_per_256_idx = np.where(noise_per_256 == 1)[0]
            labels[noise_per_256_idx] = 12
        else:
            if np.sum(noise_per_256) > 0:
                labels.append(12)

        ids = str(data["_id"])
        records = str(data["recordname"])
        dbs = data["database"]
        onsets = data["onset"]

        if self.multiple_labels:
            unique_labels = list(set(labels))
            one_hot_label = np.zeros((len(self.label_mapper),), dtype=int)
            for i in range(len(unique_labels)):
                one_hot_label[unique_labels[i]] = 1
        else:
            one_hot_label = np.array(labels)

        return {"xs":signal, "ys":one_hot_label, "records":records, "dbs":dbs, "ids":ids, "onsets":onsets, "label": one_hot_label, "sampling_probability": self.sampling_probablities[labels[0]]}       

    def random_balanced_split(self, train_ratio=0.5, trainrecords = None):
        print("Random balanced split calculation while accounting for patients with multiple records")
        recordnames = self.df["records"]
        uniquerecordnames = recordnames.unique()
        print("Unique record names: ", uniquerecordnames)

        label_counts = [0]*13
        main_labels = [[]]*len(label_counts)
        df_unique = self.df.drop_duplicates(subset='records', keep='first')
        for i in tqdm(range(len(df_unique))):
            labels = df_unique.iloc[i]["label"]
            recordname = df_unique.iloc[i]["records"]

            if self.multiple_labels:
                unique_labels = np.where(labels == 1)[0]
                if len(unique_labels) > 1 and 0 in unique_labels:
                    unique_labels = unique_labels[unique_labels != 0]
                label_counts[unique_labels[0]] += 1
                main_labels[unique_labels[0]].append(recordname)
                #self.df.loc[self.df["records"] == recordname, "main_label"] = unique_labels[0]
            else:
                unique_labels = np.unique(labels)
                label_counts[unique_labels[0]] += 1
                main_labels[unique_labels[0]].append(recordname)
                #self.df.loc[self.df["records"] == recordname, "main_label"] = unique_labels[0]

        print("Label counts: ", label_counts)
        
        #sample df using goal counts for each label
        alltrainrecords = []
        self.df["istrain"] = 0
        self.df["isval"] = 1
        for i in range(len(label_counts)):
            trainrecords = np.random.choice(main_labels[i], int(label_counts[i]*train_ratio), replace=False)
            alltrainrecords.extend(trainrecords)
            self.df.loc[self.df["records"].isin(trainrecords),"istrain"] = 1
            self.df.loc[self.df["records"].isin(trainrecords),"isval"] = 0
        
        return alltrainrecords

    def get_data_shape(self):
        return self.df.iloc[0]["xs"].shape

    def get_label_shape(self):
        return self.df.iloc[0]["ys"].shape

    def get_sampling_probabilities(self):
        return self.df["sampling_probability"].values
    
    def get_sampled_data(self, p):
        #sample df using p
        return self.df.sample(frac=p)

    def get_balanced_data(self):
        label_counts = [0]*13
        main_labels = []
        for i in range(len(self.df)):
            labels = self.df.iloc[i]["label"]
            if self.multiple_labels:
                unique_labels = np.where(labels == 1)[0]
                if len(unique_labels) > 1 and 0 in unique_labels:
                    unique_labels = unique_labels[unique_labels != 0]
                label_counts[unique_labels[0]] += 1
                main_labels.append(unique_labels[0])
            else:
                unique_labels = np.unique(labels)
                label_counts[unique_labels[0]] += 1
                main_labels.append(unique_labels[0])


        self.df["main_label"] = main_labels

        print("Label counts: ", label_counts)

        goal_counts = [min(200,c) for c in label_counts]

        #sample df using goal counts for each label
        dfs = []
        for i in range(13):
            dfs.append(self.df[self.df["main_label"] == i].sample(n=goal_counts[i]))
            label_counts[i] = len(dfs[-1])

        print("Label counts: ", label_counts)
        
        return pd.concat(dfs)

    def get_balanced_ids(self):
        label_counts = [0]*13
        main_labels = []
        for i in range(len(self.df)):
            labels = self.df.iloc[i]["label"]
            if self.multiple_labels:
                unique_labels = np.where(labels == 1)[0]
                if len(unique_labels) > 1 and 0 in unique_labels:
                    unique_labels = unique_labels[unique_labels != 0]
                main_labels.append(unique_labels[0])
                label_counts[unique_labels[0]] += 1
            else:
                unique_labels = np.unique(labels)
                main_labels.append(unique_labels[0])
                label_counts[unique_labels[0]] += 1

        self.df["main_label"] = main_labels

        #NSR,AFIB,CHB,BIG,TRI,EAR,IVR,JR,SVT,VT,WENCK,NOISE
        goal_counts = [400,300,200,200,200,200,200,200,200,200,200,200,200]

        actual_goal_counts = [min(goal_counts[i],c) for i, c in enumerate(label_counts)]
        print("Goal counts: ", actual_goal_counts)

        #sample df using goal counts for each label
        dfs = []
        for i in range(12):
            print("Sampling label ", i, " with goal count ", actual_goal_counts[i], len(self.df[self.df["main_label"] == i]))
            dfs.append(self.df[self.df["main_label"] == i].sample(n=actual_goal_counts[i]))
            label_counts[i] = len(dfs[-1])

        print("Label counts: ", label_counts)
        
        return pd.concat(dfs).index.tolist()

    def get_ids(self):
        return np.arange(len(self.df))

    def load_case(self, idx):
        row = self.df.iloc[idx]
        return row["xs"], row["ys"], {"records":row["records"], "dbs":row["dbs"], "ids":row["ids"]}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = np.array(row["xs"]).astype(np.float32)
        lbl = np.array(row["ys"]).astype(np.int16)

        if len(sig.shape) == 1:
            sig = np.expand_dims(sig, 0)
        if len(lbl.shape) == 1:
            lbl = np.expand_dims(lbl, 0)

        return {"xs":sig, "ys":lbl, "records":row["records"], "dbs":row["dbs"], "id":row["ids"]}

class FromFolderGenerator(MongoDBDatasetGenerator):
    def __init__(self, detector="gt", cachenames=[""], split=False, type="train", source_folder="", multiple_labels=False, size=256, batch_size=16, augment=False, diagnoses=[], patientsplit=False, trainsplit=None, testsplit=None, valsplit=None, train_ratio=0.5, message="Preprocessing...", folder="./data/"):
        
        self.source_folder = source_folder
        self.size = size
        self.multiple_labels = multiple_labels
        records = np.loadtxt(os.path.join(self.source_folder, "RECORDS"), dtype=str)
        db = os.path.basename(self.source_folder)

        print("Data retrieved", len(records))

        cpus = 32
        
        print("Using ", cpus, " cores")
        print(message)
        obj = {}

        processed_records = None

        with multiprocessing.get_context("spawn").Pool(cpus) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []

            for i, record in tqdm(enumerate(records), desc="Processing records"):
                #print(record)
                r.append(
                    export_pool.starmap_async(
                        self.worker,
                        ((record,db),)
                    )
                )
            
            processed_records = []
            for res in tqdm(r, desc="Collecting results"):
                if res.get()[0] is not None:
                    processed_records.extend(res.get()[0])

            #print(processed_records)
            self.df = pd.DataFrame.from_dict(processed_records)

        if source_folder.split("/")[-1] == "STANFORD":
            print("Processing Stanford labels")
            self.process_json_labels()

        print(self.df.head())

    def process_json_labels(self):

        self.stanford_label_mapper = [
            ["NSR"], #NSR
            ["AFIB", "AFL"], #AFIB/AFL
            ["SUDDEN_BRADY", "AVB_TYPE2"], #AVB2/CHB
            ["BIGEMINY"], #BIGEMINY
            ["TRIGEMINY"], #TRIGEMINY
            ["EAR"], #EAR
            ["IVR"], #IVR
            ["JUNCTIONAL"], #JR
            ["SVT"], #SVT
            ["VT"], #VT
            ["WENCKEBACH"], #Wenckebach
            ["NOISE"] #Noise
        ]

        label_256 = []
        label_2000 = []

        for i in range(len(self.df)):
            record = self.df.iloc[i]["ids"]
            annfile = glob.glob(os.path.join(self.source_folder, record + "_grp*.episodes.json"))[0]

            with open(annfile, "r") as f:
                ann = json.load(f)

            episodes = ann["episodes"]
            labels = np.zeros(self.df.iloc[i]["xs"].shape[1], dtype=int)

            for episode in episodes:
                diagnosis = episode["rhythm_name"]
                #print(diagnosis)
                diagnosis = [i for i in range(len(self.stanford_label_mapper)) if diagnosis in self.stanford_label_mapper[i]][0]
                start = episode["onset"]
                end = episode["offset"]
                labels[start:end] = diagnosis

            label_256.append(np.array([np.argmax(np.bincount(labels[i:i+256])) for i in range(0, len(labels), 256)]))
            label_2000.append(np.array([np.argmax(np.bincount(labels[i:i+2000])) for i in range(0, len(labels), 2000)]))

        self.df["label_256"] = label_256
        self.df["label_2000"] = label_2000

                # for diagnosis, start, end in matches:
                #     labelregions.append([diagnosis, float(start)*200, float(end)*200])
                #     labels.append(diagnosis)


    def preprocess_ecg_ecgfounder(self, ecg, fs):

        ecg = self.resize_signal(ecg, int(ecg.shape[-1]*(500/fs)))
        fs = 500

       #highpass filter 1hz
        b, a = butter(4, 1/(fs/2), btype='highpass')
        ecg = filtfilt(b, a, ecg)

        #lowpass filter 30hz
        b, a = butter(2, 30/(fs/2), btype='lowpass')
        ecg = filtfilt(b, a, ecg)

        #notch filter 50hz
        b, a = sps.iirnotch(50, 30, fs)
        ecg = filtfilt(b, a, ecg)
        #notch filter 60hz
        b, a = sps.iirnotch(60, 30, fs)
        ecg = filtfilt(b, a, ecg)
        
        if np.sum(ecg) == 0:
            return ecg
        ecg = ecg - np.mean(ecg)
        ecg = (ecg) / (np.std(ecg))

        return ecg

    def preprocess_ecg(self, ecg, fs):

        ecg = self.resize_signal(ecg, int(ecg.shape[-1]*(200/fs)))
        
        if np.sum(ecg) == 0:
            return ecg
        ecg = ecg - np.mean(ecg)
        ecg = (ecg) / (np.std(ecg))

        return ecg

    def to_one_hot(self, label):
        one_hot = np.zeros((len(self.label_mapper),))
        one_hot[label] = 1
        return one_hot

    def worker(self, record, db):
        
        record = os.path.join(self.source_folder, record)

        #get file without extension
        record = os.path.splitext(record)[0]

        rec = wfdb.rdrecord(record)
        ecg = rec.p_signal
        fs = rec.fs

        #monitor_memory()
        rawII = np.array(ecg)[:,0]
        if self.multiple_labels:
            rawII = self.preprocess_ecg_ecgfounder(rawII, fs)
        else:
            rawII = self.preprocess_ecg(rawII, fs)

        rawII = rawII[:int(len(rawII)/self.size)*self.size]
        outs = []
        outs.append({"xs":np.array([rawII]), "ids":record.split("/")[-1], "dbs":db})

        # for i in range(len(rawII)//self.size):
        #     signal = np.array([rawII[i*self.size:(i+1)*self.size]])

        #     ids = record.split("/")[-1]

        #     outs.append({"xs":signal, "ids":ids, "dbs":db})

        return outs

    def get_data_shape(self):
        return self.df.iloc[0]["xs"].shape

    def get_label_shape(self):
        return self.df.iloc[0]["ys"].shape

    def get_sampling_probabilities(self):
        return self.df["sampling_probability"].values

    def get_ids(self):
        return np.arange(len(self.df))

    def load_case(self, idx):
        row = self.df.iloc[idx]
        return row["xs"], row["ys"], {"records":row["records"], "dbs":row["dbs"], "ids":row["ids"]}

    def get_sampled_data(self, p):
        #sample df using p
        return self.df.sample(frac=p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = np.array(row["xs"]).astype(np.float32)

        if len(sig.shape) == 1:
            sig = np.expand_dims(sig, 0)

        return {"xs":sig, "id":row["ids"], "dbs":row["dbs"]}

class MongoDataLoader(DataLoader):
    def __init__(self,
                 data: MongoDBChunkGenerator,
                 batch_size: int,
                 do_sampling=False,
                 do_balanced=False,
                 multiple_labels=False,
                 transforms=None):
        super().__init__(data, batch_size, 1, None, True, False, True, None)

        if do_balanced:
            self.indices = list(data.get_balanced_ids())
        else:
            self.indices = list(data.get_ids())
        #self.indices = list(data.get_balanced_ids())

        #self.list_of_keys = list(self._data.get_balanced_ids())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        
        self.data_shape, self.lbl_shape = self.determine_shapes()
        self.multiple_labels = multiple_labels
        if do_sampling and not do_balanced:
            self.sampling_probabilities = list(self._data.get_sampling_probabilities())
            self.sampling_probabilities = self.sampling_probabilities / np.sum(self.sampling_probabilities)
        else:
            self.sampling_probabilities = None
        self.transforms = transforms

    def determine_shapes(self):
        data_shape = (self.batch_size, *self._data.get_data_shape())
        lbl_shape = (self.batch_size, *self._data.get_label_shape())

        return data_shape, lbl_shape

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        lbl_all = np.zeros(self.lbl_shape, dtype=np.float32)

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            data, lbl, properties = self._data.load_case(current_key)

            data_all[j] = data
            lbl_all[j] = lbl

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):

                    data_all = torch.from_numpy(data_all).float()
                    if self.multiple_labels:
                        lbl_all = torch.from_numpy(lbl_all)
                    else:
                        lbl_all = torch.from_numpy(lbl_all).to(torch.int64)
                    ecgs = []
                    lbls = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'label': seg_all[b]})
                        ecgs.append(tmp['image'])
                        lbls.append(tmp['label'])
                    data_all = torch.stack(ecgs)
                    if isinstance(segs[0], list):
                        lbl_all = [torch.stack([s[i] for s in lbls]) for i in range(len(lbls[0]))]
                    else:
                        lbl_all = torch.stack(lbls)
                    del lbls, ecgs

            return {'xs': data_all, 'ys': lbl_all, 'keys': selected_keys}

        data_all = torch.from_numpy(data_all).float()
        if self.multiple_labels:
            lbl_all = torch.from_numpy(lbl_all)
        else:
            lbl_all = torch.from_numpy(lbl_all).to(torch.int64)

        return {'xs': data_all, 'ys': lbl_all, 'keys': selected_keys}
