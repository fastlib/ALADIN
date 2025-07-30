import numpy as np
import pandas as pd
import os
import pickle
import scipy.signal as sps
from scipy.signal import butter, filtfilt
import json
import multiprocessing

from torch.utils.data import Dataset

import joblib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch

from pymongo import MongoClient
import psutil



def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

class Collator:
    def __call__(self, batch):
        assert all('xs' in x for x in batch)
        assert all('ys' in x for x in batch)
        assert all('labels' in x for x in batch)
        assert all('records' in x for x in batch)
        assert all('dbs' in x for x in batch)
        
        return {
            'xs': torch.tensor(np.array([x['xs'] for x in batch])),
            'ys': torch.tensor(np.array([x['ys'] for x in batch])),
            'labels': torch.tensor(np.array([x['labels'] for x in batch])),
            'records': np.array([x['records'] for x in batch]),
            'dbs': np.array([x['dbs'] for x in batch]),
        }
    
class GenericDatasetGenerator(Dataset):

    def __init__(self, detector="gt", cachenames=[""], split=True, type="train", size=256, batch_size=16, channels=["signal"], augment=False, patientsplit=False, trainsplit=None, testsplit=None, valsplit=None, train_ratio=0.5, message="Loading signals", folder="./data/"):
        super().__init__()
        
        self.detector = detector
        self.cachenames = cachenames
        self.cachename = "_".join(cachenames)
        self.folder = folder
        self.size = size
        self.batch_size = batch_size
        self.augment = augment
        self.patientsplit = patientsplit
        self.trainsplit = trainsplit
        self.testsplit = testsplit
        self.valsplit = valsplit
        self.channels = channels
        self.type = type

        #check if file exists
        if(os.path.isfile(self.folder+self.cachename+"_processed.pkl")):
            with open(self.folder+self.cachename+"_processed.pkl", "rb") as f:
                self.df = pickle.load(f)
        else:
            print("Create dataset")
            data = []
            for i in range(len(self.cachenames)):
                with open(self.folder+self.cachenames[i]+".pkl", "rb") as f:
                    data.extend(pickle.load(f))

            data = pd.DataFrame(data)
            if self.cachenames[0].split("/")[-1] == "stanford" or self.cachenames[0].split("/")[-1]  == "stanford_with_kappa":
                data = data[data["db"] == "STANFORD"]
            else:
                data = data[data["db"] != "STANFORD"]

            print(data.head())

            #data = self.add_rpeaks(data)

            cpus = joblib.cpu_count()-10

            #divide data into chunks
            data = np.array_split(data, cpus)
            
            print("Using ", cpus, " cores")
            print(message)
            obj = {}

            self.df = pd.DataFrame()
            with ProcessPoolExecutor(max_workers=cpus) as executor:
                for res in list(tqdm(executor.map(self.worker, data), total=len(data))):
                    reskeys = res.keys()
                    for key in reskeys:
                        if key not in obj:
                            obj[key] = []
                        obj[key].extend(res[key])




            self.df = pd.DataFrame.from_dict(obj)
            self.df["ids"] = np.arange(len(self.df))

            #print distribution of ys
            if split:
                if self.patientsplit:
                    if self.trainsplit is not None and self.testsplit is not None:
                        self.trainsplit = [str(rec) for rec in self.trainsplit]
                        self.testsplit = [str(rec) for rec in self.testsplit]
                        self.df = self.df[(self.df["records"].isin(self.trainsplit)) | (self.df["records"].isin(self.testsplit))]

                        self.predefined_patient_split(self.trainsplit, self.testsplit, self.valsplit)
                    else:
                        self.random_patient_split(train_ratio=train_ratio)
                else:
                    self.random_split(train_ratio=train_ratio)

                with open((self.folder+self.cachename+"_processed.pkl"), "wb") as f:
                    pickle.dump(self.df, f, protocol=4)

        if split:
            if(self.type=="train"):
                self.df = self.df[self.df["istrain"]==1]
                #self.shuffle(equal=True, interleave=True)
            if(self.type=="test"):
                self.df = self.df[self.df["istrain"]==0]
            if(self.type=="val"):
                self.df = self.df[self.df["isval"]==1]

        print(self.type+" set size: ", len(self.df))
        labels = self.df["labels"].unique()
        for label in labels:
            print("-", label, len(self.df[self.df["labels"]==label]))


        # print(self.type, "records: [")
        # for record in self.df["records"].unique():
        #     print(record, end=",")
        # print("]")
    
    def shuffle(self, equal=False, interleave=False):
        dbs = self.df["dbs"].unique()
        newdf = pd.DataFrame()

        if equal:
            minlength = len(self.df)
            for db in dbs:
                length = len(self.df[self.df["dbs"]==db])
                newlength = int(length/self.batch_size)*self.batch_size
                minlength = min(minlength, newlength)
        for db in dbs:
            length = len(self.df[self.df["dbs"]==db])
            if equal:
                newlength = minlength
            else:
                newlength = int(length/self.batch_size)*self.batch_size
            newdf = pd.concat([newdf,self.df[self.df["dbs"]==db].sample(newlength)])

        self.df = newdf

        if interleave:
            ntwobatch = len(self.df)//(2*self.batch_size)
            newdf = pd.DataFrame()
            for i in range(ntwobatch):
                newdf = pd.concat([newdf,self.df.iloc[(ntwobatch+i)*self.batch_size:(ntwobatch+i+1)*self.batch_size]])
                newdf = pd.concat([newdf,self.df.iloc[i*self.batch_size:(i+1)*self.batch_size]])

            self.df = newdf

    def predefined_patient_split(self, trainsplit, testsplit, valsplit=None):
        print("Predefined patient split calculation")
        self.df.loc[self.df["records"].isin(trainsplit),"istrain"] = 1
        self.df.loc[self.df["records"].isin(testsplit),"istrain"] = 0
        if valsplit is not None:
            self.df.loc[self.df["records"].isin(valsplit),"isval"] = 1
            self.df.loc[~self.df["records"].isin(valsplit),"isval"] = 0
        else:
            self.df.loc[self.df["records"].isin(testsplit),"isval"] = 1
            self.df.loc[~self.df["records"].isin(testsplit),"isval"] = 0


    def random_split(self, train_ratio=0.5, trainrecords = None):
        print("Random split calculation while accounting for patients with multiple records")
        recordnames = self.df["records"]
        uniquerecordnames = recordnames.unique()

        if trainrecords is None:
            self.df["istrain"] = 0
            self.df["isval"] = 1
            trainrecords = np.random.choice(uniquerecordnames, int(len(uniquerecordnames)*train_ratio), replace=False)
            self.df.loc[self.df["records"].isin(trainrecords),"istrain"] = 1
            self.df.loc[self.df["records"].isin(trainrecords),"isval"] = 0
            
            return trainrecords
        else:
            self.df["istrain"] = 0
            self.df.loc[self.df["records"].isin(trainrecords),"istrain"] = 1

    def random_patient_split(self, train_ratio=0.5, trainids = None):
        print("Random balanced patient split calculation")
        if trainids is None:
            if "fulllabel" not in self.df.columns:
                self.df["fulllabel"] = self.df["labels"]
            labels = self.df["fulllabel"].unique()

            for label in labels:
                self.df.loc[self.df["fulllabel"]==label,"istrain"] = 0
                self.df.loc[self.df["fulllabel"]==label,"isval"] = 1
                trainn = int(len(self.df[self.df["fulllabel"]==label])*train_ratio)
                if trainn > 0:
                    trainrecs = np.random.choice(self.df[self.df["fulllabel"]==label]["records"].values, int(len(self.df[self.df["fulllabel"]==label])*train_ratio), replace=False)
                    self.df.loc[self.df["records"].isin(trainrecs),"istrain"] = 1
                    self.df.loc[self.df["records"].isin(trainrecs),"isval"] = 0
                else:
                    sample = np.random.random(1) 
                    if sample < train_ratio:
                        self.df.loc[self.df["fulllabel"]==label,"istrain"] = 1
                        self.df.loc[self.df["fulllabel"]==label,"isval"] = 0

            trainids = self.df[self.df["istrain"]==1]["ids"].values
        
            return trainids
        else:
            self.df["istrain"] = 0
            self.df.loc[self.df["ids"].isin(trainids),"istrain"] = 1

    def resize_signal(self, signal, new_length=128):
        old_length = signal.shape[-1]
        old_indices = np.arange(old_length)
        new_indices = np.linspace(0, old_length - 1, new_length)
        #interpolate signal that has two dimensions (channels, signal)
        if(len(signal.shape)==2):
            resized_signal = np.zeros((signal.shape[0], new_length))
            for i in range(signal.shape[0]):
                resized_signal[i] = np.interp(new_indices, old_indices, signal[i])
        else:
            resized_signal = np.interp(new_indices, old_indices, signal)
        return resized_signal
    
    def moving_lambda(self, x, stride, lmbda, axis = 0):
        x = np.swapaxes(np.copy(x),0,axis)
        return np.swapaxes([lmbda(x[i:i+stride]) for i in range(0,len(x),stride)],0,axis)
    
    def preprocess_ecg(self, ecg, fs):

        
        # filter signal between 5 and 30 Hz
        highcut = 40
        lowcut = 0.2
        nyquist = 0.5 * fs
        high = highcut / nyquist
        low = lowcut / nyquist
        # b, a = butter(4, [low,high], btype='band')
        b, a = butter(4, high, btype='low')

        signal = filtfilt(b, a, ecg)
        #signal = ecg

        ms200 = int(0.2 * fs)
        ms600 = int(0.6 * fs)
        ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
        ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
        baseline = sps.medfilt(sps.medfilt(signal, ms200), ms600)
        signal = signal - baseline


        # oldsig = signal.copy()
        # oldsig = np.pad(oldsig, (int(fs*2),int(fs*2)), 'constant', constant_values=(0, 0))
        # gaussian = sps.windows.gaussian(int(fs*4), int(fs*4)//4)
        # for i in range(len(signal)):
        #     st = max(0, i)
        #     en = min(len(oldsig), int(i+fs*4))

        #     maximum = np.max(oldsig[st:en]*gaussian)
        #     minimum = np.min(oldsig[st:en]*gaussian)
        #     peak = max(abs(maximum), abs(minimum))
        #     if peak == 0:
        #         peak = 1
        #     signal[i] /= peak

        #normalize using rolling median
        # oldsig = signal.copy()
        # oldsig = np.pad(oldsig, (int(fs),int(fs)), 'constant', constant_values=(0, 0))
        # gaussian = sps.windows.gaussian(int(fs*2), int(fs*2)//4)
        # for i in range(len(signal)):
        #     maximum = np.max(oldsig[i:i+len(gaussian)]*gaussian)
        #     if signal[i] != 0:
        #         signal[i] /= maximum

        if np.sum(signal) == 0:
            return signal
        #ecg = ecg - np.mean(ecg)
        signal = signal - np.mean(signal)
        signal = (signal) / (np.std(signal))
        #ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
        return signal
    
    def worker(self, data):
        return NotImplementedError
    
    def label(self, idx, label):
        self.df.at[idx,"ys"] = label

    def save(self, filename="pool.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.df, f, protocol=4)

    def load(self, filename="pool.pkl"):
        with open(filename, "rb") as f:
            self.df = pickle.load(f)

        withlabels = 0
        for i in range(len(self.df)):
            #check if array has only ones
            ys = self.df.iloc[i]["ys"]
            if not np.all(ys[3]==1):
                withlabels += 1

        print("With labels:", withlabels)
            
    def __len__(self):
        return len(self.df)
    
    def get_df(self):
        return self.df
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = row["xs"].astype(np.float32)
        seg = row["ys"].astype(np.float32)
        if len(img.shape) == 1:
            img = np.expand_dims(img, 0)
        if len(seg.shape) == 1:
            seg = np.expand_dims(seg, 0)
        return {"xs":img, "ys":seg, "labels":row["labels"], "records":row["records"], "dbs":row["dbs"], "is_labeled":row["is_labeled"]}
    
    def __remove__(self, idx):
        self.df = self.df.drop(idx)

class MongoDBDatasetGenerator(GenericDatasetGenerator):
    def __init__(self, detector="gt", cachenames=[""], split=True, type="train", expid=None, size=256, batch_size=16, augment=False, diagnoses=[], patientsplit=False, trainsplit=None, testsplit=None, valsplit=None, train_ratio=0.5, message="Preprocessing...", folder="./data/"):
        
        self.detector = detector
        self.cachenames = cachenames
        self.cachename = "_".join(cachenames)
        self.folder = folder
        self.size = size
        self.batch_size = batch_size
        self.augment = augment
        self.patientsplit = patientsplit
        self.trainsplit = trainsplit
        self.testsplit = testsplit
        self.valsplit = valsplit
        self.type = type
        self.expid = expid
        self.diagnoses = diagnoses

        #check if file exists
        if(os.path.isfile(self.folder+self.cachename+"_processed.pkl")):
            with open(self.folder+self.cachename+"_processed.pkl", "rb") as f:
                self.df = pickle.load(f)
        else:
            print("Create dataset")

            #connect to MongoDB
            client = MongoClient('mongodb://localhost:27017/', username='root', password='j230sdncjsdf234')
            db = client['orion']
            datadb = db['data']
            aldb = db['al']
            arrdb = db['arrhythmia']

            print("Retrieving data from MongoDB...")

            #see if we need to retrieve labeled or unlabeled data
            if self.type=="train" or self.type=="lstm_train":

                #see if we were given an experiment id, in that case we need to ask to aldb for the recordings
                if self.expid is not None:
                    
                    print("Retrieve labeled records based on expid (train)")
                    recs = list(aldb.aggregate([
                        {"$match": { "_id": expid }},
                        {"$unwind": "$iterations"},
                        {"$unwind": "$iterations.recordings"},
                        {"$group": {"_id": "$_id", "recordings": {"$push": "$iterations.recordings"}}}
                    ]))

                    #if no recordings are found, we return an empty list, otherwise we retrieve the recordings from datadb
                    if recs == []:
                        records = []
                    else:
                        ids = recs[0]["recordings"]
                        records = list(datadb.find({"_id": {"$in": ids}, "status": {"$ne": "skipped"}, "database": {"$nin": ["ICENTIA"]}}))
                        notlabeled = []
                        for record in records:
                            if record["status"] != "labelled":
                                notlabeled.append(record["recordname"])

                        #we expect all records to be labelled, otherwise we raise an exception
                        if len(notlabeled) > 0:
                            raise Exception("Records not labelled ", notlabeled)
                else:
                    #we got no expid, so we retrieve all labelled records
                    print("Retrieve labeled records (train)")
                    records = list(datadb.find({"status": "labeled", "database": {"$nin": ["ICENTIA"]}}))
            elif self.type=="pool" or self.type=="lstm_pool":
                #we are prediction, so we need only unlabeled data to form the pool
                blacklisted = ["VFIB","VFLUT","PR","AAPR","AVPR","AFL"]
                includediagnoses = diagnoses if len(diagnoses) > 0 else list(arrdb.find({},{"id":1,"_id":0}))

                if self.expid is not None:
                    #we got an expid, so we need to retrieve the recordings that are not in the expid
                    print("Retrieve unlabeled records based on expid (pool)")
                    recs = list(aldb.aggregate([
                        {"$match": { "_id": expid }},
                        {"$unwind": "$iterations"},
                        {"$unwind": "$iterations.recordings"},
                        {"$group": {"_id": "$_id", "recordings": {"$push": "$iterations.recordings"}}}
                    ]))

                    #if no recordings are in aldb, we can return all unlabeled records
                    if recs == []:
                        if len(diagnoses) > 0:
                            records = list(datadb.find({"diagnoses": {"$in": includediagnoses, "$nin": blacklisted}, "database": {"$nin": ["INCART"]},"status": "unlabeled"}))
                        else:
                            records = list(datadb.find({"diagnoses": {"$nin": blacklisted}, "database": {"$nin": ["INCART"]},"status": "unlabeled"}))
                    else:
                        #otherwise we retrieve those recordings that are not in the expid
                        if len(diagnoses) > 0:
                            records = list(datadb.find({"_id": {"$nin": list(recs)[0]["recordings"]}, "diagnoses": {"$in": includediagnoses, "$nin": blacklisted}, "database": {"$nin": ["INCART"]}}))
                        else:
                            records = list(datadb.find({"_id": {"$nin": list(recs)[0]["recordings"]}, "diagnoses": {"$nin": blacklisted}, "database": {"$nin": ["INCART"]}}))
                else:
                    #we got no expid, so we retrieve all unlabeled records
                    print("Retrieve unlabeled records (pool)")
                    if len(diagnoses) > 0:
                        records = list(datadb.find({"diagnoses": {"$in": includediagnoses, "$nin": blacklisted}, "status": "unlabeled"}))
                    else:
                        records = list(datadb.find({"diagnoses": {"$nin": blacklisted}, "status": "unlabeled"}))
            elif self.type=="all":
                #we are prediction, so we need only unlabeled data to form the pool
                blacklisted = ["VFIB","VFLUT","PR","AAPR","AVPR","AFL"]

                records = list(datadb.find({"diagnoses": {"$nin": blacklisted}, "status": {"$ne": "skipped"}, "database": {"$nin": ["ICENTIA"]}}))
            elif self.type=="labeled":
                #we are prediction, so we need only unlabeled data to form the pool
                blacklisted = ["VFIB","VFLUT","PR","AAPR","AVPR","AFL"]

                records = list(datadb.find({"diagnoses": {"$nin": blacklisted}, "status": "labelled", "database": {"$nin": ["ICENTIA"]}}))
            elif self.type=="train_only_labels" or self.type=="val_only_labels":
                includediagnoses = diagnoses if len(diagnoses) > 0 else list(arrdb.find({},{"id":1,"_id":0}))

                records = list(datadb.find({"diagnoses": {"$in": includediagnoses}, "database": {"$nin": ["ICENTIA"]}}))

                    

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
                            ((record,),)
                        )
                    )
                
                processed_records = []
                for res in tqdm(r, desc="Collecting results"):
                    if res.get()[0] is not None:
                        processed_records.append(res.get()[0])

            #print(processed_records)
            self.df = pd.DataFrame.from_dict(processed_records)

            #print distribution of ys
            if split:
                if self.patientsplit:
                    if self.trainsplit is not None and self.testsplit is not None:
                        self.trainsplit = [str(rec) for rec in self.trainsplit]
                        self.testsplit = [str(rec) for rec in self.testsplit]
                        self.df = self.df[(self.df["records"].isin(self.trainsplit)) | (self.df["records"].isin(self.testsplit))]

                        self.predefined_patient_split(self.trainsplit, self.testsplit, self.valsplit)
                    else:
                        self.random_patient_split(train_ratio=train_ratio)
                else:
                    self.random_split(train_ratio=train_ratio)

                with open((self.folder+self.cachename+"_processed.pkl"), "wb") as f:
                    pickle.dump(self.df, f, protocol=4)

        #tally the database distribution
        dbs = self.df["dbs"].unique()
        for db in dbs:
            print(db, len(self.df[self.df["dbs"]==db]))

        if split:
            if(self.type=="train" or self.type=="pool" or self.type=="train_only_labels"):
                self.df = self.df[self.df["istrain"]==1]
                #self.shuffle(equal=True, interleave=True)
            if(self.type=="test"):
                self.df = self.df[self.df["istrain"]==0]
            if(self.type=="val" or self.type=="val_only_labels"):
                self.df = self.df[self.df["isval"]==1]

        print(self.type+" set size: ", len(self.df))
    
    def worker(self, data):
        fs = 204.8
        #QRS, T, P, Noise, , Abnorm
        typemapper = [1,2,0,4,0,5]

        #monitor_memory()
        rawII = np.array(data["ecg"])[:,0]
        rawV1 = np.array(data["ecg"])[:,1]
        rawV6 = np.array(data["ecg"])[:,2]
        rawII = self.preprocess_ecg(rawII, fs)
        rawV1 = self.preprocess_ecg(rawV1, fs)
        rawV6 = self.preprocess_ecg(rawV6, fs)

        raw = np.array([rawII, rawV1, rawV6])
        raw = raw[:, None, None, :]

        length = len(rawII)
        signals = raw
        del raw, rawV1, rawV6, rawII
        
        annotationsegs = data["annotations"] if "annotations" in data else []
        annotations = np.zeros((6, length))

        for ann in annotationsegs:
            annotations[typemapper[ann["type"]],int(ann["start"]*fs):int(ann["end"]*fs)] = 1

        annotations[3] = np.ones(length)
        annotations[3, np.where(annotations[0]==1)] = 0
        annotations[3, np.where(annotations[1]==1)] = 0
        annotations[3, np.where(annotations[2]==1)] = 0
        annotations[3, np.where(annotations[4]==1)] = 0
        annotations[3, np.where(annotations[5]==1)] = 0

        if self.type == "lstm_train" or self.type == "lstm_pool":
            split_inds = np.array([0, 0.25, 0.5])*length
            annotations_segments = np.array([annotations[:,s:s+(length//2)] for s in split_inds.astype(int)])
            annotations = np.array(annotations_segments)

        segmentations = annotations
        ids = str(data["_id"])
        records = str(data["recordname"])
        dbs = data["database"]
        props = {"spacing": [999,999,1]}
        onsets = data["onset"]

        if "AFL" in data["diagnoses"] or "AFIB" in data["diagnoses"] or "AFIBAFL" in data["diagnoses"]:
            labels = 1
        else:
            labels = 0

        return {"xs":signals, "ys":segmentations, "labels":labels, "records":records, "dbs":dbs, "ids":ids, "props":props, "onsets":onsets, "diagnoses":data["diagnoses"]}

    def get_items_by_id(self, ids):
        return self.df[self.df["ids"].isin(ids)]
    
    def get_items_by_index(self, idx):
        return self.df.iloc[idx]

    def get_df(self):
        return self.df
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = row["xs"].astype(np.float32)
        seg = row["ys"].astype(np.float32)

        if len(img.shape) == 1:
            img = np.expand_dims(img, 0)
        if len(seg.shape) == 1:
            seg = np.expand_dims(seg, 0)

        return {"xs":img, "ys":seg, "labels":row["labels"], "records":row["records"], "dbs":row["dbs"], "id":row["ids"]}