import numpy as np
import pandas as pd
import os
import wfdb
import pickle
from rich.console import Console
from rich.table import Table
import re
from datetime import datetime
import glob
import ast
import gzip
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import bigquery
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import NoCredentialsError

from concurrent.futures import ThreadPoolExecutor

from aladin.utils.morph import closing
from aladin.utils.helpers import get_regions
from scipy.ndimage import binary_erosion

from tqdm import tqdm
import json
import pickle


def determine_qrs_type(logits):
    merged = logits[2,:] + logits[5,:]

    binary = np.array(merged > 0.5, dtype=int)
    regions = get_regions(binary)

    qrsnorm = logits[2,:]
    qrsabnorm = logits[5,:]
    qrsabnorm[binary==0] = 0
    
    for region in regions:
        #print(np.mean(qrsabnorm[region[0]:region[1]]))
        qrsabnorm[region[0]:region[1]] = 1 if np.mean(qrsabnorm[region[0]:region[1]]) > 0.1 else 0

    logits[2,qrsabnorm==1] = 0
    logits[5,:] = qrsabnorm

    return logits

def proccess(seg, fs=2048/10):

    L1 = int(0.05*fs)

    SE1 = np.zeros((max(3,L1)))

    seg = np.pad(seg, (L1, L1), 'edge')
    seg = closing(seg, SE1)
    seg = seg[L1:-L1]

    return seg

def get_regions(seg, st=0):
    #append 0 to start and end
    seg = np.append(seg, 0)
    seg = np.insert(seg, 0, 0)
    d = np.diff(seg)
    starts = np.where(d==1)[0]
    ends = np.where(d==-1)[0]
    starts = starts - st
    ends = ends - st

    zipped = list(zip(starts, ends))
    return zipped

def resize_signal(signal, new_length=128):
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

def regions_to_binary(regions, length):
    binary = np.zeros(length)
    for (st, en) in regions:
        if np.isnan(st) or np.isnan(en):
            continue
        binary[int(st):int(en)] = 1
    return binary

class AsyncData:
    def __init__(self, name, folder, fraction=1.0):
        self.folder = folder
        self.name = name
        self.classes = []
        self.class_mapper = {}
        self.fraction = fraction
        self.objs = self.get_data()
        self.split_objs()

    def get_name(self):
        return self.name

class BaseDataLoader:
    def __init__(self, folder, asynchronous=False, fraction=1.0):
        self.folder = folder
        self.name = ""
        self.classes = []
        self.class_mapper = {}
        self.objs = {}
        self.asynchronous = asynchronous
        self.fraction = fraction
        self.basefolder = os.environ.get('benchmark_data')

    def get_name(self):
        return self.name

    def init_objects(self):
        raise NotImplementedError("This method should be implemented in the subclass")

    def get_data(self, recordname):
        raise NotImplementedError("This method should be implemented in the subclass")

    def get_data_batch(self, keys):
        """
        This method should be implemented in the subclass to handle batch data retrieval.
        It is expected to return a list of objects corresponding to the keys provided.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def __len__(self):
        return len(self.objs)

    def keys(self):
        return list(self.objs.keys())

    def upload_record(self, record):
        return

    def set_as_finished(self, keys):
        return

    def __getitem__(self, item):
        if not isinstance(item, str):
            key = list(self.objs.keys())[item]
            return self.get_data(key)
            
        return self.get_data(item)

    def batch(self, batch_size=32):
        keys = list(self.objs.keys())
        for i in range(0, len(keys), batch_size):
            if i + batch_size > len(keys):
                batch_size = len(keys) - i
            batch_keys = keys[i:i + batch_size]
            yield self.get_data_batch(batch_keys)

    def get_class_mapper(self):
        return self.class_mapper

    def get_classes(self):
        return self.classes

    def __iter__(self):
        for key in self.objs:
            yield key

class InternalData(BaseDataLoader):
    def __init__(self, folder, asynchronous=False, fraction=1.0):
        super().__init__(folder, asynchronous, fraction)
        self.name = "INTERNAL"
        self.annotation_data = {}
        self.get_annotation_data()
        self.set_class_mapper()
        self.init_objects()

    def load_arrhythmia_csv(self, path):
        df = {"Major":[],"Minor":[],"Name":[], "Abreviation":[],"Synonyms":[]}

        with open(path, "r") as f:
            for line in f:
                row = line.strip()
                cells = row.split(",")
                major = cells[0]
                minor = cells[1]
                name = cells[2]
                abbr = cells[3]
                df["Major"].append(major)
                df["Minor"].append(minor)
                df["Name"].append(name.strip().lower())
                df["Abreviation"].append(abbr)
                
                synonyms = cells[2:]
                synonymsstripped = []
                for synonym in synonyms:
                    stripped = synonym.strip().lower()
                    if stripped != "":
                        synonymsstripped.append(stripped)
                df["Synonyms"].append(synonymsstripped)

        df = pd.DataFrame(df)
        self.arrhythmia_df = df

    def get_annotation_data(self):
        annfile = self.basefolder+'/VALIDATION/val_matched.pkl'
        self.load_arrhythmia_csv(self.basefolder+'/arrhythmia_classes.csv')
        print(self.arrhythmia_df)

        dat = pickle.load(open(annfile, 'rb'))

        for d in dat:
            record_id = "record_" + str(d['record'])
            #print(d)
            labels = d['fulllabel'].split(",")
            self.annotation_data[record_id] = []
            for label in labels:
                label = label.strip().lower()
                #iterate dataframe
                for i, row in self.arrhythmia_df.iterrows():
                    if label in row["Synonyms"]:
                        self.annotation_data[record_id].append(row["Abreviation"])
                        break
                #else:
                #    print(label)
                    
            print(self.annotation_data[record_id])

    def init_objects(self):
        annfile = self.basefolder+'/VALIDATION/val_matched.pkl'
        dat = pickle.load(open(annfile, 'rb'))

        self.objs = {}

        for d in dat:
            record_id = "record_" + str(d['record'])
            labels = self.annotation_data[record_id]
            labelregions = []
            ecg = d['signal']

            #print(labels)
            if not np.any([label in labels for label in self.classes]):
                continue

            for label in labels:
                if label in self.class_mapper:
                    labelregions.append([self.class_mapper[label], 0, len(ecg)])


            self.objs[record_id] = {
                "record": record_id, 
                "path": "",
                "initialized": True,
                "signal": ecg,  
                "kappa": 0,
                "fs":204.8, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions
            }

        print("Initialized objects for INTERNAL")
        print("Number of objects:", len(self.objs))

    def get_data(self, recordname):
        
        obj = self.objs[recordname]

        if obj["initialized"]:
            return obj

    def get_data_batch(self, keys):
        return [self.objs[key] for key in keys if key in self.objs]

    def set_class_mapper(self):
        self.classes = [
            "SR",
            "SB",
            "AFIB",
            "AFL",
            "AFIBAFL",
            "AVB II",
            "CHB",
            "AVD",
            "VT",
            "WENCK",
            "SVT",
            "JR",
            "PVC",
            "TRI",
            "BIG",
            "AR",
            "AER",
            "PChange"
        ]
        self.class_mapper = {c: c for c in self.classes}
        self.class_mapper["AFIB"] = "AFIB/AFL"
        self.class_mapper["AFL"] = "AFIB/AFL"
        self.class_mapper["AFIBAFL"] = "AFIB/AFL"
        self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["SUDDEN_BRADY"] = "AVB"
        self.class_mapper["AVB II"] = "AVB"
        self.class_mapper["CHB"] = "AVB"
        self.class_mapper["AVD"] = "AVB"
        self.class_mapper["SR"] = "NSR"
        self.class_mapper["SB"] = "NSR"
        self.class_mapper["ST"] = "NSR"
        self.class_mapper["WENCK"] = "WENCKEBACH"
        self.class_mapper["JR"] = "JUNCTIONAL"
        self.class_mapper["AJR"] = "JUNCTIONAL"
        self.class_mapper["TRI"] = "TRIGEMINY"
        self.class_mapper["BIG"] = "BIGEMINY"
        self.class_mapper["AR"] = "EAR"
        self.class_mapper["AER"] = "EAR"
        self.class_mapper["PChange"] = "EAR"

class StanfordData(BaseDataLoader):
    def __init__(self, folder, asynchronous=False, fraction=1.0):
        super().__init__(folder, asynchronous, fraction)
        self.name = "STANFORD"
        credentials = service_account.Credentials.from_service_account_file(self.basefolder+'/aladin-466917-e056430d6165.json')
        self.bucket = storage.Client(credentials=credentials).get_bucket("arts-aladin")
        self.annotation_data = []
        self.get_annotation_data()
        self.init_objects()
        self.set_class_mapper()

    def get_annotation_data(self):
        annfile = self.basefolder+'/'+self.folder+'/stanford2.pkl'
        if not os.path.exists(annfile):
            print("Downloading annotation data for STANFORD")
            blob = self.bucket.blob("STANFORD/stanford2.pkl")
            blob.download_to_filename(annfile)
            
        dat = pickle.load(open(annfile, 'rb'))
        self.annotation_data = {d['record']:d for d in dat if d['db'] == 'STANFORD'}

    def init_objects(self):
        paths = np.loadtxt(self.basefolder+'/'+self.folder+'/RECORDS', dtype=str)

        if self.fraction < 1.0:
            paths = np.random.choice(paths, int(len(paths)*self.fraction), replace=False)
            print("Fraction of data: ", len(paths))

        for path in tqdm(paths, desc="Initializing Stanford records"):
            recordname = path.split("/")[-1].split(".")[0]
            obj = {"record": recordname, "path": path,"initialized": False}
            self.objs[recordname] = obj

            self.get_data(recordname)

    def get_data(self, recordname):
        
        obj = self.objs[recordname]

        if obj["initialized"]:
            return obj

        local_file_path = os.path.join(self.basefolder, self.folder, obj["path"])

        if not os.path.exists(local_file_path+".dat"):
            print("Downloading", recordname)
            blob = self.bucket.blob("STANFORD/"+obj["path"]+".dat")
            blob.download_to_filename(local_file_path+".dat")
        if not os.path.exists(local_file_path+".hea"):
            blob = self.bucket.blob("STANFORD/"+obj["path"]+".hea")
            blob.download_to_filename(local_file_path+".hea")

        d = self.annotation_data[recordname]
        pwaves = get_regions(d["segmentation"][0,:])
        qrswaves = get_regions(d["segmentation"][1,:])
        qrsabnormwaves = get_regions(d["segmentation"][5,:])
        twaves = get_regions(d["segmentation"][2,:])

        ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
        qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
        qrsas = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrsabnormwaves]))
        ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

        if len(ps) == 0:
            ps = np.array([[],[],[]])
        if len(qrss) == 0:
            qrss = np.array([[],[],[]])
        if len(qrsas) == 0:
            qrsas = np.array([[],[],[]])
        if len(ts) == 0:
            ts = np.array([[],[],[]])

        labellines = d["fulllabel"].split("\n")
        labelregions = []

        for line in labellines:
            regex = r"(\S+) (\d+\.\d+)s -(\d+\.\d+)s"
            matches = re.findall(regex, line)

            for diagnosis, start, end in matches:
                labelregions.append([diagnosis, float(start)*200, float(end)*200])

        rec = wfdb.rdrecord(local_file_path)
        sig = rec.p_signal[:,0]

        self.objs[recordname] = {
            "record": d["record"], 
            "path": obj["path"],
            "initialized": True,
            "signal": sig,  
            "kappa": 0,
            "fs":200, 
            "p": ps,
            "qrs": qrss, 
            "qrsabnorm": qrsas,
            "t": ts,
            "p_binary": d["segmentation"][0,:],
            "qrs_binary": d["segmentation"][1,:],
            "qrsabnorm_binary": d["segmentation"][5,:],
            "t_binary": d["segmentation"][2,:],
            "labelregions": labelregions
        }
    
    def get_data_batch(self, keys):
        return [self.objs[key] for key in keys if key in self.objs]

    def set_class_mapper(self):
        self.classes = ["AFIB", "AFL", "AVB_TYPE2", "BIGEMINY", "SUDDEN_BRADY", "EAR", "IVR", "JUNCTIONAL", "NOISE", "NSR", "SVT", "TRIGEMINY", "VT", "WENCKEBACH"]
        self.class_mapper = {c: c for c in self.classes}
        self.class_mapper["AFIB"] = "AFIB/AFL"
        self.class_mapper["AFL"] = "AFIB/AFL"
        self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["SUDDEN_BRADY"] = "AVB"
        self.class_mapper["AVB_TYPE1"] = "NSR"
        self.class_mapper["BRADYCARDIA"] = "NSR"
        self.class_mapper["TACHYCARDIA"] = "NSR"
        self.class_mapper["AIVR"] = "IVR"

class CINCData(BaseDataLoader):
    def __init__(self, folder, asynchronous=False, fraction=1.0):
        super().__init__(folder, asynchronous, fraction)
        self.name = "CINC"
        self.boto_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.annotation_data = []
        self.num_workers = 32
        self.get_annotation_data()
        self.init_objects()
        self.set_class_mapper()

    def get_annotation_data(self):
        self.annotation_data = pd.read_csv(self.basefolder+'/'+self.folder+'/REFERENCE-v3.csv', header=None, names=["Recording", "Label"])
        #self.annotation_data = self.annotation_data[(self.annotation_data["Label"] == "N") | (self.annotation_data["Label"] == "A")]
        #print(len(self.annotation_data))

    def init_objects(self):
        paths = np.loadtxt(self.basefolder+'/'+self.folder+'/RECORDS', dtype=str)

        if self.fraction < 1.0:
            paths = np.random.choice(paths, int(len(paths)*self.fraction), replace=False)
            print("Fraction of data: ", len(paths))

        for path in tqdm(paths, desc="Initializing CINC records"):
            recordname = path
            recordname.replace("/", "_")
            if self.annotation_data[self.annotation_data["Recording"] == recordname].empty:
                continue
            obj = {"record": recordname, "path": path,"initialized": False}
            self.objs[recordname] = obj

            if not self.asynchronous:
                self.get_data(recordname)

    def get_data(self, recordname):

        obj = self.objs[recordname]

        if obj["initialized"]:
            return obj

        local_file_path = os.path.join(self.basefolder, self.folder, obj["path"])

        # Define the S3 bucket and the local directory
        bucket_name = 'physionet-open'
        prefix = 'challenge-2017/1.0.0/training/'

        if not os.path.exists(local_file_path+".mat"):
            print("Downloading", recordname)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".mat", local_file_path+".mat")
        if not os.path.exists(local_file_path+".hea"):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".hea", local_file_path+".hea")

        rec = wfdb.rdrecord(local_file_path)
        ecg = rec.p_signal[:,0]
        fs = rec.fs
        #ecg = resize_signal(ecg, int(ecg.shape[-1]*(200/fs)))

        #print(csvanns["Recording"])
        label = self.annotation_data[self.annotation_data["Recording"] == recordname]["Label"].values[0]

        labelregions = [[label, 0, len(ecg)]]

        self.objs[recordname] = {
            "record": recordname, 
            "path": obj["path"],
            "initialized": True,
            "signal": ecg,  
            "kappa": 0,
            "fs":300, 
            "p": None,
            "qrs": None, 
            "qrsabnorm": None,
            "t": None,
            "p_binary": None,
            "qrs_binary": None,
            "qrsabnorm_binary": None,
            "t_binary": None,
            "labelregions": labelregions
        }

        return self.objs[recordname]

    def download_file(self, obj, pbar):
        bucket_name = 'physionet-open'
        prefix = 'challenge-2017/1.0.0/training/'
        os.makedirs(os.path.dirname(obj["local_file_path"]), exist_ok=True)

        self.boto_client.download_file(bucket_name, prefix + obj["path"]+".mat", obj["local_file_path"]+".mat")
        self.boto_client.download_file(bucket_name, prefix + obj["path"]+".hea", obj["local_file_path"]+".hea")
        pbar.update(1)

    def get_data_batch(self, keys):
        
        #first check if the files are already downloaded
        download_objs = []
        for key in keys:
            obj = self.objs[key]
            fp = os.path.join(self.basefolder, self.folder, obj["path"])
            obj["local_file_path"] = fp

            if not os.path.exists(fp+".mat") or not os.path.exists(fp+".hea"):
                download_objs.append(obj)


        if len(download_objs) > 0:
            with tqdm(total=len(download_objs), desc="Downloading files") as pbar:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit each download task and update the progress bar as tasks complete
                    futures = [executor.submit(self.download_file, file_info, pbar) for file_info in download_objs]
                    # Wait for all tasks to complete
                    for future in futures:
                        future.result()

        #now read the files
        for key in keys:

            if self.objs[key]["initialized"]:
                continue

            local_file_path = os.path.join(self.basefolder, self.folder, self.objs[key]["path"])
            rec = wfdb.rdrecord(local_file_path)
            ecg = rec.p_signal[:,0]
            fs = rec.fs

            label = self.annotation_data[self.annotation_data["Recording"] == key]["Label"].values[0]

            labelregions = [[label, 0, len(ecg)]]

            self.objs[key] = {
                "record": key,
                "path": self.objs[key]["path"],
                "initialized": True,
                "signal": ecg,  
                "kappa": 0,
                "fs":300, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions
            }

        return [self.objs[key] for key in keys if key in self.objs]


    def set_class_mapper(self):
        self.classes = ["N", "A", "O", "~"]
        self.class_mapper = {c: c for c in self.classes}

class ICENTIAData(BaseDataLoader):
    def __init__(self, folder, asynchronous=False, fraction=1.0):
        super().__init__(folder, asynchronous, fraction)
        self.name = "ICENTIA"
        self.boto_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.num_workers = 16
        self.set_class_mapper()
        self.credentials = service_account.Credentials.from_service_account_file(self.basefolder+"/aladin-466917-e056430d6165.json")
        self.bigquery_client = bigquery.Client(credentials=self.credentials, project=self.credentials.project_id)
        self.bucket = storage.Client(credentials=self.credentials).get_bucket("arts-aladin")
        self.init_objects()
        self.table_id = "benchmarks.ICENTIA-Cleaned"

    def init_objects(self):
        recordname = "p0_p00000_p00000_s00"
        self.objs[recordname] = {"record": recordname, "path": "p00/p00000/p00000_s00", "initialized": False}


        print("Number of records:", len(self.objs))
        
    def get_data(self, recordname):

        obj = self.objs[recordname]

        if obj["initialized"]:
            return obj

        local_file_path = os.path.join(self.basefolder, self.folder, obj["path"])

        # Define the S3 bucket and the local directory
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'

        if not os.path.exists(local_file_path+".dat"):
            print("Downloading", recordname)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".dat", local_file_path+".dat")
        if not os.path.exists(local_file_path+".hea"):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".hea", local_file_path+".hea")
        if not os.path.exists(local_file_path+".atr"):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".atr", local_file_path+".atr")

        rec = wfdb.rdrecord(local_file_path)
        ecg = rec.p_signal[:,0]
        fs = rec.fs
        #ecg = resize_signal(ecg, int(ecg.shape[-1]*(200/fs)))

        anns = wfdb.rdann(local_file_path, 'atr', return_label_elements=['symbol'])

        labelregions = []
        beattypes = [a for a in anns.symbol if a != '+']
        beattypes = "".join(beattypes)
        beattypes = beattypes.replace('S', 'N')
        #beattypes = beattypes.replace('Q', 'N')
        #print(beattypes)

        big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
        for big_match in big_matches:
            start = anns.sample[big_match.start()]
            end = anns.sample[big_match.end()-1]
            #print("BIG pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
            if (end-start)/fs > 10:
                labelregions.append(["BIGEMINY", start, end])

        tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
        for tri_match in tri_matches:
            start = anns.sample[tri_match.start()]
            end = anns.sample[tri_match.end()-1]
            #print("TRI pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
            if (end-start)/fs > 10:
                labelregions.append(["TRIGEMINY", start, end])

        vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
        for vt_ivr_match in vt_ivr_matches:
            start = anns.sample[vt_ivr_match.start()]
            end = anns.sample[vt_ivr_match.end()-1]
            #print("VT/IVR pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
            if (end-start)/fs > 10:
                labelregions.append(["VT", start, end])
        #search for VVV pattern

        rhythm_type = ""
        rhythm_start = 0
        #foundafib = False

        for idx, beat in enumerate(anns.symbol):
            if beat == 'N':
                continue
            if beat == 'V':
                start = anns.sample[idx] - fs*0.15
                end = anns.sample[idx] + fs*0.15
                #labelregions.append(["PVC", start, end])
            elif beat == 'S':
                start = anns.sample[idx] - fs*0.15
                end = anns.sample[idx] + fs*0.15
                #labelregions.append(["SVPB", start, end])
            elif beat == '+':
                if anns.aux_note[idx] == '(N':
                    rhythm_type = "NSR"
                    rhythm_start = anns.sample[idx]
                elif anns.aux_note[idx] == '(AFIB':
                    rhythm_type = "AFIB/AFL"
                    rhythm_start = anns.sample[idx]
                    foundafib = True
                    #print("AFIB/AFL", rhythm_start)
                elif anns.aux_note[idx] == '(AFL':
                    rhythm_type = "AFIB/AFL"
                    rhythm_start = anns.sample[idx]
                    foundafib = True
                elif anns.aux_note[idx] == ')':
                    if (anns.sample[idx]-rhythm_start)/fs > 30:
                        labelregions.append([rhythm_type, rhythm_start, anns.sample[idx]])
                        # if rhythm_type != "NSR":
                        #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")
                    rhythm_type = ""
                    rhythm_start = 0

        self.objs[recordname] = {
            "record": recordname, 
            "path": obj["path"],
            "initialized": True,
            "signal": ecg,  
            "kappa": 0,
            "fs":250, 
            "p": None,
            "qrs": None, 
            "qrsabnorm": None,
            "t": None,
            "p_binary": None,
            "qrs_binary": None,
            "qrsabnorm_binary": None,
            "t_binary": None,
            "labelregions": labelregions
        }

        return self.objs[recordname]

    def download_file(self, obj, pbar):
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        print("Downloading", prefix + obj["path"]+".dat", "dir:", os.path.dirname(obj["local_file_path"]))
        os.makedirs(os.path.dirname(obj["local_file_path"]), exist_ok=True)

        try:
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".dat", obj["local_file_path"]+".dat")
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".hea", obj["local_file_path"]+".hea")
            self.boto_client.download_file(bucket_name, prefix + obj["path"]+".atr", obj["local_file_path"]+".atr")
            pbar.update(1)
            return True
        except:
            pbar.update(1)
            print("Error downloading", obj["path"])
            return obj["record"]

    def download_record_index(self, path):
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        local_file_path = os.path.join(self.basefolder, self.folder, path, "RECORDS")
        print(prefix + path +"RECORDS")
        if not os.path.exists(local_file_path):
            print("Downloading record index of ", path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.boto_client.download_file(bucket_name, prefix + path +"RECORDS", local_file_path)

    def get_data_batch(self, paths):
        
        #delete self.objs
        self.objs = {}

        for path in paths:
            recordname = path
            recordname = recordname.split("/")[-1]
            obj = {"record": recordname, "path": path, "initialized": False, "done": False}
            self.objs[recordname] = obj

        #first check if the files are already downloaded
        download_objs = []
        for path in paths:
            recordname = path
            recordname = recordname.split("/")[-1]
            fp = os.path.join(self.basefolder, self.folder, self.objs[recordname]["path"])
            self.objs[recordname]["local_file_path"] = fp
            print("Checking", fp)

            if not os.path.exists(fp+".dat") or not os.path.exists(fp+".hea") or not os.path.exists(fp+".atr"):
                download_objs.append({"record": recordname, "path": self.objs[recordname]["path"], "local_file_path": fp})

        errors = []
        with tqdm(total=len(download_objs), desc="Downloading files") as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit each download task and update the progress bar as tasks complete
                futures = [executor.submit(self.download_file, file_info, pbar) for file_info in download_objs]
                # Wait for all tasks to complete
                for future in futures:
                    res = future.result()
                    if res != True:
                        errors.append(res)

        print("Files downloaded, now reading...")

        res = []
        #now read the files
        for path in paths:
            recordname = path
            recordname = recordname.split("/")[-1]
            if recordname in errors:
                print("Error downloading", recordname)
                continue
            local_file_path = os.path.join(self.basefolder, self.folder, self.objs[recordname]["path"])
            rec = wfdb.rdrecord(local_file_path)
            ecg = rec.p_signal[:,0]
            fs = rec.fs

            anns = wfdb.rdann(local_file_path, 'atr', return_label_elements=['symbol'])

            labelregions = []
            beattypes = [a for a in anns.symbol if a != '+']
            beattypes = "".join(beattypes)
            beattypes = beattypes.replace('S', 'N')
            #beattypes = beattypes.replace('Q', 'N')
            #print(beattypes)

            big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
            for big_match in big_matches:
                start = anns.sample[big_match.start()]
                end = anns.sample[big_match.end()-1]
                #print("BIG pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
                if (end-start)/fs > 10:
                    labelregions.append(["BIGEMINY", start, end])

            tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
            for tri_match in tri_matches:
                start = anns.sample[tri_match.start()]
                end = anns.sample[tri_match.end()-1]
                #print("TRI pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
                if (end-start)/fs > 10:
                    labelregions.append(["TRIGEMINY", start, end])

            vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
            for vt_ivr_match in vt_ivr_matches:
                start = anns.sample[vt_ivr_match.start()]
                end = anns.sample[vt_ivr_match.end()-1]
                #print("VT/IVR pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
                if (end-start)/fs > 10:
                    labelregions.append(["VT", start, end])
            #search for VVV pattern

            rhythm_type = ""
            rhythm_start = 0
            #foundafib = False

            for idx, beat in enumerate(anns.symbol):
                if beat == 'N':
                    continue
                if beat == 'V':
                    start = anns.sample[idx] - fs*0.15
                    end = anns.sample[idx] + fs*0.15
                    #labelregions.append(["PVC", start, end])
                elif beat == 'S':
                    start = anns.sample[idx] - fs*0.15
                    end = anns.sample[idx] + fs*0.15
                    #labelregions.append(["SVPB", start, end])
                elif beat == '+':
                    if anns.aux_note[idx] == '(N':
                        rhythm_type = "NSR"
                        rhythm_start = anns.sample[idx]
                    elif anns.aux_note[idx] == '(AFIB':
                        rhythm_type = "AFIB/AFL"
                        rhythm_start = anns.sample[idx]
                        foundafib = True
                        #print("AFIB/AFL", rhythm_start)
                    elif anns.aux_note[idx] == '(AFL':
                        rhythm_type = "AFIB/AFL"
                        rhythm_start = anns.sample[idx]
                        foundafib = True
                    elif anns.aux_note[idx] == ')':
                        if (anns.sample[idx]-rhythm_start)/fs > 30:
                            labelregions.append([rhythm_type, rhythm_start, anns.sample[idx]])
                            # if rhythm_type != "NSR":
                            #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")
                        rhythm_type = ""
                        rhythm_start = 0

            self.objs[recordname] = {
                "record": recordname,
                "path": self.objs[recordname]["path"],
                "initialized": True,
                "signal": ecg,  
                "kappa": 0,
                "fs":300, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions, 
                "done": False
            }
            res.append(self.objs[recordname])

        return res

    def upload_record(self, record):
        key = record.recordname
        diagnoses = []
        delineations = {"p": [], "qrs": [], "t": [], "noise": [], "afib": [], "afib_uncertain": []}

        for j in range(len(record.diagnosis)):
            diagnoses.append({"type":record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})

        for j in range(len(record.subdiagnosis)):
            diagnoses.append({"type":record.subdiagnosis[j].name, "onset": record.subdiagnosis[j].onset, "offset": record.subdiagnosis[j].offset})

        for pwave in record.p:
            delineations["p"].append((pwave.onset, pwave.offset))

        for qrs in record.qrs:
            delineations["qrs"].append((qrs.onset, qrs.offset, qrs.abnormal))

        for twave in record.t:
            delineations["t"].append((twave.onset, twave.offset))

        noiseregions = get_regions(record.delineations.noise.binary)
        afibregions = get_regions(record.delineations.afib.binary)
        afibuncertainregions = get_regions(record.delineations.afib.logits<0.25)

        for noise in noiseregions:
            delineations["noise"].append((noise[0], noise[1]))

        for afib in afibregions:
            delineations["afib"].append((afib[0], afib[1]))

        for afib_uncertain in afibuncertainregions:
            delineations["afib_uncertain"].append((afib_uncertain[0], afib_uncertain[1]))

        data = {
            "diagnosis": diagnoses,
            "delineations": delineations
        }
        with gzip.open(f'{record.recordname}.pkl.gz', 'wb') as f:
            pickle.dump(data, f)

        blob = self.bucket.blob(f"ICENTIA-Cleaned/{self.objs[key]['path']}.pkl.gz")
        blob.upload_from_filename(f'{record.recordname}.pkl.gz')
        print(f"Uploaded {key} to {blob.public_url}")

        #remove the local file
        if os.path.exists(f'{record.recordname}.pkl.gz'):
            os.remove(f'{record.recordname}.pkl.gz')

    def run_bigquery(self, query):

        retries = 0
        max_retries = 10
        timeout = 5

        while retries < max_retries:
            try:
                # Initialize BigQuery client
                client = bigquery.Client(credentials=self.credentials, project=self.credentials.project_id)

                # Run the query with the specified timeout
                res = client.query(query, timeout=timeout).result()

                # If successful, break out of the loop
                return res

            except GatewayTimeout as e:
                # Handle the timeout exception
                print(f"Timeout on attempt {retries + 1}, retrying...")
                retries += 1
                time.sleep(2)  # Wait for 2 seconds before retrying

            except Exception as e:
                # Handle other exceptions
                print(f"Error on attempt {retries + 1}: {e}")
                retries += 1
                time.sleep(2)  # Wait for 2 seconds before retrying

        # If maximum retries exceeded, log it
        if retries == max_retries:
            print("Query failed after 10 attempts.")
            return False

    def set_as_finished(self, keys):
        paths = [self.objs[key]["path"] for key in keys if key in self.objs]

        #update the status of the records to 'processing'
        if len(paths) > 0:
            
            self.run_bigquery(f"""UPDATE `{self.credentials.project_id}.{self.table_id}` SET status='done', last_updated='{datetime.utcnow().isoformat()}' WHERE record_id IN UNNEST({paths})""")

        for key in keys:
            if key in self.objs:
                self.objs[key]["done"] = True
                #remove file from disk
                # local_file_path = self.objs[key]["local_file_path"]+".dat"
                # if os.path.exists(local_file_path):
                #     os.remove(local_file_path)
                # local_file_path = self.objs[key]["local_file_path"]+".hea"
                # if os.path.exists(local_file_path):
                #     os.remove(local_file_path)
                # local_file_path = self.objs[key]["local_file_path"]+".atr"
                # if os.path.exists(local_file_path):
                #     os.remove(local_file_path)

    def cleanup(self):

        #update the status of all records to 'done'
        paths = [self.objs[key]["path"] for key in self.objs if not self.objs[key]["done"]]
        if len(paths) > 0:
            self.run_bigquery(f"""UPDATE `{self.credentials.project_id}.{self.table_id}` SET status='unprocessed' WHERE record_id IN UNNEST({paths})""")

        print("All records set to unprocessed again.")

    def batch(self, batch_size=32):

        while True:
            query = f"""SELECT record_id, status FROM `{self.credentials.project_id}.{self.table_id}` WHERE status='unprocessed' ORDER BY RAND() LIMIT {batch_size} """
            # query = f"""
            #         SELECT record_id, status FROM `{self.credentials.project_id}.{self.table_id}` AS t
            #         WHERE EXISTS (
            #             SELECT 1
            #             FROM UNNEST(JSON_EXTRACT_ARRAY(t.diagnosis)) AS tag
            #             WHERE JSON_EXTRACT_SCALAR(tag, '$') IN ('AVB_TYPE2', 'VT', 'IVR', 'WENCKEBACH', 'SUDDEN_BRADY', 'BIGEMINY', 'TRIGEMINY')
            #         )
            #         ORDER BY RAND() LIMIT {batch_size} """
            
            rows = self.run_bigquery(query)
            if not rows:
                break
            
            results = [dict(row) for row in rows]
            paths = [row['record_id'] for row in results]

            #update the status of the records to 'processing'
            if len(paths) > 0:
                update_query = f"""UPDATE `{self.credentials.project_id}.{self.table_id}` SET status='processing' WHERE record_id IN UNNEST({paths})"""
                self.run_bigquery(update_query)

            if len(paths) == 0:
                break

            yield self.get_data_batch(paths)

    def set_class_mapper(self):
        self.classes = ["NSR", "AFIB/AFL", "AVB", "BIGEMINY", "TRIGEMINY", "VT", "PVC", "SVPB", "NOISE"]
        self.class_mapper = {c: c for c in self.classes}


class ICENTIASAMPLEData(ICENTIAData):
    def __init__(self, folder, sample, annfile, allfile, asynchronous=False, fraction=1.0):
        self.folder = folder
        self.classes = []
        self.class_mapper = {}
        self.case_mapper = {}
        self.objs = {}
        self.asynchronous = asynchronous
        self.fraction = fraction
        self.sample = sample
        self.annfile = annfile
        self.allfile = allfile
        self.basefolder = os.environ.get('benchmark_data')
        self.name = "ICENTIA"
        self.boto_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.num_workers = 16
        #self.set_class_mapper()
        self.set_class_mapper_triage()
        self.annotation_data = {}
        self.credentials = service_account.Credentials.from_service_account_file(self.basefolder+"/aladin-466917-e056430d6165.json")
        self.bigquery_client = bigquery.Client(credentials=self.credentials, project=self.credentials.project_id)
        self.bucket = storage.Client(credentials=self.credentials).get_bucket("arts-aladin")
        self.get_annotation_data()
        self.init_objects()
        self.table_id = "benchmarks.ICENTIA-Cleaned"

    def init_objects(self):
        
        with open(self.sample, 'r') as f:    
            records = json.load(f)

        with open(self.allfile, 'r') as f:
            all_results_per_arrhythmia = json.load(f)

        all_records = []
        for ID, rec in records.items():
            # if rec["type"] == "NSR" or rec["type"] == "NOISE":
            #     continue
            # if not rec["patient"] == "p09998":
            #     continue
            rec["arrhythmia"] = rec["type"]
            rec["recordname"] = "record_"+str(ID)
            if rec["recordname"] in self.annotation_data:
                all_records.append(rec)

        print("Total number of records: ", len(all_records))

        for record in tqdm(all_records, desc="Initializing ICENTIA records"):
            path = record["path"]
            recordname = path.split("/")[-1].split(".")[0]
            patient = recordname
            patient = patient.split("_")[0]
            onset = record["onset"]
            offset = record["offset"]
            #recordname = recordname+":" + str(onset) + "-" + str(offset)
            recordname = record["recordname"]

            with_same_arrhythmia = all_results_per_arrhythmia[record["arrhythmia"]]
            same_patient = [r for r in with_same_arrhythmia if r["patient"] == patient]
            seen_elsewhere = int(np.any([c["aladin"] for c in same_patient if c["path"] != path]))

            obj = {"record": recordname, "path": path, "onset":onset, "offset":offset, "arrhythmia": record["arrhythmia"], "seen_by_human": record["human"], "seen_elsewhere":seen_elsewhere, "initialized": False}
            self.objs[recordname] = obj

        self.get_data(list(self.objs.keys())[0])  # Initialize the first record to set fs and other parameters
        print("Initialized ", len(self.objs), " records from ICENTIA-2 dataset")

    def get_annotation_data(self):

        #read excel file
        annfile = self.annfile
        if not os.path.exists(annfile):
            print("Annotation file not found:", annfile)
            return

        if annfile.endswith('.xlsx'):
            dat = pd.read_excel(annfile, engine='openpyxl', sheet_name=0, skiprows=3, names=["ID","Label1","Label2","Label3","Comment"]).to_dict(orient='records')
        elif annfile.endswith('.json'):
            dat = pd.read_json(annfile).to_dict(orient='records')
        elif annfile.endswith('.csv'):
            dat = pd.read_csv(annfile).to_dict(orient='records')
        elif annfile.endswith('.ods'):
            dat = pd.read_excel(annfile, engine='odf', sheet_name=0, skiprows=0, names=["ID","Label1"], usecols="A,G").to_dict(orient='records')

        for ann in dat:
            recordname = "record_" + str(int(ann["ID"]))
            
            if "Label1" in ann and not pd.isna(ann["Label1"]):
                label = ann["Label1"]
                if label == "??":
                    continue
                self.annotation_data[recordname] = []
                label = label.replace(" ","").strip()
                labels = label.split(",")
                for l in labels:
                    l = l.strip()
                    if l not in self.annotation_data[recordname]:
                        self.annotation_data[recordname].append(l)

            if "Label2" in ann and not pd.isna(ann["Label2"]):
                label = ann["Label2"]
                label = label.replace(" ","").strip().lower()
                self.annotation_data[recordname].append(label)

            if "Label3" in ann and not pd.isna(ann["Label3"]):
                label = ann["Label3"]
                label = label.replace(" ","").strip().lower()
                self.annotation_data[recordname].append(label)

        print(len(self.annotation_data), "records with annotations found in the annotation file.")
        
    def upload_record(self, record):
        return

    def set_as_finished(self, keys):
        return

    def get_data(self, recordname):

        obj = self.objs[recordname]

        if obj["initialized"]:
            return obj

        local_file_path = os.path.join(self.basefolder, self.folder, obj["path"])

        rec = wfdb.rdrecord(local_file_path)
        ecg = rec.p_signal[:,0]
        midpoint = (int(obj["onset"]) + int(obj["offset"])) // 2
        window = rec.fs * 30  # 30 seconds window
        obj["window_start"] = max(0, midpoint - window // 2)
        obj["window_end"] = min(len(ecg), midpoint + window // 2)
        ecg = ecg[obj["window_start"]:obj["window_end"]]

        labels = self.annotation_data[obj["record"]]
        labelregions = []
        for label in labels:
            labelregions.append([label, 0, len(ecg)])

        self.objs[recordname] = {
            "record": recordname, 
            "path": obj["path"],
            "onset": obj["onset"],
            "offset": obj["offset"],
            "arrhythmia": obj["arrhythmia"],
            "seen_by_human": obj["seen_by_human"],
            "seen_elsewhere": obj["seen_elsewhere"],
            "initialized": True,
            "signal": ecg,  
            "kappa": 0,
            "fs":rec.fs, 
            "p": None,
            "qrs": None, 
            "qrsabnorm": None,
            "t": None,
            "p_binary": None,
            "qrs_binary": None,
            "qrsabnorm_binary": None,
            "t_binary": None,
            "labelregions": labelregions,
            "false_positive": not obj["seen_by_human"],
            "segment": [obj["onset"], obj["offset"]]
        }

    def get_data_batch(self, keys):

        objs = []
        
        for key in keys:

            obj = self.objs[key]

            local_file_path = os.path.join(self.basefolder, self.folder, obj["path"])

            midpoint = (int(obj["onset"]) + int(obj["offset"])) // 2
            window = 250 * 30  # 30 seconds window
            header = wfdb.rdheader(local_file_path)
            size = header.sig_len
            obj["window_start"] = max(0, midpoint - window // 2)
            obj["window_end"] = min(size, midpoint + window // 2)
            rec = wfdb.rdrecord(local_file_path, sampfrom=obj["window_start"], sampto=obj["window_end"])
            ecg = rec.p_signal[:,0]
            
            labels = self.annotation_data[obj["record"]]
            labelregions = []
            for label in labels:
                labelregions.append([label, 0, len(ecg)])

            self.objs[key] = {
                "record": obj["record"], 
                "path": obj["path"],
                "initialized": True,
                "arrhythmia": obj["arrhythmia"],
                "seen_elsewhere": obj["seen_elsewhere"],
                "signal": ecg,  
                "kappa": 0,
                "fs":rec.fs, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions,
                "false_positive": not obj["seen_by_human"],
                "segment": [obj["onset"], obj["offset"]]
            }
            #self.objs[key]["labelregions"] = [[label, 0, len(ecg)]]

        return [self.objs[key] for key in keys if key in self.objs]

    def batch(self, batch_size=32):
        keys = list(self.objs.keys())
        for i in range(0, len(keys), batch_size):
            if i + batch_size > len(keys):
                batch_size = len(keys) - i
            batch_keys = keys[i:i + batch_size]
            yield self.get_data_batch(batch_keys)

    def set_class_mapper_triage(self):
        self.classes = ["NORMAL","NONCRITICAL","CRITICAL"]
        self.class_mapper["AFIB"] = "CRITICAL"
        self.class_mapper["AFL"] = "CRITICAL"
        self.class_mapper["SUDDEN_BRADY"] = "CRITICAL"
        self.class_mapper["CHB"] = "CRITICAL"
        self.class_mapper["SVT>30s"] = "CRITICAL"
        self.class_mapper["VT>10s"] = "CRITICAL"

        self.class_mapper["VT<10s"] = "NONCRITICAL"
        self.class_mapper["AVB_TYPE2"] = "NONCRITICAL"
        self.class_mapper["WENCKEBACH"] = "NONCRITICAL"
        self.class_mapper["BIGEMINY"] = "NONCRITICAL"
        self.class_mapper["TRIGEMINY"] = "NONCRITICAL"
        self.class_mapper["AIVR"] = "NONCRITICAL"
        self.class_mapper["IVR"] = "NONCRITICAL"

        self.class_mapper["NSR"] = "NORMAL"
        self.class_mapper["NOISE"] = "NORMAL"
        self.class_mapper["AVB_TYPE1"] = "NONCRITICAL"
        self.class_mapper["BRADYCARDIA"] = "NONCRITICAL"
        self.class_mapper["TACHYCARDIA"] = "NONCRITICAL"

        self.case_mapper = self.class_mapper


    def set_class_mapper(self):
        self.classes = ["AFIB", "AFL", "AVB_TYPE2", "BIGEMINY", "SUDDEN_BRADY", "IVR", "NOISE", "NSR", "SVT>30s", "TRIGEMINY", "VT<10s", "VT>10s", "WENCKEBACH"]
        self.class_mapper = {c: c for c in self.classes}
        self.class_mapper["AFIB"] = "AFIB"
        self.class_mapper["AFL"] = "AFIB"
        self.class_mapper["AVB_TYPE1"] = "NSR"
        self.class_mapper["BRADYCARDIA"] = "NSR"
        self.class_mapper["TACHYCARDIA"] = "NSR"
        self.class_mapper["AVB2"] = "AVB_TYPE2"
        self.class_mapper["CHB"] = "SUDDEN_BRADY"
        self.class_mapper["SVT"] = "SVT>30s"
        self.class_mapper["Wenckebach"] = "WENCKEBACH"
        # self.class_mapper["WENCEKBACH"] = "AVB"
        # self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["BIG"] = "BIGEMINY"
        self.class_mapper["TRI"] = "TRIGEMINY"
        self.class_mapper["AIVR"] = "IVR"

        self.case_mapper = {c: [c] for c in self.classes}
        self.case_mapper["IVR"].append("VT<10s") #we also accept VT<10s when we inspect IVR

class Data:

    def __init__(self, name, folder, fraction=1.0):
        self.folder = folder
        self.name = name
        self.classes = []
        self.class_mapper = {}
        self.fraction = fraction
        self.objs = self.get_data()
        self.split_objs()

    def get_data(self):
        objs = {}

        if self.name == "STANFORD":
            return self.get_stanford()

        if self.name == "CINC":
            return self.get_cinc()
        
        if self.name == "VAL":
            return self.get_validationset()

        if self.name == "VAL_males":
            return self.get_validationset_per_sex("Male")
        
        if self.name == "VAL_females":
            return self.get_validationset_per_sex("Female")
        
        if self.name == "LUDB":
            return self.get_ludb()
        
        if self.name == "RDB":
            return self.get_rdb()

        if self.name == "ICENTIA":
            return self.get_icentia()

        if self.name == "MIT-NORMAL":
            return self.get_mit_normal()

        if self.name == "LONGST":
            return self.get_longst()

        return objs

    def get_name(self):
        return self.name

    def get_ludb(self):
        dat = pickle.load(open('/data/ludb.pkl', 'rb'))
        arrhythmia = pd.read_csv(os.path.join(self.folder, "ludb.csv"))

        print('Data db: ', np.unique([d['db'] for d in dat]))

        objs = {}
        labels = []

        if self.fraction < 1.0:
            dat = np.random.choice(dat, int(len(dat)*self.fraction), replace=False)
            print("Fraction of data: ", len(dat))

        for d in dat:

            recid = d["record"].split("/")[1]
            diagnosisrow = arrhythmia[arrhythmia["ID"] == int(recid)]
            diagnosisrow = diagnosisrow.fillna("")
            rhythm = diagnosisrow["Rhythms"].values[0]
            axis = diagnosisrow["Electric axis of the heart"].values[0]
            conduction = diagnosisrow["Conduction abnormalities"].values[0]
            extrasystoles = diagnosisrow["Extrasystolies"].values[0]
            hypertrophies = diagnosisrow["Hypertrophies"].values[0]
            pacing = diagnosisrow["Cardiac pacing"].values[0]
            ischemia = diagnosisrow["Ischemia"].values[0]

            extrasystolessimple = ""
            for line in extrasystoles.split("\n"):
                if re.match(r'^Atrial extrasystole', extrasystoles):
                    extrasystolessimple += "PAC, "
                if re.match(r'^Ventricular extrasystole', extrasystoles):
                    extrasystolessimple += "PVC, "

            axis = re.sub(r'^Electric axis of the heart: ', '', axis)
            if axis == "normal":
                axis = ""
            if axis == "vertical":
                axis = "vertical axis deviation"
            if axis == "horizontal":
                axis = ""

            ischemiasimple = ""
            for line in ischemia.split("\n"):
                if re.match(r'^Ischemia: inferior', ischemia):
                    ischemiasimple += "Inferior Ischemia, "
                if re.match(r'^Ischemia: posterior', ischemia):
                    ischemiasimple += "Posterior Ischemia, "
                if re.match(r'^Ischemia: lateral', ischemia):
                    ischemiasimple += "Lateral Ischemia, "
                if re.match(r'^Ischemia: apical', ischemia):
                    ischemiasimple += "Apical Ischemia, "
                if re.match(r'^Ischemia: septal', ischemia):
                    ischemiasimple += "Septal Ischemia, "
                if re.match(r'^Stemi:', ischemia):
                    ischemiasimple += "STEMI, "
                if re.match(r'{.nstemi}', ischemia):
                    ischemiasimple += "NSTEMI, "

            #print(rhythm, axis, conduction, extrasystoles, hypertrophies, pacing, ischemia)
            diagnosis = rhythm + ", " + axis + ", " + conduction + ", " + extrasystolessimple + ", " + hypertrophies + ", " + pacing + ", " + ischemiasimple

            pwaves = get_regions(d["segmentation"][0,:])
            qrswaves = get_regions(d["segmentation"][1,:])
            twaves = get_regions(d["segmentation"][2,:])



            ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
            qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
            ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

            if len(ps) == 0:
                ps = np.array([[],[],[]])
            if len(qrss) == 0:
                qrss = np.array([[],[],[]])
            if len(ts) == 0:
                ts = np.array([[],[],[]])

            #get signal
            obj = { "record": d["record"], 
                    "signal": d["signalII"], 
                    "fs":204.8, 
                    "p": ps,
                    "qrs": qrss, 
                    "t": ts,
                    "p_binary": d["segmentation"][0,:],
                    "qrs_binary": d["segmentation"][1,:],
                    "t_binary": d["segmentation"][2,:],
                    "diagnosis": diagnosis,
                    "labelregions": []
            }
            objs[d["record"]] = obj

        return objs

    def get_stanford(self):
        basefolder = os.environ.get('benchmark_data')
        records = np.loadtxt(basefolder+'/STANFORD/RECORDS', dtype=str)

        dat = pickle.load(open(basefolder+'/STANFORD/stanford2.pkl', 'rb'))
        signalsfolder = 'data/STANFORD'


        dat = [d for d in dat if d['db'] == 'STANFORD']
        dat = [d for d in dat if d['record'] in records]

        objs = {}
        labels = []
        classes = []

        for d in dat:
            pwaves = get_regions(d["segmentation"][0,:])
            qrswaves = get_regions(d["segmentation"][1,:])
            qrsabnormwaves = get_regions(d["segmentation"][5,:])
            twaves = get_regions(d["segmentation"][2,:])

            ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
            qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
            qrsas = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrsabnormwaves]))
            ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

            if len(ps) == 0:
                ps = np.array([[],[],[]])
            if len(qrss) == 0:
                qrss = np.array([[],[],[]])
            if len(qrsas) == 0:
                qrsas = np.array([[],[],[]])
            if len(ts) == 0:
                ts = np.array([[],[],[]])

            labellines = d["fulllabel"].split("\n")
            labelregions = []

            for line in labellines:
                regex = r"(\S+) (\d+\.\d+)s -(\d+\.\d+)s"
                matches = re.findall(regex, line)

                for diagnosis, start, end in matches:
                    labelregions.append([diagnosis, float(start)*200, float(end)*200])
                    labels.append(diagnosis)

            rec = wfdb.rdrecord(os.path.join(basefolder+"/STANFORD", d["record"]))
            sig = rec.p_signal[:,0]

            obj = {"record": d["record"], 
                "signal": sig,  
                "kappa": 0,
                "fs":200, 
                "p": ps,
                "qrs": qrss, 
                "qrsabnorm": qrsas,
                "t": ts,
                "p_binary": d["segmentation"][0,:],
                "qrs_binary": d["segmentation"][1,:],
                "qrsabnorm_binary": d["segmentation"][5,:],
                "t_binary": d["segmentation"][2,:],
                "labelregions": labelregions
            }

            objs[d["record"]] = obj


        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        self.classes = uniquelabels
        self.class_mapper = {c: c for c in uniquelabels}
        self.class_mapper["AFIB"] = "AFIB/AFL"
        self.class_mapper["AFL"] = "AFIB/AFL"
        self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["SUDDEN_BRADY"] = "AVB"
        self.class_mapper["AVB_TYPE1"] = "NSR"
        self.class_mapper["BRADYCARDIA"] = "NSR"
        self.class_mapper["TACHYCARDIA"] = "NSR"

        return objs

        # basefolder = os.environ.get('benchmark_data')
        # records = np.loadtxt(basefolder+'/STANFORD/RECORDS', dtype=str)

        # objs = {}
        # labels = []
        # classes = []

        # if self.fraction < 1.0:
        #     records = np.random.choice(records, int(len(records)*self.fraction), replace=False)
        #     print("Fraction of data: ", len(records))

        # for record in records:
            
        #     labelregions = []

        #     annfile = glob.glob(basefolder+"/STANFORD/"+record+"_grp*.episodes.json")[0]
        #     with open(annfile, 'r') as f:
        #         ann = json.load(f)
        #         for episode in ann["episodes"]:
        #             labelregions.append([episode["rhythm_name"], float(episode["onset"]), float(episode["offset"])])
        #             labels.append(episode["rhythm_name"])

        #     rec = wfdb.rdrecord(basefolder+"/STANFORD/"+record)
        #     sig = rec.p_signal[:,0]

        #     obj = {"record": record, 
        #         "signal": sig,  
        #         "kappa": 0,
        #         "fs":200,
        #         "labelregions": labelregions
        #     }

        #     objs[record] = obj


        # labels = np.array(labels)
        # uniquelabels = np.unique(labels)
        # self.classes = uniquelabels
        # self.class_mapper = {c: c for c in uniquelabels}
        # self.class_mapper["AFIB"] = "AFIB/AFL"
        # self.class_mapper["AFL"] = "AFIB/AFL"
        # self.class_mapper["AVB_TYPE2"] = "AVB"
        # self.class_mapper["SUDDEN_BRADY"] = "AVB"
        # self.class_mapper["AVB_TYPE1"] = "NSR"
        # self.class_mapper["BRADYCARDIA"] = "NSR"
        # self.class_mapper["TACHYCARDIA"] = "NSR"

        # return objs
    
    def get_rdb(self):
        basefolder = os.environ.get('benchmark_data')
        records = np.loadtxt(basefolder+'/RDB/RECORDS', dtype=str)

        objs = {}
        labels = []
        classes = []

        if self.fraction < 1.0:
            records = np.random.choice(records, int(len(records)*self.fraction), replace=False)
            print("Fraction of data: ", len(records))

        for record in records:
            
            labelregions = []

            annfile = glob.glob(basefolder+"/RDB/ann_txt/"+record+".ii.txt")[0]

            file = os.path.join(basefolder+"/RDB/dat_csv/"+record+".csv")
            df = pd.read_csv(file, header=None)
            sig = df.iloc[:, 1].values
            fs = 500

            category = record[:2]

            if category == "AF" or category == "AFIB":
                diagnosis = "AFIB/AFL"
            elif category == "AT":
                diagnosis = "EAR"
            elif category == "SB":
                diagnosis = "BRADYCARDIA"
            elif category == "ST":
                diagnosis = "TACHYCARDIA"
            elif category == "SR":
                diagnosis = "NSR"
            elif category == "VT":
                diagnosis = "VT"

            with open(annfile, 'r') as f:
                anns = pd.read_csv(f)

                pwaves = [(anns["START"][i], anns["END"][i]) for i in range(len(anns)) if anns["TYPE"][i] == 0]
                qrswaves = [(anns["START"][i], anns["END"][i]) for i in range(len(anns)) if anns["TYPE"][i] == 1]
                twaves = [(anns["START"][i], anns["END"][i]) for i in range(len(anns)) if anns["TYPE"][i] == 2]

                ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
                qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
                ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

                if len(ps) == 0:
                    ps = np.array([[],[],[]])
                if len(qrss) == 0:
                    qrss = np.array([[],[],[]])
                if len(ts) == 0:
                    ts = np.array([[],[],[]])

                p_binary = regions_to_binary(pwaves, len(sig))
                qrs_binary = regions_to_binary(qrswaves, len(sig))
                t_binary = regions_to_binary(twaves, len(sig))


            labelregions = [[diagnosis, 0, len(sig)]]

            obj = {"record": record, 
                "signal": sig, 
                "fs":fs, 
                "p": ps,
                "qrs": qrss, 
                "t": ts,
                "p_binary": p_binary,
                "qrs_binary": qrs_binary,
                "t_binary": t_binary,
                "diagnosis": diagnosis, 
                "labelregions": labelregions
            }

            objs[record] = obj


        labels = np.array(labels)
        uniquelabels = np.unique(labels)

        return objs
    
    def get_cinc(self):

        basefolder = os.environ.get('benchmark_data')
        records = np.loadtxt(basefolder+'/CINC/training/RECORDS', dtype=str)
        csvanns = pd.read_csv(basefolder+'/CINC/training/REFERENCE-v3.csv', header=None, names=["Recording", "Label"])

        objs = {}
        labels = []
        classes = []

        if self.fraction < 1.0:
            records = np.random.choice(records, int(len(records)*self.fraction), replace=False)
            print("Fraction of data: ", len(records))

        for record in records:

            file = basefolder+"/CINC/training/"+record

            rec = wfdb.rdrecord(file)
            ecg = rec.p_signal[:,0]
            fs = rec.fs
            #ecg = resize_signal(ecg, int(ecg.shape[-1]*(200/fs)))

            recordingname = record
            #print(csvanns["Recording"])
            label = csvanns[csvanns["Recording"] == recordingname]["Label"].values[0]

            labelregions = [[label, 0, len(ecg)]]
            labels.append(label)

            obj = {"record": record, 
                "signal": ecg,  
                "kappa": 0,
                "fs":300, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions
            }

            objs[record] = obj

        
        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        self.classes = uniquelabels
        self.class_mapper = {c: c for c in uniquelabels}

        return objs

    def get_validationset(self):
        basefolder = os.environ.get('benchmark_data')
        dat = pickle.load(open(basefolder+'/VALIDATION/val_matched.pkl', 'rb'))
        print('Data db: ', np.unique([d['db'] for d in dat]))

        objs = {}
        labels = []

        if self.fraction < 1.0:
            dat = np.random.choice(dat, int(len(dat)*self.fraction), replace=False)
            print("Fraction of data: ", len(dat))

        for d in dat:
            
            pwaves = get_regions(d["segmentation"][0,:])
            qrswaves = get_regions(d["segmentation"][1,:])
            twaves = get_regions(d["segmentation"][2,:])

            ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
            qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
            ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

            if len(ps) == 0:
                ps = np.array([[],[],[]])
            if len(qrss) == 0:
                qrss = np.array([[],[],[]])
            if len(ts) == 0:
                ts = np.array([[],[],[]])

            labellines = d["fulllabel"].split("\n")
            labelregions = []

            for line in labellines:
                regex = r"(\S+) (\d+\.\d+)s -(\d+\.\d+)s"
                matches = re.findall(regex, line)

                for diagnosis, start, end in matches:
                    labelregions.append([diagnosis, float(start)*200, float(end)*200])
                    labels.append(diagnosis)

            diagnosis = d["fulllabel"]

            obj = {"record": d["record"], 
                "signal": d["signal"], 
                "fs":204.8, 
                "p": ps,
                "qrs": qrss, 
                "t": ts,
                "p_binary": d["segmentation"][0,:],
                "qrs_binary": d["segmentation"][1,:],
                "t_binary": d["segmentation"][2,:],
                "diagnosis": diagnosis,
                "labelregions": labelregions
            }

            objs[d["record"]] = obj


        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        print(uniquelabels)

        return objs
    
    def get_validationset_per_sex(self, sex):
        basefolder = os.environ.get('benchmark_data')
        dat = pickle.load(open(basefolder+'/VALIDATION/val_matched.pkl', 'rb'))
        print('Data db: ', np.unique([d['db'] for d in dat]))

        print(dat[0].keys())
        objs = {}
        labels = []

        for d in dat:

            if not d["sex"] == sex:
                continue
            
            pwaves = get_regions(d["segmentation"][0,:])
            qrswaves = get_regions(d["segmentation"][1,:])
            twaves = get_regions(d["segmentation"][2,:])

            ps = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in pwaves]))
            qrss = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in qrswaves]))
            ts = np.transpose(np.array([[k[0],(k[0]+k[1])//2,k[1]] for k in twaves]))

            if len(ps) == 0:
                ps = np.array([[],[],[]])
            if len(qrss) == 0:
                qrss = np.array([[],[],[]])
            if len(ts) == 0:
                ts = np.array([[],[],[]])

            labellines = d["fulllabel"].split("\n")
            labelregions = []

            for line in labellines:
                regex = r"(\S+) (\d+\.\d+)s -(\d+\.\d+)s"
                matches = re.findall(regex, line)

                for diagnosis, start, end in matches:
                    labelregions.append([diagnosis, float(start)*200, float(end)*200])
                    labels.append(diagnosis)

            diagnosis = d["fulllabel"]

            obj = {"record": d["record"], 
                "signal": d["signal"], 
                "fs":204.8, 
                "p": ps,
                "qrs": qrss, 
                "t": ts,
                "p_binary": d["segmentation"][0,:],
                "qrs_binary": d["segmentation"][1,:],
                "t_binary": d["segmentation"][2,:],
                "diagnosis": diagnosis,
                "labelregions": labelregions
            }

            objs[d["record"]] = obj


        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        print(uniquelabels)

        return objs
    
    def get_icentia(self):

        basefolder = os.environ.get('benchmark_data')
        patient_folders = np.loadtxt(basefolder+'/ICENTIA/random_patients.txt', dtype=str)

        objs = {}
        labels = []
        classes = []

        if self.fraction < 1.0:
            records = np.random.choice(records, int(len(records)*self.fraction), replace=False)
            print("Fraction of data: ", len(records))

        for patient_folder in tqdm(patient_folders[:100]):

            #check if folder exists
            if not os.path.exists(basefolder+"/ICENTIA/"+patient_folder):
                print("Folder does not exist: ", basefolder+"/ICENTIA/"+patient_folder)
                continue

            folder = basefolder+"/ICENTIA/"+patient_folder
            records = np.loadtxt(folder+"/RECORDS", dtype=str)

            numrecords = len(records)
            targetnumber = 21
            st = max(0,(numrecords - targetnumber) // 2)
            en = min(numrecords, st + targetnumber)

            for record in records[st:en]:
                print("Processing record: ", record)
                file = folder+record

                if not os.path.exists(file+".dat"):
                    print("File does not exist: ", file+".dat")
                    continue

                rec = wfdb.rdrecord(file)
                ecg = rec.p_signal[:,0]
                fs = rec.fs

                if not os.path.exists(file+".atr"):
                    continue

                recordingname = record
                anns = wfdb.rdann(file, 'atr', return_label_elements=['symbol'])

                labelregions = []
                beattypes = [a for a in anns.symbol if a != '+']
                beattypes = "".join(beattypes)
                beattypes = beattypes.replace('S', 'N')
                #beattypes = beattypes.replace('Q', 'N')
                #print(beattypes)

                big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
                for big_match in big_matches:
                    start = anns.sample[big_match.start()]
                    end = anns.sample[big_match.end()-1]
                    print("BIG pattern found in", recordingname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
                    if (end-start)/fs > 10:
                        labelregions.append(["BIGEMINY", start, end])

                tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
                for tri_match in tri_matches:
                    start = anns.sample[tri_match.start()]
                    end = anns.sample[tri_match.end()-1]
                    print("TRI pattern found in", recordingname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
                    if (end-start)/fs > 10:
                        labelregions.append(["TRIGEMINY", start, end])

                vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
                for vt_ivr_match in vt_ivr_matches:
                    start = anns.sample[vt_ivr_match.start()]
                    end = anns.sample[vt_ivr_match.end()-1]
                    print("VT/IVR pattern found in", recordingname, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
                    if (end-start)/fs > 10:
                        labelregions.append(["VT", start, end])
                #search for VVV pattern

                rhythm_type = ""
                rhythm_start = 0
                #foundafib = False

                for idx, beat in enumerate(anns.symbol):
                    if beat == 'N':
                        continue
                    if beat == 'V':
                        start = anns.sample[idx] - fs*0.15
                        end = anns.sample[idx] + fs*0.15
                        #labelregions.append(["PVC", start, end])
                    elif beat == 'S':
                        start = anns.sample[idx] - fs*0.15
                        end = anns.sample[idx] + fs*0.15
                        #labelregions.append(["SVPB", start, end])
                    elif beat == '+':
                        if anns.aux_note[idx] == '(N':
                            rhythm_type = "NSR"
                            rhythm_start = anns.sample[idx]
                        elif anns.aux_note[idx] == '(AFIB':
                            rhythm_type = "AFIB/AFL"
                            rhythm_start = anns.sample[idx]
                            foundafib = True
                            #print("AFIB/AFL", rhythm_start)
                        elif anns.aux_note[idx] == '(AFL':
                            rhythm_type = "AFIB/AFL"
                            rhythm_start = anns.sample[idx]
                            foundafib = True
                        elif anns.aux_note[idx] == ')':
                            if (anns.sample[idx]-rhythm_start)/fs > 30:
                                labelregions.append([rhythm_type, rhythm_start, anns.sample[idx]])
                                if rhythm_type != "NSR":
                                    print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")
                            rhythm_type = ""
                            rhythm_start = 0

            # label = csvanns[csvanns["Recording"] == recordingname]["Label"].values[0]

            # labelregions = [[label, 0, len(ecg)]]
            # labels.append(label)

            #Normal
            #AFIB >30s
            #SVT >30s (using streaks of S beats)
            #VT >10s (using streaks of V beats)
            #asystole > 3.5s (using pauses in N beats)
            #CHB (using severe bradycardia + chaotic p wave interval + manual check)


                lbls = np.unique([lt[0] for lt in labelregions])
                labels.extend(lbls)

                obj = {"record": record, 
                    "signal": ecg,  
                    "kappa": 0,
                    "fs":fs, 
                    "p": None,
                    "qrs": None, 
                    "qrsabnorm": None,
                    "t": None,
                    "p_binary": None,
                    "qrs_binary": None,
                    "qrsabnorm_binary": None,
                    "t_binary": None,
                    "labelregions": labelregions
                }

                objs[record] = obj

        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        self.classes = ['NSR', 'AFIB/AFL', 'VT', 'TRIGEMINY', 'BIGEMINY']
        self.class_mapper = {c: c for c in self.classes}
        self.class_mapper["IVR"] = "VT"
        self.class_mapper["AFL"] = "AFIB/AFL"
        self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["SUDDEN_BRADY"] = "AVB"
        self.class_mapper["AVB_TYPE1"] = "NSR"
        self.class_mapper["BRADYCARDIA"] = "NSR"
        self.class_mapper["TACHYCARDIA"] = "NSR"

        return objs

    def get_longst(self):

        basefolder = os.environ.get('benchmark_data')
        records = np.loadtxt(basefolder+'/LONGST/RECORDS', dtype=str)

        objs = {}
        labels = []
        classes = []

        if self.fraction < 1.0:
            records = np.random.choice(records, int(len(records)*self.fraction), replace=False)
            print("Fraction of data: ", len(records))

        for record in records[:10]:
            print("Processing record: ", record)
            file = basefolder+"/LONGST/"+record

            rec = wfdb.rdrecord(file)
            ecg = rec.p_signal[:,0]
            fs = rec.fs

            if not os.path.exists(file+".atr"):
                continue

            recordingname = record
            anns = wfdb.rdann(file, 'atr')

            labelregions = []
            beattypes = [a for a in anns.symbol if a != '+']
            beattypes = "".join(beattypes)
            beattypes = beattypes.replace('S', 'N')
            #beattypes = beattypes.replace('Q', 'N')
            #print(beattypes)

            # big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
            # for big_match in big_matches:
            #     start = anns.sample[big_match.start()]
            #     end = anns.sample[big_match.end()-1]
            #     if (end-start)/fs > 10:
            #         print("BIG pattern found in", recordingname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
            #         labelregions.append(["BIGEMINY", start, end])

            # tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
            # for tri_match in tri_matches:
            #     start = anns.sample[tri_match.start()]
            #     end = anns.sample[tri_match.end()-1]
            #     if (end-start)/fs > 10:
            #         print("TRI pattern found in", recordingname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
            #         labelregions.append(["TRIGEMINY", start, end])


            rhythm_type = ""
            rhythm_start = 0
            #foundafib = False

            for idx, note in enumerate(anns.aux_note):
                if note != '':
                    if rhythm_type != "":
                        if rhythm_type == "(AFIB" and (anns.sample[idx]-rhythm_start)/fs > 30:
                            labelregions.append(["AFIB/AFL", rhythm_start, anns.sample[idx]])
                        if rhythm_type == "(VT" and (anns.sample[idx]-rhythm_start)/fs > 10:
                            labelregions.append(["VT", rhythm_start, anns.sample[idx]])
                        if rhythm_type == "(SVTA" and (anns.sample[idx]-rhythm_start)/fs > 30:
                            labelregions.append(["SVT", rhythm_start, anns.sample[idx]])
                        if rhythm_type == "(IVR" and (anns.sample[idx]-rhythm_start)/fs > 10:
                            labelregions.append(["IVR", rhythm_start, anns.sample[idx]])
                        if rhythm_type == "(T" and (anns.sample[idx]-rhythm_start)/fs > 10:
                            labelregions.append(["TRIGEMINY", rhythm_start, anns.sample[idx]])
                        if rhythm_type == "(B" and (anns.sample[idx]-rhythm_start)/fs > 10:
                            labelregions.append(["BIGEMINY", rhythm_start, anns.sample[idx]])
                        print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")

                    rhythm_type = note
                    rhythm_start = anns.sample[idx]

        # label = csvanns[csvanns["Recording"] == recordingname]["Label"].values[0]

        # labelregions = [[label, 0, len(ecg)]]
        # labels.append(label)

        #Normal
        #AFIB >30s
        #SVT >30s (using streaks of S beats)
        #VT >10s (using streaks of V beats)
        #asystole > 3.5s (using pauses in N beats)
        #CHB (using severe bradycardia + chaotic p wave interval + manual check)


            lbls = np.unique([lt[0] for lt in labelregions])
            labels.extend(lbls)

            obj = {"record": record, 
                "signal": ecg,  
                "kappa": 0,
                "fs":fs, 
                "p": None,
                "qrs": None, 
                "qrsabnorm": None,
                "t": None,
                "p_binary": None,
                "qrs_binary": None,
                "qrsabnorm_binary": None,
                "t_binary": None,
                "labelregions": labelregions
            }

            objs[record] = obj

        labels = np.array(labels)
        uniquelabels = np.unique(labels)
        self.classes = uniquelabels
        self.class_mapper = {c: c for c in self.classes}
        self.class_mapper["IVR"] = "VT"
        self.class_mapper["AFL"] = "AFIB/AFL"
        self.class_mapper["AVB_TYPE2"] = "AVB"
        self.class_mapper["SUDDEN_BRADY"] = "AVB"
        self.class_mapper["AVB_TYPE1"] = "NSR"
        self.class_mapper["BRADYCARDIA"] = "NSR"
        self.class_mapper["TACHYCARDIA"] = "NSR"

        return objs 

    def split_objs(self):

        objs = self.objs
        for record in list(objs.keys()):
            duration = len(objs[record]["signal"]) / objs[record]["fs"]

            if duration > 7200:
                for st in range(0, len(objs[record]["signal"]), objs[record]["fs"]*3600):
                    end = min(st + objs[record]["fs"]*3600, len(objs[record]["signal"]))
                    new_record = f"{record}_{st//objs[record]['fs']}"

                    new_labelregions = []
                    for labelregion in objs[record]["labelregions"]:
                        if labelregion[2] < st or labelregion[1] > end:
                            continue
                        new_labelregions.append([labelregion[0], max(labelregion[1]-st, 0), min(labelregion[2]-st, end-st)])

                    objs[new_record] = {
                        "record": new_record,
                        "signal": objs[record]["signal"][st:end],
                        "kappa": objs[record]["kappa"],
                        "fs": objs[record]["fs"],
                        "p": objs[record]["p"],
                        "qrs": objs[record]["qrs"],
                        "qrsabnorm": objs[record]["qrs"],
                        "t": objs[record]["t"],
                        "p_binary": objs[record]["p_binary"][st:end] if objs[record]["p_binary"] is not None else None,
                        "qrs_binary": objs[record]["qrs_binary"][st:end] if objs[record]["qrs_binary"] is not None else None,
                        "qrsabnorm_binary": objs[record]["qrs_binary"][st:end] if objs[record]["qrs_binary"] is not None else None,
                        "t_binary": objs[record]["t_binary"][st:end] if objs[record]["t_binary"] is not None else None,
                        "labelregions": new_labelregions
                    }

                del objs[record]

        print("Split objects into smaller segments based on duration > 7200 seconds.")
        print(len(self.objs.keys()), "records remaining after splitting.")

    def get_classes(self):
        return self.classes
    
    def get_class_mapper(self):
        if self.class_mapper == {}:
            for i, c in enumerate(self.classes):
                self.class_mapper[c] = c
        return self.class_mapper

    def __iter__(self):
        return iter(self.objs)
    
    def __getitem__(self, key):
        return self.objs[key]
    
    def __len__(self):
        return len(self.objs)
    
    def keys(self):
        return self.objs.keys()
    
class Model:
    
    def __init__(self):
        self.output_binary = False
        self.name = ""
        self.predict_abnorm = False
        self.gpu_batchsize = 1
        self.cpu_batchsize = 1

    def predict(self, signal, fs, meta=None, preprocess=False):
        raise NotImplementedError

    def predict_batch(self, data):
        raise NotImplementedError

    def calculate_batchsizes(self, data):
        self.gpu_batchsize = 32
        self.cpu_batchsize = 32


    def load_checkpoint(self, checkpoint):
        raise NotImplementedError
    
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
    
    def visualize(self, signal, p, qrs, t, fs):
        raise NotImplementedError

    def tapered_pred(self, dest, pred, taper=100, trim=0):
        if trim > 0:
            desttaper = np.concatenate((np.ones(trim),np.linspace(1,0,taper-trim*2),np.zeros(trim)))
            predtaper = np.concatenate((np.zeros(trim),np.linspace(0,1,taper-trim*2),np.ones(trim)))
        else:
            desttaper = np.linspace(1,0,taper)
            predtaper = np.linspace(0,1,taper)

        #use cubic spline interpolation to smooth the transition

        dest[:taper] *= desttaper
        pred[:taper] *= predtaper

        return dest+pred

    def masked_pred(self, dest, pred, taper=100):
        def gaussian(x, mu, sig):
            return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
        
        mask = gaussian(np.linspace(-3,3,len(pred)), 0, 1)
        mask = mask/np.max(mask)

        pred *= mask

        #use cubic spline interpolation to smooth the transition

        return dest+pred

class BaseBenchmark():
    def __init__(self, data, model, trim=0):
        self.data = data
        self.model = model
        self.trim = trim
        self.trimleft = 0
        self.trimright = 0

    def initialize(self):
        return NotImplementedError

    def run(self):
        return NotImplementedError
       
class DelineationBenchmark(BaseBenchmark):

    def __init__(self, data, model, trim=0, plt=False, fiducials=["p_onset", "p_center", "p_offset", "qrs_onset", "qrs_center", "qrs_offset", "t_onset", "t_center", "t_offset"]):#, "qrsabnorm", "afib"]):
        super().__init__(data, model, trim)
        
        self.fiducials = fiducials
        self.plt = plt

        self.initialize()

    def initialize(self):
        self.fiducial_tex = {
            "p_onset": "$P_{on}$",
            "p_center": "$P_{cen}$",
            "p_offset": "$P_{off}$",
            "qrs_onset": "$QRS_{on}$",
            "qrs_center": "$QRS_{cen}$",
            "qrs_offset": "$QRS_{off}$",
            "t_onset": "$T_{on}$",
            "t_center": "$T_{cen}$",
            "t_offset": "$T_{off}$",
            "qrsabnorm": "$QRS_{abn}$"
        }
        self.macros = {}
        self.micro_se = {}
        self.micro_pp = {}
        self.micro_f1 = {}
        self.micro_iou = {}
        self.micro_dice = {}
        self.micro_biou = {}
        self.micro_bdice = {}
        self.micro_support = {}
        self.macro_pixel = {}
        self.sd = {}

        self.kappas = []
        self.certainties = []
        self.idx = 0
        self.fig = None
        self.axs = None

        for fiducial in self.fiducials:
            self.macros[fiducial] = {"tp": 0, "fp": 0, "fn": 0}
            self.micro_se[fiducial] = []
            self.micro_pp[fiducial] = []
            self.micro_f1[fiducial] = []
            self.micro_dice[fiducial] = []
            self.micro_biou[fiducial] = []
            self.micro_bdice[fiducial] = []
            self.micro_support[fiducial] = []
            self.sd[fiducial] = []

        self.regions = {}
        self.gt_regions = {}

        os.makedirs("results", exist_ok=True)

    def run(self):
        results = {}
        for record in tqdm(self.data):
            obj = self.data[record]
            signal = obj["signal"]
            fs = obj["fs"]
            preprocess = False if self.data.name == "STANFORD" else True
            if self.model.savelogits:
                #check if function returns one or five values
                res = self.model.predict(signal, fs, meta=obj, preprocess=True)
                if res is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = res
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = res
            else:
                res = self.model.predict(signal, fs, meta=obj, preprocess=True)
                if res is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = res
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = res

            if "kappa" in obj:
                #self.certainties.append(1.0-np.mean(uncertainty))
                self.kappas.append(obj["kappa"])

            self.trimleft = self.trim*fs
            self.trimright = len(signal)-self.trim*fs

            if self.model.predict_abnorm:
                qrs = self.merge_qrs(qrs, qrs_abnorm)


            self.get_performance(obj, p, obj["p"], obj["p_binary"], fs, "p", isbinary=self.model.output_binary, record=record)
            self.get_performance(obj, qrs, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=record)
            self.get_performance(obj, t, obj["t"], obj["t_binary"], fs, "t", isbinary=self.model.output_binary, record=record)

            #print(str(obj["record"])+":")
            # self.get_performance(signal, p, obj["p"], obj["p_binary"], fs, "p", isbinary=self.model.output_binary, record=record)
            # if self.model.predict_abnorm:
            #     print(obj.keys())
            #     self.get_performance(signal, self.merge_qrs(qrs, qrs_abnorm), self.merge_qrs(obj["qrs"], obj["qrsabnorm"]), self.merge_qrs(obj["qrs_binary"],obj["qrsabnorm_binary"]), fs, "qrs", isbinary=self.model.output_binary, record=record)
            # else:
            #     self.get_performance(signal, qrs, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=record)
            # #self.get_performance(signal, qrs_abnorm, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=record)
            # #self.get_performance(signal, qrs_abnorm, obj["qrsabnorm"], obj["qrsabnorm_binary"], fs, "qrsabnorm", isbinary=self.model.output_binary, record=record)
            # self.get_performance(signal, t, obj["t"], obj["t_binary"], fs, "t", isbinary=self.model.output_binary, record=record)
            # if "qrsabnorm" in self.fiducials and qrs_abnorm is not None:
            #     self.get_qrs_abnormal_performance(signal, qrs_abnorm, obj["qrsabnorm"], fs, "qrsabnorm", isbinary=self.model.output_binary, record=record)
            # if "afib" in self.fiducials and hasafib is not None:
            #     self.get_afib_performance(signal, hasafib, obj["labelregions"], fs, "afib", record=record)


            # if self.plt:
            #     self.plot(signal, p, qrs, t, uncertainty, isbinary=self.model.output_binary)

        self.aggregate()
        self.save_macro_to_csv()
        self.save_macro_to_json()
        #self.save_per_file_macro_to_csv("results/delineation/macro_per_file_"+self.model.name+"_"+self.data.get_name()+".csv")
        self.report()
        #self.results_to_barchart()
        #self.save_preds("results/delineation/preds_"+self.model.name+".pkl")

        if "kappa" in obj:
            self.scatter_plot()

        return results

    def run_batch(self):

        preds = self.model.predict_batch(self.data)

        #print(preds)

        for recordname in list(preds.keys()):
            pred = preds[recordname]
            obj = self.data[recordname]
            signal = obj["signal"]
            fs = obj["fs"]
            if self.model.savelogits:
                #check if function returns one or five values
                if pred is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = pred["p"], pred["qrs"], pred["qrs_abnorm"], pred["t"], pred["hasafib"]
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = pred["p"], pred["qrs"], pred["t"]
            else:
                if pred is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = pred["p"], pred["qrs"], pred["qrs_abnorm"], pred["t"], pred["hasafib"]
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = pred["p"], pred["qrs"], pred["t"]

            if "kappa" in obj:
                #self.certainties.append(1.0-np.mean(uncertainty))
                self.kappas.append(obj["kappa"])

            self.trimleft = self.trim*fs
            self.trimright = len(signal)-self.trim*fs

            if self.model.predict_abnorm:
                qrs = self.merge_qrs(qrs, qrs_abnorm)

            #print(recordname)
            self.get_performance(obj, p, obj["p"], obj["p_binary"], fs, "p", isbinary=self.model.output_binary, record=recordname)
            self.get_performance(obj, qrs, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=recordname)
            self.get_performance(obj, t, obj["t"], obj["t_binary"], fs, "t", isbinary=self.model.output_binary, record=recordname)

        #print(self.macros["p_onset"]["tp"], self.macros["p_onset"]["fp"], self.macros["p_onset"]["fn"])
        self.aggregate()
        self.save_macro_to_csv()
        self.save_macro_to_json()
        self.report()

    def aggregate(self):

        self.micro_aggregated = {}
        self.macro_aggregated = {}
        self.sd_aggregated = {}

        for fiducial in self.fiducials:
            self.micro_aggregated[fiducial] = {
                "se": np.nanmean(self.micro_se[fiducial]),
                "pp": np.nanmean(self.micro_pp[fiducial]),
                "f1": np.nanmean(self.micro_f1[fiducial]),
                "dice": np.nanmean(self.micro_dice[fiducial]),
                "biou": np.nanmean(self.micro_biou[fiducial]),
                "bdice": np.nanmean(self.micro_bdice[fiducial]),
                "support": np.nanmean(self.micro_support[fiducial])
            }
            self.macro_aggregated[fiducial] = {
                "se": self.macros[fiducial]["tp"]/(self.macros[fiducial]["tp"]+self.macros[fiducial]["fn"]) if self.macros[fiducial]["tp"]+self.macros[fiducial]["fn"] > 0 else 0,
                "pp": self.macros[fiducial]["tp"]/(self.macros[fiducial]["tp"]+self.macros[fiducial]["fp"]) if self.macros[fiducial]["tp"]+self.macros[fiducial]["fp"] > 0 else 0,
                "f1": 2*self.macros[fiducial]["tp"]/(2*self.macros[fiducial]["tp"]+self.macros[fiducial]["fp"]+self.macros[fiducial]["fn"])
            }

            self.sd_aggregated[fiducial] = {
                "mean" : np.nanmean(self.sd[fiducial]),
                "sd" : np.nanstd(self.sd[fiducial])
            }

    def region_to_binary(self, regions, length):
        binary = np.zeros(length)
        for (st, en) in regions:
            binary[int(st):int(en)] = 1
        return binary

    def get_boundary(self, arr, erosion_iters=1):
        """
        Extract the boundary of a binary array.
        The boundary is the difference between the array and its erosion.
        
        Parameters:
        - arr: binary 1D array (numpy array) of ground truth or predicted segmentation.
        - erosion_iters: number of iterations for binary erosion to extract boundaries.
        
        Returns:
        - boundary: binary array where boundary pixels are marked as 1.
        """
        eroded_arr = binary_erosion(arr, iterations=erosion_iters)
        boundary = np.logical_xor(arr,eroded_arr)
        return boundary.astype(int)

    def boundary_iou(self, gt, pred, tolerance=1):
        """
        Compute the Boundary IoU between two binary 1D arrays.
        
        Parameters:
        - gt: binary 1D ground truth array (numpy array).
        - pred: binary 1D predicted array (numpy array).
        - tolerance: number of pixels of tolerance for matching boundaries.
        
        Returns:
        - boundary_iou: Boundary IoU score.
        """
        # Get boundaries of both arrays
        gt_boundary = self.get_boundary(gt, erosion_iters=1)
        pred_boundary = self.get_boundary(pred, erosion_iters=1)
        
        # Apply tolerance by dilating the ground truth boundary
        gt_dilated = np.zeros_like(gt_boundary)
        for i in range(len(gt_boundary)):
            if gt_boundary[i] == 1:
                start = max(0, i - tolerance)
                end = min(len(gt_boundary), i + tolerance + 1)
                gt_dilated[start:end] = 1

        pred_dilated = np.zeros_like(pred_boundary)
        for i in range(len(pred_boundary)):
            if pred_boundary[i] == 1:
                start = max(0, i - tolerance)
                end = min(len(pred_boundary), i + tolerance + 1)
                pred_dilated[start:end] = 1
        
        # Compute boundary IoU
        intersection = np.sum((gt_dilated == 1) & (pred_dilated == 1))
        union = np.sum((gt_dilated == 1) | (pred_dilated == 1))

        support = np.sum(gt)/len(gt)
        
        if union == 0:
            return 1.0, 1.0 if intersection == 0 else 0.0, support
        
        boundary_iou = intersection / union
        
        boundary_pred_sum = np.sum(pred_dilated)
        boundary_gt_sum = np.sum(gt_dilated)
    
        # Avoid division by zero
        if boundary_pred_sum + boundary_gt_sum == 0:
            return 1.0, 1.0 if intersection == 0 else 0.0, support
        
        boundary_dice = (2 * intersection) / (boundary_pred_sum + boundary_gt_sum)

        return boundary_iou, boundary_dice, support

    def get_qrs_abnormal_performance(self, sig, binary, true_qrsabnorm, fs, key, isbinary=False, record=""):

        true_centers = true_qrsabnorm[1]
        true_centers = [x for x in true_centers if not np.isnan(x) and x > self.trimleft and x < self.trimright]

        if isbinary:
            binary = proccess(binary, fs)
            regions = get_regions(binary)
        else:
            regions = binary
            binary = self.region_to_binary(regions, len(sig))

        pred_centers = []

        for (st,en) in regions:
            if not np.isnan(st) and not np.isnan(en) and st > self.trimleft and st < self.trimright and en > self.trimleft and en < self.trimright:
                pred_centers.append((st+en)//2)

        tp, fp, fn, _ = self.get_tp_fp_fn(pred_centers, true_centers, fs, isnan=False)

        self.macros[key]["tp"] += tp
        self.macros[key]["fp"] += fp
        self.macros[key]["fn"] += fn
        
        if len(true_centers) > 0:
            se = tp/(tp+fn) if tp+fn > 0 else 0
            pp = tp/(tp+fp) if tp+fp > 0 else 0
            f1 = 2*tp/(2*tp+fp+fn)
        else:
            se = np.nan
            pp = np.nan
            f1 = np.nan

        self.micro_se[key].append(se)
        self.micro_pp[key].append(pp)
        self.micro_f1[key].append(f1)

    def get_afib_performance(self, sig, pred_afib, true_labels, fs, key, record=""):

        true_afib = np.any([x[0] == "AFIB" or x[0] == "AFL" or x[0] == "AFIB/AFL" for x in true_labels])

        if pred_afib and true_afib:
            self.macros[key]["tp"] += 1
        elif pred_afib and not true_afib:
            self.macros[key]["fp"] += 1
        elif not pred_afib and true_afib:
            self.macros[key]["fn"] += 1

    def get_performance(self, obj, binary, true, true_binary, fs, key, isbinary=False, record=""):

        true_onsets = true[0]
        true_centers = true[1]
        true_offsets = true[2]

        if record not in self.gt_regions:
            self.gt_regions[record] = {}
        self.gt_regions[record][key] = list(zip(true_onsets, true_offsets))

        true_onsets = [x for x in true_onsets if not np.isnan(x) and x > self.trimleft and x < self.trimright]
        true_centers = [x for x in true_centers if not np.isnan(x) and x > self.trimleft and x < self.trimright]
        true_offsets = [x for x in true_offsets if not np.isnan(x) and x > self.trimleft and x < self.trimright]

        if isbinary:
            binary = proccess(binary, fs)
            regions = get_regions(binary)
        else:
            regions = binary
            binary = self.region_to_binary(regions, len(obj["signal"]))

        #print(binary, len(sig))

        smooth = 1e-6

        binary = binary[int(self.trimleft):int(self.trimright)]
        true_binary = true_binary[int(self.trimleft):int(self.trimright)]

        intersection = np.sum(binary*true_binary)
        self.micro_dice[key+"_onset"].append((2*intersection+smooth)/(np.sum(binary)+np.sum(true_binary)+smooth))
        self.micro_dice[key+"_center"].append(0)
        self.micro_dice[key+"_offset"].append(0)

        biou, bdice, support = self.boundary_iou(true_binary, binary, tolerance=int(0.075*fs))
        self.micro_biou[key+"_onset"].append(biou)
        self.micro_biou[key+"_center"].append(0)
        self.micro_biou[key+"_offset"].append(0)
        self.micro_bdice[key+"_onset"].append(bdice)
        self.micro_bdice[key+"_center"].append(0)
        self.micro_bdice[key+"_offset"].append(0)

        pred_onsets = []
        pred_centers = []
        pred_offsets = []

        for (st,en) in regions:
            if not np.isnan(st) and st > self.trimleft and st < self.trimright:
                pred_onsets.append(st)
            if not np.isnan(en) and en > self.trimleft and en < self.trimright:
                pred_offsets.append(en)
            if not np.isnan(st) and not np.isnan(en) and st > self.trimleft and st < self.trimright and en > self.trimleft and en < self.trimright:
                pred_centers.append((st+en)//2)
        
        if record not in self.regions:
            self.regions[record] = {}
        self.regions[record][key] = regions

        print(pred_onsets, true_onsets)

        tp, fp, fn, sds = self.get_tp_fp_fn(pred_onsets, true_onsets, fs, isnan=False)
        self.macros[key+"_onset"]["tp"] += tp
        self.macros[key+"_onset"]["fp"] += fp
        self.macros[key+"_onset"]["fn"] += fn
        self.sd[key+"_onset"].extend(sds)

        tp, fp, fn, sds = self.get_tp_fp_fn(pred_centers, true_centers, fs, isnan=False)
        self.macros[key+"_center"]["tp"] += tp
        self.macros[key+"_center"]["fp"] += fp
        self.macros[key+"_center"]["fn"] += fn
        self.sd[key+"_center"].extend(sds)

        tp, fp, fn, sds = self.get_tp_fp_fn(pred_offsets, true_offsets, fs, isnan=False)
        self.macros[key+"_offset"]["tp"] += tp
        self.macros[key+"_offset"]["fp"] += fp
        self.macros[key+"_offset"]["fn"] += fn
        self.sd[key+"_offset"].extend(sds)

        tp, fp, fn, _ = self.get_tp_fp_fn(pred_onsets, true_onsets, fs, isnan=True)
        if len(true_onsets) > 0:
            se = tp/(tp+fn) if tp+fn > 0 else 0
            pp = tp/(tp+fp) if tp+fp > 0 else 0
            f1 = 2*tp/(2*tp+fp+fn)
        else:
            se = np.nan
            pp = np.nan
            f1 = np.nan
        
        self.micro_se[key+"_onset"].append(se)
        self.micro_pp[key+"_onset"].append(pp)
        self.micro_f1[key+"_onset"].append(f1)

        tp, fp, fn, _ = self.get_tp_fp_fn(pred_centers, true_centers, fs, isnan=True)
        if len(true_centers) > 0:
            se = tp/(tp+fn) if tp+fn > 0 else 0
            pp = tp/(tp+fp) if tp+fp > 0 else 0
            f1 = 2*tp/(2*tp+fp+fn)
        else:
            se = np.nan
            pp = np.nan
            f1 = np.nan
        #print(key+"_center", f1)
        self.micro_se[key+"_center"].append(se)
        self.micro_pp[key+"_center"].append(pp)
        self.micro_f1[key+"_center"].append(f1)

        tp, fp, fn, _ = self.get_tp_fp_fn(pred_offsets, true_offsets, fs, isnan=True)
        if len(true_offsets) > 0:
            se = tp/(tp+fn) if tp+fn > 0 else 0
            pp = tp/(tp+fp) if tp+fp > 0 else 0
            f1 = 2*tp/(2*tp+fp+fn)
        else:
            se = np.nan
            pp = np.nan
            f1 = np.nan
        self.micro_se[key+"_offset"].append(se)
        self.micro_pp[key+"_offset"].append(pp)
        self.micro_f1[key+"_offset"].append(f1)

    def merge_qrs(self, qrs, qrs_abnorm):

        if qrs.ndim == 1:
            return np.logical_or(qrs, qrs_abnorm)
        
        qrs = qrs.T
        qrs_abnorm = qrs_abnorm.T
        merged = np.concatenate((qrs, qrs_abnorm), axis=0)
        #sort by first tuple element
        merged = sorted(merged, key=lambda x: x[0])
        merged = np.array(merged).T

        if len(merged) == 0:
            return np.array([[],[],[]])
        
        return merged
            
    def create_fig(self, nrow, ncol):
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*12, nrow*1.5), dpi=300)
        return fig, axs
    
    def save_preds(self, filename):
        #save as pickle file
        with open(filename, 'wb') as f:
            pickle.dump({"regions":self.regions, "gt":self.gt_regions}, f)

    def plot(self, sig, p, qrs, t, uncertainty, isbinary=False):
        if self.idx % 10 == 0:
            if self.idx > 0 and self.idx < len(self.data.keys())-1:
                plt.tight_layout()
                plt.savefig("results/"+str(self.idx//10).zfill(2)+"_plot.png")
                plt.close()
            self.fig, self.axs = self.create_fig(5, 2)
            self.axs = self.axs.reshape(-1)

        minecg = np.min(sig)
        maxecg = np.max(sig)
        localidx = self.idx % 10
        self.axs[localidx].plot(sig, color='black')
        #self.axs[localidx].plot((uncertainty*(maxecg-minecg))+minecg, color='red')
        self.axs[localidx].set_xlim(0, len(sig))
        self.axs[localidx].set_axis_off()

        cols = ['#27ae60','#2980b9','#8e44ad']

        #print(qrs)
        for wave, col in zip([qrs, t, p],cols):
            if isbinary:
                wave = proccess(wave)
                regions = get_regions(wave)
            else:
                regions = wave

            for region in regions:
                meanuncertainty = np.mean(uncertainty[int(region[0]):int(region[1])])
                self.axs[localidx].axvspan(region[0], region[1], alpha=max(0.0,1.0-meanuncertainty*2), color=col, zorder=0)

        if self.idx == len(self.data.keys())-1:
            plt.tight_layout()
            plt.savefig("results/"+str((self.idx//10)+1).zfill(2)+"_plot.png")
            plt.close()

        self.idx += 1

    def to_latex(self, table):

        cols = table.columns
        latex = "\\begin{tabular}{" + "l|" + "l"*(len(table.columns)-1) + "} \n"

        for col in cols:
            texheader = self.fiducial_tex[col.header] if col.header in self.fiducial_tex else col.header
            latex += texheader + " & "
        latex = latex[:-2] + " \\\\ \\hline\n"

        nrow = len(cols[0]._cells)
        for i in range(nrow):
            for col in cols:
                latex += col._cells[i] + " & "
            latex = latex[:-2] + " \\\\ \n"

        latex = latex[:-1] + " \\hline \n"
        latex += "\\end{tabular}"

        return latex
    
    def to_csv(self, table, filename):

        cols = table.columns
        f = open(filename, "w")
        csv = ""

        for col in cols:
            header = col.header
            csv += header + ","
            
        csv = csv[:-1] +"\n"
        f.write(csv)

        nrow = len(cols[0]._cells)
        for i in range(nrow):
            csv = ""
            for col in cols:
                csv += col._cells[i] + ","
            csv = csv[:-1] + "\n"
            f.write(csv)

        f.close()

        return filename

    def report(self):
        
        table = Table(title=self.data.get_name()+" Macro Aggregated Performance")
        metrics = ["se", "pp", "f1"]
        rows = []
        for metric in metrics:
            row = [metric]
            for fiducial in self.fiducials:
                row.append(str(np.round(self.macro_aggregated[fiducial][metric]*1000)/10))
            rows.append(row)

        row = ["$m \pm \sigma$"]
        for fiducial in self.fiducials:
            row.append(str(np.round(self.sd_aggregated[fiducial]["mean"]*10000)/10)+"$\pm$"+str(np.round(self.sd_aggregated[fiducial]["sd"]*10000)/10))
        rows.append(row)

        columns = ["Metric"] + self.fiducials

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)

        print(self.to_latex(table))


        table = Table(title=self.data.get_name()+" Micro Aggregated Performance")
        metrics = ["se", "pp", "f1", "dice", "biou", "bdice"]
        rows = []
        for metric in metrics:
            row = [metric]
            for fiducial in self.fiducials:
                row.append(str(np.round(self.micro_aggregated[fiducial][metric]*1000)/10))
            rows.append(row)

        columns = ["Metric"] + self.fiducials

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)
        print(self.to_latex(table))

    def get_tp_fp_fn(self, pred, true, fs, isnan=False):
        #filter out nans
        tp = 0
        fp = 0
        fn = 0
        sds = []

        if len(true) == 0 and not isnan:
            return 0, len(pred), 0, [np.nan]
        
        if len(true) == 0 and isnan:
            return np.nan, np.nan, np.nan, [np.nan]

        if len(pred) == 0 or np.isnan(pred).all() == True:
            return 0, 0, len(true), [np.nan]

        for p in true:
            diff = pred-p
            idx = np.argmin(np.abs(diff))
            val = np.min(np.abs(diff))
            if val < 0.150*fs:
                tp += 1
                sds.append(diff[idx]/fs)
            else:
                fn += 1

        for p in pred:
            val = np.min(np.abs(true-p))
            if val > 0.150*fs:
                fp += 1

        return tp, fp, fn, sds
    
    def get_sensitivity_specificity(self, pred, true, qrss, fs, isnan=False):
        #filter out nans
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(0, len(qrss)-1):
            if i > 0 and i < len(qrss)-1:
                ons = qrss[i]-min(int(2*fs),(qrss[i]-qrss[i-1])/2)
                offs = qrss[i]+min(int(2*fs),(qrss[i+1]-qrss[i])/2)
                window = [ons,offs]
                #windows[1] = [qrss[i], qrss[i+1]+(qrss[i]+qrss[i+1])/2]
            elif i == 0:
                ons = qrss[i]-min(int(2*fs),(qrss[i+1]-qrss[i])/2)
                offs = qrss[i]+min(int(2*fs),(qrss[i+1]-qrss[i])/2)
                window = [ons,offs]
                #windows[1] = [qrss[i], qrss[i+1]+(qrss[i]+qrss[i+1])/2]
            else:
                ons = qrss[i]-min(int(2*fs),(qrss[i]-qrss[i-1])/2)
                offs = qrss[i]+min(int(2*fs),(qrss[i]-qrss[i-1])/2)
                window = [ons,offs]
                #windows[1] = [qrss[i], qrss[i]+int(fs*0.5)]
            
            true_in_window = np.array([x for x in true if x > window[0] and x < window[1]])
            pred_in_window = np.array([x for x in pred if x > window[0] and x < window[1]])

            # if len(true_in_window) > 0:
            #     if len(pred_in_window) > 0:
            #         for p in true_in_window:
            #             val = np.min(np.abs(pred_in_window-p))
            #             if val < 0.15*fs:
            #                 tp += 1
            #             else:
            #                 fn += 1

            #         for p in pred_in_window:
            #             val = np.min(np.abs(true_in_window-p))
            #             if val > 0.15*fs:
            #                 fp += 1
            #     else:
            #         fn += len(true_in_window)
            # else:
            #     if len(pred_in_window) == 0:
            #         tn += 1
            #     else:
            #         fp += len(pred_in_window)

            if len(true_in_window) > 0:
                if len(pred_in_window) > 0:
                    tp += 1
                    for p in true_in_window:
                        val = np.min(np.abs(pred_in_window-p))
                        if val < 0.15*fs:
                            tp += 1
                        else:
                            fn += 1

                    for p in pred_in_window:
                        val = np.min(np.abs(true_in_window-p))
                        if val > 0.15*fs:
                            fp += 1
                else:
                    #fn += 1
                    fn += len(true_in_window)
            else:
                if len(pred_in_window) == 0:
                    tn += 1
                else:
                    #fp += 1
                    fp += len(pred_in_window)

        se = tp/(tp+fn) if tp+fn > 0 else 0
        sp = tn/(tn+fp) if tn+fp > 0 else 0

        return se, sp

    def save_macro_to_csv(self):

        cols = ["Condition", "SE", "PP", "F1", "Error Mean", "Error SD"]

        basefolder = os.environ.get('benchmark_results')
        filename = basefolder+"/delineation/macro_"+self.model.name+"_"+self.data.get_name()+".csv"
        f = open(filename, "w")
        csv = ""

        for col in cols:
            csv += col + ","
            
        csv = csv[:-1] +"\n"
        f.write(csv)
    
        for fiducial in self.fiducials:
            row = [fiducial]
            row.append(str(np.round(self.macro_aggregated[fiducial]["se"]*1000)/10))
            row.append(str(np.round(self.macro_aggregated[fiducial]["pp"]*1000)/10))
            row.append(str(np.round(self.macro_aggregated[fiducial]["f1"]*1000)/10))
            row.append(str(np.round(self.sd_aggregated[fiducial]["mean"]*10000)/10))
            row.append(str(np.round(self.sd_aggregated[fiducial]["sd"]*10000)/10))

            csv = ""
            for col in row:
                csv += col + ","
            csv = csv[:-1] + "\n"
            f.write(csv)

        f.close()

        return filename

    def save_macro_to_json(self):
        
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_compressed = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data = {
            "date": date,
            "model": self.model.name,
            "dataset": self.data.get_name(),
            "modelpaths": self.model.modelpaths if hasattr(self.model, "modelpaths") else [],
            "results": {}
        }
        for fiducial in self.fiducials:
            data["results"][fiducial] = {
                "se": np.round(self.macro_aggregated[fiducial]["se"]*1000)/10,
                "pp": np.round(self.macro_aggregated[fiducial]["pp"]*1000)/10,
                "f1": np.round(self.macro_aggregated[fiducial]["f1"]*1000)/10,
                "error_mean": np.round(self.sd_aggregated[fiducial]["mean"]*10000)/10,
                "error_sd": np.round(self.sd_aggregated[fiducial]["sd"]*10000)/10
            }

        basefolder = os.environ.get('benchmark_results')
        filename = basefolder+"/delineation/macro_"+self.model.name+"_"+self.data.get_name()+"_["+date_compressed+"].json"

        with open(filename, 'w') as f:
            json.dump(data, f)

        return filename

    def save_per_file_macro_to_csv(self, filename):

        cols = ["Condition", "SE", "PP", "F1", "Error Mean", "Error SD"]
        f = open(filename, "w")
        csv = ""

        for col in cols:
            csv += col + ","
            
        csv = csv[:-1] +"\n"
        f.write(csv)

        for i,record in enumerate(self.data.keys()):
            row = [record]
            for fiducial in self.fiducials:
                row.append(str(np.round(self.micro_pp[fiducial][i]*1000)/10))
            csv = ""
            for col in row:
                csv += col + ","
            csv = csv[:-1] + "\n"
            f.write(csv)

        f.close()

        return filename

class PerClassBenchmark(DelineationBenchmark):
    
    def __init__(self, data, model, trim=0, fiducials=["p_onset", "p_center", "p_offset", "qrs_onset", "qrs_center", "qrs_offset", "t_onset", "t_center","t_offset"]):
        self.classes = ["AFIB",
                        "AFL",
                        "AVB_TYPE2",
                        "BIGEMINY",
                        "SUDDEN_BRADY",
                        "EAR",
                        "IVR",
                        "JUNCTIONAL",
                        "NOISE",
                        "NSR",
                        "SVT",
                        "TRIGEMINY",
                        "VT",
                        "WENCKEBACH"]
        
        super().__init__(data, model, trim, fiducials)
        self.fiducials = fiducials

    def initialize(self):
        fiducials = ["p_onset", "p_center", "p_offset", "qrs_onset", "qrs_center", "qrs_offset", "t_onset", "t_center", "t_offset"]
        self.fiducial_tex = {
            "p_onset": "$P_{on}$",
            "p_center": "$P_{cen}$",
            "p_offset": "$P_{off}$",
            "qrs_onset": "$QRS_{on}$",
            "qrs_center": "$QRS_{cen}$",
            "qrs_offset": "$QRS_{off}$",
            "t_onset": "$T_{on}$",
            "t_center": "$T_{cen}$",
            "t_offset": "$T_{off}$",
            "P Presence": "$P_{sen}$",
            "P Precision": "$P_{prec}$",
            "QRS Timing": "$QRS_{f1}$",
            "P Onset Error": "$P_{error}$ ($m \pm \sigma$)",
            "QRS Width": "$QRS_{width}$ (ms)"
        }
        self.macros = {}
        self.support = {}
        self.micro_se = {}
        self.micro_pp = {}
        self.micro_f1 = {}
        self.micro_sp = {}
        self.micro_pixel_se = {}
        self.micro_pixel_sp = {}
        self.micro_iou = {}
        self.micro_dice = {}
        self.micro_biou = {}
        self.micro_bdice = {}
        self.micro_support = {}
        self.sd = {}
        self.width = {}
        self.softmax = {}
        self.preds = {}
        self.counts = {}

        for cls in self.classes:
            self.macros[cls] = {}
            self.support[cls] = {"p": 0, "qrs": 0, "t": 0}
            self.micro_se[cls] = {}
            self.micro_pp[cls] = {}
            self.micro_f1[cls] = {}
            self.micro_sp[cls] = {}
            self.micro_pixel_se[cls] = {}
            self.micro_pixel_sp[cls] = {}
            self.micro_iou[cls] = {}
            self.micro_dice[cls] = {}
            self.micro_biou[cls] = {}
            self.micro_bdice[cls] = {}
            self.micro_support[cls] = {}
            self.sd[cls] = {}
            self.width[cls] = {}
            self.softmax[cls] = {"p": [], "qrs": [], "t": [], "logits": []}
            self.preds[cls] = {"p": [], "qrs": [], "t": [], "p_hat": [], "qrs_hat": [], "t_hat": []}
            self.counts[cls] = 0

            for fiducial in fiducials:
                self.macros[cls][fiducial] = {"tp": 0, "fp": 0, "fn": 0}
                self.micro_se[cls][fiducial] = []
                self.micro_pp[cls][fiducial] = []
                self.micro_f1[cls][fiducial] = []
                self.micro_sp[cls][fiducial] = []
                self.micro_pixel_se[cls][fiducial] = []
                self.micro_pixel_sp[cls][fiducial] = []
                self.micro_iou[cls][fiducial] = []
                self.micro_dice[cls][fiducial] = []
                self.micro_biou[cls][fiducial] = []
                self.micro_bdice[cls][fiducial] = []
                self.micro_support[cls][fiducial] = []

            for wave in ["p","qrs","t"]:
                self.width[cls][wave] = []
                self.sd[cls][wave] = []

        self.regions = {}
        os.makedirs("results", exist_ok=True)

    def run(self):
        results = {}
        for record in tqdm(self.data):
            obj = self.data[record]
            signal = obj["signal"]
            fs = obj["fs"]
            if self.model.savelogits:
                #check if function returns one or five values
                res = self.model.predict(signal, fs, meta=obj, preprocess=True)
                if res is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, _ = res
                else:
                    qrs_abnorm = None
                    p, qrs, t = res
            else:
                res = self.model.predict(signal, fs, meta=obj, preprocess=True)
                if res is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, _ = res
                else:
                    qrs_abnorm = None
                    p, qrs, t = res

            self.trimleft = self.trim*fs
            self.trimright = len(signal)-self.trim*fs

            if self.model.predict_abnorm:
                qrs = self.merge_qrs(qrs, qrs_abnorm)

            #print(str(obj["record"])+":")
            self.get_performance(obj, p, obj["p"], obj["p_binary"], fs, "p", isbinary=self.model.output_binary, record=record)
            self.get_performance(obj, qrs, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=record)
            self.get_performance(obj, t, obj["t"], obj["t_binary"], fs, "t", isbinary=self.model.output_binary, record=record)

        basefolder = os.environ.get('benchmark_results')
        self.aggregate()
        self.report()
        self.save_preds(basefolder+"/delineation/preds_"+self.model.name+".pkl")
        self.save_micro_to_csv(basefolder+"/delineation/micros_"+self.model.name+"_"+self.data.get_name()+".csv")

        return results

    def run_batch(self):

        results = {}

        preds = self.model.predict_batch(self.data)

        #print(preds)

        for recordname in list(preds.keys()):
            pred = preds[recordname]
            obj = self.data[recordname]
            signal = obj["signal"]
            fs = obj["fs"]
            if self.model.savelogits:
                #check if function returns one or five values
                if pred is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = pred["p"], pred["qrs"], pred["qrs_abnorm"], pred["t"], pred["hasafib"]
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = pred["p"], pred["qrs"], pred["t"]
            else:
                if pred is None:
                    continue

                if self.model.predict_abnorm:
                    p, qrs, qrs_abnorm, t, hasafib = pred["p"], pred["qrs"], pred["qrs_abnorm"], pred["t"], pred["hasafib"]
                else:
                    qrs_abnorm = None
                    hasafib = None
                    p, qrs, t = pred["p"], pred["qrs"], pred["t"]

            self.trimleft = self.trim*fs
            self.trimright = len(signal)-self.trim*fs

            if self.model.predict_abnorm:
                qrs = self.merge_qrs(qrs, qrs_abnorm)

            #print(recordname)
            self.get_performance(obj, p, obj["p"], obj["p_binary"], fs, "p", isbinary=self.model.output_binary, record=recordname)
            self.get_performance(obj, qrs, obj["qrs"], obj["qrs_binary"], fs, "qrs", isbinary=self.model.output_binary, record=recordname)
            self.get_performance(obj, t, obj["t"], obj["t_binary"], fs, "t", isbinary=self.model.output_binary, record=recordname)

        
        basefolder = os.environ.get('benchmark_results')
        self.aggregate()
        self.report()
        self.save_preds(basefolder+"/delineation/preds_"+self.model.name+".pkl")
        self.save_micro_to_csv(basefolder+"/delineation/micros_"+self.model.name+"_"+self.data.get_name()+".csv")

        return results

    def save_logits(self, obj, logits, fs):
        trim = self.trim*fs
        labelregions = obj["labelregions"]

        p_binary = obj["p_binary"]
        qrs_binary = obj["qrs_binary"]
        t_binary = obj["t_binary"]

        for cls in self.classes:
            for label in labelregions:
                if label[0] == cls:
                    st = int(label[1])
                    en = int(label[2])

                    p = p_binary[st:en]
                    qrs = qrs_binary[st:en]
                    t = t_binary[st:en]
                    lgts = logits[:,st:en]

                    self.softmax[cls]["p"].append(p)
                    self.softmax[cls]["qrs"].append(qrs)
                    self.softmax[cls]["t"].append(t)
                    self.softmax[cls]["logits"].append(lgts)


    def save_hats(self, obj, p_pred, qrs_pred, t_pred, fs):
        trim = self.trim*fs
        labelregions = obj["labelregions"]

        p_binary = obj["p_binary"]
        qrs_binary = obj["qrs_binary"]
        t_binary = obj["t_binary"]

        p_pred_binary = np.zeros(len(obj["signal"]))
        qrs_pred_binary = np.zeros(len(obj["signal"]))
        t_pred_binary = np.zeros(len(obj["signal"]))

        if not self.model.output_binary:
            if len(p_pred) > 0:
                for st,en in list(zip(p_pred[:,0],p_pred[:,1])):
                    if not np.isnan(st) and st > trim and st < len(obj["signal"])-trim:
                        p_pred_binary[int(st):int(en)] = 1
            if len(qrs_pred) > 0:
                for st,en in list(zip(qrs_pred[:,0],qrs_pred[:,1])):
                    if not np.isnan(st) and st > trim and st < len(obj["signal"])-trim:
                        qrs_pred_binary[int(st):int(en)] = 1

            if len(t_pred) > 0:
                for st,en in list(zip(t_pred[:,0],t_pred[:,1])):
                    if not np.isnan(st) and st > trim and st < len(obj["signal"])-trim:
                        t_pred_binary[int(st):int(en)] = 1
        else:
            p_pred_binary = p_pred
            qrs_pred_binary = qrs_pred
            t_pred_binary = t_pred

        for cls in self.classes:
            for label in labelregions:
                if label[0] == cls:
                    st = int(label[1])
                    en = int(label[2])

                    p = p_binary[st:en]
                    qrs = qrs_binary[st:en]
                    t = t_binary[st:en]
                    p_hat = p_pred_binary[st:en]
                    qrs_hat = qrs_pred_binary[st:en]
                    t_hat = t_pred_binary[st:en]

                    self.preds[cls]["p"].append(p)
                    self.preds[cls]["qrs"].append(qrs)
                    self.preds[cls]["t"].append(t)
                    self.preds[cls]["p_hat"].append(p_hat)
                    self.preds[cls]["qrs_hat"].append(qrs_hat)
                    self.preds[cls]["t_hat"].append(t_hat)

    def get_performance(self, obj, binary, true, true_binary, fs, key, isbinary=False, record=""):
        if isbinary:
            binary = proccess(binary)
            regions = get_regions(binary)
        else:
            regions = binary
            binary = self.region_to_binary(regions, len(obj["signal"]))

        smooth = 1e-6

        binary = binary[self.trimleft:self.trimright]
        true_binary = true_binary[self.trimleft:self.trimright]
        
        intersection = np.sum(binary*true_binary)
        dice = (2*intersection)/(np.sum(binary)+np.sum(true_binary)+smooth)
        iou = (intersection)/(np.sum(binary)+np.sum(true_binary)-intersection+smooth)
        biou, bdice, support = self.boundary_iou(true_binary, binary, tolerance=int(0.075*fs))
        tp = np.sum((true_binary == 1) & (binary == 1))
        tn = np.sum((true_binary == 0) & (binary == 0))
        fp = np.sum((true_binary == 0) & (binary == 1))
        fn = np.sum((true_binary == 1) & (binary == 0))

        # Calculate sensitivity and specificity
        se = tp / (tp + fn) if (tp + fn) != 0 else 0
        sp = tn / (tn + fp) if (tn + fp) != 0 else 0

        all_pred_onsets = []
        all_pred_centers = []
        all_pred_offsets = []
        all_pred_widths = []
        trim = self.trim*fs
        sig = obj["signal"]
        labelregions = obj["labelregions"]

        for (st,en) in regions:
            if not np.isnan(st) and st > self.trimleft and st < self.trimright:
                all_pred_onsets.append(st)
            if not np.isnan(en) and en > self.trimleft and en < self.trimright:
                all_pred_offsets.append(en)
            if not np.isnan(st) and not np.isnan(en) and st > self.trimleft and st < self.trimright and en > self.trimleft and en < self.trimright:
                all_pred_centers.append((st+en)//2)
                all_pred_widths.append(((st+en)//2,(en-st)/fs))

        if record not in self.regions:
            self.regions[record] = {}

        self.regions[record][key] = regions

        all_true_onsets = true[0]
        all_true_centers = true[1]
        all_true_offsets = true[2]
        all_true_widths = []
        for i in range(len(all_true_onsets)):
            ons = all_true_onsets[i]
            endsearch = all_true_onsets[i+1] if i < len(all_true_onsets)-1 else len(sig)
            for i in range(len(all_true_offsets)):
                if all_true_offsets[i] > ons and all_true_offsets[i] < endsearch:
                    off = all_true_offsets[i]
                    all_true_widths.append(((ons+off)/2, (off-ons)/fs))

        all_true_onsets = [x for x in all_true_onsets if not np.isnan(x) and x > trim and x < len(sig)-trim]
        all_true_centers = [x for x in all_true_centers if not np.isnan(x) and x > trim and x < len(sig)-trim]
        all_true_offsets = [x for x in all_true_offsets if not np.isnan(x) and x > trim and x < len(sig)-trim]

        all_qrs_onsets = [x for x in obj["qrs"][0] if x > trim and x < len(sig)-trim]
        all_qrs_centers = [x for x in obj["qrs"][1] if x > trim and x < len(sig)-trim]
        all_qrs_offsets = [x for x in obj["qrs"][2] if x > trim and x < len(sig)-trim]

        if key == "p":
            unique_classes = []
            for label in labelregions:
                cls = label[0]
                if cls not in unique_classes:
                    unique_classes.append(cls)
                    if cls != "NSR":
                        self.counts[cls] += 1
            
            if unique_classes == ["NSR"]:
                self.counts["NSR"] += 1

        for cls in self.classes:
            for label in labelregions:
                if label[0] == cls:
                    true_onsets = [x for x in all_true_onsets if x > label[1] and x < label[2]]
                    true_centers = [x for x in all_true_centers if x > label[1] and x < label[2]]
                    true_offsets = [x for x in all_true_offsets if x > label[1] and x < label[2]]
                    true_widths = [x for x in all_true_widths if x[0] > label[1] and x[0] < label[2]]

                    pred_onsets = [x for x in all_pred_onsets if x > label[1] and x < label[2]]
                    pred_centers = [x for x in all_pred_centers if x > label[1] and x < label[2]]
                    pred_offsets = [x for x in all_pred_offsets if x > label[1] and x < label[2]]
                    pred_widths = [x for x in all_pred_widths if x[0] > label[1] and x[0] < label[2]]
                    
                    tp, fp, fn, sds_onset = self.get_tp_fp_fn(pred_onsets, true_onsets, fs, isnan=False)
                    self.macros[cls][key+"_onset"]["tp"] += tp
                    self.macros[cls][key+"_onset"]["fp"] += fp
                    self.macros[cls][key+"_onset"]["fn"] += fn
                    
                    tp, fp, fn, _ = self.get_tp_fp_fn(pred_centers, true_centers, fs, isnan=False)
                    self.macros[cls][key+"_center"]["tp"] += tp
                    self.macros[cls][key+"_center"]["fp"] += fp
                    self.macros[cls][key+"_center"]["fn"] += fn
                    
                    tp, fp, fn, sds_offset = self.get_tp_fp_fn(pred_offsets, true_offsets, fs, isnan=False)
                    self.macros[cls][key+"_offset"]["tp"] += tp
                    self.macros[cls][key+"_offset"]["fp"] += fp
                    self.macros[cls][key+"_offset"]["fn"] += fn
                    self.sd[cls][key].extend(sds_onset)
                    self.sd[cls][key].extend(sds_offset)

                    self.micro_pixel_se[cls][key+"_center"].append(se)
                    self.micro_pixel_sp[cls][key+"_center"].append(sp)
                    se2, sp2 = self.get_sensitivity_specificity(pred_centers, true_centers, all_qrs_centers, fs, isnan=False)
                    #print(cls, key+"_onset", se, sp)
                    self.micro_se[cls][key+"_center"].append(se2)
                    self.micro_sp[cls][key+"_center"].append(sp2)

                    self.micro_dice[cls][key+"_center"].append(dice)
                    self.micro_iou[cls][key+"_center"].append(iou)
                    self.micro_biou[cls][key+"_center"].append(biou)
                    self.micro_bdice[cls][key+"_center"].append(bdice)
                    self.micro_support[cls][key+"_center"].append(support)

                    wers = self.get_width_error(pred_widths, true_widths, fs)
                    self.width[cls][key].extend(wers)

                    self.support[cls][key] += len(true_centers)

    def aggregate(self):

        self.micro_aggregated = {}
        self.macro_aggregated = {}
        self.sd_aggregated = {}
        self.width_aggregated = {}

        for cls in self.classes:
            self.micro_aggregated[cls] = {}
            self.macro_aggregated[cls] = {}
            self.sd_aggregated[cls] = {}
            self.width_aggregated[cls] = {}

            self.softmax[cls]["p"] = np.concatenate(self.softmax[cls]["p"], axis=0) if len(self.softmax[cls]["p"]) > 0 else np.array([])
            self.softmax[cls]["qrs"] = np.concatenate(self.softmax[cls]["qrs"], axis=0) if len(self.softmax[cls]["qrs"]) > 0 else np.array([])
            self.softmax[cls]["t"] = np.concatenate(self.softmax[cls]["t"], axis=0) if len(self.softmax[cls]["t"]) > 0 else np.array([])
            self.softmax[cls]["logits"] = np.concatenate(self.softmax[cls]["logits"], axis=1) if len(self.softmax[cls]["logits"]) > 0 else np.array([])

            self.preds[cls]["p"] = np.concatenate(self.preds[cls]["p"], axis=0) if len(self.preds[cls]["p"]) > 0 else np.array([])
            self.preds[cls]["qrs"] = np.concatenate(self.preds[cls]["qrs"], axis=0) if len(self.preds[cls]["qrs"]) > 0 else np.array([])
            self.preds[cls]["t"] = np.concatenate(self.preds[cls]["t"], axis=0) if len(self.preds[cls]["t"]) > 0 else np.array([])
            self.preds[cls]["p_hat"] = np.concatenate(self.preds[cls]["p_hat"], axis=0) if len(self.preds[cls]["p_hat"]) > 0 else np.array([])
            self.preds[cls]["qrs_hat"] = np.concatenate(self.preds[cls]["qrs_hat"], axis=0) if len(self.preds[cls]["qrs_hat"]) > 0 else np.array([])
            self.preds[cls]["t_hat"] = np.concatenate(self.preds[cls]["t_hat"], axis=0) if len(self.preds[cls]["t_hat"]) > 0 else np.array([])

            for fiducial in self.fiducials:
                self.micro_aggregated[cls][fiducial] = {
                    "se": np.nanmean(self.micro_se[cls][fiducial]),
                    "pp": np.nanmean(self.micro_pp[cls][fiducial]),
                    "f1": np.nanmean(self.micro_f1[cls][fiducial]),
                    "sp": np.nanmean(self.micro_sp[cls][fiducial]),
                    "pixel_se": np.nanmean(self.micro_pixel_se[cls][fiducial]),
                    "pixel_sp": np.nanmean(self.micro_pixel_sp[cls][fiducial]),
                    "iou": np.nanmean(self.micro_iou[cls][fiducial]),
                    "dice": np.nanmean(self.micro_dice[cls][fiducial]),
                    "biou": np.nanmean(self.micro_biou[cls][fiducial]),
                    "bdice": np.nanmean(self.micro_bdice[cls][fiducial]),
                    "support": np.nanmean(self.micro_support[cls][fiducial])
                }

                self.macro_aggregated[cls][fiducial] = {
                    "se": self.macros[cls][fiducial]["tp"]/(self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fn"]) if self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fn"] > 0 else 0,
                    "pp": self.macros[cls][fiducial]["tp"]/(self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fp"]) if self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fp"] > 0 else 0,
                    "f1": 2*self.macros[cls][fiducial]["tp"]/(2*self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fp"]+self.macros[cls][fiducial]["fn"]) if self.macros[cls][fiducial]["tp"]+self.macros[cls][fiducial]["fn"]+self.macros[cls][fiducial]["fp"] > 0 else 0
                }

            for wave in ["p","qrs","t"]:
                self.sd_aggregated[cls][wave+"_center"] = {
                    "mean" : np.nanmean(self.sd[cls][wave]),
                    "sd" : np.nanstd(self.sd[cls][wave])
                }
                self.sd_aggregated[cls][wave+"_onset"] = {"mean" : 0,"sd" : 0}
                self.sd_aggregated[cls][wave+"_offset"] = {"mean" : 0,"sd" : 0}

            for wave in ["p","qrs","t"]:
                self.width_aggregated[cls][wave] = {
                    "mean" : np.nanmean(self.width[cls][wave]),
                    "sd" : np.nanstd(self.width[cls][wave])
                }
        
    def save_preds(self, filename):
        print("Save preds per class")
        #save as pickle file
        if self.model.savelogits:
            with open(filename, 'wb') as f:
                pickle.dump({"regions":self.regions, "logits":self.softmax}, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump({"regions":self.regions, "preds":self.preds}, f)
    
    def get_width_error(self, pred, true, fs):
        widtherrors = []

        if len(pred) == 0:
            return []
        
        if len(true) == 0:
            return []
        
        for p in true:
            diff = np.array([abs(x[0]-p[0]) for x in pred])
            idx = np.argmin(diff)
            val = diff[idx]
            if val < 0.15*fs:
                width1 = pred[idx][1]
                width2 = p[1]
                widtherrors.append(np.abs(width1-width2))

        return widtherrors

    def report(self):
        
        table = Table(title=self.data.get_name()+" F1")
        rows = []
        for cls in self.classes:
            row = [cls]
            for fiducial in self.fiducials:
                row.append(str(np.round(self.macro_aggregated[cls][fiducial]["f1"]*1000)/10))
            rows.append(row)

        columns = ["Class"] + self.fiducials

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)

        table = Table(title=self.data.get_name()+" Boundary Dice Coefficient")
        rows = []
        metrics = ["P", "QRS", "T", "P_sup","P_se","P_sp"]
        for cls in self.classes:
            row = [str.replace(cls,"_"," ")]
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["qrs_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["t_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["support"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["se"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["sp"]*1000)/10))
            rows.append(row)

        columns = ["Arrhythmia"] + metrics

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)
        print(self.to_latex(table))
        print(self.to_csv(table, filename="results/results_"+self.model.name+"_STANFORD.csv"))

    def save_micro_to_csv(self, filename):

        cols = ["Condition", "Fiducial", "SE", "PP", "F1", "SP", "Pixel SE", "Pixel SP", "IoU", "Dice", "BIoU", "BDice", "Support", "Error Mean", "Error SD"]
        f = open(filename, "w")
        csv = ""

        for col in cols:
            csv += col + ","
            
        csv = csv[:-1] +"\n"
        f.write(csv)

        for cls in self.classes:
            for fiducial in self.fiducials:
                row = [cls, fiducial]
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["se"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["pp"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["f1"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["sp"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["pixel_se"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["pixel_sp"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["iou"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["dice"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["biou"]*1000)/10))
                row.append(str(np.round(self.micro_aggregated[cls][fiducial]["bdice"]*1000)/10))
                row.append(str(self.micro_aggregated[cls][fiducial]["support"]))
                row.append(str(np.round(self.sd_aggregated[cls][fiducial]["mean"]*10000)/10))
                row.append(str(np.round(self.sd_aggregated[cls][fiducial]["sd"]*10000)/10))

                csv = ""
                for col in row:
                    csv += col + ","
                csv = csv[:-1] + "\n"
                f.write(csv)

        f.close()

        return filename

class PerArrhythmiaBenchmark(PerClassBenchmark):
    def __init__(self, data, model, trim=0, fiducials=["p_onset", "p_center", "p_offset", "qrs_onset", "qrs_center", "qrs_offset", "t_onset", "t_center", "t_offset"]):
        
        self.data = data
        self.model = model
        self.trim = trim
        self.trimleft = 0
        self.trimright = 0
        self.fiducials = fiducials
        self.arrhythmias = {}
        self.classes = []
        self.arrhythmia_table = {}
        self.get_arrhythmias_from_labels(data)

        self.initialize()

    def load_arrhythmia_csv(self, path):
        df = {"Major":[],"Minor":[],"Name":[], "Abreviation":[],"Synonyms":[]}

        with open(path, "r") as f:
            for line in f:
                row = line.strip()
                cells = row.split(",")
                major = cells[0]
                minor = cells[1]
                name = cells[2]
                abbr = cells[3]
                df["Major"].append(major)
                df["Minor"].append(minor)
                df["Name"].append(name)
                df["Abreviation"].append(abbr)
                
                synonyms = cells[2:]
                synonymsstripped = []
                for synonym in synonyms:
                    stripped = synonym.strip().lower()
                    if stripped != "":
                        synonymsstripped.append(stripped)
                df["Synonyms"].append(synonymsstripped)

        df = pd.DataFrame(df)
        self.arrhythmia_df = df

    def get_arrhythmias_from_labels(self, data):
        
        basefolder = os.environ.get('benchmark_data')
        self.load_arrhythmia_csv(basefolder+"/arrhythmia_classes.csv")
        for record in data:
            self.arrhythmias[record] = {
                'matched': False,
                'arrhythmias': []
            }
        self.classes = []
        self.arrhythmia_table = {}

        for i in range(len(self.arrhythmia_df)):
            arrhythmia_major = self.arrhythmia_df['Major'][i]
            arrhythmia_minor = self.arrhythmia_df['Minor'][i]
            name = self.arrhythmia_df['Name'][i]
            abbr = self.arrhythmia_df['Abreviation'][i]
            synonyms = self.arrhythmia_df['Synonyms'][i]

            for record in data:
                fulllabel = data[record]['diagnosis'].strip()
                obj = {}
                for syn in synonyms:
                    syn = syn.strip()
                    regex = r'(^{syn}$|^{syn},|^{syn}\s?(and|or)|,\s?{syn},|,\s?{syn}$)'.format(syn=syn)
                    if re.search(regex, fulllabel, re.IGNORECASE):
                        if abbr not in self.classes:
                            self.classes.append(abbr)
                            self.arrhythmia_table[abbr] = {
                                "name": name,
                                "abbr": abbr,
                                "major": arrhythmia_major,
                                "minor": arrhythmia_minor,
                                "count": 1
                            }
                        else:
                            self.arrhythmia_table[abbr]["count"] += 1
                        self.arrhythmias[record]['matched'] = True
                        self.arrhythmias[record]['arrhythmias'].append(abbr)
                        break
        
        notmatched = 0
        matched = 0
        for i, rec in enumerate(self.arrhythmias):
            if not self.arrhythmias[rec]["matched"]:
                notmatched += 1
                print("start:|",data[rec]['diagnosis'], "|end")
        print("Not matched: ", notmatched)

    def get_performance(self, obj, binary, true, true_binary, fs, key, isbinary=False, record=""):
        if isbinary:
            binary = proccess(binary, fs)
            regions = get_regions(binary)
        else:
            regions = binary
            binary = self.region_to_binary(regions, len(obj["signal"]))
    
        smooth = 1e-6



        binary = binary[int(self.trimleft):int(self.trimright)]
        true_binary = true_binary[int(self.trimleft):int(self.trimright)]

        intersection = np.sum(binary*true_binary)
        dice = (2*intersection)/(np.sum(binary)+np.sum(true_binary)+smooth)
        iou = (intersection)/(np.sum(binary)+np.sum(true_binary)-intersection+smooth)
        biou, bdice, support = self.boundary_iou(true_binary, binary, tolerance=int(0.075*fs))
        tp = np.sum((true_binary == 1) & (binary == 1))
        tn = np.sum((true_binary == 0) & (binary == 0))
        fp = np.sum((true_binary == 0) & (binary == 1))
        fn = np.sum((true_binary == 1) & (binary == 0))

        # Calculate sensitivity and specificity
        se = tp / (tp + fn) if (tp + fn) != 0 else 0
        sp = tn / (tn + fp) if (tn + fp) != 0 else 0

        all_pred_onsets = []
        all_pred_centers = []
        all_pred_offsets = []
        all_pred_widths = []
        trim = self.trim*fs
        sig = obj["signal"]
        arrhythmias = self.arrhythmias[obj["record"]]['arrhythmias']

        for (st,en) in regions:
            if not np.isnan(st) and st > trim and st < len(sig)-trim:
                all_pred_onsets.append(st)
            if not np.isnan(en) and en > trim and en < len(sig)-trim:
                all_pred_offsets.append(en)
            if not np.isnan(st) and not np.isnan(en) and st > trim and st < len(sig)-trim and en > trim and en < len(sig)-trim:
                all_pred_centers.append((st+en)//2)
                all_pred_widths.append(((st+en)//2,(en-st)/fs))

        if record not in self.regions:
            self.regions[record] = {}

        self.regions[record][key] = regions

        all_true_onsets = true[0]
        all_true_centers = true[1]
        all_true_offsets = true[2]
        all_true_widths = []
        for i in range(len(all_true_onsets)):
            ons = all_true_onsets[i]
            endsearch = all_true_onsets[i+1] if i < len(all_true_onsets)-1 else len(sig)
            for i in range(len(all_true_offsets)):
                if all_true_offsets[i] > ons and all_true_offsets[i] < endsearch:
                    off = all_true_offsets[i]
                    all_true_widths.append(((ons+off)/2, (off-ons)/fs))

        all_true_onsets = [x for x in all_true_onsets if not np.isnan(x) and x > trim and x < len(sig)-trim]
        all_true_centers = [x for x in all_true_centers if not np.isnan(x) and x > trim and x < len(sig)-trim]
        all_true_offsets = [x for x in all_true_offsets if not np.isnan(x) and x > trim and x < len(sig)-trim]

        all_qrs_onsets = [x for x in obj["qrs"][0] if x > trim and x < len(sig)-trim]
        all_qrs_centers = [x for x in obj["qrs"][1] if x > trim and x < len(sig)-trim]
        all_qrs_offsets = [x for x in obj["qrs"][2] if x > trim and x < len(sig)-trim]

        for arrhythmia in arrhythmias:
            tp, fp, fn, sds_onset = self.get_tp_fp_fn(all_pred_onsets, all_true_onsets, fs, isnan=False)
            self.macros[arrhythmia][key+"_onset"]["tp"] += tp
            self.macros[arrhythmia][key+"_onset"]["fp"] += fp
            self.macros[arrhythmia][key+"_onset"]["fn"] += fn
            
            tp, fp, fn, _ = self.get_tp_fp_fn(all_pred_centers, all_true_centers, fs, isnan=False)
            self.macros[arrhythmia][key+"_center"]["tp"] += tp
            self.macros[arrhythmia][key+"_center"]["fp"] += fp
            self.macros[arrhythmia][key+"_center"]["fn"] += fn
            
            tp, fp, fn, sds_offset = self.get_tp_fp_fn(all_pred_offsets, all_true_offsets, fs, isnan=False)
            self.macros[arrhythmia][key+"_offset"]["tp"] += tp
            self.macros[arrhythmia][key+"_offset"]["fp"] += fp
            self.macros[arrhythmia][key+"_offset"]["fn"] += fn
            self.sd[arrhythmia][key].extend(sds_onset)
            self.sd[arrhythmia][key].extend(sds_offset)

            self.micro_pixel_se[arrhythmia][key+"_center"].append(se)
            self.micro_pixel_sp[arrhythmia][key+"_center"].append(sp)
            se2, sp2 = self.get_sensitivity_specificity(all_pred_centers, all_true_centers, all_qrs_centers, fs, isnan=False)
            #print(cls, key+"_onset", se, sp)
            self.micro_se[arrhythmia][key+"_center"].append(se2)
            self.micro_sp[arrhythmia][key+"_center"].append(sp2)
            self.micro_dice[arrhythmia][key+"_center"].append(dice)
            self.micro_iou[arrhythmia][key+"_center"].append(iou)
            self.micro_biou[arrhythmia][key+"_center"].append(biou)
            self.micro_bdice[arrhythmia][key+"_center"].append(bdice)
            self.micro_support[arrhythmia][key+"_center"].append(support)

            wers = self.get_width_error(all_pred_widths, all_true_widths, fs)
            self.width[arrhythmia][key].extend(wers)

            self.support[arrhythmia][key] += len(all_true_centers)

    def report(self):
        
        basefolder = os.environ.get('benchmark_results')
        table = Table(title=self.data.get_name()+" F1")
        rows = []
        for cls in self.classes:
            row = [cls]
            for fiducial in self.fiducials:
                row.append(str(np.round(self.macro_aggregated[cls][fiducial]["f1"]*1000)/10))
            rows.append(row)

        columns = ["Class"] + self.fiducials

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)

        table = Table(title=self.data.get_name()+" Boundary Dice Coefficient")
        rows = []
        metrics = ["P", "QRS", "T", "P_sup","P_se","P_sp"]
        for cls in self.classes:
            row = [str.replace(cls,"_"," ")]
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["qrs_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["t_center"]["bdice"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["support"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["se"]*1000)/10))
            row.append(str(np.round(self.micro_aggregated[cls]["p_center"]["sp"]*1000)/10))
            rows.append(row)

        columns = ["Arrhythmia"] + metrics

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)
        print(self.to_latex(table))
        print(self.to_csv(table, filename=basefolder+"/delineation/arrhythmia_specific_results_"+self.model.name+"_"+self.data.get_name()+".csv"))

        json.dump(self.arrhythmia_table, open(basefolder+"/delineation/arrhythmia_table_"+self.data.get_name()+".json", "w"))


class DiagnosticBenchmark(BaseBenchmark):
    def __init__(self, data, models):

        if not isinstance(models, list):
            models = [models]

        super().__init__(data, models[0], 0)
        self.models = models

        self.initialize()

    def initialize(self):
        self.set_predictions = {}
        for model in self.models:
            self.set_predictions[model.name] = []
            if not hasattr(model, "save_output"):
                model.save_output = False
            if not hasattr(model, "savelogits"):
                model.savelogits = False
            if not hasattr(model, "modelpaths"):
                model.modelpaths = []
            if not hasattr(model, "name"):
                model.name = "unknown_model"
        
    def run(self, overwrite=False):
        
        #check if a csv exists
        basefolder = os.environ.get('benchmark_results')
        modelnames = ("_".join([model.name for model in self.models]))
        resfiles = glob.glob(f"{basefolder}/diagnosis/set_level_diagnosis*_{modelnames}.csv")
        filename = ""
        if len(resfiles) == 0 or overwrite:

            #get date and time for unique filename
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

            if np.any([model.save_output for model in self.models]):
                filename = self.create_json()

            for i, record in tqdm(enumerate(self.data)):
                self.data.get_data(record)
                obj = self.data[record]
                signal = obj["signal"]
                fs = obj["fs"]
                true_episodes = obj["labelregions"]
                for model in self.models:
                    if model.name not in self.set_predictions:
                        self.set_predictions[model.name] = []
                    pred, rest = model.predict(signal, fs, meta=obj)
                    self.get_performance(obj, model, pred, true_episodes, fs, record=record)

                    for k,v in rest.items():
                        self.set_predictions[model.name][-1][k] = v

                    if model.save_output:
                        self.append_to_json(filename, model.name, [self.set_predictions[model.name][-1]], rest)
                
                #if i > 10:
                #    break

        self.aggregate()
        self.report()

        if np.any([m.savelogits for m in self.models]):
            self.find_and_apply_optimal_threshold()
            self.append_to_json(filename, model.name, self.set_predictions[model.name], {})

        self.aggregate()
        self.report()

    def run_batch(self, overwrite=False):
        
        #check if a csv exists
        basefolder = os.environ.get('benchmark_results')
        modelnames = ("_".join([model.name for model in self.models]))
        resfiles = glob.glob(f"{basefolder}/diagnosis/set_level_diagnosis*_{modelnames}.csv")
        filename = ""

        if len(resfiles) == 0 or overwrite:

            #get date and time for unique filename
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

            if np.any([model.save_output for model in self.models]):
                filename = self.create_json()

            for model in self.models:
                if model.name not in self.set_predictions:
                    self.set_predictions[model.name] = []
                
                preds = model.predict_batch(self.data)

                for pred in preds:
                    print(pred["record"])
                    obj = self.data[pred["record"]]
                    true_episodes = obj["labelregions"]
                    fs = obj["fs"]
                    self.get_performance(obj, model, pred["diagnoses"], true_episodes, fs, record=pred["record"])
                    del pred["diagnoses"]
                    del pred["record"]
                    
                #print(preds)

                if model.save_output:
                    self.append_to_json(filename, model.name, self.set_predictions[model.name], preds)

        self.aggregate()
        self.report()

        if np.any([m.savelogits for m in self.models]):
            self.find_and_apply_optimal_threshold()
            self.append_to_json(filename, model.name, self.set_predictions[model.name], {})

        self.aggregate()
        self.report()

    def get_performance(self, obj, model, predictions, true, fs, record=""):

        set_level_trues = [d[0] for d in true]
        set_level_trues = [self.data.class_mapper[d] if d in self.data.class_mapper else d for d in set_level_trues]

        if predictions != None:
            if model.savelogits and "logits" in predictions:
                set_level_logits = predictions["logits"]
                #turn into list of lists
                set_level_logits = [[float(x) for x in logits] for logits in set_level_logits]
                set_level_predictions = [d["type"] for d in predictions["predictions"]] 
            else:
                set_level_logits = []
                set_level_predictions = [d["type"] for d in predictions]

            set_level_predictions = [self.data.class_mapper[d] if d in self.data.class_mapper else d for d in set_level_predictions]
            #if "arrhythmia" in obj:
            #    set_level_predictions = [d for d in set_level_predictions if d == self.data.class_mapper[obj["arrhythmia"]]]
        else:
            set_level_logits = []
            set_level_predictions = ["NA"]

        row = {
            "record": record,
            "true": set_level_trues,
            "predicted": set_level_predictions,
            "logits": set_level_logits
        }

        self.set_predictions[model.name].append(row)

    def ci_interval(self, values, ci=95):
            return np.nanmean(values), np.nanpercentile(values, (100 - ci) / 2), np.nanpercentile(values, 100 - (100 - ci) / 2)

    def aggregate(self, filename="", bootstrap=False, usecache=False, triage=False, sampled=False):

        if filename != "":
            with open(filename, 'r') as f:
                data = json.load(f)
            if "model" not in data["results"][0]:
                data["results"] = [{"model": data["model"], "modelpaths": data["modelpaths"], "results": data["results"]}]

            meta = {
                "date": data["date"], 
                "models": [m["model"] for m in data["results"]], 
                "modelpaths": [m["modelpaths"] for m in data["results"]],
                #"type": (data["modelpaths"][0].split("_")[0] + data["modelpaths"][0].split("1d")[1]) if len(data["modelpaths"]) > 0 else ""
            }
        elif np.any([model.save_output for model in self.models]):
            basefolder = os.environ.get('benchmark_results')
            modelnames = ("_".join([model.name for model in self.models]))
            files = glob.glob(f"{basefolder}/diagnosis/set_level_diagnosis*_{modelnames}_{self.data.name}_*.json")
            files.sort(reverse=True)

            if len(files) > 1:
                print("Multiple files found, using most recent file")
            elif len(files) == 0:
                print("No files found")
                return
            
            with open(files[0], 'r') as f:
                data = json.load(f)
        else:
            resobj = []
            for model in self.models:
                resobj.append({"model": model.name, "results": self.set_predictions[model.name]})

            data = {"results": resobj}
            meta = {
                "models": [model.name for model in self.models], 
                "modelpaths": [model.modelpaths if hasattr(model, "modelpaths") else [] for model in self.models],
                #"type": (self.model.modelpaths[0].split("1d")[1]) if hasattr(self.model, "modelpaths") else ""
            }

        results = data["results"]
        #print(results)
        if triage:
            class_mapper = {
                    "AFIB": "NOTACUTE",
                    "AFL": "NOTACUTE",
                    "AFIB/AFL": "NOTACUTE",
                    "AVB": "ACUTE",
                    "AVB_TYPE2": "SUBACUTE",
                    "AVB_TYPE1": "NORMAL",
                    "SUDDEN_BRADY": "ACUTE",
                    "BIGEMINY": "NORMAL",
                    "TRIGEMINY": "NORMAL",
                    "EAR": "NOTACUTE",
                    "IVR": "ACUTE",
                    "JUNCTIONAL": "ACUTE",
                    "NOISE": "NORMAL",
                    "NSR": "NORMAL",
                    "SVT": "ACUTE",
                    "VT": "ACUTE",
                    "WENCKEBACH": "SUBACUTE",
                    "N": "NORMAL",
                    "A": "NOTACUTE",
                    "O": "NOTACUTE",
                    "~": "NORMAL"
                }
            arrhythmias = ["NORMAL", "NOTACUTE", "SUBACUTE", "ACUTE"]
        else:
            class_mapper = self.data.get_class_mapper()
            arrhythmias = list(set(class_mapper.values()))
            val_only = [a for a in arrhythmias if a not in list(class_mapper.keys())]
            for v in val_only:
                class_mapper[v] = v
            arrhythmias.sort()

        y_true_all = []
        y_pred_all = []

        for m in range(len(results)):
            modelresults = results[m]["results"]
            y_true = []
            y_pred = []
            #print(len(modelresults), "records for model", results[m]["model"])
            for i in range(len(modelresults)):
                one_hot = np.zeros(len(arrhythmias)+1)
                if triage:
                    predicted_scores = [arrhythmias.index(class_mapper[d]) for d in modelresults[i]["predicted"] if d in class_mapper]
                    if len(predicted_scores) == 0:
                        predicted_scores = [0]
                    worst_score = np.max(predicted_scores)
                    one_hot[worst_score] = 1
                elif sampled:
                    #the arrhythmia we are interested in 
                    focused_arrhythmia = modelresults[i]["arrhythmia"]
                    #the arrhythmia we will accept to be correct also
                    accepted_arrhythmia = self.data.case_mapper[focused_arrhythmia] if focused_arrhythmia in self.data.case_mapper else [focused_arrhythmia]

                    #get the arrhythmias from the model results
                    if "raw" in modelresults[i]:
                        preds, _ = self.model.process_diagnoses(modelresults[i]["raw"])
                        preds = [p["type"] for p in preds]
                    else:
                        preds = modelresults[i]["predicted"]

                    #map the arrhythmias to the class mapper
                    predicted_arrhythmias = [class_mapper[d] for d in preds if d in class_mapper]

                    #filter the predicted arrhythmias to only those that are in the accepted arrhythmia
                    predicted_arrhythmias = [d for d in predicted_arrhythmias if d in accepted_arrhythmia]

                    if len(predicted_arrhythmias) > 0:
                        predicted_arrhythmias = [focused_arrhythmia]

                    # if focused_arrhythmia == "VT>10s":
                    #     print("pred:", modelresults[i]["record"], accepted_arrhythmia, predicted_arrhythmias)


                    for d in predicted_arrhythmias:
                        if d in class_mapper:
                            one_hot[arrhythmias.index(class_mapper[d])] = 1
                else:
                    for d in modelresults[i]["predicted"]:
                        if d in class_mapper:
                            one_hot[arrhythmias.index(class_mapper[d])] = 1

                #set activity flag
                if len(modelresults[i]["predicted"]) == 1 and modelresults[i]["predicted"][0] == "NA":
                    #print("NOT ACTIVE")
                    one_hot[len(arrhythmias)] = 0
                else:
                    #print(modelresults[i]["record"], "ACTIVE")
                    one_hot[len(arrhythmias)] = 1

                y_pred.append(one_hot)

                one_hot = np.zeros(len(arrhythmias)+1)
                if triage:
                    true_scores = [arrhythmias.index(class_mapper[d]) for d in modelresults[i]["true"] if d in class_mapper]
                    if len(true_scores) == 0:
                        true_scores = [0]
                    worst_score = np.max(true_scores)
                    one_hot[worst_score] = 1
                elif sampled:
                    focused_arrhythmia = modelresults[i]["arrhythmia"]
                    accepted_arrhythmia = self.data.case_mapper[focused_arrhythmia] if focused_arrhythmia in self.data.case_mapper else [focused_arrhythmia]
                    trues = self.data.annotation_data[modelresults[i]["record"]]
                    true_arrhythmias = [class_mapper[d] for d in trues if d in class_mapper]
                    true_arrhythmias = [d for d in true_arrhythmias if d in accepted_arrhythmia]

                    if len(true_arrhythmias) > 0:
                        true_arrhythmias = [focused_arrhythmia]

                    # if focused_arrhythmia == "VT>10s":
                    #     print("true", modelresults[i]["record"], accepted_arrhythmia, true_arrhythmias)

                    for d in true_arrhythmias:
                        if d in class_mapper:
                            one_hot[arrhythmias.index(class_mapper[d])] = 1
                else:
                    for d in modelresults[i]["true"]:
                        if d in class_mapper:
                            one_hot[arrhythmias.index(class_mapper[d])] = 1
                y_true.append(one_hot)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        self.set_metrics = {}
        self.set_distributions = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.true_positives = {}
        self.confusion_matrix = np.zeros((len(arrhythmias), len(arrhythmias)))

        bootstrap_n = 1000 if bootstrap else 1
        cacheexists = False
        self.bootstrap_indices = {}
        num_splits = 1

        for m in range(len(results)):
            modelresults = results[m]["results"]
            model = results[m]["model"]
            self.false_negatives[model] = {}
            self.false_positives[model] = {}
            self.true_positives[model] = {}

            for i in range(len(modelresults)):
                
                y_true_arrhythmia = y_true_all[m,i,:-1]
                y_pred_arrhythmia = y_pred_all[m,i,:-1]

                self.confusion_matrix += np.outer(y_true_arrhythmia, y_pred_arrhythmia)



        for arrhythmia in arrhythmias:
            
            acc_vals = []
            se_vals = []
            sp_vals = []
            ppv_vals = []
            npv_vals = []
            fnr_vals = []
            f1_vals = []
            tp_vals = []
            fp_vals = []
            fn_vals = []

            for m in range(len(results)):
                modelresults = results[m]["results"]
                model = results[m]["model"]
                self.false_negatives[model] = {}
                self.false_positives[model] = {}
                self.true_positives[model] = {}
                for i in range(len(modelresults)):
                    
                    y_true_arrhythmia = y_true_all[m,i,arrhythmias.index(arrhythmia)]
                    y_pred_arrhythmia = y_pred_all[m,i,arrhythmias.index(arrhythmia)]

                    #if "seen_elsewhere" in modelresults[i] and modelresults[i]["seen_elsewhere"] == 1 and y_pred_arrhythmia == 0:
                    #    y_pred_all[m,i,arrhythmias.index(arrhythmia)] = 1

                    # if np.sum(y_pred_arrhythmia) == 0:
                    #     continue

                    if y_true_arrhythmia == 1 and y_pred_arrhythmia == 0:
                        #print(f"False negative for {arrhythmia} in record {modelresults[i]['record']}")
                        #if arrhythmia == "N":
                        #    print(modelresults[i]["record"])#, arrhythmias[np.where(y_true_all[m,i,:])[0][0]])
                        #print(modelresults[i]["record"])
                        if arrhythmia not in self.false_negatives[model]:
                            self.false_negatives[model][arrhythmia] = []
                        self.false_negatives[model][arrhythmia].append(modelresults[i]["record"])
                    
                    if y_true_arrhythmia == 0 and y_pred_arrhythmia == 1:
                        # if arrhythmia == "N":
                        #    print(modelresults[i]["record"])#, arrhythmias[np.where(y_true_all[m,i,:])[0][0]])
                        if arrhythmia not in self.false_positives[model]:
                            self.false_positives[model][arrhythmia] = []
                        self.false_positives[model][arrhythmia].append(modelresults[i]["record"])
                    

                    if y_true_arrhythmia == 1 and y_pred_arrhythmia == 1:
                        if arrhythmia not in self.true_positives[model]:
                            self.true_positives[model][arrhythmia] = []
                        self.true_positives[model][arrhythmia].append(modelresults[i]["record"])

            # if arrhythmia == "AVB_TYPE2":
            #     model = results[0]["model"]
            #     # print(self.false_negatives)
            #     print(f"False negatives for {arrhythmia}", len(self.false_negatives[model][arrhythmia]))
            #     print(f"False positives for {arrhythmia}", len(self.false_positives[model][arrhythmia]))
            #     # print(f"True positives for {arrhythmia}", len(self.true_positives[model][arrhythmia]))

            #     for rec in self.false_negatives[model][arrhythmia]:
            #         print("FN", rec)

            #     for rec in self.false_positives[model][arrhythmia]:
            #         print("FP", rec)

            y_true_arrhythmia = y_true_all[:,:,arrhythmias.index(arrhythmia)]
            y_pred_arrhythmia = y_pred_all[:,:,arrhythmias.index(arrhythmia)]
            use_prediction = y_pred_all[:,:,len(arrhythmias)] == 1


            if not cacheexists:
                self.bootstrap_indices[arrhythmia] = []

            for i in range(bootstrap_n):
                
                modelinds = np.random.randint(0, y_true_arrhythmia.shape[0], size=y_true_arrhythmia.shape[1])
                mask = use_prediction[modelinds,np.arange(y_true_arrhythmia.shape[1])]
                y_true_arrhythmia_bootstrap = y_true_arrhythmia[modelinds,np.arange(y_true_arrhythmia.shape[1])]
                y_pred_arrhythmia_bootstrap = y_pred_arrhythmia[modelinds,np.arange(y_true_arrhythmia.shape[1])]

                y_true_arrhythmia_bootstrap = y_true_arrhythmia_bootstrap[mask]
                y_pred_arrhythmia_bootstrap = y_pred_arrhythmia_bootstrap[mask]

                if bootstrap:
                    if cacheexists:
                        indices = self.bootstrap_indices[arrhythmia][i]
                    else:
                        indices = np.random.choice(np.arange(y_true_arrhythmia_bootstrap.shape[0]), y_true_arrhythmia_bootstrap.shape[0], replace=True)
                        self.bootstrap_indices[arrhythmia].append([int(s) for s in indices])
                else:
                    indices = np.arange(y_true_arrhythmia_bootstrap.shape[0])

                y_true_arrhythmia_bootstrap = y_true_arrhythmia_bootstrap[indices]
                y_pred_arrhythmia_bootstrap = y_pred_arrhythmia_bootstrap[indices]

                tp = np.sum(np.logical_and(y_true_arrhythmia_bootstrap, y_pred_arrhythmia_bootstrap))
                fp = np.sum(np.logical_and((1-y_true_arrhythmia_bootstrap),y_pred_arrhythmia_bootstrap))
                fn = np.sum(np.logical_and(y_true_arrhythmia_bootstrap,(1-y_pred_arrhythmia_bootstrap)))
                tn = np.sum(np.logical_and((1-y_true_arrhythmia_bootstrap),(1-y_pred_arrhythmia_bootstrap)))


                # print("Arrhythmia:", arrhythmia)

                # if tp + fp + fn == 0:
                #     acc = se = sp = ppv = npv = fnr = f1 = 0
                #     tp = fp = fn = 0

                # else:
                acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else np.nan
                se = tp / (tp + fn) if (tp + fn) != 0 else np.nan
                sp = tn / (tn + fp) if (tn + fp) != 0 and tp > 0 else np.nan
                ppv = tp / (tp + fp) if (tp + fp) != 0 else np.nan
                npv = tn / (tn + fp) if (tn + fp) != 0 else np.nan
                fnr = fn / (fn + tp) if (fn + tp) != 0 else np.nan
                f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) != 0 else np.nan

                if not np.isnan(acc):
                    acc_vals.append(acc)
                if not np.isnan(se):
                    se_vals.append(se)
                if not np.isnan(sp):
                    sp_vals.append(sp)
                if not np.isnan(ppv):
                    ppv_vals.append(ppv)
                if not np.isnan(npv):
                    npv_vals.append(npv)
                if not np.isnan(fnr):
                    fnr_vals.append(fnr)
                if not np.isnan(f1):
                    f1_vals.append(f1)
                if not np.isnan(tp):
                    tp_vals.append(tp)
                if not np.isnan(fp):
                    fp_vals.append(fp)
                if not np.isnan(fn):
                    fn_vals.append(fn)

            if len(acc_vals) == 0:
                acc_vals = [0]
            if len(se_vals) == 0:
                se_vals = [0]
            if len(sp_vals) == 0:
                sp_vals = [0]
            if len(ppv_vals) == 0:
                ppv_vals = [0]
            if len(npv_vals) == 0:
                npv_vals = [0]
            if len(fnr_vals) == 0:
                fnr_vals = [0]
            if len(f1_vals) == 0:
                f1_vals = [0]
            if len(tp_vals) == 0:
                tp_vals = [0]
            if len(fp_vals) == 0:
                fp_vals = [0]
            if len(fn_vals) == 0:
                fn_vals = [0]

            acc_m, acc_low, acc_high = self.ci_interval(acc_vals)
            se_m, se_low, se_high = self.ci_interval(se_vals)
            sp_m, sp_low, sp_high = self.ci_interval(sp_vals)
            ppv_m, ppv_low, ppv_high = self.ci_interval(ppv_vals)
            npv_m, npv_low, npv_high = self.ci_interval(npv_vals)
            fnr_m, fnr_low, fnr_high = self.ci_interval(fnr_vals)
            f1_m, f1_low, f1_high = self.ci_interval(f1_vals)
            tp_m, tp_low, tp_high = self.ci_interval(tp_vals)
            fp_m, fp_low, fp_high = self.ci_interval(fp_vals)
            fn_m, fn_low, fn_high = self.ci_interval(fn_vals)

            self.set_distributions[arrhythmia] = {
                "acc": acc_vals,
                "se": se_vals,
                "sp": sp_vals,
                "ppv": ppv_vals,
                "npv": npv_vals,
                "fnr": fnr_vals,
                "f1": f1_vals,
                "tp": tp_vals,
                "fp": fp_vals,
                "fn": fn_vals
            }

            self.set_metrics[arrhythmia] = {
                "acc": [acc_m, acc_low, acc_high],
                "se": [se_m, se_low, se_high],
                "sp": [sp_m, sp_low, sp_high],
                "ppv": [ppv_m, ppv_low, ppv_high],
                "npv": [npv_m, npv_low, npv_high],
                "fnr": [fnr_m, fnr_low, fnr_high],
                "f1": [f1_m, f1_low, f1_high],
                "tp": [tp_m, tp_low, tp_high],
                "fp": [fp_m, fp_low, fp_high],
                "fn": [fn_m, fn_low, fn_high]
            }

        return self.set_metrics, self.set_distributions
            #print(arrhythmia)
            #print(self.metrics[arrhythmia])

    def find_and_apply_optimal_threshold(self):

        model = self.models[0]
        tasks = list(model.output_to_label.keys())
        n_tasks = np.array(self.set_predictions[model.name][0]["logits"]).shape[1]
        optimal_thresholds = []
        new_labels = []

        for i in range(len(self.set_predictions[model.name])):
            self.set_predictions[model.name][i]["predicted"] = []

        for taski in range(n_tasks):
            best_ba = -1
            best_thresh = 0.5
            logits = np.array([np.array(self.set_predictions[model.name][i]["logits"])[:,taski].max() for i in range(len(self.set_predictions[model.name]))])
            gt = [self.set_predictions[model.name][i]["true"] for i in range(len(self.set_predictions[model.name]))]
            print(self.data.class_mapper[model.output_to_label[tasks[taski]]])
            gt_binary = [int(self.data.class_mapper[model.output_to_label[tasks[taski]]] in gti) for gti in gt]
            #use task_label to make two binary arrays
            
            for thresh in np.linspace(0.01, 0.99, 99):
                pred_labels = (logits > thresh).astype(int)
                if np.all(gt_binary == 0) and np.all(pred_labels == 0):
                    continue
                # print(tasks[taski], gt_binary, pred_labels)
                ba = balanced_accuracy_score(gt_binary, pred_labels)
                if ba > best_ba:
                    best_ba = ba
                    best_thresh = thresh
            optimal_thresholds.append(best_thresh)

            # Apply the best threshold
            pred_labels = (logits > best_thresh).astype(int)
            for i in range(len(self.set_predictions[model.name])):
                if pred_labels[i]:
                    #print(self.set_predictions[model.name][i]["predicted"], task_label[task])
                    self.set_predictions[model.name][i]["predicted"] = self.data.class_mapper[model.output_to_label[tasks[taski]]]

        for i in range(len(self.set_predictions[model.name])):
            self.set_predictions[model.name][i]["predicted"] = list(set(self.set_predictions[model.name][i]["predicted"]))
            # if len(self.set_predictions[model.name][i]["predicted"]) == 0:
            #     self.set_predictions[model.name][i]["predicted"] = ["~"]
            #print(self.set_predictions[model.name][i]["predicted"])

        return optimal_thresholds

    def export_table(self, metrics):

        def format_ci(ci, percentage=True):
            if percentage:
                return str(np.round(ci[0]*1000)/10) + " (" + str(np.round(ci[1]*1000)/10) + "-" + str(np.round(ci[2]*1000)/10) + ")"
            return str(np.round(ci[0], 2)) + " (" + str(np.round(ci[1], 2)) + "-" + str(np.round(ci[2], 2)) + ")"

        table = {}
        for arrhythmia in self.set_metrics.keys():
            table[arrhythmia] = {metric: format_ci(self.set_metrics[arrhythmia][metric]) for metric in metrics}

        return table

    def print_latex_table(self):

        table = Table(title=self.data.get_name()+" Performance per Arrhythmia")
        columns = ["Arrhythmia", "Accuracy", "SE", "PPV", "NPV", "F1"]
        metrics = ["acc", "sp", "se", "ppv", "npv", "f1"]
        rows = []
        arrhythmias = list(self.set_metrics.keys())

        def format_ci(ci, percentage=True):
            if percentage:
                return str(np.round(ci[0]*1000)/10) + " (" + str(np.round(ci[1]*1000)/10) + "-" + str(np.round(ci[2]*1000)/10) + ")"
            return str(np.round(ci[0], 2)) + " (" + str(np.round(ci[1], 2)) + "-" + str(np.round(ci[2], 2)) + ")"

        averages = {}
        for metric in metrics:
            averages[metric] = 0

        for arrhythmia in arrhythmias:
            row = [arrhythmia]
            for metric in metrics:
                averages[metric] += self.set_metrics[arrhythmia][metric][0]
                row.append(format_ci(self.set_metrics[arrhythmia][metric], percentage=True))
            rows.append(row)

        row = ["Average"]
        for metric in metrics:
            row.append(str(np.round(averages[metric]/len(arrhythmias)*1000)/10))
        rows.append(row)

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)

    def report(self):
        
        table = Table(title=self.data.get_name()+" Macro Performance per Arrhythmia")
        columns = ["Arrhythmia", "SP", "SE", "PPV", "NPV", "F1", "TP", "FP", "FN"]
        metrics = ["sp", "se", "ppv", "npv", "f1", "tp", "fp", "fn"]
        rows = []
        arrhythmias = list(self.set_metrics.keys())

        patients_per_arrhythmia = {
            "NSR": 55,
            "NOISE": 74,
            "AFIB": 395,
            "VT>10s": 4735,
            "VT<10s": 399,
            "SVT>30s": 277,
            "IVR": 1009,
            "TRIGEMINY": 264,
            "BIGEMINY": 268,
            "WENCKEBACH": 3750,
            "AVB_TYPE2": 1387,
            "SUDDEN_BRADY": 4835,
            "NORMAL": 125,
            "NONCRITICAL": 6799,
            "CRITICAL": 10388
        }

        def format_ci(ci, percentage=True):
            if percentage:
                return str(np.round(ci[0]*1000)/10) + " (" + str(np.round(ci[1]*1000)/10) + "-" + str(np.round(ci[2]*1000)/10) + ")"
            return str(np.round(ci[0], 2)) + " (" + str(np.round(ci[1], 2)) + "-" + str(np.round(ci[2], 2)) + ")"

        averages = {}
        for metric in metrics:
            averages[metric] = 0

        for arrhythmia in arrhythmias:
            row = [arrhythmia]
            for metric in metrics:
                if metric in ["tp", "fp", "fn"]:
                    self.set_metrics[arrhythmia][metric][0] *= 1000 / patients_per_arrhythmia[arrhythmia]
                    self.set_metrics[arrhythmia][metric][1] *= 1000 / patients_per_arrhythmia[arrhythmia]
                    self.set_metrics[arrhythmia][metric][2] *= 1000 / patients_per_arrhythmia[arrhythmia]
                    averages[metric] += self.set_metrics[arrhythmia][metric][0] 
                    row.append(format_ci(self.set_metrics[arrhythmia][metric], percentage=False))
                else:
                    averages[metric] += self.set_metrics[arrhythmia][metric][0] 
                    row.append(format_ci(self.set_metrics[arrhythmia][metric], percentage=True))
            rows.append(row)

        row = ["Average"]
        for metric in metrics:
            if metric in ["tp", "fp", "fn"]:
                row.append(str(np.round(averages[metric]/len(arrhythmias)*10)/10))
            else:
                row.append(str(np.round(averages[metric]/len(arrhythmias)*1000)/10))
        rows.append(row)

        for column in columns:
            table.add_column(column)

        for row in rows:
            table.add_row(*row, style='bright_green')

        console = Console()
        console.print(table)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.confusion_matrix, cmap='Blues')
        ax.set_xticks(np.arange(len(arrhythmias)))
        ax.set_yticks(np.arange(len(arrhythmias)))
        ax.set_xticklabels(arrhythmias)
        ax.set_yticklabels(arrhythmias)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(arrhythmias)):
            for j in range(len(arrhythmias)):
                text = ax.text(j, i, int(self.confusion_matrix[i, j]), ha="center", va="center", color="black")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        modelnames = ("_".join([model.name for model in self.models]))
        plt.savefig(os.environ.get('benchmark_results') + f"/diagnosis/confusion_matrix_{modelnames}_{self.data.get_name()}.png")

        #print(self.to_latex(table))

    def create_json(self):
        
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_compressed = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data = {
            "date": date,
            "dataset": self.data.get_name(),
            "results": [
                {
                    "model": model.name,
                    "modelpaths": model.modelpaths if hasattr(model, "modelpaths") else [],
                    "results": []
                } for model in self.models
            ]
        }
        basefolder = os.environ.get('benchmark_results')
        modelnames = ("_".join([model.name for model in self.models]))
        filename = f"{basefolder}/diagnosis/set_level_diagnosis_{modelnames}_{self.data.get_name()}_[{date_compressed}].json"

        with open(filename, 'w') as f:
            json.dump(data, f)

        return filename

    def append_to_json(self, filename, model_name,predictions, rest={}):

        with open(filename, 'r') as f:
            data = json.load(f)

        if len(predictions) > 1:
            model_index = -1
            for i, res in enumerate(data["results"]):
                if res["model"] == model_name:
                    model_index = i
                    break
            if model_index != -1:
                data["results"][model_index]["results"] = predictions

                if type(rest) == list and len(rest) > 0:
                    for i in range(len(data["results"][model_index]["results"])):
                        for key, value in rest[i].items():
                            data["results"][model_index]["results"][i][key] = value
            else:
                print(f"Model {model_name} not found in results.")
        else:
            model_index = -1
            for i, res in enumerate(data["results"]):
                if res["model"] == model_name:
                    model_index = i
                    break
            data["results"][model_index]["results"].append(predictions[0])

            if len(rest) > 0:
                for obj in rest:
                    for key, value in obj.items():
                        data["results"][model_index]["results"][-1][key] = value

        with open(filename, 'w') as f:
            json.dump(data, f)

    def create_csv(self, filename, columns):
        
        f = open(filename, "w")
        csv = ""

        for col in columns:
            csv += col + ";"

        csv = csv[:-1] + "\n"
        f.write(csv)
        f.close()

    def append_to_csv(self, filename, data):

        f = open(filename, "a")
        csv = ""

        for col in data:
            #if col is array, convert to string
            if type(col) == list:
                col = '"' + str(col) + '"'
            csv += str(col) + ";"

        csv = csv[:-1] + "\n"
        f.write(csv)
        f.close

    def predictions_to_csv(self, filename):

        cols = self.set_predictions_df.columns
        f = open(filename, "w")
        csv = ""

        for col in cols:
            csv += col + ","
            
        csv = csv[:-1] +"\n"
        f.write(csv)

        nrow = len(self.set_predictions_df)
        for i in range(nrow):
            csv = ""
            for col in cols:
                csv += str(self.set_predictions_df[col][i]) + ","
            csv = csv[:-1] + "\n"
            f.write(csv)

        f.close()

        return filename



    


        