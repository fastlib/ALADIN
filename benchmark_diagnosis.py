import numpy as np
import os
import neurokit2 as nk 
import scipy.signal as sps
import torch
import pickle
import time
import asyncio
from tqdm.asyncio import tqdm

import atexit

import scipy.signal as sps
from scipy.signal import butter, filtfilt, medfilt, iirnotch
from scipy.interpolate import interp1d

from aladin import ALADIN
from aladin.utils.benchmark_utils import Data, Model, DiagnosticBenchmark, StanfordData, CINCData, ICENTIAData, ICENTIASAMPLEData, InternalData
from aladin.utils.helpers import resize_signal
from aladin.core import Record, RecordCollection
from aladin.backend.segmenter import UNetSegmenter
from aladin.logicengine.logic import LogicEngine
from aladin.selfreflection.reflection import Reflection
from aladin.utils.martinez_v2 import QRSdelineation
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, nnUNetWithClassificationPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join
import psutil
import pickle

import concurrent.futures

import argparse

import sys
sys.path.append('./benchmark')
from inference import Predictor, ECGFounderPredictor

def get_memory_usage_bytes() -> int:
    """Returns the RSS (resident set size) in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


class ECGFounderModel(ECGFounderPredictor):
    def __init__(self, model_path):
        super().__init__(model_path)

        if isinstance(self.model_path, str):
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        new_state_dict = {}
        if 'network_weights' in checkpoint:
            for k, value in checkpoint['network_weights'].items():
                #if k == "model.dense.weight" or k == "model.dense.bias":
                #    continue
                new_state_dict[k] = value

        elif 'state_dict' in checkpoint:
            for k, value in checkpoint['state_dict'].items():
                #if k == "model.dense.weight" or k == "model.dense.bias":
                #    continue
                new_state_dict["model."+k] = value

        if self.network.custom_head:
            self.network.load_state_dict(new_state_dict)

        self.network = self.network.to(self.device)
        self.network.eval()

    def get_unique_diagnoses(self, file):

        json_file = glob.glob(file + "_grp*.episodes.json")
        if len(json_file) == 0:
            return []

        json_file = json_file[0]
        with open(json_file, "r") as f:
            diagnose_data = json.load(f)

        diagnoses = diagnose_data["episodes"]
        diagnoses = [d["rhythm_name"] for d in diagnoses]
        unique_diagnoses = list(set(diagnoses))

        return unique_diagnoses

    def analyse_result_per_diagnosis(self, result, diagnosis):

        preds = result["pred"]
        trues = result["true"]

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(preds)):
            isdiagnosed = False
            for d in preds[i]:
                if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                    print("AVB")
                    isdiagnosed = True
                    break
                if d == diagnosis:
                    isdiagnosed = True
                    break
            isannotated = False
            for d in trues[i]:
                if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                    isannotated = True
                    break
                if diagnosis == "AFIB/AFL" and (d == "AFIB" or d == "AFL"):
                    isannotated = True
                    break
                if d == diagnosis:
                    isannotated = True
                    break

            if isdiagnosed and isannotated:
                tp += 1
            elif isdiagnosed and not isannotated:
                #print("False positive:", dat["case"][i], trues[i])
                fp += 1
            elif not isdiagnosed and isannotated:
                #print("False negative:", dat["case"][i])
                fn += 1

        if tp > 0:
            se = tp / (tp + fn)
            pp = tp / (tp + fp)
            f1 = 2 * (se * pp) / (se + pp)

            print(tp, fp, fn)
            print(diagnosis, "F1: ", f1, "(", tp, "/", fp, "/", fn, ")")
        else:
            print(diagnosis, "No true positives")

    def predict_on_folder(self, folder):

        data = FromFolderGenerator(source_folder=folder)
        res = {"pred": [], "true": []}

        # Predict
        self.network.eval()
        with torch.no_grad():
            for i, rec in enumerate(data):
                unique_gt_label = self.get_unique_diagnoses(os.path.join(folder, rec["id"]))
                inp = torch.from_numpy(rec["xs"]).to(self.device)
                inp = inp.unsqueeze(0)
                output = self.network(inp)[0]
                preds = torch.argmax(output, dim=1)
                preds = preds.cpu().numpy()
                unique_predictions = np.unique(preds)
                unique_predictions_label = [self.trainer.labels[up] for up in unique_predictions]
                res["pred"].append(unique_predictions_label)
                res["true"].append(unique_gt_label)

        self.analyse_result_per_diagnosis(res, "NSR") 
        self.analyse_result_per_diagnosis(res, "IVR")
        self.analyse_result_per_diagnosis(res, "AFIB/AFL")
        self.analyse_result_per_diagnosis(res, "TRIGEMINY")
        self.analyse_result_per_diagnosis(res, "BIGEMINY")
        self.analyse_result_per_diagnosis(res, "VT")
        self.analyse_result_per_diagnosis(res, "SVT")
        self.analyse_result_per_diagnosis(res, "JUNCTIONAL")
        self.analyse_result_per_diagnosis(res, "EAR")
        self.analyse_result_per_diagnosis(res, "AVB")
        self.analyse_result_per_diagnosis(res, "WENCKEBACH")
        self.analyse_result_per_diagnosis(res, "NOISE")

    def get_diagnoses_from_array(self, predictions, output_to_label, custom_head=True):
        diagnoses = []
        print("predictions: ", predictions)

        for i in range(len(predictions)):

            # if predictions[i] == 5 or predictions[i] == 32:
            #     diagnoses.append({"type":"AFIB/AFL", "onset": 0, "offset": 1})
            # elif predictions[i] == 1 or predictions[i] == 2 or predictions[i] == 3:
            #     diagnoses.append({"type":"NSR", "onset": 0, "offset": 1})
            # elif predictions[i] == 39:
            #     diagnoses.append({"type":"NOISE", "onset": 0, "offset": 1})
            # elif predictions[i]>1:
            #     diagnoses.append({"type":"OTHER", "onset": 0, "offset": 1})
            if custom_head:
                diagnoses.append({"type":self.trainer.labels[predictions[i]], "onset": 0, "offset": 1})
            else:
                if predictions[i] in output_to_label:
                    diagnoses.append({"type":output_to_label[predictions[i]], "onset": 0, "offset": 1})
                # if predictions[i] == 5:# or predictions[i] == 32:
                #     diagnoses.append({"type":"AFIB/AFL", "onset": 0, "offset": 1})
                # elif predictions[i] == 3:
                #     diagnoses.append({"type":"NSR", "onset": 0, "offset": 1})
                # elif predictions[i] == 39:
                #     diagnoses.append({"type":"NOISE", "onset": 0, "offset": 1})

        # if not custom_head and len(diagnoses) == 0:
        #     diagnoses.append({"type":"OTHER", "onset": 0, "offset": 1})
        print(diagnoses)

        return diagnoses

    def predict_on_array(self, sig, fs, output_to_label):

        # Remove power-line interference
        # fs = 300
        # b, a = iirnotch(50, 30, fs)
        # filtered_signal = np.zeros_like(sig)
        # filtered_signal = filtfilt(b, a, sig)

        # # Simple bandpass filter
        # b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
        # filtered_signal = filtfilt(b, a, filtered_signal)

        # # # Remove baseline wander
        # baseline = np.zeros_like(filtered_signal)
        # kernel_size = int(0.4 * fs) + 1
        # if kernel_size % 2 == 0:
        #     kernel_size += 1  # Ensure kernel size is odd
        # baseline = medfilt(filtered_signal, kernel_size=kernel_size)
        # filter_ecg = filtered_signal - baseline

        #filter_ecg = resize_signal(filter_ecg, int(filter_ecg.shape[-1]*(500/fs)))
        sig = resize_signal(sig, int(sig.shape[-1]*(500/fs)))
        fs = 500

        #highpass filter 1hz
        b, a = butter(4, 0.5/(fs/2), btype='highpass')
        ecg = filtfilt(b, a, sig)

        #lowpass filter 30hz
        b, a = butter(2, 50/(fs/2), btype='lowpass')
        ecg = filtfilt(b, a, ecg)

        #notch filter 50hz
        b, a = sps.iirnotch(50, 30, fs)
        ecg = filtfilt(b, a, ecg)
        #notch filter 60hz
        b, a = sps.iirnotch(60, 30, fs)
        ecg = filtfilt(b, a, ecg)

        baseline = np.zeros_like(ecg)
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline = medfilt(ecg, kernel_size=kernel_size)
        ecg = ecg - baseline

        # ecg = ecg - np.mean(ecg)
        # ecg = (ecg) / (np.std(ecg))

        size = 1000
        ecg = ecg[:int(len(ecg)/size)*size]
        diagnoses = []
        logits = []
        with torch.no_grad():
            inp = torch.from_numpy(ecg.copy()).float().to(self.device)
            inp = inp[None,None,:]
            st = 0

            unique_predictions = []
            while st < inp.shape[-1]:
                en = min(st + 5000, inp.shape[-1])
                signal = inp[:,:,st:en]
                #pad to 5000 samples with zero
                if signal.shape[-1] < 5000:
                    padding = torch.zeros((1, 1, 5000 - signal.shape[-1]), device=self.device)
                    signal = torch.cat([signal, padding], dim=-1)
                signal -= torch.mean(signal)
                signal /= torch.std(signal)
                output = self.network(signal)[0]
                output = torch.sigmoid(output)
                output = output.cpu().numpy()
                logits.append(output.reshape(1, -1))
                output = np.array([o > 0.5 for k, o in enumerate(output)])
                unique_prediction = np.where(output)[0]
                unique_predictions.extend(unique_prediction)
                st += 5000

            logits = np.concatenate(logits, axis=0)
            unique_predictions = list(set(unique_predictions))
            print(unique_predictions)
            return self.get_diagnoses_from_array(unique_predictions, output_to_label=output_to_label, custom_head=self.network.custom_head), logits

class HannunModel(Predictor):
    def __init__(self, model_path):
        super().__init__(model_path)

        if isinstance(self.model_path, str):
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            new_state_dict[k] = value

        self.network.load_state_dict(new_state_dict)
        self.network = self.network.to(self.device)
        self.network.eval()

    def get_unique_diagnoses(self, file):

        json_file = glob.glob(file + "_grp*.episodes.json")
        if len(json_file) == 0:
            return []

        json_file = json_file[0]
        with open(json_file, "r") as f:
            diagnose_data = json.load(f)

        diagnoses = diagnose_data["episodes"]
        diagnoses = [d["rhythm_name"] for d in diagnoses]
        unique_diagnoses = list(set(diagnoses))

        return unique_diagnoses

    def analyse_result_per_diagnosis(self, result, diagnosis):

        preds = result["pred"]
        trues = result["true"]

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(preds)):
            isdiagnosed = False
            for d in preds[i]:
                if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                    print("AVB")
                    isdiagnosed = True
                    break
                if d == diagnosis:
                    isdiagnosed = True
                    break
            isannotated = False
            for d in trues[i]:
                if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                    isannotated = True
                    break
                if diagnosis == "AFIB/AFL" and (d == "AFIB" or d == "AFL"):
                    isannotated = True
                    break
                if d == diagnosis:
                    isannotated = True
                    break

            if isdiagnosed and isannotated:
                tp += 1
            elif isdiagnosed and not isannotated:
                #print("False positive:", dat["case"][i], trues[i])
                fp += 1
            elif not isdiagnosed and isannotated:
                #print("False negative:", dat["case"][i])
                fn += 1

        if tp > 0:
            se = tp / (tp + fn)
            pp = tp / (tp + fp)
            f1 = 2 * (se * pp) / (se + pp)

            print(tp, fp, fn)
            print(diagnosis, "F1: ", f1, "(", tp, "/", fp, "/", fn, ")")
        else:
            print(diagnosis, "No true positives")

    def predict_on_folder(self, folder):

        data = FromFolderGenerator(source_folder=folder)
        res = {"pred": [], "true": []}

        # Predict
        self.network.eval()
        with torch.no_grad():
            for i, rec in enumerate(data):
                unique_gt_label = self.get_unique_diagnoses(os.path.join(folder, rec["id"]))
                inp = torch.from_numpy(rec["xs"]).to(self.device)
                inp = inp.unsqueeze(0)
                output = self.network(inp)[0]
                preds = torch.argmax(output, dim=1)
                preds = preds.cpu().numpy()
                unique_predictions = np.unique(preds)
                unique_predictions_label = [self.trainer.labels[up] for up in unique_predictions]
                res["pred"].append(unique_predictions_label)
                res["true"].append(unique_gt_label)

        self.analyse_result_per_diagnosis(res, "NSR") 
        self.analyse_result_per_diagnosis(res, "IVR")
        self.analyse_result_per_diagnosis(res, "AFIB/AFL")
        self.analyse_result_per_diagnosis(res, "TRIGEMINY")
        self.analyse_result_per_diagnosis(res, "BIGEMINY")
        self.analyse_result_per_diagnosis(res, "VT")
        self.analyse_result_per_diagnosis(res, "SVT")
        self.analyse_result_per_diagnosis(res, "JUNCTIONAL")
        self.analyse_result_per_diagnosis(res, "EAR")
        self.analyse_result_per_diagnosis(res, "AVB")
        self.analyse_result_per_diagnosis(res, "WENCKEBACH")
        self.analyse_result_per_diagnosis(res, "NOISE")

    def get_diagnoses_from_array(self, predictions):
        diagnoses = []
        cur_episode = -1
        cur_episode_start = 0
        print("predictions: ", predictions)
        for i in range(len(predictions)):
            if predictions[i] != cur_episode:
                if cur_episode != -1:
                    diagnoses.append({"type":self.trainer.labels[cur_episode], "onset": (cur_episode_start*256)/200, "offset": (i*256)/200})
                cur_episode = predictions[i]
                cur_episode_start = i

        diagnoses.append({"type":self.trainer.labels[cur_episode], "onset": (cur_episode_start*256)/200, "offset": (len(predictions)*256)/200})
        return diagnoses

    def predict_on_array(self, sig, fs):
        
        sig = resize_signal(sig, int(sig.shape[-1]*(200/fs)))
        
        sig = sig - np.mean(sig)
        sig = (sig) / (np.std(sig))

        size = 256
        sig = sig[:int(len(sig)/size)*size]

        diagnoses = []
        with torch.no_grad():
            inp = torch.from_numpy(sig).float().to(self.device)
            inp = inp[None,None,:]
            output = self.network(inp)[0]
            logits = output.cpu().numpy()
            output = output.argmax(1)
            output = output.cpu().numpy()

            return self.get_diagnoses_from_array(output), logits


class ALADINModel(Model):
    def __init__(self, modelpaths=[]):
        super().__init__()
        self.output_binary = True
        self.name = "ALADIN"
        self.save_output = True
        self.modelpaths = modelpaths if len(modelpaths) > 0 else ["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"]

        self.aladin = ALADIN(modelpaths=self.modelpaths)

    def predict(self, sig, fs, meta=None, preprocess=False): 

        recordname = str(meta["record"]).replace("/", "_")

        record = Record(sig, fs, "bench", recordname)
        self.aladin.analyse(record)

        diagnoses = []
        for i in range(len(record.diagnosis)):
            diagnoses.append({"type":record.diagnosis[i].name, "onset": record.diagnosis[i].onset, "offset": record.diagnosis[i].offset})
        return diagnoses

    def calculate_batchsizes(self, data):
        ram = psutil.virtual_memory().available
        vram = torch.cuda.get_device_properties(0).total_memory
        #vram = (1024*1024*1024) * 1

        length = data[0]["signal"].shape[-1]
        ram_per_record = 6*length*4*5 #6 channels, 4 bytes per float, 5 for preprocessing and storing
        max_cpu_records_per_batch = int(ram / ram_per_record)
        print("Max records per batch (RAM): ", max_cpu_records_per_batch)
        self.cpu_batchsize = max_cpu_records_per_batch

        duration = length / data[0]["fs"]
        short_chunks_per_record = int(np.ceil(duration / 9)) if duration > 10 else 1
        print(short_chunks_per_record, "chunks per record")
        results_vram_per_chunk = (2048*6*4*5) # 2048 samples, 64 features, 2 bytes per float
        ratio_results_vram = 0.5

        max_gpu_records_per_batch = int((vram*ratio_results_vram) / ((results_vram_per_chunk) * short_chunks_per_record))
        print("Max records per batch (VRAM): ", max_gpu_records_per_batch)
        self.gpu_batchsize = max_gpu_records_per_batch

        large_chunks_per_record = int(np.ceil(duration / 27)) if duration > 30 else 1
        results_vram_per_chunk = (6144*6*4*5) # 6144 samples, 64 features, 2 bytes per float
        max_gpu_records_2_per_batch = int((vram*ratio_results_vram) / ((results_vram_per_chunk) * large_chunks_per_record))
        print("Max records per batch (VRAM, 2 per record): ", max_gpu_records_2_per_batch)

        self.cpu_batchsize = min(self.cpu_batchsize, max_gpu_records_per_batch, max_gpu_records_2_per_batch)
        print("Final CPU batch size: ", self.cpu_batchsize)

    def process_diagnoses(self, record: Record):
        diagnoses = []
        raw_diagnoses = []
        for j in range(len(record.diagnosis)):
            diagnoses.append({"type":record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})
            raw_diagnoses.append({"type":record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})

        for j in range(len(record.subdiagnosis)):
            raw_diagnoses.append({"type":record.subdiagnosis[j].name})

        return diagnoses, raw_diagnoses

    def preprocess_batch_on_cpu(self, recs, batch_index):
        print("Preprocessing on CPU start: ", batch_index, flush=True)
        records = []
        ids = []
        for record in recs:
            records.append(Record(record["signal"], record["fs"], "bench", record["record"]))
            ids.append(record["record"])

        # collection = RecordCollection(records)
        # collection.preprocess()
        print("Preprocessing on CPU done: ", batch_index, flush=True)

        return records

    def segment_on_gpu(self, recs):
        self.aladin.segmenter.batch(recs)

    def process_reflection_and_diagnosis_on_cpu(self, recs, batch_index):
        print("Reflection on CPU: ", batch_index, flush=True)
        self.aladin.reflection.batch(recs)
        
        print("Reflection on CPU done: ", batch_index, flush=True)
        print("Diagnosing records on CPU: ", batch_index, flush=True)
        for record in tqdm(recs):
            #reflection = Reflection(debug = False)
            #reflection.reflect(record)
            logic = LogicEngine(debug=False)
            logic.diagnose(record)

        print("Reflection and diagnosis on CPU done: ", batch_index, flush=True)
        return recs

    def analyze_batch(self, recs, batch_index):
        print("Segmenting on GPU: ", batch_index, flush=True)
        # Segment on GPU
        self.segment_on_gpu(recs)
        print("Segmenting on GPU done: ", batch_index, flush=True)

        return recs

    def predict_batch(self, data):
        
        self.calculate_batchsizes(data)
        #self.cpu_batchsize = 2
        num_batches = len(data) // self.cpu_batchsize + (1 if len(data) % self.cpu_batchsize > 0 else 0)
        out = []


        for idx, batch in enumerate(data.batch(self.cpu_batchsize)):
            if idx > 5:
                data.cleanup()
                exit(0)

            print(len(batch), "records in batch ", idx)
            st = time.time()

            processed_batch = self.preprocess_batch_on_cpu(batch, idx)
            self.analyze_batch(processed_batch, idx)

            self.process_reflection_and_diagnosis_on_cpu(processed_batch, idx)

            for i, record in enumerate(processed_batch):
                #print("Record: ", record.recordname)
                diagnoses, raw_diagnoses = self.process_diagnoses(record)
                print(record.recordname, diagnoses)
                out.append({"record": record.recordname, "diagnoses": diagnoses, "raw": raw_diagnoses, "arrhythmia": batch[i]["arrhythmia"]})
                data.upload_record(record)

            data.set_as_finished([record.recordname for record in processed_batch])
            print("Batch ", idx, " processed in ", time.time() - st, " seconds", flush=True)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        #     future_to_batch = {}
        #     prev_processed_batch = None

        #     # CPU thread preps next batch, GPU processes current batch
        #     for idx, batch in enumerate(data.batch(self.cpu_batchsize)):
        #         print(len(batch), "records in batch ", idx)

        #         batch = self.preprocess_batch_on_cpu(batch, idx)
        #         # If there's a previously processed batch, start processing reflection/diagnosis on CPU
        #         if prev_processed_batch:
        #             # Handle reflection and diagnosis for the previous batch (CPU)
        #             future = executor.submit(self.process_reflection_and_diagnosis_on_cpu, prev_processed_batch, idx-1)
        #             future_to_batch[future] = prev_processed_batch
        #             #future_to_batch[executor.submit(self.process_reflection_and_diagnosis_on_cpu, prev_processed_batch)] = prev_processed_batch
                
        #         # Start the GPU segmentation for the current batch
        #         prev_processed_batch = self.analyze_batch(batch, idx)  # GPU handles segmentation

        #         # if idx == num_batches-1:  # Check if this is the last batch
        #         #     #print(f"Last batch: Processing reflection and diagnosis for batch {prev_processed_batch}")
        #         #     future = executor.submit(self.process_reflection_and_diagnosis_on_cpu, prev_processed_batch, idx)
        #         #     future_to_batch[future] = prev_processed_batch

        #         # Wait for any CPU tasks (reflection/diagnosis) to finish and process the results
        #         done, not_done = concurrent.futures.wait(future_to_batch.keys(), timeout=None, return_when=concurrent.futures.ALL_COMPLETED)

        #         for future in done:
        #             processed_records = future.result()
                    
        #             # Process results on CPU
        #             for i, record in enumerate(processed_records):
        #                 #print("Record: ", record.recordname)
        #                 diagnoses, raw_diagnoses = self.process_diagnoses(record)
        #                 out.append({"record": record.recordname, "diagnoses": diagnoses, "raw": raw_diagnoses})
        #                 data.upload_record(record)

        #             data.set_as_finished([record.recordname for record in processed_records])

        #             # Remove completed futures
        #             future_to_batch.pop(future)

        return out


        # out = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #     future_to_batch = {}

        #     for batch in tqdm(data.batch(self.cpu_batchsize)):
        #         print(len(batch), "records in batch")

        #         # Start processing the next batch while the current batch is being processed
        #         future = executor.submit(self.analyze_batch, batch)
        #         future_to_batch[future] = batch

        #         # Check if any tasks are finished and process them
        #         done, not_done = concurrent.futures.wait(future_to_batch.keys(), timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED)

        #         for future in done:
        #             processed_records = future.result()

        #             # Process results on CPU
        #             for i, record in enumerate(processed_records):
        #                 #print("Record: ", record.recordname)
        #                 diagnoses = self.process_diagnoses(record)
        #                 out.append({"record": list(data.keys())[i], "diagnoses": diagnoses, "raw": record.diagnosis + record.subdiagnosis})

        #             # Remove completed futures
        #             future_to_batch.pop(future)

        # return out

class ALADINModelForCinc(ALADINModel):
    def __init__(self, modelpaths=[]):
        super().__init__(modelpaths=modelpaths)
    
    def predict(self, sig, fs, meta=None, preprocess=False):

        recordname = str(meta["record"]).replace("/", "_")
        db = "bench"
        
        record = Record(sig, None, None, fs, db, recordname)
        record = self.aladin.analyse(record)

        totallength = len(record.ecg)
        diagnoses = []

        # mapper = {
        #     "EAR": "NSR",
        #     "JUNCTIONAL": "NSR",
        # }
       
        for d in record.diagnoses:
            type = mapper[d["type"]] if d["type"] in mapper else d["type"]
            duration = (d["offset"]-d["onset"])
            diagnoses.append({"type":type, "duration":duration / totallength})

        #sort diagnoses by duration
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
        per_type = {}
        for d in diagnoses:
            if d["type"] in per_type:
                per_type[d["type"]] += d["duration"]
            else:
                per_type[d["type"]] = d["duration"]

        diagnoses = [{"type":k, "duration":v} for k,v in per_type.items()]
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
    
        subdiagnoses = [d["type"] for d in record.subdiagnoses]
        diagnose = ""

        #return diagnoses + subdiagnoses
        
        if len(diagnoses) == 0:
            diagnose = "O"
        elif diagnoses[0]["type"] == "NOISE":
            diagnose = "~"
        elif diagnoses[0]["type"] == "AFIB":
            diagnose = "A"
        elif np.all([d["type"] == "NSR" or d["type"] == "NOISE" for d in diagnoses]) \
            and not "TACHYCARDIA" in subdiagnoses \
            and not "BRADYCARDIA" in subdiagnoses \
            and not "PVC" in subdiagnoses \
            and not "PAC" in subdiagnoses \
            and not "IVB" in subdiagnoses:

            diagnose = "N"
        else:
            diagnose = "O"

        res = [{"type":diagnose}]

        return res

    def process_diagnoses(self, record: Record):
        totallength = len(record.ecg)
        diagnoses = []

        mapper = {
            "EAR": "NSR",
            "JUNCTIONAL": "NSR",
        }

        for j in range(len(record.diagnosis)):
            type = mapper[record.diagnosis[j].name] if record.diagnosis[j].name in mapper else record.diagnosis[j].name
            duration = (record.diagnosis[j].offset-record.diagnosis[j].onset)
            diagnoses.append({"type":type, "duration":duration / totallength})

        #sort diagnoses by duration
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
        per_type = {}
        for d in diagnoses:
            if d["type"] in per_type:
                per_type[d["type"]] += d["duration"]
            else:
                per_type[d["type"]] = d["duration"]

        diagnoses = [{"type":k, "duration":v} for k,v in per_type.items()]
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
    
        subdiagnoses = [record.subdiagnosis[j].name for j in range(len(record.subdiagnosis))]
        diagnose = ""

        mostnoise = diagnoses[0]["type"] == "NOISE" and diagnoses[0]["duration"] > 0.5 if len(diagnoses) > 0 else False
        mostafib = diagnoses[0]["type"] == "AFIB" if len(diagnoses) > 0 else False
        haspac = "PAC" in subdiagnoses
        haspvc = "PVC" in subdiagnoses
        hasivb = "IVB" in subdiagnoses
        hastachycardia = "TACHYCARDIA" in subdiagnoses
        hasbradycardia = "BRADYCARDIA" in subdiagnoses
        onlynsr = np.all([d["type"] == "NSR" or d["type"] == "NOISE" for d in diagnoses])
        abnormality = haspac or haspvc or hasivb or hastachycardia or hasbradycardia
        #return diagnoses + subdiagnoses
        
        if mostnoise:
            diagnose = "~"
        elif abnormality:
            diagnose = "O"
        elif onlynsr:
            diagnose = "N"
        elif mostafib:
            diagnose = "A"
        else:
            diagnose = "O"

        return [{"type":diagnose, "onset":0, "offset":1}], diagnoses + subdiagnoses

class ALADINModelForICENTIA(ALADINModel):
    def __init__(self, modelpaths=[]):
        super().__init__(modelpaths=modelpaths)

    def process_diagnoses(self, record: Record):

        filtered_diagnoses = []
        diagnoses = record.diagnosis
        raw_diagnoses = []

        for diagnosis in diagnoses:
            onset = diagnosis.onset
            offset = diagnosis.offset
            duration = (offset - onset) / 250  # Convert to seconds

            if diagnosis.name == "AFIB" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "AFIB", "onset": onset, "offset": offset})
            elif diagnosis.name == "VT" and duration >= 10:
                filtered_diagnoses.append({"type": "VT>10s", "onset": onset, "offset": offset})
            elif diagnosis.name == "VT" and duration < 10:
                filtered_diagnoses.append({"type": "VT<10s", "onset": onset, "offset": offset})
            elif diagnosis.name == "SVT" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "SVT>30s", "onset": onset, "offset": offset})
            elif diagnosis.name == "IVR":
                filtered_diagnoses.append({"type": "IVR", "onset": onset, "offset": offset})
            elif diagnosis.name == "TRIGEMINY":
                filtered_diagnoses.append({"type": "TRIGEMINY", "onset": onset, "offset": offset})
            elif diagnosis.name == "BIGEMINY":
                filtered_diagnoses.append({"type": "BIGEMINY", "onset": onset, "offset": offset})
            elif diagnosis.name == "NOISE" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "NOISE", "onset": onset, "offset": offset})
            elif diagnosis.name == "NSR" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "NSR", "onset": onset, "offset": offset})
            elif diagnosis.name == "WENCKEBACH" or \
                    diagnosis.name == "AVB" or \
                    diagnosis.name == "IVR" or \
                    diagnosis.name == "AVB_TYPE2" or \
                    diagnosis.name == "SUDDEN_BRADY":
                filtered_diagnoses.append({"type": diagnosis.name, "onset": onset, "offset": offset})


        for j in range(len(record.diagnosis)):
            raw_diagnoses.append({"type": record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})

        for j in range(len(record.subdiagnosis)):
            raw_diagnoses.append({"type": record.subdiagnosis[j].name, "onset": record.subdiagnosis[j].onset, "offset": record.subdiagnosis[j].offset})

        return filtered_diagnoses, raw_diagnoses

        # diagnoses = []
        # rawdiagnoses = []
        # for j in range(len(record.diagnosis)):
        #     append = True
        #     if record.diagnosis[j].name == "AFIB":
        #         if record.diagnosis[j].offset - record.diagnosis[j].onset < 25*record.fs:
        #             append = False
                
        #     if record.diagnosis[j].name == "NOISE":
        #         if record.diagnosis[j].offset - record.diagnosis[j].onset < 25*record.fs:
        #             append = False
            
        #     if record.diagnosis[j].name == "SVT":
        #         if record.diagnosis[j].offset - record.diagnosis[j].onset < 25*record.fs:
        #             append = False

        #     if append:
        #         diagnoses.append({"type":record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})

        #     rawdiagnoses.append({"type":record.diagnosis[j].name, "onset": record.diagnosis[j].onset, "offset": record.diagnosis[j].offset})

        # for j in range(len(record.subdiagnosis)):
        #     rawdiagnoses.append({"type":record.subdiagnosis[j].name, "onset": record.subdiagnosis[j].onset, "offset": record.subdiagnosis[j].offset})

        # onlynsr = np.all([d["type"] == "NSR" or d["type"] == "AVB_TYPE1" or d["type"] == "NOISE" for d in diagnoses])
        # print("Only NSR: ", onlynsr)
        # if not onlynsr:
        #     diagnoses = [d for d in diagnoses if d["type"] != "NSR" and d["type"] != "AVB_TYPE1" and d["type"] != "NOISE"]
        #     rawdiagnoses = [d for d in rawdiagnoses if d["type"] != "NSR" and d["type"] != "AVB_TYPE1" and d["type"] != "NOISE"]
        #     print(diagnoses)


        # return diagnoses, rawdiagnoses

    def predict(self, sig, fs, meta=None, preprocess=False): 
        
        mem_bytes = get_memory_usage_bytes()
        print(f"Memory usage: {mem_bytes / (1024**2):.2f} MB")

        recordname = str(meta["record"]).replace("/", "_")
        db = "bench"
        
        record = Record(sig, fs, db, recordname)

        self.aladin.analyse(record)

        diagnoses = []
        for i in range(len(record.diagnosis)):
            diagnoses.append(self.process_diagnoses(record.diagnosis[i]))

        diagnoses = [d for d in diagnoses if d is not False]

        return diagnoses

    def predict_batch(self, data):
        
        self.calculate_batchsizes(data)
        #self.cpu_batchsize = 2
        num_batches = len(data) // self.cpu_batchsize + (1 if len(data) % self.cpu_batchsize > 0 else 0)
        out = []


        for idx, batch in enumerate(data.batch(self.cpu_batchsize)):
            print(len(batch), "records in batch ", idx)
            st = time.time()

            processed_batch = self.preprocess_batch_on_cpu(batch, idx)
            self.analyze_batch(processed_batch, idx)

            self.process_reflection_and_diagnosis_on_cpu(processed_batch, idx)

            for i, record in enumerate(processed_batch):
                #print("Record: ", record.recordname)
                diagnoses, raw_diagnoses = self.process_diagnoses(record)
                print(record.recordname, diagnoses)
                out.append({"record": record.recordname, "diagnoses": diagnoses, "raw": raw_diagnoses, "arrhythmia": batch[i]["arrhythmia"], "seen_elsewhere": batch[i]["seen_elsewhere"]})
                data.upload_record(record)

            data.set_as_finished([record.recordname for record in processed_batch])
            print("Batch ", idx, " processed in ", time.time() - st, " seconds", flush=True)

            if idx>10:
                print("Stopping after 10 batches for testing purposes")
                break

        return out

    def process_reflection_and_diagnosis_on_cpu(self, recs, batch_index):
        print("Processing reflection and diagnosis on CPU start: ", batch_index, flush=True)
        self.aladin.reflection.batch(recs)
        
        print("Reflection on CPU done: ", batch_index, flush=True)
        print("Diagnosing records on CPU: ", batch_index, flush=True)
        for record in tqdm(recs):
            logic = LogicEngine(debug=False, customarrhythmia={"PAC_PVC": False})
            logic.diagnose(record)

        print("Processing reflection and diagnosis on CPU done: ", batch_index, flush=True)
        return recs



class ECGFounderWrapper(Model):
    def __init__(self, modelpaths):
        super().__init__()
        self.name = "ECGFounder"
        self.save_output = True
        self.savelogits = True
        self.modelpaths = modelpaths if len(modelpaths) > 0 else ["./benchmark/weights/ECGFounderNet_checkpoint_best.pth"]
        self.model = ECGFounderModel(self.modelpaths[0])
        self.output_to_label = {
            3: "NSR",
            5: "AFIB",
            32: "AFL",
            38: "EAR",
            49: "JUNCTIONAL",
            79: "NSR", #AVB1
            81: "NSR", #AVB1
            91: "BIGEMINY",
            93: "SVT",
            98: "VT",
            #118: "WENCKEBACH",
            121: "SUDDEN_BRADY",
            126: "JUNCTIONAL",
            133: "IVR",
            #134: "AVB_TYPE2",
            145: "SVT",
            146: "WENCKEBACH",
            147: "AVB_TYPE2",
            148: "SUDDEN_BRADY"
        }

    def predict(self, sig, fs, meta=None, preprocess=False):
        preds, logits = self.model.predict_on_array(sig, fs, self.output_to_label)
        logits = logits[:,[k for k in list(self.output_to_label.keys())]]

        return {"predictions": preds, "logits": logits}, {}

class ECGFounderWrapperForCinc(Model):
    def __init__(self, modelpaths):
        super().__init__()
        self.name = "ECGFounder"
        self.save_output = True
        self.savelogits = True
        self.modelpaths = modelpaths if len(modelpaths) > 0 else ['./benchmark/weights/ECGFounderNet_checkpoint_best.pth']
        #self.modelpaths = modelpaths if len(modelpaths) > 0 else ['./benchmark/weights/1_lead_ECGFounder.pth']
        self.model = ECGFounderModel(self.modelpaths[0])
        self.output_to_label = {
            0: "O", #ABNORMAL ECG
            6: "O",  # SINUS TACHYCARDIA
            9: "O",  # PREMATURE VENTRICULAR COMPLEXES
            # #11: "O", # RIGHT BUNDLE BRANCH BLOCK
            16: "O", # PREMATURE ATRIAL COMPLEXES
            19: "O", # PREMATURE SUPRAVENTRICULAR COMPLEXES
            # #20: "O", # LEFT BUNDLE BRANCH BLOCK
            # #32: "O", # ATRIAL FLUTTER
            33: "O", # MARKED SINUS BRADYCARDIA
            #38: "O", # ECTOPIC ATRIAL RHYTHM
            #49: "O", # JUNCTIONAL RHYTHM
            #51: "O", # ABERRANT CONDUCTION
            #58: "O", # WIDE QRS RHYTHM
            #59: "O", # WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES
            #65: "O", # BIFASCICULAR BLOCK
            #69: "O", # PREMATURE ECTOPIC COMPLEXES
            79: "O", # WITH 1ST DEGREE AV BLOCK
            81: "O", # WITH PROLONGED AV CONDUCTION
            #83: "O", # WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY
            #90: "O", # PREMATURE VENTRICULAR AND FUSION COMPLEXES
            91: "O", # IN A PATTERN OF BIGEMINY
            93: "O", # SUPRAVENTRICULAR TACHYCARDIA
            98: "O", # VENTRICULAR TACHYCARDIA
            112: "O", # NONSPECIFIC INTRAVENTRICULAR BLOCK
            121: "O", # WITH COMPLETE HEART BLOCK
            #126: "O", # JUNCTIONAL BRADYCARDIA
            133: "O", # IDIOVENTRICULAR RHYTHM
            145: "O", # SUPRAVENTRICULAR COMPLEXES
            146: "O", # WITH 2ND DEGREE AV BLOCK MOBITZ I
            147: "O", # WITH 2:1 AV CONDUCTION
            148: "O", # WITH AV DISSOCIATION
            #149: "O", # MULTIFOCAL ATRIAL TACHYCARDIA
            1: "N",  # SINUS RHYTHM
            5: "A",  # ATRIAL FIBRILLATION
            #39: "~"
        }
        
    def predict(self, sig, fs, meta=None, preprocess=False):

        recorddiagnoses, logits = self.model.predict_on_array(sig, fs, self.output_to_label)
        diagnoses = []
        totallength = len(sig)

        mapper = {
            "EAR": "NSR",
            "JUNCTIONAL": "NSR",
        }

        for d in recorddiagnoses:
            type = mapper[d["type"]] if d["type"] in mapper else d["type"]
            duration = (d["offset"]-d["onset"])
            diagnoses.append(type)

        print("Record diagnoses: ", diagnoses)
        
        if len(diagnoses) == 0:
            diagnose = "N"
        elif "NOISE" in diagnoses and len(diagnoses) == 1:
            diagnose = "~"
        elif "AFIB/AFL" in diagnoses:
            diagnose = "A"
        elif ("NSR" in diagnoses and len(diagnoses) == 1) or ("NSR" in diagnoses and "NOISE" in diagnoses and len(diagnoses) == 2):
            diagnose = "N"
        else:
            diagnose = "O"

        res = [{"type":diagnose}]

        logits = logits[:,[k for k in list(self.output_to_label.keys())]]

        return {"predictions": res, "logits": logits}, {}

class HannunWrapper(Model):
    def __init__(self, modelpaths):
        super().__init__()
        self.name = "Hannun"
        self.savelogits = False
        self.save_output = True
        self.modelpaths = modelpaths if len(modelpaths) > 0 else ['./benchmark/weights/HannunNet_checkpoint_best.pth']
        self.model = HannunModel(self.modelpaths[0])
        self.labels = ["NSR","AFIB/AFL","AVB_TYPE2","SUDDEN_BRADY","BIGEMINY","TRIGEMINY","EAR","IVR","JUNCTIONAL","SVT","VT","WENCKEBACH","NOISE"]
        self.output_to_label = {
            0: "NSR",
            1: "AFIB/AFL",
            2: "AVB_TYPE2",
            3: "SUDDEN_BRADY",
            4: "BIGEMINY",
            5: "TRIGEMINY",
            6: "EAR",
            7: "IVR",
            8: "JUNCTIONAL",
            9: "SVT",
            10: "VT",
            11: "WENCKEBACH",
            12: "NOISE"
        }

    def predict(self, sig, fs, meta=None, preprocess=False):
        preds, logits = self.model.predict_on_array(sig, fs)
        logits = logits[:,[k for k in list(self.output_to_label.keys())]]
        print(logits)

        return {"predictions": preds, "logits": logits}, {}

class HannunWrapperForCinc(Model):
    def __init__(self, modelpaths):
        super().__init__()
        self.name = "Hannun"
        self.save_output = True
        self.modelpaths = modelpaths if len(modelpaths) > 0 else ['./benchmark/weights/HannunNet_checkpoint_best.pth']
        self.model = HannunModel(self.modelpaths[0])

    def predict(self, sig, fs, meta=None, preprocess=False):

        recorddiagnoses = self.model.predict_on_array(sig, fs)
        diagnoses = []
        totallength = len(sig)/fs

        mapper = {
            "EAR": "NSR",
            "JUNCTIONAL": "NSR",
        }

        for d in recorddiagnoses:
            type = mapper[d["type"]] if d["type"] in mapper else d["type"]
            duration = (d["offset"]-d["onset"])
            diagnoses.append({"type":type, "duration":duration / totallength})

        #sort diagnoses by duration
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
        per_type = {}
        for d in diagnoses:
            if d["type"] in per_type:
                per_type[d["type"]] += d["duration"]
            else:
                per_type[d["type"]] = d["duration"]

        diagnoses = [{"type":k, "duration":v} for k,v in per_type.items()]
        diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
    
        diagnose = ""

        mostnoise = diagnoses[0]["type"] == "NOISE" and diagnoses[0]["duration"] > 0.5 if len(diagnoses) > 0 else False
        mostafib = diagnoses[0]["type"] == "AFIB" if len(diagnoses) > 0 else False
        onlynsr = np.all([d["type"] == "NSR" or d["type"] == "NOISE" for d in diagnoses])
        #return diagnoses + subdiagnoses
        
        if mostnoise:
            diagnose = "~"
        elif onlynsr:
            diagnose = "N"
        elif mostafib:
            diagnose = "A"
        else:
            diagnose = "O"

        return [{"type":diagnose}], {}
        
        # mapper = {
        #     "EAR": "NSR",
        #     "JUNCTIONAL": "NSR",
        # }

        # #print("Record diagnoses: ", recorddiagnoses)
       
        # for d in recorddiagnoses:
        #     type = mapper[d["type"]] if d["type"] in mapper else d["type"]
        #     duration = (d["offset"]-d["onset"])
        #     diagnoses.append({"type":type, "duration":duration / totallength})

        # #print("Diagnoses: ", diagnoses)

        # #sort diagnoses by duration
        # diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)
        # per_type = {}
        # for d in diagnoses:
        #     if d["type"] in per_type:
        #         per_type[d["type"]] += d["duration"]
        #     else:
        #         per_type[d["type"]] = d["duration"]

        # diagnoses = [{"type":k, "duration":v} for k,v in per_type.items()]
        # diagnoses = sorted(diagnoses, key=lambda x: x["duration"], reverse=True)

        
        # if len(diagnoses) == 0:
        #     diagnose = "N"
        # elif diagnoses[0]["type"] == "NOISE" and diagnoses[0]["duration"] > 0.5:
        #     diagnose = "~"
        # elif diagnoses[0]["type"] == "AFIB/AFL":
        #     diagnose = "A"
        # elif np.all([d["type"] == "NSR" or d["type"] == "NOISE" for d in diagnoses]):
        #     diagnose = "N"
        # else:
        #     diagnose = "O"

        # res = [{"type":diagnose}]

        # return res
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--method', type=str, help='ALADIN or Martinez', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset used to benchmark (STANFORD, LUDB, VAL)', required=True)
    #add overwrite flag
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    #modelpaths is a list of strings
    parser.add_argument('--modelpaths', nargs='+', help='Paths to the models used in the benchmark', required=False, default=[])

    args = parser.parse_args()
    method = args.method
    dataset = args.dataset
    modelpaths = args.modelpaths
    overwrite = args.overwrite

    resultsfolder = os.environ.get('benchmark_results')
    datafolder = os.environ.get('benchmark_data')

    if resultsfolder is None:
        print("ERROR: Please set the environment variable benchmark_results to the folder where you want to save the results")
        exit(1)
    if datafolder is None:
        print("ERROR: Please set the environment variable benchmark_data to the folder where the data is located")
        exit(1)

    print("Method: ", method)
    print("Dataset: ", dataset)

    if dataset == "STANFORD":
        data = StanfordData("STANFORD", asynchronous=True)

        if method == "ALADIN":
            model = ALADINModel(modelpaths=modelpaths)
        elif method == "ECGFounder":
            data.class_mapper["TRIGEMINY"] = "NSR" #account for the fact that ECGFounder cannot detect trigeminy
            model = ECGFounderWrapper(modelpaths=modelpaths)
        elif method == "Hannun":
            model = HannunWrapper(modelpaths=modelpaths)

    elif dataset == "INTERNAL":
        data = InternalData("INTERNAL", asynchronous=True)

        if method == "ALADIN":
            model = ALADINModel(modelpaths=modelpaths)

    elif dataset == "CINC":
        data = CINCData("CINC", asynchronous=True)

        if method == 'ALADIN':
            model = ALADINModelForCinc(modelpaths=modelpaths)
        elif method == "ECGFounder":
            model = ECGFounderWrapperForCinc(modelpaths=modelpaths)
        elif method == "Hannun":
            model = HannunWrapperForCinc(modelpaths=modelpaths)

    elif dataset == "ICENTIA":
        data = ICENTIAData("ICENTIA", asynchronous=True)

        atexit.register(data.cleanup)

        if method == "ALADIN":
            model = ALADINModelForICENTIA(modelpaths=modelpaths)
        # elif method == "ECGFounder":
        #     model = ECGFounderModel(modelpaths=modelpaths)
        # elif method == "Hannun":
        #     model = HannunModel(modelpaths=modelpaths)

        # if method == 'ALADIN':
        #     model = ALADINModelForCinc(modelpaths=modelpaths)
        # elif method == "ECGFounder":
        #     model = ECGFounderModelForCinc(modelpaths=modelpaths)
        # elif method == "Hannun":
        #     model = HannunModelForCinc(modelpaths=modelpaths)

    elif dataset == "ICENTIA-SAMPLE":
        #check gabor data
        data = ICENTIASAMPLEData("ICENTIA", 
            sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", 
            annfile="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods",
            allfile="/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA/samples_xxl.json",
            asynchronous=True)
        #data = ICENTIASAMPLEData("ICENTIA", sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", annfile="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_lukas.xlsx",asynchronous=True)

        if method == "ALADIN":
            model = ALADINModelForICENTIA(modelpaths=modelpaths)

    resultsfolder = os.environ.get('benchmark_results')
    if not os.path.exists(resultsfolder+"/diagnosis"):
        os.makedirs(resultsfolder+"/diagnosis")

    exp = DiagnosticBenchmark(data, model)
    exp.run_batch(overwrite=overwrite)
    #exp.run(overwrite=overwrite)

    #experiment1 = PerClassBenchmark(stanford, nnunet, trim=1)
    #experiment1.run()
    # nnunetstanford = nnUNetModel(folder="Stanford_9")
    # experiment1 = Benchmark(stanford, nnunetstanford, trim=1)
    # experiment1.run()
    # nnunetludb = nnUNetModel(folder="ludb")
    # experiment2 = PerArrhythmiaBenchmark(ludb, nnunetludb, trim=2)
    # experiment2.run()

    #Macro benchmarks
    # experiment4 = Benchmark(ludb, cwt, trim=2)
    # experiment4.run()
    # experiment5 = Benchmark(validationset, cwt, trim=1)
    # experiment5.run()
    # experiment6 = Benchmark(stanford, cwt, trim=1)
    # experiment6.run()

    # nnunetludb = nnUNetModel(folder="ludb_2")
    # nnunetval = nnUNetModel(folder="val_2")
    # nnunetstanford = nnUNetModel(folder="stanford_2")

    # experiment2 = Benchmark(stanford, nnunetstanford, trim=1)
    # experiment2.run()
    # experiment2 = Benchmark(ludb, nnunetludb, trim=2)
    # experiment2.run()
    # experiment2 = Benchmark(validationset, nnunetval, trim=1)
    # experiment2.run()