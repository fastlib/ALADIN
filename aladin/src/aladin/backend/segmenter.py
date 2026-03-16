import numpy as np
import os
import torch
from tqdm import tqdm
import wfdb
import time

import aladin._main as cpp_backend
from aladin.core import Record
from aladin.utils.helpers import resize_signal, get_regions
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from aladin.utils.morphological import closingcentered, openingcentered
import scipy.signal as sps
import subprocess
import subprocess

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import aladin.configuration
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, nnUNetWithClassificationPredictor, nnUNetLSTMWithClassificationPredictor
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiments, preprocess, extract_fingerprints
from nnunetv2.run.run_training import run_training
from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json

from baal.active.heuristics import BALD


class SegmenterBase():
    def __init__(self, debug=False):
        self.debug = debug
        pass

    def segment(self, ecg: np.ndarray):
        raise NotImplementedError

    def plot(self, record: Record, filename=""):
        sig = record.filtered_ecg

        cols = ['#e74c3c','#2ecc71','#3498DB','#f1c40f','#908894']
        wave_names = ["p", "qrs", "t", "abnormal_qrs", "noise"]

        fig, ax = plt.subplots(2, 1, figsize=(len(sig)/(record.fs), 6), dpi=200)
        ax[0].plot(sig, color='black', label="ECG Signal")
        ax[0].set_xticks(np.arange(0, len(sig), record.fs*10))
        ax[0].set_xticklabels(np.arange(0, len(sig)/record.fs, 10))

        p_waves = record.cpp_record.delineations.p.binary
        qrs_waves = record.cpp_record.delineations.qrs.binary
        t_waves = record.cpp_record.delineations.t.binary
        abnormal_qrs_waves = record.cpp_record.delineations.abnormal_qrs.binary
        noise_waves = record.cpp_record.delineations.noise.binary

        wave_types = [p_waves, qrs_waves, t_waves, abnormal_qrs_waves, noise_waves]

        for i, waves in enumerate(wave_types):
            regions = get_regions(waves)
            for region in regions:
                ax[0].axvspan(region[0], region[1], alpha=0.4, color=cols[i], zorder=0)

        afibregions = get_regions(record.cpp_record.delineations.afib.binary)
        for region in afibregions:
            ax[0].axvspan(region[0], region[1], alpha=0.4, color='#8e44ad', zorder=0)
        
        # p_waves = get_regions(record.delineation["p"]["binary"])
        # p_centers = [(region[0]+region[1])//2 for region in p_waves]
        # y = [p_centers[i]-p_centers[i-1] for i in range(1,len(p_centers))]
        # x = [p_centers[i] for i in range(1,len(p_centers))]

        ax[1].plot(record.cpp_record.delineations.afib.logits, color='blue', linestyle='--', label="afib_uncertainty")
        #ax[1].plot(record.cpp_record.delineations.noise.logits, color='red', linestyle='--', label="afib_uncertainty")
        #ax[1].plot(np.log(record.cpp_record.delineations.noise.uncertainty), color='yellow', linestyle='--', label="afib_uncertainty")
        ax[1].plot(record.cpp_record.delineations.afib.uncertainty, color='purple', linestyle='--', label="afib_uncertainty")

        #ax[0].axis('off')
        ax[0].set_xlim(0, len(sig))
        ax[1].set_xlim(0, len(sig))

        if filename == "":
            filename = "segmentation.png"

        plt.savefig(filename)

class UNetSegmenter(SegmenterBase):
    def __init__(self, modelpaths=[], usefullcontext=True, debug=False, cache=False, embed=False):
        super().__init__(debug=debug)

        print("UNetSegmenter initialized")
        print("Number of processors: ", multiprocessing.cpu_count())

        self.targetfs = 204.8
        self.thresholds = [0.25, 0.25, 0.5, 0.5, 0.1, 0.5]
        self.cache = cache
        self.num_workers = multiprocessing.cpu_count()

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get the current device (GPU)
            self.device = torch.device('cuda:0')
            dev_id = torch.cuda.current_device()
            # Get the total memory available in bytes
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            res = result.stdout.split("\n")[dev_id]
            # Extract memory information
            _, ram_available, _ = map(int, res.split(","))
            if ram_available < 1024:
                print("Warning: Available GPU memory is less than 1GB. We switch to CPU device instead.")
                self.device = torch.device('cpu')
            print("Using GPU device:", torch.cuda.get_device_name(dev_id), "with", ram_available, "MB free memory")
        else:
            self.device = torch.device('cpu')

        self.sliding_models = []
        self.fullcontext_models = []
        self.n_leads = []

        for modelpath in modelpaths:
            model = nnUNetWithClassificationPredictor(
                tile_step_size=0.9,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=True,
                device=self.device,
                verbose=True,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            model.initialize_from_trained_model_folder(
                os.path.join(aladin.configuration.aladin_model_folder, modelpath),
                use_folds=(0,1,2,3,4),
                checkpoint_name='checkpoint_best.pth',
            )
            self.sliding_models.append(model)

            dataset_json = load_json(join(os.path.join(aladin.configuration.aladin_model_folder, modelpath), 'dataset.json'))
            self.n_leads.append(len(dataset_json['channel_names']))

        if usefullcontext:
            for modelpath in modelpaths:
                model = nnUNetLSTMWithClassificationPredictor(
                    tile_step_size=0.9,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_device=True,
                    device=self.device,
                    verbose=True,
                    verbose_preprocessing=False,
                    allow_tqdm=False
                )
                model.initialize_from_trained_model_folder(
                    os.path.join(aladin.configuration.aladin_model_folder, modelpath),
                    use_folds=(0,1,2,3,4),
                    checkpoint_name='checkpoint_best.pth',
                )
                self.fullcontext_models.append(model)

        self.heuristic = BALD()

        if len(set(self.n_leads)) > 1:
            print("Warning: The models have different number of input leads. This might lead to suboptimal performance. Make sure all models have the same number of input leads for best performance.")
        else:
            self.n_leads = self.n_leads[0]
    

    def preprocess(self, record: Record):

        fs = record.fs

        signal = record.ecg.copy()

        channels = 12
        record.cpp_record.normalized_ecg = np.zeros((channels, len(signal[0])))

        for ch in range(channels):
            
            #if lead is not available, skip processing for this channel and leave it as zeros in the normalized_ecg and filtered_ecg properties
            if not record.available_leads[ch]:
                continue

            sig = signal[ch, :]
            # filter signal between 5 and 30 Hz

            lowcut = 0.2
            highcut = 30
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(4, high, btype='low')
            sig_band = filtfilt(b, a, sig.copy())

            ms200 = int(0.2 * fs)
            ms600 = int(0.6 * fs)
            ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
            ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
            baseline = sps.medfilt(sps.medfilt(sig_band, ms200), ms600)
            sig_band = sig_band - baseline
            if ch == 1: #use lead II for symbolic reasoning
                record.cpp_record.ecg_bandpass = sig_band.copy()
            
            # norm_signal = sig.copy()
            # norm_signal -= np.mean(norm_signal)
            # norm_signal /= np.std(norm_signal)
            
            # record.cpp_record.normalized_ecg[ch,:] = norm_signal

            highcut = 40
            nyquist = 0.5 * fs
            high = highcut / nyquist
            b, a = butter(4, high, btype='low')
            sig = filtfilt(b, a, sig)

            ms200 = int(0.2 * fs)
            ms600 = int(0.6 * fs)
            ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
            ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
            baseline = sps.medfilt(sps.medfilt(sig, ms200), ms600)
            sig = sig - baseline

            #signal -= np.mean(signal)
            sig /= np.std(sig)

            record.cpp_record.normalized_ecg[ch,:] = sig.copy()

            if ch == 1: #use lead II for symbolic reasoning
                record.cpp_record.filtered_ecg = sig.copy()
                record.cpp_record.ecg_no_qrst = sig.copy()

            highcut = 10
            nyquist = 0.5 * fs
            high = highcut / nyquist
            b, a = butter(4, high, btype='high')
            sig = filtfilt(b, a, sig)
            
            if ch == 1: #use lead II for symbolic reasoning
                record.cpp_record.ecg_noise = sig.copy()

        return record

    def preprocess_batch(self, records: list):

        #print("Preprocessing batch of records")
        
        for i in tqdm(range(len(records))):
            records[i] = self.preprocess(records[i])

            # fs = records[i].fs

            # signal = records[i].ecg.copy()
            # assert len(signal.shape) == 1

            # lowcut = 0.2
            # highcut = 30
            # nyquist = 0.5 * fs
            # low = lowcut / nyquist
            # high = highcut / nyquist
            # b, a = butter(4, high, btype='low')
            # sig_band = filtfilt(b, a, signal.copy())

            # ms200 = int(0.2 * fs)
            # ms600 = int(0.6 * fs)
            # ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
            # ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
            # baseline = sps.medfilt(sps.medfilt(sig_band, ms200), ms600)
            # sig_band = sig_band - baseline
            # records[i].cpp_record.ecg_bandpass = sig_band.copy()

            # highcut = 40
            # nyquist = 0.5 * fs
            # high = highcut / nyquist
            # b, a = butter(4, high, btype='low')
            # signal = filtfilt(b, a, signal)

            # ms200 = int(0.2 * fs)
            # ms600 = int(0.6 * fs)
            # ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
            # ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
            # baseline = sps.medfilt(sps.medfilt(signal, ms200), ms600)
            # signal = signal - baseline

            # signal -= np.mean(signal)
            # signal /= np.std(signal)

            # records[i].cpp_record.filtered_ecg = signal.copy()
            # records[i].cpp_record.ecg_no_qrst = signal.copy()

            # highcut = 10
            # nyquist = 0.5 * fs
            # high = highcut / nyquist
            # b, a = butter(4, high, btype='high')
            # signal = filtfilt(b, a, signal)

            # records[i].cpp_record.ecg_noise = signal.copy()

        return records

    def prepare_ecg(self, record: Record):
        sig = record.normalized_ecg.copy()
        if self.n_leads == 1:
            sig = sig[1:2,:] #use lead II for single lead processing
            print("Warning: The model accepts 1 lead, we will use only lead II for processing.")
        elif self.n_leads == 3:
            sig = sig[[1,6,11],:]
            print("Warning: The model accepts 3 leads, we will use leads II, V1 and V6 for processing.")
        else:
            raise ValueError("Currently ALADIN works only on 1-lead and 3-lead data, but the provided model(s) expect a different number of leads. Please provide models that are trained on the same number of leads as your data for best performance.")

        record.original_length = sig.shape[1]
        record.before_padding = sig.shape[1]

        #resize if necessary
        if record.fs != self.targetfs:
            newlength = int(sig.shape[1] * self.targetfs / record.fs)
            sig = resize_signal(sig, newlength)
            record.before_padding = sig.shape[1]

        if sig.shape[-1] % 256 != 0:
            sig = np.pad(sig, ((0,0), (0, 256 - sig.shape[1] % 256)), 'constant', constant_values=(0, 0))
            #print("Padded signal to ", sig.shape[-1], "from ", record.before_padding)

        #add channel dimension
        if len(sig.shape) == 1:
            sig = sig[None, None, None, :]
        elif len(sig.shape) == 2:
            sig = sig[:, None, None, :]

        props = {"spacing": [999,999,1]}

        return sig, props
    
    def prepare_batch(self, records: list):

        ecgs = []
        maxlen = 0
        for record in records:
            sig = record.normalized_ecg.copy()
            if self.n_leads == 1:
                sig = sig[1:2,:] #use lead II for single lead processing
                print("Warning: The model accepts 1 lead, we will use only lead II for processing.")
            elif self.n_leads == 3:
                sig = sig[[1,6,11],:]
                print("Warning: The model accepts 3 leads, we will use leads II, V1 and V6 for processing.")
            else:
                raise ValueError("Currently ALADIN works only on 1-lead and 3-lead data, but the provided model(s) expect a different number of leads. Please provide models that are trained on the same number of leads as your data for best performance.")
                
            record.original_length = sig.shape[1]
            record.before_padding = sig.shape[1]

            #resize if necessary
            if record.fs != self.targetfs:
                newlength = int(sig.shape[1] * self.targetfs / record.fs)
                sig = resize_signal(sig, newlength)
                record.before_padding = sig.shape[1]

            if sig.shape[-1] % 256 != 0:
                sig = np.pad(sig, ((0,0), (0, 256 - sig.shape[1] % 256)), 'constant', constant_values=(0, 0))

            ecgs.append(sig)
            # if len(sig) > maxlen:
            #     maxlen = len(sig)
        
        sigs = []
        props = []
        for i, s in enumerate(ecgs):
            props.append({"spacing": [999,999,1]})
            tmp = np.zeros((s.shape[0], 1, 1, s.shape[1]))
            tmp[:, 0, 0, :] = s
            # if len(s) < maxlen:
            #     tmp[0, 0, 0, len(s):] = 0
            sigs.append(tmp)

        return sigs, props

    def calculate_uncertainties(self, rets):
        
        #folds = [r["seg"][1][i] for r in rets for i in range(len(r))]
        folds = [r["seg"][1][i] for r in rets for i in range(len(r["seg"][1]))]
        nchannels = 6
        nfolds = len(folds)
        #print(len(rets))
        n_sampels = rets[0]["seg"][1][0][1].shape[-1]
        probs = np.zeros((nfolds, nchannels, n_sampels))

        #loop over all folds
        for f, fr in enumerate(folds):
            logits = fr[1][:,0,0,:]

            probs[f] = logits


        #transpose to have correct shape for the BALD heuristic
        probs = probs.transpose(1,2,0)
        uncertainties = np.zeros((6, n_sampels))

        for t in range(probs.shape[0]):
            uncertainties[t] = self.heuristic.compute_score(probs[t][None, None, :, :])[0]

        return uncertainties

    def calculate_afib_uncertainties(self, pred):

        #calculate entropy of softmax output
        epsilon = 1e-10
        pred = np.clip(pred, epsilon, 1 - epsilon)

        # Calculate entropy for each prediction
        entropy = -np.sum(pred * np.log2(pred), axis=0)
        return entropy

    def undo_resampling(self, sig, record: Record):

        if record.fs != self.targetfs:
            newlength = int(record.before_padding * record.fs / self.targetfs)
            sig = resize_signal(sig[:, :record.before_padding], newlength)
            if sig.shape[-1] > record.original_length:
                sig = sig[:,:record.original_length]
            elif sig.shape[-1] < record.original_length:
                #pad only last dimension
                sig = np.pad(sig, ((0,0),(0, record.original_length - sig.shape[-1])), 'constant', constant_values=(0, 0))

                #sig = np.pad(sig, (0, record.original_length - sig.shape[-1]), 'constant', constant_values=(0, 0))
    
        return sig

    def process_ensemble(self, sliding_rets, fullcontext_rets, record: Record, length=6144, pbar=None):
        
        avg_afib_logits = np.zeros((2, length))
        avg_afib_softmax = np.zeros((2, length))
        avg_inverse_logits = np.zeros((2, length))
        avg_logits = np.zeros((6, length))
        uncertainties = self.calculate_uncertainties(sliding_rets + fullcontext_rets)

        for i in range(len(sliding_rets)):
            #ret = [avg_rets = (seg,prob), all_rets=[(seg,prob),...,(seg,prob)]]
            seg = sliding_rets[i]["seg"]
            afib_logits = sliding_rets[i]["cls"][0][:,:,0,0,:]
            inverse_logits = sliding_rets[i]["cls"][1][:,:,0,0,:]

            #print(length, afib_logits.shape)

            avg_afib_logits += torch.sigmoid(afib_logits.mean(0)).detach().cpu().numpy()
            avg_afib_softmax += torch.softmax(afib_logits.mean(0), dim=0).detach().cpu().numpy()
            avg_inverse_logits += torch.softmax(inverse_logits.mean(0), dim=0).detach().cpu().numpy()
            avg_logits += seg[0][1][:,0,0,:]


        for i in range(len(fullcontext_rets)):
            #ret = [avg_rets = (seg,prob), all_rets=[(seg,prob),...,(seg,prob)]]
            seg = fullcontext_rets[i]["seg"]
            afib_logits = fullcontext_rets[i]["cls"][0][:,:,0,0,:]
            inverse_logits = fullcontext_rets[i]["cls"][1][:,:,0,0,:]
            
            avg_afib_logits += torch.sigmoid(afib_logits.mean(0)).detach().cpu().numpy()
            avg_afib_softmax += torch.softmax(afib_logits.mean(0), dim=0).detach().cpu().numpy()
            avg_inverse_logits += torch.softmax(inverse_logits.mean(0), dim=0).detach().cpu().numpy()
            avg_logits += seg[0][1][:,0,0,:]

        avg_afib_logits /= len(sliding_rets) + len(fullcontext_rets)
        avg_afib_softmax /= len(sliding_rets) + len(fullcontext_rets)
        avg_inverse_logits /= len(sliding_rets) + len(fullcontext_rets)
        avg_logits /= len(sliding_rets) + len(fullcontext_rets)

        if record.fs != self.targetfs:
            avg_logits = self.undo_resampling(avg_logits, record)
            avg_afib_logits = self.undo_resampling(avg_afib_logits, record)
            avg_afib_softmax = self.undo_resampling(avg_afib_softmax, record)
            avg_inverse_logits = self.undo_resampling(avg_inverse_logits, record)
            uncertainties = self.undo_resampling(uncertainties, record)

        p_hat = np.array(avg_logits[1] > self.thresholds[1], dtype=bool)
        qrs_hat = np.array(avg_logits[2] > self.thresholds[2], dtype=bool)
        t_hat = np.array(avg_logits[3] > self.thresholds[3], dtype=bool)
        noise_hat = np.array(avg_logits[4] > self.thresholds[4], dtype=bool)
        abnormal_qrs_hat = np.array(avg_logits[5] > self.thresholds[5], dtype=bool)
        afib_hat = np.array(avg_afib_softmax.argmax(0) == 1, dtype=bool)

        uncertainties_afib = self.calculate_afib_uncertainties(avg_afib_logits)

        p_wave_delineation = cpp_backend.Delineation(avg_logits[1], uncertainties[1],  p_hat)
        qrs_delineation = cpp_backend.Delineation(avg_logits[2], uncertainties[2], qrs_hat)
        t_wave_delineation = cpp_backend.Delineation(avg_logits[3], uncertainties[3], t_hat)
        noise_delineation = cpp_backend.Delineation(avg_logits[4], uncertainties[4], noise_hat)
        abnormal_qrs_delineation = cpp_backend.Delineation(avg_logits[5], uncertainties[5], abnormal_qrs_hat)
        afib_delineation = cpp_backend.Delineation(avg_afib_logits[1], uncertainties_afib, afib_hat)

        record.cpp_record.delineations = cpp_backend.Delineations(p_wave_delineation, qrs_delineation, abnormal_qrs_delineation, t_wave_delineation, noise_delineation, afib_delineation)
        #print(record.cpp_record.delineations.p.size)

        if pbar is not None:
            pbar.update(1)

        if np.mean(np.array(avg_inverse_logits.argmax(0)) > 0.75):
            return False

        return True


        # sliding_ret = sliding_rets[index]
        # fullcontext_ret = fullcontext_rets[index]

        # avg_afib_logits = np.zeros((2, length))
        # avg_afib_softmax = np.zeros((2, length))
        # avg_inverse_logits = np.zeros((2, length))
        # avg_logits = np.zeros((6, length))
        # uncertainties = self.calculate_uncertainties(sliding_ret + fullcontext_ret)

        # for i in range(len(sliding_ret)):
        #     #ret = [avg_rets = (seg,prob), all_rets=[(seg,prob),...,(seg,prob)]]
        #     seg = sliding_ret[i]["seg"]
        #     afib_logits = sliding_ret[i]["cls"][0][:,:,0,0,:]
        #     avg_afib_logits += torch.sigmoid(afib_logits.mean(0)).detach().cpu().numpy()
        #     avg_afib_softmax += torch.softmax(afib_logits.mean(0), dim=0).detach().cpu().numpy()
        #     avg_logits += seg[0][1][:,0,0,:]


        # for i in range(len(fullcontext_ret)):
        #     #ret = [avg_rets = (seg,prob), all_rets=[(seg,prob),...,(seg,prob)]]
        #     seg = fullcontext_ret[i]["seg"]
        #     afib_logits = fullcontext_ret[i]["cls"][0][:,:,0,0,:]
        #     inverse_logits = fullcontext_ret[i]["cls"][1][:,:,0,0,:]
            
        #     avg_afib_logits += torch.sigmoid(afib_logits.mean(0)).detach().cpu().numpy()
        #     avg_afib_softmax += torch.softmax(afib_logits.mean(0), dim=0).detach().cpu().numpy()
        #     avg_inverse_logits += torch.softmax(inverse_logits.mean(0), dim=0).detach().cpu().numpy()
        #     avg_logits += seg[0][1][:,0,0,:]

        # avg_afib_logits /= len(sliding_ret) + len(fullcontext_ret)
        # avg_afib_softmax /= len(sliding_ret) + len(fullcontext_ret)
        # avg_inverse_logits /= len(fullcontext_ret)
        # avg_logits /= len(sliding_ret) + len(fullcontext_ret)

        # if record.fs != self.targetfs:
        #     avg_logits = self.undo_resampling(avg_logits, record)
        #     avg_afib_logits = self.undo_resampling(avg_afib_logits, record)
        #     avg_afib_softmax = self.undo_resampling(avg_afib_softmax, record)
        #     avg_inverse_logits = self.undo_resampling(avg_inverse_logits, record)
        #     uncertainties = self.undo_resampling(uncertainties, record)

        # p_hat = np.array(avg_logits[1] > self.thresholds[1], dtype=bool)
        # qrs_hat = np.array(avg_logits[2] > self.thresholds[2], dtype=bool)
        # t_hat = np.array(avg_logits[3] > self.thresholds[3], dtype=bool)
        # noise_hat = np.array(avg_logits[4] > self.thresholds[4], dtype=bool)
        # abnormal_qrs_hat = np.array(avg_logits[5] > self.thresholds[5], dtype=bool)
        # afib_hat = np.array(avg_afib_softmax.argmax(0) == 1, dtype=bool)

        # uncertainties_afib = self.calculate_afib_uncertainties(avg_afib_logits)

        # p_wave_delineation = cpp_backend.Delineation(avg_logits[1], uncertainties[1],  p_hat)
        # qrs_delineation = cpp_backend.Delineation(avg_logits[2], uncertainties[2], qrs_hat)
        # t_wave_delineation = cpp_backend.Delineation(avg_logits[3], uncertainties[3], t_hat)
        # noise_delineation = cpp_backend.Delineation(avg_logits[4], uncertainties[4], noise_hat)
        # abnormal_qrs_delineation = cpp_backend.Delineation(avg_logits[5], uncertainties[5], abnormal_qrs_hat)
        # afib_delineation = cpp_backend.Delineation(avg_afib_logits[1], uncertainties_afib, afib_hat)

        # record.cpp_record.delineations = cpp_backend.Delineations(p_wave_delineation, qrs_delineation, abnormal_qrs_delineation, t_wave_delineation, noise_delineation, afib_delineation)

        # if np.mean(np.array(avg_inverse_logits.argmax(0)) > 0.75):
        #     return False

        # return out

    def segment(self, record: Record):
            
        record = self.preprocess(record)
        sig, props = self.prepare_ecg(record)

        print(sig.shape)

        sliding_rets = []
        for model in self.sliding_models:
            sliding_rets.append(model.predict_single_npy_array(sig, props, None, None, True))
        fullcontext_rets = []
        for model in self.fullcontext_models:
            fullcontext_rets.append(model.predict_single_npy_array(sig, props, None, None, True))
        
        if(not self.process_ensemble(sliding_rets, fullcontext_rets, record, length=sig.shape[-1])):

            #print("Lead reversal detected. Reversing signal and segmenting again")
            record.cpp_record.reverse()
            sig, props = self.prepare_ecg(record)
            sliding_rets = []
            for model in self.sliding_models:
                sliding_rets.append(model.predict_single_npy_array(sig, props, None, None, True))
            fullcontext_rets = []
            for model in self.fullcontext_models:
                fullcontext_rets.append(model.predict_single_npy_array(sig, props, None, None, True))
            
            self.process_ensemble(sliding_rets, fullcontext_rets, record, length=sig.shape[-1])

        del sig

        if self.debug:
            self.plot(record)

    def batch(self, records: list):
        
        records = self.preprocess_batch(records)
        sig, props = self.prepare_batch(records)

        #print("Batch processing ", len(records), "records with 10s sliding window")
        sliding_rets = []
        for model in self.sliding_models:
            sliding_rets.append(model.predict_from_list_of_npy_arrays(sig, None, props, None, 32, True))

        #print("Batch processing ", len(records), "records with 30s sliding window")
        fullcontext_rets = []
        for model in self.fullcontext_models:
            fullcontext_rets.append(model.predict_from_list_of_npy_arrays(sig, None, props, None, 32, True))
        
        #print(len(records), len(sig), sliding_rets, len(fullcontext_rets))
        reversed_records = []

        st = time.time()
        for i in tqdm(range(len(records)), desc="Processing GPU results"):
            if(not self.process_ensemble([sliding_rets[k][i] for k in range(len(sliding_rets))], [fullcontext_rets[k][i] for k in range(len(fullcontext_rets))], records[i], length=sig[i].shape[-1])):
                records[i].cpp_record.reverse()
                reversed_records.append(i)
        #print("Processing took ", time.time() - st, " seconds")

        if len(reversed_records) > 0:
            #print("Lead reversal detected in records: ", reversed_records)
            sig, props = self.prepare_batch([records[i] for i in reversed_records])

            #print("Batch processing ", len(reversed_records), "records with 10s sliding window")
            sliding_rets = []
            for model in self.sliding_models:
                sliding_rets.append(model.predict_from_list_of_npy_arrays(sig, None, props, None, 32, True))

            #print("Batch processing ", len(reversed_records), "records with 30s sliding window")
            fullcontext_rets = []
            for model in self.fullcontext_models:
                fullcontext_rets.append(model.predict_from_list_of_npy_arrays(sig, None, props, None, 32, True))

            for idx, i in enumerate(reversed_records):
                self.process_ensemble([sliding_rets[k][idx] for k in range(len(sliding_rets))], [fullcontext_rets[k][idx] for k in range(len(fullcontext_rets))], records[i], length=sig[idx].shape[-1])

                # print(i, "Lead reversal detected. Reversing signal and segmenting again")
                # records[i].cpp_record.reverse()
                # tmpsig, tmpprops = self.prepare_ecg(records[i])
                # tmpsig = np.pad(tmpsig, ((0,0),(0,0),(0,0),(0,sig[i].shape[-1]-tmpsig.shape[-1])), 'constant', constant_values=(0, 0))
                # sliding_ret = []
                # for model in self.sliding_models:
                #     sliding_ret.append(model.predict_single_npy_array(tmpsig, tmpprops, None, None, True))
                # fullcontext_ret = []
                # for model in self.fullcontext_models:
                #     fullcontext_ret.append(model.predict_single_npy_array(tmpsig, tmpprops, None, None, True))
                
                # self.process_ensemble(sliding_ret, fullcontext_ret, records[i], length=sig[i].shape[-1])

        del sig

        if self.debug:
            for i in range(len(records)):
                recname = records[i].recordname.split("/")[-1]
                self.plot(records[i], filename=f"segmentation_{recname}.png")


    def embed(self, record: Record):

        record = self.preprocess(record)
        sig, props = self.prepare_ecg(record)

        embedding = self.fullcontext_models[0].embed_single_npy_array(sig, props, latent_layer=0)

        embed = embedding["embedding"][:,0,0,:]
        embed = torch.mean(embed, axis=-1).squeeze()
        record["embedding"] = embed.detach().cpu().numpy()

        return record

    
    def embed_batch(self, records: list):

        records = self.preprocess_batch(records)
        sig, props = self.prepare_batch(records)

        res = []
        embeddings = self.sliding_models[0].embed_from_list_of_npy_arrays(sig, None, props, None, 32, True, latent_layer=0)

        for i, embedding in enumerate(embeddings):
            embed = embedding["embedding"][:,0,0,:]
            print("Embedding shape: ", embed.shape)
            embed = torch.mean(embed, axis=-1).squeeze()
            records[i]._embedding = embed.detach().cpu().numpy()
            res.append(embed.detach().cpu().numpy())

        return np.array(res)