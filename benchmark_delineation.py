import numpy as np
import os
import neurokit2 as nk 
import scipy.signal as sps
import torch

import scipy.signal as sps

from aladin import ALADIN
from aladin.utils.benchmark_utils import Data, Model, DelineationBenchmark, PerArrhythmiaBenchmark
from aladin.core import Record, RecordCollection
from aladin.utils.martinez_v2 import QRSdelineation
import psutil
import skimage
import dill

import argparse
import math

import sak
import sak.signal.wavelet
import sak.data.augmentation
import sak.data.preprocessing
import sak.visualization.signal
import sak.torch.nn
import sak.torch.train
import sak.torch.data
import sak.torch.models
import sak.torch.models.lego
import sak.torch.models.variational
import sak.torch.models.classification

import faulthandler
faulthandler.enable()


def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

class ALADINModel(Model):
    def __init__(self, modelpaths=[]):
        super().__init__()
        self.output_binary = True
        self.name = "ALADIN"
        self.savelogits = False
        self.modelpaths = modelpaths
        self.predict_abnorm = True

        self.aladin = ALADIN(modelpaths=modelpaths)

    def predict(self, sig, fs, meta=None, preprocess=False): 
        
        #monitor_memory()
        recordname = str(meta["record"]).replace("/", "_")

        record = Record(sig, fs, "bench", recordname)
        self.aladin.segmenter.segment(record)
        
        p_hat = record.delineations.p.binary
        qrs_hat = record.delineations.qrs.binary
        qrs_abnorm_hat = record.delineations.abnormal_qrs.binary
        t_hat = record.delineations.t.binary
        hasafib = record.delineations.afib.binary

        del record

        return p_hat, qrs_hat, qrs_abnorm_hat, t_hat, hasafib
    
    def predict_batch(self, data):
        """
        Predicts the delineation for a batch of data.
        """
        results = {}
        records = []
        for recordname in data:
            record = data[recordname]
            sig = record["signal"]
            fs = record["fs"]
            records.append(Record(sig, fs, "bench", recordname))

        recordCollection = RecordCollection(records)
        recordCollection.preprocess()
        self.aladin.segmenter.batch(records)

        for record in records:
            
            results[record.recordname] = {
                "p": record.delineations.p.binary,
                "qrs": record.delineations.qrs.binary,
                "qrs_abnorm": record.delineations.abnormal_qrs.binary,
                "t": record.delineations.t.binary,
                "hasafib": record.delineations.afib.binary
            }
        
        return results

class Martinez(Model):
        
        def __init__(self):
            super().__init__()
            self.name = "martinez"
            self.savelogits = False
            self.predict_abnorm = False
            self.output_binary = False
    
        def predict(self, sig, fs, meta=None, preprocess=False):
            
            ms200 = int(0.2 * fs)
            ms600 = int(0.6 * fs)
            ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
            ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
            baseline = sps.medfilt(sps.medfilt(sig, ms200), ms600)
            signal = sig - baseline

            b, a = sps.butter(3, 40/(fs/2), btype='lowpass')
            sig = sps.filtfilt(b, a, signal)
            sig -= np.mean(sig)
            sig /= np.std(sig)

            rpeaks, _ = QRSdelineation(sig, fs)

            if len(rpeaks) > 3:
                #get nearest power of 2
                po2 = 2**int(np.ceil(np.log2(len(sig))))
                sig = np.pad(sig, (0, po2-len(sig)), 'constant')
                _, waves = nk.ecg_delineate(sig, rpeaks, sampling_rate=fs)
                
                p_on = np.array(waves["ECG_P_Onsets"])
                p_off = np.array(waves["ECG_P_Offsets"])
                qrs_on = np.array(waves["ECG_R_Onsets"])
                qrs_off = np.array(waves["ECG_R_Offsets"])
                t_on = np.array(waves["ECG_T_Onsets"])
                t_off = np.array(waves["ECG_T_Offsets"])

            else:
                p_on = np.array([])
                p_off = np.array([])
                qrs_on = np.array([])
                qrs_off = np.array([])
                t_on = np.array([])
                t_off = np.array([])

            #zip p_on and p_off to get the regions
            p = np.array(list(zip(p_on, p_off)))
            qrs = np.array(list(zip(qrs_on, qrs_off)))
            t = np.array(list(zip(t_on, t_off)))

            p = p[~np.isnan(p).any(axis=1)] if len(p) > 0 else np.array([])
            qrs = qrs[~np.isnan(qrs).any(axis=1)] if len(qrs) > 0 else np.array([])
            t = t[~np.isnan(t).any(axis=1)] if len(t) > 0 else np.array([])

            return p, qrs, t

class DelineatorSwitchAndCompose(Model):
     
    def __init__(self):
        super().__init__()
        self.output_binary = True
        self.name = 'DelineatorSwitchAndCompose'
        self.savelogits = False
        self.basedir = "./DelineatorSwitchAndCompose"
        self.predict_abnorm = False

        self.valid_folds = sak.load_data(os.path.join(self.basedir,'TrainedModels',"v0",'validation_files.csv'),dtype=None)
        self.fold_of_file = {fname: k for k in self.valid_folds for fname in self.valid_folds[k]}

        #########################################################################
        # Load models
        self.models = {}

    def predict_mask(self, signal, N, stride, model, thr_dice, batch_size = 16):
        # Data structure for computing the segmentation
        windowed_signal = skimage.util.view_as_windows(signal,(N,1),(stride,1))

        # Flat batch shape
        new_shape = (windowed_signal.shape[0]*windowed_signal.shape[1],*windowed_signal.shape[2:])
        windowed_signal = np.reshape(windowed_signal,new_shape)

        # Exchange channel position
        windowed_signal = np.swapaxes(windowed_signal,1,2)

        # Output structures
        windowed_mask = np.zeros((windowed_signal.shape[0],3,windowed_signal.shape[-1]),dtype=float)
        
        # Compute segmentation for all leads independently
        with torch.no_grad():
            for i in range(0,windowed_signal.shape[0],batch_size):
                inputs = {"x": torch.tensor(windowed_signal[i:i+batch_size]).cuda().float()}
                windowed_mask[i:i+batch_size] = model.cuda()(inputs)["sigmoid"].cpu().detach().numpy() > thr_dice

        # Retrieve mask as 1D
        counter = np.zeros((signal.shape[0]), dtype=int)
        segmentation = np.zeros((3,signal.shape[0]))

        for i in range(windowed_mask.shape[0]):
            counter[i*stride:i*stride+N] += 1
            segmentation[:,i*stride:i*stride+N] += windowed_mask[i]
        segmentation = (segmentation/counter).astype(int)

        return segmentation

    def predict(self, signal, fs, meta=None, preprocess=False):

        window_size = 2**11
        stride = 128
        thr_dice = 0.8
        ptg_voting = 0.5
        target_fs = 250.
        batch_size = 16
        use_morph = False

        oldoldlength = signal.shape[0]
        oldsignal = signal
        signal = self.resize_signal(signal, int(signal.shape[0]*500/fs))
        fs = 500

        down_factor = int(fs/250.)
        signal = sps.decimate(signal,down_factor,axis=0)
        
        # Filter signal
        signal = sps.filtfilt(*sps.butter(4, 75.0/250.,  'low'),signal.T).T

        # Compute moving operation for matching amplitude criteria to development set
        ampl = np.median(sak.signal.moving_lambda(signal,200,lambda x: np.max(x,axis=0)-np.min(x,axis=0)),axis=0)

        signal = np.expand_dims(signal,1)
        oldlength = signal.shape[0]

        # Normalize and pad signal for inputing in algorithm
        if signal.shape[0] < window_size:
            signal = np.pad(signal,((0,math.ceil(signal.shape[0]/window_size)*window_size-signal.shape[0]),(0,0)),mode='edge')
        if (signal.shape[0]-window_size)%stride != 0:
            signal = np.pad(signal,((0,math.ceil((signal.shape[0]-window_size)/stride)*stride-(signal.shape[0]%window_size)),(0,0)),mode='edge')
        
        # Correct amplitudes
        signal = signal/ampl
        
        # Obtain segmentation
        segmentation = np.zeros((signal.shape[1],3,signal.shape[0]),dtype=int)
        for l,lead in enumerate(signal.T):
            sig = lead[:,None]
            for j,fold in enumerate(self.models):
                m = self.models[fold]
                segmentation[l,...] += self.predict_mask(sig, window_size, stride, m, thr_dice, batch_size)
                
        segmentation = (segmentation >= 3)

        # Morphological operations
        if use_morph:
            p                   = cv2.morphologyEx(segmentation[0,0,:].astype('float32'), cv2.MORPH_CLOSE, np.ones((5,))).squeeze()
            qrs                 = cv2.morphologyEx(segmentation[0,1,:].astype('float32'), cv2.MORPH_CLOSE, np.ones((5,))).squeeze()
            t                   = cv2.morphologyEx(segmentation[0,2,:].astype('float32'), cv2.MORPH_CLOSE, np.ones((5,))).squeeze()
            segmentation[0,0,:] = cv2.morphologyEx(p,   cv2.MORPH_OPEN,  np.ones((5,))).squeeze().astype(bool)
            segmentation[0,1,:] = cv2.morphologyEx(qrs, cv2.MORPH_OPEN,  np.ones((5,))).squeeze().astype(bool)
            segmentation[0,2,:] = cv2.morphologyEx(t,   cv2.MORPH_OPEN,  np.ones((5,))).squeeze().astype(bool)

        p = np.array(segmentation[0,0,:], dtype=int)[:oldlength]
        qrs = np.array(segmentation[0,1,:], dtype=int)[:oldlength]
        t = np.array(segmentation[0,2,:], dtype=int)[:oldlength]

        p = np.array(self.resize_signal(p, oldoldlength), dtype=int)
        qrs = np.array(self.resize_signal(qrs, oldoldlength), dtype=int)
        t = np.array(self.resize_signal(t, oldoldlength), dtype=int)
        #self.visualize(oldsignal, p, qrs, t, fs)

        return p, qrs, t

    def load_checkpoint(self, file):
        for i in range(5):
            if os.path.isfile(os.path.join(self.basedir,'TrainedModels',"v0",'fold_{}'.format(i+1),'model_best.model')):
                self.models['fold_{}'.format(i+1)] = torch.load(os.path.join(self.basedir,'TrainedModels',"v0",'fold_{}'.format(i+1),'model_best.model'),pickle_module=dill).eval().float()
            else:
                print("File for fold {} not found. Continuing...".format(i+1))

    def visualize(self, signal, p, qrs, t, fs):
        plt.figure(figsize=(15, 5))
        plt.plot(signal)
        plt.plot(p)
        plt.plot(qrs)
        plt.plot(t)

        plt.savefig('ecg.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--method', type=str, help='ALADIN, Martinez or Jiminez', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset used to benchmark (VAL, RDB)', required=True)
    parser.add_argument('--perarrhythmia', action='store_true', help='Run per arrhythmia benchmark')
    #modelpaths is a list of strings
    parser.add_argument('--modelpaths', nargs='+', help='Paths to the models used in the benchmark', required=False, default=["Dataset201_all_101/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"])

    args = parser.parse_args()
    method = args.method
    dataset = args.dataset
    models = args.modelpaths
    perarrhythmia = args.perarrhythmia

    resultsfolder = os.environ.get('benchmark_results')
    datafolder = os.environ.get('benchmark_data')

    if resultsfolder is None:
        print("ERROR: Please set the environment variable benchmark_results to the folder where you want to save the results")
        exit(1)
    if datafolder is None:
        print("ERROR: Please set the environment variable benchmark_data to the folder where the data is located")
        exit(1)

    if method == "ALADIN":
        model = ALADINModel(modelpaths=models)
    elif method == "Martinez":
        model = Martinez()
    elif method == "Jiminez":
        model = DelineatorSwitchAndCompose()
        model.load_checkpoint('')

    if dataset == "VAL":
        data = Data("VAL", "")
    elif dataset == "RDB":
        data = Data("RDB", "")
    else:
        print("ERROR: Please select a valid dataset (VAL, RDB)")
        exit(1)

    if not os.path.exists(resultsfolder+"/delineation"):
        os.makedirs(resultsfolder+"/delineation")
    
    if perarrhythmia:
        #Per arrhythmia benchmarks
        experiment = PerArrhythmiaBenchmark(data, model, trim=1)
        experiment.run()
    else:
        #Overall benchmark
        experiment = DelineationBenchmark(data, model, trim=1)
        #experiment.run()
        experiment.run_batch()