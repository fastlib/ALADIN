import numpy as np
from tqdm import tqdm
import os
import time
import multiprocessing
import json
import pickle
from scipy.signal import hilbert

from aladin.utils.bald import BALD
import torch

import aladin.configuration
import matplotlib.pyplot as plt
from aladin.utils.morphological import closingcentered, openingcentered
from scipy.signal import butter, lfilter, filtfilt
import scipy.signal as sps
from dtw import *


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils":
            module = "aladin.utils"  # Redirect to the new module path
        return super().find_class(module, name)

class Beat():
    def __init__(self, onset, offset, qrs_onset, qrs_offset, signal, abnormal, fs):
        self.onset = onset
        self.offset = offset
        self.qrs = [qrs_onset, qrs_offset]
        self.width = qrs_offset - qrs_onset
        self.t = None
        self.p = None
        self.pr = np.nan
        self.signal = signal
        self.envelope = None
        self.abnormal = abnormal
        self.part_of = ""
        self.junctional = False
        self.d_signal = np.diff(signal)
        self.fs = fs
        self.curvature = []
        self.dominant_points = []
        self.oldlength = 0

    def get_representative_point(self):

        max_delta = -np.inf
        max_point = None

        for dp in self.dominant_points:
            if np.abs(dp["delta"]) > max_delta:
                max_delta = dp["delta"]
                max_point = dp

        return max_point

    def get_r_wave(self):
        mp = self.get_representative_point()
        if mp is None:
            return (self.qrs[0] + self.qrs[1])//2
        
        if hasattr(self, 'oldlength'):
            samp_ratio = self.oldlength/len(self.signal)
        else:
            samp_ratio = 1

        return int(mp["j"]*samp_ratio) + self.onset

    
    def get_support(self):
        mp = self.get_representative_point()
        return (mp["d"][1] - mp["d"][0])

    def get_width_test(self, signal, name=""):

        midpoint = self.get_r_wave()
        winl = int(0.3*self.fs)
        winr = int(0.15*self.fs)
        calwin = int(0.05*self.fs)
        signal = np.pad(signal.copy(), (winl, winr), 'edge')
        midpoint += winl
        sig = signal[midpoint-winl:midpoint+winr]


        hilb = hilbert(sig)
        self.envelope = np.sqrt(np.real(hilb)*np.real(hilb) + np.imag(hilb)*np.imag(hilb))
        self.envelope[0] = self.envelope[1]
        self.envelope[-1] = self.envelope[-2]
        r = 1
        tmp = np.pad(self.envelope, (2*r, 2*r), 'edge')
        for i in range(2*r,len(tmp)-2*r):
            self.envelope[i-2*r] = (1/10) * (2*(tmp[i-2*r] - tmp[i+2*r]) + 2*(tmp[i+r] - tmp[i-r]))
        
        self.envelope = 2*(self.envelope**2)

        u0l = np.mean(self.envelope[:calwin])
        u0r = np.mean(self.envelope[-calwin:])
        u1l = np.mean(self.envelope[:winl])
        u1r = np.mean(self.envelope[-winl:])
        m = np.max(self.envelope)
        thresl = np.max(self.envelope[:calwin])
        thresr = np.max(self.envelope[-calwin:])
        self.thresl = thresl
        self.thresr = thresr

        self.cumsuml, onsets = self.page_hinkley(self.envelope[:winl], thresl, m*0.5)
        self.cumsuml = np.pad(self.cumsuml, (0, len(self.envelope)-winl), 'constant', constant_values=(0, 0))
        self.cumsumr, offsets = self.page_hinkley(self.envelope[winl:][::-1], thresr, m*0.5)
        self.cumsumr = self.cumsumr[::-1]
        self.cumsumr = np.pad(self.cumsumr, (winl, 0), 'constant', constant_values=(0, 0))
        self.cumsum = self.cumsuml + self.cumsumr
        offsets = [len(self.envelope)-o for o in offsets]

        if name != "":
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
            ax.plot((sig-np.min(sig))/(np.max(sig)-np.min(sig)), color='black', zorder=1)
            #ax.plot(self.envelope, color='purple', zorder=1)
            ax.plot(self.cumsum, color='blue', zorder=1)
            #ax.set_ylim([0,2000])
            ax.axvline(x=onsets[0], color='red', linestyle='--', zorder=0)
            ax.axvline(x=offsets[0], color='red', linestyle='--', zorder=0)
            ax.text(onsets[0], 0, "Width: "+str((offsets[0]-onsets[0])/self.fs), color='red')
            plt.savefig(name+".png")

        self.width = (self.qrs[1] - self.qrs[0])/self.fs

        return self.width

    def get_width(self):
        
        sig = (self.signal-np.mean(self.signal))/np.std(self.signal)
        imagpart = hilbert(self.signal)
        self.envelope = np.sqrt(np.real(imagpart)*np.real(imagpart) + np.imag(imagpart)*np.imag(imagpart))
        self.envelope[0] = self.envelope[1]
        self.envelope[-1] = self.envelope[-2]
        r = 1
        tmp = np.pad(self.envelope, (2*r, 2*r), 'edge')
        for i in range(2*r,len(tmp)-2*r):
            self.envelope[i-2*r] = (1/10) * (2*(tmp[i-2*r] - tmp[i+2*r]) + 2*(tmp[i+r] - tmp[i-r]))
        
        self.envelope = 2*(self.envelope**2)

        #get middle point
        midpoint = 20
        thresl = np.max(self.envelope[:int(0.05*self.fs)])
        thresr = np.max(self.envelope[-int(0.15*self.fs):])
        self.thresl = thresl
        self.thresr = thresr

        self.cumsuml, onsets = self.page_hinkley(self.envelope[:midpoint], 0, thresl)
        self.cumsuml = np.pad(self.cumsuml, (0, len(self.envelope)-midpoint), 'constant', constant_values=(0, 0))
        self.cumsumr, offsets = self.page_hinkley(self.envelope[midpoint:][::-1], 0, thresr)
        self.cumsumr = self.cumsumr[::-1]
        self.cumsumr = np.pad(self.cumsumr, (midpoint, 0), 'constant', constant_values=(0, 0))
        self.cumsum = self.cumsuml + self.cumsumr
        offsets = [len(self.envelope)-o-1 for o in offsets]

        if len(onsets) == 0 or len(offsets) == 0:
            return self.offset - self.onset
        self.qrs[0] = onsets[0]
        self.qrs[1] = offsets[0]
        self.width = offsets[0] - onsets[0]

        return self.width

    def page_hinkley(self, data, delta, lambda_):
        """
        Page-Hinkley Test for changepoint detection.

        Parameters:
        - data: array-like, input time series or signal (1D NumPy array)
        - delta: tolerance for small changes (sensitivity parameter)
        - lambda_: allowable threshold for the cumulative sum
        - threshold: absolute change threshold for declaring a changepoint

        Returns:
        - changepoints: list of indices where changepoints are detected
        """
        mean = 0
        cumulative_sum = 0
        cumsum = []
        changepoints = []
        lastzero = 0
        
        for t in range(len(data)):
            # Incrementally update the mean
            mean = mean + (data[t] - mean) / (t + 1)
            
            # Update the cumulative sum
            cumulative_sum += data[t] - mean
            signed_cumulative_sum = cumulative_sum if cumulative_sum > 0 else 0
            lastzero = t if signed_cumulative_sum == 0 else lastzero
            cumsum.append(signed_cumulative_sum)

            if signed_cumulative_sum > lambda_:
                changepoints.append(lastzero)
                cumulative_sum = 0

        if len(changepoints) == 0:
            changepoints.append(len(data)-1)

        return cumsum, changepoints

    def plot(self, title="Beat"):
        
        #self.get_width()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
        norm = (self.signal-np.min(self.signal))/(np.max(self.signal)-np.min(self.signal))
        #normenv = (self.envelope-np.min(self.envelope))/(np.max(self.envelope)-np.min(self.envelope))
        #normcumsum = (self.cumsum-np.min(self.cumsum))/(np.max(self.cumsum)-np.min(self.cumsum))

        ax.plot(self.signal, color='black', zorder=1)
        # ax.plot(normcumsum, color='blue', zorder=1)
        # ax.plot(normenv, color='purple', zorder=1)
        # ax.axvline(x=self.qrs[0], color='red', linestyle='--', zorder=0)
        # ax.axvline(x=self.qrs[1], color='red', linestyle='--', zorder=0)
        #ax.axhline(y=self.thresl, color='green', linestyle='--', zorder=0)
        #ax.axhline(y=self.thresr, color='green', linestyle='--', zorder=0)

        #ax.plot([curv["j"] for curv in self.curvature], [curv["c"] for curv in self.curvature], color='red', zorder=1)
        for dp in self.dominant_points:
            ax.plot([dp["sup"][0],dp["j"],dp["sup"][1]], [self.signal[dp["sup"][0]],self.signal[dp["j"]],self.signal[dp["sup"][1]]], marker = 'o', zorder=2)
            #ax.plot([dp["i"],dp["j"],dp["k"]], [self.signal[dp["i"]],self.signal[dp["j"]],self.signal[dp["k"]]], marker = 'o', zorder=2)
            #ax.axvspan(dp["sup"][0], dp["sup"][1], alpha=0.4, color="#c0392b", zorder=0)


        # nplots = len(self.curvature)
        # nrows = np.floor(np.sqrt(nplots))
        # ncols = np.ceil(nplots/nrows)
        # fig, ax = plt.subplots(int(nrows), int(ncols), figsize=(12, 8), dpi=200)
        # ax = ax.flatten()

        # for i, curv in enumerate(self.curvature):
        #     ax[i].plot(self.signal, color='black', zorder=1)
        #     ax[i].plot([curv["i"],curv["j"],curv["k"]], [self.signal[curv["i"]],self.signal[curv["j"]],self.signal[curv["k"]]], marker = 'o', zorder=2)
        #     ax[i].axvline(x=curv["d"][0], color='red', linestyle='--', zorder=0)
        #     ax[i].axvline(x=curv["d"][1], color='red', linestyle='--', zorder=0)
        #     ax[i].axvline(x=curv["s"][0], color='blue', linestyle='-.', zorder=0)
        #     ax[i].axvline(x=curv["s"][1], color='blue', linestyle='-.', zorder=0)


        plt.savefig("beat_"+title+".png")

class Cluster():
    def __init__(self, initial_beat: Beat):
        self.beats = [initial_beat]
        self.template = initial_beat
        self.abnormality = int(initial_beat.abnormal)
        self.p_qrs_ratio = int(initial_beat.p is not None)

    def update_template(self, beat: Beat, template: Beat):
        self.beats.append(beat)
        self.abnormality = np.mean([beat.abnormal for beat in self.beats])
        self.template = template
        self.calc_p_qrs_ratio()

    def identify_abnormality(self, record, otherclusters):
        
        for beat in self.beats:
            
            no_atrial_activity = False
            if beat.p is None:
                no_atrial_activity = True
            elif beat.p is not None and beat.qrs[0] < beat.p[0]:
                no_atrial_activity = True
            elif beat.p is not None and (beat.qrs[0] - beat.p[0]) < 0.075*beat.fs:
                no_atrial_activity = True
            else:
                no_atrial_activity = False

            beat.no_atrial_activity = no_atrial_activity

            #print("Abnormal uncertainty: ", beat.abnormal_uncertainty, "Abnormal logit:", beat.abnormal_logit, "Width: ", beat.width, "No atrial activity: ", beat.no_atrial_activity, "p", beat.p)
            if beat.abnormal_logit > 0.25:
                beat.abnormal = True
            elif beat.abnormal_logit > 0.1 and beat.no_atrial_activity:
                beat.abnormal = True
            elif beat.abnormal_uncertainty > -2.5 and beat.no_atrial_activity:
                beat.abnormal = True
            else:
                beat.abnormal = False
            #print("Abnormal: ", beat.abnormal)


        if np.sum([beat.abnormal for beat in self.beats]) > 1:# and np.mean([beat.width for beat in self.beats]) >= 0.12:
            #print("At least two abnormal beats")
            for beat in self.beats:
                if beat.abnormal_uncertainty > -2.5:
                    #print("Abnormal uncertainty: ", beat.abnormal_uncertainty, "Width: ", beat.width)
                    beat.abnormal = True

        if np.mean([beat.abnormal for beat in self.beats]) >= 0.66:
            #print("More than 75% abnormal beats")
            for beat in self.beats: 
                beat.abnormal = True# if beat.width >= 0.12 else False


        if len(otherclusters) > 0:
            #print("More than one cluster, so now analyse uncertainty")
            meanuncertainty = np.mean([beat.abnormal_uncertainty for beat in self.beats])
            if meanuncertainty > -5:
                self.abnormality = 1
                for beat in self.beats:

                    no_atrial_activity = False
                    if beat.p is None:
                        no_atrial_activity = True
                    elif beat.p is not None and beat.qrs[0] < beat.p[0]:
                        no_atrial_activity = True
                    elif beat.p is not None and (beat.qrs[0] - beat.p[0]) < 0.075*beat.fs:
                        no_atrial_activity = True
                    else:
                        no_atrial_activity = False

                    if beat.abnormal_uncertainty > -5 and beat.width >= 0.12 and no_atrial_activity:
                        beat.abnormal = True
        else:
            #last defence for LBBB and RBBB
            if self.p_qrs_ratio > 0.75:
                for beat in self.beats:
                    beat.abnormal = False
    
    def set_width(self, signal):
        ws = []

        for beat in self.beats:
            beat.get_width_test(signal)
            ws.append(beat.width)

        #print("Widths: ", ws)
        self.width = np.mean(ws)
        #print("Mean width: ", self.width)

    def calc_p_qrs_ratio(self):
        p = 0
        qrs = 0
        for beat in self.beats:
            p += 1 if beat.p is not None else 0
            qrs += 1
        self.p_qrs_ratio = p/qrs

    def set_abnormal(self, abnormal):
        for beat in self.beats:
            beat.abnormal = abnormal

    def get_support(self):
        min_support = np.inf
        max_support = -np.inf

        for dp in self.template.dominant_points:
            min_support = min(min_support, dp["sup"][0])
            max_support = max(max_support, dp["sup"][1])

        return (min_support, max_support)


class Record():
    def __init__(self, ecg: np.ndarray, lateralone: np.ndarray, lateralsix: np.ndarray, fs: int, db: str, recordname: str, delineation = None):
        self.ecg = ecg
        self.lateralone = lateralone
        self.lateralsix = lateralsix
        self.fs = fs
        self.db = db
        self.recordname = recordname
        self.delineation = delineation
        self.pqrsts = []
        self.diagnoses = []
        self.p_wave_groups = []
        self.p_wave_polarity = np.ones(len(ecg))
        self.p_wave_unmatched = np.zeros(len(ecg))
        self.p_wave_group = np.zeros(len(ecg))
        self.average_p_waves = []
        self.beats = []

    def get_pqrst(self):
        return self.pqrsts

    def get_diagnoses(self):
        return self.diagnoses

    def get_beats(self):
        return self.beats

    def get_beats_inside_region(self, onset, offset):
        return [beat for beat in self.beats if beat.onset >= onset and beat.offset <= offset]
    
    def save(self, filename):
        with open(filename, 'wb') as pickle_file:
            # Save only the properties (attributes)
            pickle.dump(self.__dict__, pickle_file)

    # Method to load properties from a pickle file into the object
    def load(self, filename):
        with open(filename, 'rb') as pickle_file:
            # Load the properties (attributes) and update the object's __dict__
            self.__dict__.update(CustomUnpickler(pickle_file).load())

class MorphologicalClustering():
    def __init__(self, debug, cache=True):
        self.debug = debug
        self.cache = cache
        pass

    def preprocess(self, record: Record):

        fs = record.fs

        #fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=200)

        signal = record.ecg.copy()
        assert len(signal.shape) == 1

        #ax.plot(signal, color='black', zorder=1)

        lowcut = 0.2
        highcut = 30
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, high, btype='low')
        signal = filtfilt(b, a, signal)

        ms200 = int(0.2 * fs)
        ms600 = int(0.6 * fs)
        ms200 = ms200 if ms200 % 2 == 1 else ms200 + 1
        ms600 = ms600 if ms600 % 2 == 1 else ms600 + 1
        baseline = sps.medfilt(sps.medfilt(signal, ms200), ms600)
        signal = signal - baseline

        record.bandpass_ecg = signal

        return record

    def plot(self, record: Record):

        fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=200)
        ax.plot(record.bandpass_ecg, color='black', zorder=1)
        for beat in record.beats:
            if beat.abnormal:
                ax.axvspan(beat.onset, beat.offset, color='#f1c40f', alpha=0.5, zorder=0)
            else:
                ax.axvspan(beat.onset, beat.offset, color='#2ecc71', alpha=0.5, zorder=0)
        plt.savefig("clustering.png")

    def identify_beats(self, record: Record):

        self.preprocess(record)

        qrs_normal_logit = record.delineation["qrs"]["logits"]
        qrs_abnormal_logit = record.delineation["qrs_abnormal"]["logits"]
        abnormal_uncertainty = np.log(record.delineation["qrs_abnormal"]["uncertainty"]+0.0001)
        qrs_logit = (qrs_normal_logit+qrs_abnormal_logit)/2
        qrs_binary = qrs_logit > 0.25

        qrs_binary = closingcentered(qrs_binary, np.ones(int(0.075 * record.fs)))

        
        qrs = get_regions(qrs_binary)
        win_min = int(0.1*record.fs)
        win_plus = int(0.2*record.fs)
        beats = []
        prevloc = 0

        for region in qrs:
            mid = (region[0] + region[1])//2
            start = mid-win_min
            end = mid+win_plus

            noise_mask = record.noise_mask[start:end]

            if np.any(noise_mask):
                #print("Noise detected in beat")
                continue

            if start < 0 or end > len(record.bandpass_ecg):
                continue
            
            abnormal_log = np.max(qrs_abnormal_logit[start:end])
            abnormal = abnormal_log > 0.1
            abnormal_uncert = np.max(abnormal_uncertainty[start:end])

            sig = record.bandpass_ecg[start:end]
            oldlength = len(sig)
            sig = resize_signal(sig, 64)

            beat = Beat(start, end, region[0], region[1], sig, abnormal, record.fs)
            beat.abnormal_uncertainty = abnormal_uncert
            beat.abnormal_logit = abnormal_log
            beat.prev_beat = prevloc
            beat.oldlength = oldlength
            prevloc = region[1]
            beats.append(beat)
            #print("Added beat at ", mid/record.fs)

        p_waves = get_regions(record.delineation["p"]["logits"] > 0.01)

        for i in range(len(beats)):
            p_search_start = beats[i].prev_beat+int(0.2*record.fs)
            p_search_end = beats[i].qrs[1]
            candidates = [p for p in p_waves if p[0] > p_search_start and p[1] < p_search_end]
            if len(candidates) == 0:
                continue

            lastcandidate = candidates[-1]
            dist = np.abs(beats[i].qrs[0] - lastcandidate[0])
            #print("Distance to last P-wave: ", dist, "threshold: ", 0.3*record.fs, int(0.075*record.fs))
            if dist <= 0.3*record.fs and dist >= int(0.075*record.fs) and lastcandidate[1] - lastcandidate[0] >= 0.05*record.fs:
                beats[i].p = lastcandidate

        return beats

    def calculate_I_sets(self, q, j, theta, rho_min):
        """
        Calculate I_j^- and I_j^+ for a given index j in the array q.
        
        Parameters:
            q (list or numpy array): The array containing the values of q.
            j (int): The index j for which to compute the sets.
            theta (int): The range parameter.
            rho_min (float): The threshold parameter.
            
        Returns:
            I_minus (list): The set I_j^-.
            I_plus (list): The set I_j^+.
        """
        q = np.array(q)
        n = len(q)
        
        # Function to calculate Δq_j,b
        def delta_q(j, b):
            return np.abs(q[j] - q[b])

        # Calculate I_j^-
        I_minus_candidates = np.arange(max(0, j - theta), j)
        valid_minus = []
        for i in I_minus_candidates:
            indices_a = np.arange(i + 1, j)
            if indices_a.size == 0:  # Skip if no elements in the range
                valid_minus.append(i)
                continue
            delta_q_a = delta_q(j, indices_a)
            delta_q_b = delta_q(j, np.arange(indices_a[0], j)[:, None])
            max_diff = np.max(delta_q_b - delta_q_a[:, None], axis=1)
            if np.all(max_diff < rho_min):
                valid_minus.append(i)

        # Calculate I_j^+
        I_plus_candidates = np.arange(j + 1, min(n, j + theta + 1))
        valid_plus = []
        for k in I_plus_candidates:
            indices_a = np.arange(j + 1, k)
            if indices_a.size == 0:  # Skip if no elements in the range
                valid_plus.append(k)
                continue
            delta_q_a = delta_q(j, indices_a)
            delta_q_b = delta_q(j, np.arange(j + 1, indices_a[-1] + 1)[:, None])
            max_diff = np.max(delta_q_b - delta_q_a[:, None], axis=1)
            if np.all(max_diff < rho_min):
                valid_plus.append(k)

        return valid_minus, valid_plus


        # Calculate I_j^-
        I_minus = []
        for i in range(max(0, j - theta), j):
            valid = True
            for a in range(i + 1, j):  # (i, j)
                max_diff = max([delta_q(j, b) - delta_q(j, a) for b in range(a, j)])  # b ∈ (a, j)
                if max_diff >= rho_min:
                    valid = False
                    break
            if valid:
                I_minus.append(i)

        # Calculate I_j^+
        I_plus = []
        for k in range(j + 1, min(n, j + theta + 1)):
            valid = True
            for a in range(j + 1, k):  # (j, k)
                max_diff = max([delta_q(j, b) - delta_q(j, a) for b in range(j + 1, a + 1)])  # b ∈ (j, a)
                if max_diff >= rho_min:
                    valid = False
                    break
            if valid:
                I_plus.append(k)

        #print(j, I_minus, I_plus)

        return I_minus, I_plus

    def calculate_curvature_and_dominance(self, q, j, theta, rho_min):
        """
        Calculate the curvature K(q_j, q_n) for a 1D time series at index j.
        
        Parameters:
            q (numpy array): A 1D array representing the time series values.
            j (int): The index j for which to calculate the curvature.
            I_minus (list): The set I^-_j containing indices i for the backward interval.
            I_plus (list): The set I^+_j containing indices k for the forward interval.
        
        Returns:
            curvature (float): The curvature K(q_j, q_n).
        """
        def cosine_angle(q, q_i, q_j, q_k):
            """
            Calculate the cosine of the angle formed by q_i, q_j, and q_k.
            """
            # Vectors representing the segments
            v_ij = [q_i - q_j, q[q_i] - q[q_j]]
            v_jk = [q_k - q_j, q[q_k] - q[q_j]]

            dot_product = v_ij[0] * v_jk[0] + v_ij[1] * v_jk[1]

            # Compute the cosine of the angle
            denominator = np.sqrt(v_ij[0] ** 2 + v_ij[1] ** 2) * np.sqrt(v_jk[0] ** 2 + v_jk[1] ** 2)

            if denominator == 0:
                return 1  # Treat as collinear (flat line)

            return dot_product / denominator
        
        I_min_j, I_plus_j = self.calculate_I_sets(q, j, theta, rho_min)
        search_min = np.min(I_min_j)
        search_max = np.max(I_plus_j)

        max_cosine = -np.inf  # Initialize the maximum cosine value
        max_cosine_i = None
        max_cosine_k = None
        
        t = time.time()
        # Iterate over all pairs (i, k) from I^-_j and I^+_j
        for i in I_min_j:
            for k in I_plus_j:
                cos_value = cosine_angle(q, i, j, k)
                # if j == 27:
                #     print(i, j, k, cos_value)
                max_cosine = max(max_cosine, cos_value)
                if max_cosine == cos_value:
                    max_cosine_i = i
                    max_cosine_k = k
        

        v_ij = [max_cosine_i - j, q[max_cosine_i] - q[j]]
        v_jk = [max_cosine_k - j, q[max_cosine_k] - q[j]]

        dominance_min_set = []
        dominance_plus_set = []
        for k in I_plus_j:
            dominance_max_cos = -np.inf
            dominance_max_cos_i = None
            for i in I_min_j:
                cos_value = cosine_angle(q, i, j, k)
                dominance_max_cos = max(dominance_max_cos, cos_value)
                if dominance_max_cos == cos_value:
                    dominance_max_cos_i = i
            dominance_min_set.append(dominance_max_cos_i)

        for i in I_min_j:
            dominance_max_cos = -np.inf
            dominance_max_cos_k = None
            for k in I_plus_j:
                cos_value = cosine_angle(q, i, j, k)
                dominance_max_cos = max(dominance_max_cos, cos_value)
                if dominance_max_cos == cos_value:
                    dominance_max_cos_k = k
            dominance_plus_set.append(dominance_max_cos_k)

        dominance_min = np.min(dominance_min_set)
        dominance_plus = np.max(dominance_plus_set)
        
        return max_cosine, max_cosine_i, max_cosine_k, dominance_min, dominance_plus, search_min, search_max

    def calculate_dominant_points(self, beat: Beat, rho_min):

        dominant_points = []

        for j in range(1,len(beat.signal)-1): 
            start = beat.curvature[j]["d"][0]
            end = beat.curvature[j]["d"][1]
            mid = beat.curvature[j]["j"]
            max_curvature = beat.curvature[j]["c"]
            #print(j, start, end, mid, max_curvature)

            is_dominant = True
            for a in range(start, end):
                if beat.curvature[a]["c"] > max_curvature:
                    is_dominant = False
                    break

            if is_dominant:
                #print(mid, start, end, max_curvature)
                mins = [beat.signal[mid] - beat.signal[start], beat.signal[mid] - beat.signal[end]]
                min_delta_qj_ind = np.argmin(np.abs(mins))
                min_delta_qj = abs(mins[min_delta_qj_ind])

                if min_delta_qj > rho_min:
                    dp = beat.curvature[j]
                    dp["delta"] = min_delta_qj
                    dp["real_delta"] = mins[min_delta_qj_ind]
                    dp["convex"] = (beat.signal[start] > beat.signal[mid] and beat.signal[end] > beat.signal[mid])
                    dominant_points.append(dp)

        beat.dominant_points = dominant_points

        #relevant_points = []

        # for dp in dominant_points:
        #     if dp["delta"] > rho_min*3:
        #         relevant_points.append(dp)

        #beat.dominant_points = relevant_points

        return beat

    def determine_support_region(self, beat: Beat, rho_min):

        q = beat.signal

        # Function to calculate Δq_j,b
        def delta_q(j, b):
            return np.abs(q[j] - q[b])

        for dp in beat.dominant_points:
            r_min = dp["d"][0]
            r_max = dp["d"][1]
            i_min = dp["s"][0]
            i_max = dp["s"][1]
            j = dp["j"]
            j_min = r_min
            j_max = r_max

            for i in range(i_min, r_min):
                valid = True
                for a in range(i, r_min):
                    if delta_q(j, a) < delta_q(j, a+1):
                        valid = False
                        break
                if valid:
                    j_min = i
                    break
            
            for k in range(r_max, i_max+1):
                valid = True
                for a in range(r_max, k+1):
                    if delta_q(j, a-1) > delta_q(j, a):
                        valid = False
                        break
                if valid:
                    j_max = k

            dp["sup"] = [j_min, j_max]

        return beat

    def qrs_characterization(self, beat: Beat, theta, rho_min):

        curvature = [{"i":0,"j":0,"k":0,"c":-1,"d":[0,0], "s":[0,0]}]
        #remove nan values
        beat.signal = beat.signal[~np.isnan(beat.signal)]

        #print("Beat length: ", len(beat.signal))

        #fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=200)
        for j in range(1,len(beat.signal)-1):
            cur, i, k, d_min, d_max, s_min, s_max = self.calculate_curvature_and_dominance(beat.signal, j, theta, rho_min)
            curvature.append({"i": i, "j": j, "k": k, "c": cur, "d": [d_min, d_max], "s": [s_min, s_max]})
        
        curvature.append({"i":len(beat.signal)-1,"j":len(beat.signal)-1,"k":len(beat.signal)-1,"c":-1,"d":[len(beat.signal)-1,len(beat.signal)-1], "s":[len(beat.signal)-1,len(beat.signal)-1]})

        beat.curvature = curvature

        return beat

    def calculate_concordance_ratio(self, sig1, sig2, q_j_min, q_j, q_j_max, qc_j_min, qc_j_max, isconvex, rho_min):

        q_min = sig1[q_j_min]
        q_peak = sig1[q_j]
        q_max = sig1[q_j_max]
        delta_q = min(abs(q_min - q_peak), abs(q_max - q_peak))

        qc_peak_index = qc_j_min + np.argmin(sig2[qc_j_min:qc_j_max]) if isconvex else qc_j_min + np.argmax(sig2[qc_j_min:qc_j_max])
        qc_min_index = qc_j_min + np.argmax(sig2[qc_j_min:qc_peak_index+1]) if isconvex else qc_j_min + np.argmin(sig2[qc_j_min:qc_peak_index+1])
        qc_max_index = qc_peak_index + np.argmax(sig2[qc_peak_index:qc_j_max+1]) if isconvex else qc_peak_index + np.argmin(sig2[qc_peak_index:qc_j_max+1])
        
        qc_peak = sig2[qc_peak_index]
        qc_min = sig2[qc_min_index]
        qc_max = sig2[qc_max_index]
        delta_qc = min(abs(qc_min - qc_peak), abs(qc_max - qc_peak))

        # print("Q: ", q_min, q_peak, q_max)
        # print("Qc: ", qc_min, qc_peak, qc_max)

        # print("Delta q: ", delta_q)
        # print("Delta qc: ", delta_qc)

        #check if concord
        if delta_qc <= rho_min:
            return 0, qc_min_index, qc_peak_index, qc_max_index

        concordance_ratio = min(delta_q, delta_qc) / max(delta_q, delta_qc)

        return concordance_ratio, qc_min_index, qc_peak_index, qc_max_index

    def calculate_local_disimmilarity(self, sig1, sig2, q, qc, q_j_min, q_j, q_j_max, isconvex):

        # fig, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=200)
        # ax[0].plot(q, color='blue', zorder=1)
        # ax[0].plot(qc, color='red', zorder=1)
        # ax[1].plot(q, color='blue', zorder=1)
        # ax[1].plot(qc, color='red', zorder=1)
        # ax[1].axvline(x=q_j_min, color='green', linestyle='--', zorder=0)
        # ax[1].axvline(x=q_j, color='red', linestyle='--', zorder=0)
        # ax[1].axvline(x=q_j_max, color='green', linestyle='--', zorder=0)

        a = np.abs(q - qc)
        median_min = np.median(a[q_j_min:q_j])
        median_plus = np.median(a[q_j:q_j_max])

        # print("Median min: ", median_min)
        # print("Median plus: ", median_plus)

        # ax[0].plot(list(range(q_j_min,q_j)),a[q_j_min:q_j]-median_min, color='black', zorder=1)
        # ax[0].plot(list(range(q_j,q_j_max)),a[q_j:q_j_max]-median_plus, color='black', zorder=1)
        

        delta_Aj_min = 0
        for k in range(q_j_min, q_j+1):
            delta_Aj_min += np.abs(a[k] - median_min)
        delta_Aj_min += (a[q_j_min] + q[q_j])/2# - (q_j-q_j_min)*median_min

        #print("Delta Aj min: ", delta_Aj_min)

        delta_Aj_plus = 0
        for k in range(q_j, q_j_max+1):
            delta_Aj_plus += np.abs(a[k] - median_plus)
        delta_Aj_plus += (a[q_j_max] + q[q_j])/2# - (q_j_max-q_j)*median_plus

        #print("Delta Aj plus: ", delta_Aj_plus)

        A_min = 0
        for k in range(q_j_min, q_j+1):
            A_min += q[k] 
        A_min += (q[q_j_min] + q[q_j])/2 - (q_j-q_j_min) * (np.max(q[q_j_min:q_j+1]) if isconvex else np.min(q[q_j_min:q_j+1]))
        A_min = np.abs(A_min)

        #print("A min: ", A_min)
        # ax[1].axhline(y=np.max(q[q_j_min:q_j+1]) if isconvex else np.min(q[q_j_min:q_j+1]), color='green', linestyle='--', zorder=0)

        A_plus = 0
        for k in range(q_j, q_j_max+1):
            A_plus += q[k] 
        A_plus += (q[q_j] + q[q_j_max])/2 - (q_j_max-q_j) * (np.max(q[q_j:q_j_max+1]) if isconvex else np.min(q[q_j:q_j_max+1]))
        A_plus = np.abs(A_plus)

        # ax[1].axhline(y=np.max(q[q_j:q_j_max+1]) if isconvex else np.min(q[q_j:q_j_max+1]), color='green', linestyle='--', zorder=0)
        #print("A plus: ", A_plus)

        local_dissimilarity = (delta_Aj_min**2 / A_min)
        local_dissimilarity += (delta_Aj_plus**2 / A_plus)
        local_dissimilarity *= 1.0 / (A_min + A_plus)

        # print(delta_Aj_min**2, A_min)
        # print(delta_Aj_plus**2, A_plus)
        # print(1.0 / (A_min + A_plus))

        # plt.savefig("local_dissimilarity.png")

        return local_dissimilarity

    def sigmoid(self, x, alpha=1):
        return 1 - (alpha*x) / np.sqrt(1 + (alpha*x)**2)

    def piecewise_simmilarity(self, beat1, beat2, rho_min, alpha):
        
        sig1 = beat1.signal
        sig2 = beat2.signal
        dsig1 = np.array([sig1[i+1] - sig1[i] for i in range(len(sig1)-1)])
        dsig2 = np.array([sig2[i+1] - sig2[i] for i in range(len(sig2)-1)])

        window_type = "sakoechiba"
        window_size = int(0.1*beat1.fs)
        slope_constraint = rabinerJuangStepPattern(6, "c")

        alignment = dtw(
            dsig1, 
            dsig2, 
            keep_internals=True,
            step_pattern=slope_constraint,
            window_type=window_type, 
            window_args={"window_size":window_size}
        )

        aligned_dsig1 = dsig1[alignment.index1]
        aligned_dsig2 = dsig2[alignment.index2]
        aligned_dsig1 = np.insert(aligned_dsig1, 0, sig1[0])
        aligned_dsig2 = np.insert(aligned_dsig2, 0, sig2[0])

        aligned_sig1 = np.cumsum(aligned_dsig1)
        aligned_sig2 = np.cumsum(aligned_dsig2)

        piecewise_similarity = 0
        nonconcordant_dissimilarities = [0]

        #print(alignment.index1)

        for i, dp in enumerate(beat1.dominant_points):

            #fig, ax = plt.subplots(5, 1, figsize=(8, 8), dpi=200)
            convex = dp["convex"]
            # ax[0].plot(beat1.signal, color='blue', zorder=1)
            # ax[0].plot([dp["sup"][0],dp["j"],dp["sup"][1]], [beat1.signal[dp["sup"][0]],beat1.signal[dp["j"]],beat1.signal[dp["sup"][1]]], marker = 'o', zorder=2)
            
            #print(alignment.index1)
            #print(dp["sup"][0], dp["j"], dp["sup"][1])

            qhat_j = np.max(np.where(alignment.index1 == dp["j"])[0])
            qhat_j_min = np.min(np.where(alignment.index1 == dp["sup"][0])[0])
            qhat_j_max = np.max(np.where(alignment.index1 == dp["sup"][1]-1)[0])
            # ax[1].plot(aligned_sig1, color='blue', zorder=1)
            # ax[1].plot([qhat_j_min,qhat_j,qhat_j_max], [aligned_sig1[qhat_j_min],aligned_sig1[qhat_j],aligned_sig1[qhat_j_max]], marker = 'o', zorder=2)

            # ax[2].plot(beat2.signal, color='red', zorder=1)
            # ax[2].plot([qhat_j_min,qhat_j,qhat_j_max], [aligned_sig2[qhat_j_min],aligned_sig2[qhat_j],aligned_sig2[qhat_j_max]], marker = 'o', zorder=2)
            
            qc_j = alignment.index2[qhat_j]
            qc_j_min = alignment.index2[qhat_j_min]
            qc_j_max = alignment.index2[qhat_j_max]

            concordance_ratio, newq_min, new_q, newq_max = self.calculate_concordance_ratio(beat1.signal, beat2.signal, dp["sup"][0], dp["j"], dp["sup"][1], qc_j_min, qc_j_max, convex, rho_min)
            local_dissimilarity = self.calculate_local_disimmilarity(beat1.signal, beat2.signal, aligned_sig1, aligned_sig2, qhat_j_min, qhat_j, qhat_j_max, convex)
            # print("Concordance ratio: ", concordance_ratio, "for dominant point", dp)
            # print("Local dissimilarity: ", local_dissimilarity, "for dominant point", dp)

            # ax[2].plot([newq_min,new_q,newq_max], [beat2.signal[newq_min],beat2.signal[new_q],beat2.signal[newq_max]], marker = 'o', zorder=2)
            # ax[2].axvline(x=qc_j_min, color='green', linestyle='--', zorder=0)
            # ax[2].axvline(x=qc_j_max, color='green', linestyle='--', zorder=0)

            if local_dissimilarity != 0:
                piecewise_similarity += concordance_ratio * self.sigmoid(local_dissimilarity, alpha=alpha)
                # print("sigmoid", self.sigmoid(local_dissimilarity, alpha=alpha))
            else:
                nonconcordant_dissimilarities.append(local_dissimilarity)

            # ax[3].plot(beat2.signal, color='red', zorder=1)
            # ax[3].plot([qc_j_min,qc_j,qc_j_max], [beat2.signal[qc_j_min],beat2.signal[qc_j],beat2.signal[qc_j_max]], marker = 'o', zorder=2)
            
            # ax[4].plot(beat1.signal, color='blue', zorder=1)
            # ax[4].plot(beat2.signal, color='red', zorder=1)
            # ax[4].plot([dp["sup"][0],dp["j"],dp["sup"][1]], [beat1.signal[dp["sup"][0]],beat1.signal[dp["j"]],beat1.signal[dp["sup"][1]]], marker = 'o', zorder=2)
            # ax[4].plot([qc_j_min,qc_j,qc_j_max], [beat2.signal[qc_j_min],beat2.signal[qc_j],beat2.signal[qc_j_max]], marker = 'o', zorder=2)

            # plt.savefig("dp_"+str(i)+".png")


        piecewise_similarity -= np.max(nonconcordant_dissimilarities)

        return piecewise_similarity

    def compare_two_beats(self, beat1: Beat, beat2: Beat, rho_min, alpha):

        ps_1_to_2 = self.piecewise_simmilarity(beat1, beat2, rho_min, alpha)
        ps_2_to_1 = self.piecewise_simmilarity(beat2, beat1, rho_min, alpha)
        normalized_ps_1_to_2 = ps_1_to_2 / len(beat1.dominant_points)
        normalized_ps_2_to_1 = ps_2_to_1 / len(beat2.dominant_points)

        similarity = (ps_1_to_2 + ps_2_to_1)
        normalized_similarity = similarity / (len(beat1.dominant_points) + len(beat2.dominant_points))


        return similarity, normalized_similarity

    def update_cluster(self, cluster: Cluster, beat: Beat, beta, theta, rho_min, step=0):

        sig1 = beat.signal
        sig2 = cluster.template.signal
        dsig1 = np.array([sig1[i+1] - sig1[i] for i in range(len(sig1)-1)])
        dsig2 = np.array([sig2[i+1] - sig2[i] for i in range(len(sig2)-1)])

        t = time.time()
        window_type = "sakoechiba"
        window_size = int(0.1*beat.fs)
        slope_constraint = rabinerJuangStepPattern(6, "c")

        alignment = dtw(
            dsig1, 
            dsig2, 
            keep_internals=True,
            step_pattern=slope_constraint,
            window_type=window_type, 
            window_args={"window_size":window_size}
        )
        #print("Alignment took", time.time()-t, "seconds")
        t = time.time()

        # fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
        # ax.scatter(alignment.index1, alignment.index2)
        # plt.savefig("alignment.png")

        xs = alignment.index1
        ys = alignment.index2

        aligned_dsig1 = dsig1[xs]
        aligned_dsig2 = dsig2[ys]
        merged_d = ((1 - beta) * aligned_dsig2 + beta * aligned_dsig1)

        # Step 5: Use the average derivative to reconstruct new arrays
        # Align the average derivative to the second array's length
        merged_d = np.interp(
            np.linspace(0, len(merged_d) - 1, len(dsig2)),
            np.arange(len(merged_d)),
            merged_d,
        )

        # Reconstruct array1 and array2 using cumulative integration
        merged_sig = np.cumsum(np.insert(merged_d, 0, sig2[0]))
        #print("Reconstruction took", time.time()-t, "seconds")

        #print(merged_sig)
        newtemplate = Beat(0, len(merged_sig), 0, len(merged_sig), merged_sig, False, beat.fs)
        newtemplate = self.qrs_characterization(newtemplate, theta, rho_min)
        newtemplate = self.calculate_dominant_points(newtemplate, rho_min)
        newtemplate = self.determine_support_region(newtemplate, rho_min)


        # fig, ax = plt.subplots(3, 1, figsize=(8, 8), dpi=200)
        # ax[0].plot(sig1, color='blue', zorder=1)
        # ax[1].plot(sig2, color='red', zorder=1)
        # ax[2].plot(merged_d, color='red', zorder=1)
        # ax[2].plot(merged_sig, color='red', zorder=1)
        # plt.savefig("cluster_update"+str(step)+".png")
        t = time.time()
        if len(newtemplate.dominant_points) == 0:
            cluster.update_template(beat, cluster.template)
        else:
            cluster.update_template(beat, newtemplate)

        return cluster

    def get_median_beat_range(self, beats):

        ranges = []
        for beat in beats:
            minimum = np.nanmin(beat.signal)
            maximum = np.nanmax(beat.signal)
            ranges.append(maximum-minimum)

        median_range = np.median(ranges)

        return median_range

    def get_beats_from_clusters(self, clusters):
        beats = []
        for cluster in clusters:
            beats += cluster.beats

        #sort beats by onset
        beats = sorted(beats, key=lambda x: x.onset)

        return beats

    def process_beat(self, beat: Beat, theta, rho_min, alpha):
        t = time.time()
        beat = self.qrs_characterization(beat,theta, rho_min)
        #print("QRS characterization took", time.time()-t, "seconds")
        t = time.time()
        beat = self.calculate_dominant_points(beat, rho_min)
        #print("Dominant points calculation took", time.time()-t, "seconds")
        t = time.time()
        beat = self.determine_support_region(beat, rho_min)
        #print("Support region determination took", time.time()-t, "seconds")
        t = time.time()
        return beat

    def cluster(self, record: Record):

        if self.cache:
            cachefile = aladin.configuration.aladin_cache_folder+"/clusters_"+record.recordname+".npy"
            if os.path.exists(cachefile):
                print("Loading clusters from cache")
                clusters = np.load(cachefile, allow_pickle=True)
                allbeats = self.identify_beats(record)
                unmatched_beats = []
                beats = self.get_beats_from_clusters(clusters) 
                for beat in allbeats:
                    pos = (beat.qrs[0]+beat.qrs[1])//2
                    if not any([np.abs(pos - (b.qrs[0]+b.qrs[1])/2) < 0.15*record.fs for b in beats]):
                        unmatched_beats.append(beat)

                return clusters, unmatched_beats
                

        theta = int(0.1*record.fs)
        alpha = 4
        beta = 1.0/8.0

        st = time.time()
        clusters = []
        beats = self.identify_beats(record)
        #print("Identified beats in", time.time()-st, "seconds")
        st = time.time()
        #return [], beats

        if len(beats) == 0:
            return [], []
            
        med_range = self.get_median_beat_range(beats)
        rho_min = 0.1*med_range


        beat = self.process_beat(beats[0], theta, rho_min, alpha)
        print("Segmented first beat in", time.time()-st, "seconds")

        clusters.append(Cluster(beat))

        if len(beats) < 2:
            return clusters, [beat]

        print("Calculating points of QRS complexes")
        st = time.time()
        processed_beats = None

        num_cores = max(2,int(np.floor(np.sqrt(len(beats)))))

        with multiprocessing.get_context("spawn").Pool(num_cores) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []

            for i, beat in enumerate(beats[1:]):
                r.append(
                    export_pool.starmap_async(
                        self.process_beat,
                        ((beat, theta, rho_min, alpha),)
                    )
                )
            
            processed_beats = [res.get()[0] for res in r]

        print("Finished processing beats in", time.time()-st, "seconds")
        st = time.time()
        skippedbeats = []

        for i, beat in tqdm(enumerate(processed_beats), total=len(processed_beats)-1):
            
            #beat.plot("beat_"+str(i+1))
            if len(beat.dominant_points) == 0:
                #print("Skipped beat with no dominant points")
                skippedbeats.append(beat)
                continue

            added = False
            best_similarity = 0
            best_normalized_similarity = 0
            best_cluster = None
            for cluster in clusters:
                similarity, normalized_similarity = self.compare_two_beats(cluster.template, beat, rho_min, alpha)
                cluster_pqrs_ratio = cluster.p_qrs_ratio
                beat_has_p = int(beat.p is not None)
                p_ratio_diff = np.abs(cluster_pqrs_ratio - beat_has_p) * 0.3
                normalized_similarity -= p_ratio_diff
                #print("Similarity: ", normalized_similarity, "between beat", i, "and cluster", clusters.index(cluster), "p ratio diff", p_ratio_diff, "beat has p", beat_has_p, "cluster p ratio", cluster.p_qrs_ratio, "beat pos", (beat.qrs[0]+beat.qrs[1])//2)
                if normalized_similarity > best_normalized_similarity:
                    best_similarity = similarity
                    best_normalized_similarity = normalized_similarity
                    best_cluster = cluster


            if best_cluster is not None and best_normalized_similarity > 0.3:
                self.update_cluster(best_cluster, beat, beta, theta, rho_min, step=i)
                added = True

            if not added:
                clusters.append(Cluster(beat))
                

        if self.debug:
            self.plot(record)

        if self.cache:
            np.save(cachefile, clusters)

        
        print("Finished clustering in", time.time()-st, "seconds")

        return clusters, skippedbeats

def proccess(seg, fs):

    L1 = int(0.075*fs)

    SE1 = np.zeros((max(3,L1)))

    seg = np.pad(seg, (L1, L1), 'edge')
    seg = closing(seg, SE1)
    seg = seg[L1:-L1]

    return seg


def regions_to_binary(regions, length):
    binary = np.zeros(length)
    for (st, en) in regions:
        if np.isnan(st) or np.isnan(en):
            continue
        binary[int(st):int(en)] = 1
    return binary

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

def resize_1d_binary(arr):
    # Identify change points
    change_points = np.where(np.diff(arr) != 0)[0] + 1
    
    # Include start and end points
    change_points = np.concatenate(([0], change_points, [len(arr)]))
    
    # Calculate new change points in the resized signal
    new_change_points = np.round(change_points * (target_length / len(arr))).astype(int)
    
    # Ensure the points are within bounds and remove duplicates
    new_change_points = np.unique(np.clip(new_change_points, 0, target_length))
    
    # Reconstruct resized binary signal
    resized_arr = np.zeros(target_length, dtype=int)
    current_value = arr[0]
    
    for i in range(len(new_change_points) - 1):
        start, end = new_change_points[i], new_change_points[i + 1]
        resized_arr[start:end] = current_value
        current_value = 1 - current_value  # Toggle between 0 and 1
    
    return resized_arr

def resize_binary_signal(signal, target_length):
    """
    Resizes a binary signal along the last axis to a specified target length
    by identifying change points, recalculating their new positions, and reconstructing the signal.
    
    Parameters:
    signal (array-like): The original binary signal. Can be 1D, 2D, or higher.
    target_length (int): The desired length of the resized binary signal along the last axis.
    
    Returns:
    np.ndarray: Resized binary signal with the last axis of length `target_length`.
    """
    # Convert input to a numpy array if it's not already
    signal = np.asarray(signal)
    
    # Get the original length of the last axis
    original_length = signal.shape[-1]
    
    # Function to resize a single 1D binary signal
    
    # Apply resizing along the last axis
    if signal.ndim == 1:
        # Handle 1D case
        return resize_1d_binary(signal)
    else:
        # Handle multi-dimensional case, resizing each 1D signal along the last axis
        reshaped_signal = np.apply_along_axis(resize_1d_binary, axis=-1, arr=signal)
        return reshaped_signal

def calculate_uncertainties(rets):
        
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
        heuristic = BALD()

        for t in range(probs.shape[0]):
            uncertainties[t] = heuristic.compute_score(probs[t][None, None, :, :])[0]

        return uncertainties

def calculate_afib_uncertainties(pred):

        #calculate entropy of softmax output
        epsilon = 1e-10
        pred = np.clip(pred, epsilon, 1 - epsilon)

        # Calculate entropy for each prediction
        entropy = -np.sum(pred * np.log2(pred), axis=0)
        return entropy

def undo_resampling(sig, targetfs, recordinfo):
        
        #recordinfo[0] = fs
        #recordinfo[1] = original_length
        #recordinfo[2] = before_padding

        if recordinfo[0] != targetfs:
            newlength = int(recordinfo[1] * recordinfo[0] / targetfs)
            sig = resize_signal(sig[:, :recordinfo[2]], newlength)
            if sig.shape[-1] > recordinfo[1]:
                sig = sig[:,:recordinfo[1]]
            elif sig.shape[-1] < recordinfo[1]:
                #pad only last dimension
                sig = np.pad(sig, ((0,0),(0, recordinfo[1] - sig.shape[-1])), 'constant', constant_values=(0, 0))

                #sig = np.pad(sig, (0, record.original_length - sig.shape[-1]), 'constant', constant_values=(0, 0))
    
        return sig

def process_segmentation(args): #sliding_rets, fullcontext_rets, thresholds, recordinfo, targetfs, length=6144, pbar=None):
        
        sliding_rets, fullcontext_rets, thresholds, recordinfo, targetfs, length = args

        avg_afib_logits = np.zeros((2, length))
        avg_afib_softmax = np.zeros((2, length))
        avg_inverse_logits = np.zeros((2, length))
        avg_logits = np.zeros((6, length))
        uncertainties = calculate_uncertainties(sliding_rets + fullcontext_rets)

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

        if recordinfo[0] != targetfs:
            avg_logits = undo_resampling(avg_logits, targetfs, recordinfo)
            avg_afib_logits = undo_resampling(avg_afib_logits, targetfs, recordinfo)
            avg_afib_softmax = undo_resampling(avg_afib_softmax, targetfs, recordinfo)
            avg_inverse_logits = undo_resampling(avg_inverse_logits, targetfs, recordinfo)
            uncertainties = undo_resampling(uncertainties, targetfs, recordinfo)

        p_hat = np.array(avg_logits[1] > thresholds[1], dtype=bool)
        qrs_hat = np.array(avg_logits[2] > thresholds[2], dtype=bool)
        t_hat = np.array(avg_logits[3] > sthresholds[3], dtype=bool)
        noise_hat = np.array(avg_logits[4] > thresholds[4], dtype=bool)
        abnormal_qrs_hat = np.array(avg_logits[5] > thresholds[5], dtype=bool)
        afib_hat = np.array(avg_afib_softmax.argmax(0) == 1, dtype=bool)

        uncertainties_afib = calculate_afib_uncertainties(avg_afib_logits)

        p_wave_delineation = cpp_backend.Delineation(avg_logits[1], uncertainties[1],  p_hat)
        qrs_delineation = cpp_backend.Delineation(avg_logits[2], uncertainties[2], qrs_hat)
        t_wave_delineation = cpp_backend.Delineation(avg_logits[3], uncertainties[3], t_hat)
        noise_delineation = cpp_backend.Delineation(avg_logits[4], uncertainties[4], noise_hat)
        abnormal_qrs_delineation = cpp_backend.Delineation(avg_logits[5], uncertainties[5], abnormal_qrs_hat)
        afib_delineation = cpp_backend.Delineation(avg_afib_logits[1], uncertainties_afib, afib_hat)
        
        #record.cpp_record.delineations = cpp_backend.Delineations(p_wave_delineation, qrs_delineation, abnormal_qrs_delineation, t_wave_delineation, noise_delineation, afib_delineation)
        #print(record.cpp_record.delineations.p.size)

        # if pbar is not None:
        #     pbar.update(1)

        if np.mean(np.array(avg_inverse_logits.argmax(0)) > 0.75):
            return False

        return p_wave_delineation, qrs_delineation, abnormal_qrs_delineation, t_wave_delineation, noise_delineation, afib_delineation