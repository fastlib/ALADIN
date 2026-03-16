import numpy as np
import time
import os

from aladin.core import Record
from aladin.backend.segmenter import UNetSegmenter
from aladin.selfreflection.reflection import Reflection
from aladin.logicengine.logic import LogicEngine
from aladin.logicengine.embed import Embedder

from aladin.utils.helpers import Cluster, Beat, resize_signal, get_regions, MorphologicalClustering
from aladin.utils.morphological import closingcentered, openingcentered
from scipy.signal import butter, lfilter, filtfilt
import scipy.signal as sps

import multiprocessing
from tqdm import tqdm

import matplotlib.pyplot as plt

def run_logic_thread(record: Record):
    logic = LogicEngine(debug = False)
    logic.diagnose(record)
    return record

class ALADIN():
    def __init__(self, modelpaths=[], debug={}, usefullcontext=True, usecache=False, embed=False):

        self.debug = {
            "preprocessor": False,
            "segmenter": False,
            "afibdetector": False,
            "reflection": False,
            "total": False,
            "logic": False
        }

        self.init_debug(debug)

        self.segmenter = UNetSegmenter(modelpaths, debug = self.debug["segmenter"], usefullcontext=usefullcontext, cache=usecache, embed=embed)
        self.reflection = Reflection(debug = self.debug["reflection"])
        self.logic = LogicEngine(debug = self.debug["logic"])
        self.embedder = Embedder(debug = self.debug["logic"])

    def init_debug(self, debug):
        for key in debug:
            if key in self.debug:
                self.debug[key] = debug[key]

    def plot(self, record: Record, name="", xlim=None, annotations=None):

        if xlim is None:
            xlim = [0, len(record.ecg) / record.fs]
        
        dsig = np.diff(record.filtered_ecg)
        #dsig = closingcentered(dsig, np.ones(5))
        #find outliers in dsig
        interquartile_range = np.percentile(dsig, 95) - np.percentile(dsig, 5)
        lower_bound = np.percentile(dsig, 5) - 1.5 * interquartile_range
        upper_bound = np.percentile(dsig, 95) + 1.5 * interquartile_range
        sig = record.filtered_ecg[int(xlim[0]*record.fs):int(xlim[1]*record.fs)]
        dsig = np.diff(sig)
        #median filter of 3 samples
        dsig = np.where(np.abs(dsig) > upper_bound, 0, dsig)
        #calculate signal from dsig
        fig, ax = plt.subplots(1, 1, figsize=(len(sig)/(record.fs), 4), dpi=200)
        ax.plot(np.arange(xlim[0]*record.fs, xlim[1]*record.fs), sig, color='black', label="ECG Signal")
        #ax.plot(np.arange(xlim[0]*record.fs, xlim[1]*record.fs)[1:], dsig, color='red', label="Derivative of ECG Signal")
        #ax.axhline(upper_bound, color='blue', linestyle='--', label="Upper Bound")
        #ax.axhline(lower_bound, color='blue', linestyle='--', label="Lower Bound")
        ax.set_xticks(np.arange(xlim[0]*record.fs, xlim[1]*record.fs, record.fs*2))
        ax.set_xticklabels(np.arange(xlim[0], xlim[1], 2))
        #ax.set_ylim(-2,2)

        pcolor = '#e74c3c'
        tcolor = '#3498DB'

        ymax = 0.9
        ymin = 0

        for beat in record.qrs:
            if beat.onset > xlim[1]*record.fs or beat.offset < xlim[0]*record.fs:
                continue

            qrscolor = '#f1c40f' if beat.abnormal else '#2ecc71'
            
            ax.axvspan(beat.onset, beat.offset, ymax = ymax, ymin = ymin, color=qrscolor, alpha=0.5)
            if beat.t is not None:
                ax.axvspan(beat.t.onset, beat.t.offset, ymax = ymax, ymin = ymin, color=tcolor, alpha=0.5)
            if beat.junctional:
                ax.scatter((beat.onset+beat.offset)/2, 0.5, color="purple", s=30, marker='o')

        for wave in record.p:
            if wave.onset > xlim[1]*record.fs or wave.offset < xlim[0]*record.fs:
                continue

            ax.axvspan(wave.onset, wave.offset, ymax = ymax, ymin = ymin, color=pcolor, alpha=0.5)
            if wave.unmatched:
                ax.scatter((wave.onset+wave.offset)/2, 0.5, color=pcolor, s=30, marker='s')

        afib_regions = get_regions(record.delineations.afib.binary)
        for region in afib_regions:

            if region[0] > xlim[1]*record.fs or region[1]< xlim[0]*record.fs:
                continue

            region_st = max(region[0], xlim[0]*record.fs)
            region_en = min(region[1], xlim[1]*record.fs)

            ax.axvspan(region_st, region_en, ymax = ymax, ymin = ymin, color='#8e44ad', alpha=0.4)


        # for diagnosis in record.diagnoses:
        #     x0 = diagnosis["onset"]
        #     x1 = diagnosis["offset"]
        #     y0 = np.max(record.normalized_ecg) * 1.1
        #     y1 = np.max(record.normalized_ecg) * 1.1
        #     dy = np.max(record.normalized_ecg) * 0.05

        #     ax.plot([x0, x1], [y0, y1], color='black', linewidth=1)
        #     ax.plot([x0, x0], [y0-dy, y1+dy], color='black', linewidth=1)
        #     ax.plot([x1, x1], [y0-dy, y1+dy], color='black', linewidth=1)
        #     ax.text((x0+x1)/2, y0+dy, diagnosis["type"], fontsize=8, ha='center', va='bottom', color='black')

        # qrs_module = self.reflection.qrs_module
        # for ic, cluster in enumerate(qrs_module.clusters):
        #     for beat in cluster.beats:
        #         ax[1].text((beat.qrs[0]+beat.qrs[1])/2, 0, str(ic), fontsize=10)

        # for pg, group in enumerate(record.p_wave_groups):
        #     for pwave in group:
        #         print("P-wave", pwave)
        #         ax[1].text((pwave["onset"]+pwave["offset"])/2, 0.1, str(pg), fontsize=10)

        # for beat in record.beats:
        #     ax[2].scatter(beat.qrs[0], (beat.snr), color='r', s=10)
        #     #ax[2].scatter(beat.qrs[0], beat.pr, color='r', s=10)
        # ax[2].axhline(y=20, color='r', linestyle='--')
        #ax[2].axhline(y=100, color='r', linestyle='--')

        noise_regions = get_regions(record.delineations.noise.binary)
        for region in noise_regions:
            ax.axvspan(region[0], region[1], color='#908894', alpha=0.4)

        if annotations is not None:
            for annotation in annotations:
                ax.text(annotation[1], ymax, annotation[0], fontsize=8, color="black", ha='center', va='center')

        ax.set_xlim(xlim[0]*record.fs, xlim[1]*record.fs)
        #ax.set_ylim(-2,2)

        ax.set_ylabel("Amplitude (mV)")
        ax.set_xlabel("Time (s)")
        #hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        if name == "":
            plt.savefig("aladin_result.png")
        else:
            plt.savefig(name + ".png")
        plt.close()

    def analyse_batch(self, records: list, arrhythmia=None, preprocess=True):
        print("Analyse batch", len(records))

        st = time.time()
        self.segmenter.batch(records, preprocess=preprocess)
        segmenttime = time.time()-st 
        st = time.time()
        self.reflection.batch(records)
        reflecttime = time.time()-st
        st = time.time()

        for record in records:
            logic = LogicEngine(debug = False, customarrhythmia=arrhythmia)
            logic.diagnose(record)
            logic.print_diagnoses(record)

        logictime = time.time()-st
        print("Segmenter", segmenttime)
        print("Reflection", reflecttime)
        print("Logic", logictime)

    def analyse(self, record: Record, preprocess=True):
        print("Analyse record", record.recordname)

        st = time.time()
        self.segmenter.segment(record, preprocess=preprocess)
        print("Segmenter", time.time()-st)
        st = time.time()
        self.reflection.reflect(record)
        print("Reflection", time.time()-st)
        st = time.time()
        self.logic.diagnose(record)
        print("Logic", time.time()-st)

        self.logic.print_diagnoses(record)

        self.reflection.reset()
        
        if self.debug["total"]:
            self.plot(record)
            full_explanation = self.logic.get_full_explanation(record)
            with open("aladin_explanation.txt", "w") as f:
                f.write(full_explanation)

        return record

    def calculate_median(self, record: Record, time_before_peak=0.4, time_after_peak=0.6, gaussian_time=0.1):
        
        median_beat = np.zeros((len(record.available_lead_names),int(record.fs)))
        rpeak_before = int(time_before_peak*record.fs)
        rpeak_after = int(time_after_peak*record.fs)
        smooth_interval = int(gaussian_time*record.fs)

        cluster_ids = [qrs.cluster_id for qrs in record.qrs]
        largest_cluster_id = max(set(cluster_ids), key=cluster_ids.count)
        largest_cluster_size = cluster_ids.count(largest_cluster_id)

        for beat in record.qrs:
            if beat.cluster_id != largest_cluster_id:
                continue

            onset = beat.r - rpeak_before
            offset = beat.r + rpeak_after

            if beat.p is not None:
                onset = beat.p.onset-smooth_interval
                #print("P-wave onset", beat.p.onset)
            if beat.t is not None:
                offset = beat.t.offset+smooth_interval
                #print("T-wave offset", beat.t.offset)

            if onset < 0 or offset > len(record.filtered_ecg):
                continue

            i = 0
            for lead_idx, available in enumerate(record.available_leads):
                if not available:
                    continue

                pqrst = record.normalized_ecg[lead_idx, onset:offset]
                gaussian_mask = np.zeros_like(pqrst)
                gaussian_mask[:smooth_interval*2] = sps.windows.gaussian(smooth_interval*2, std=smooth_interval/3)
                gaussian_mask[-smooth_interval*2:] = sps.windows.gaussian(smooth_interval*2, std=smooth_interval/3)[::-1]
                gaussian_mask[(smooth_interval):(len(pqrst)-smooth_interval)] = 1

                pqrst = pqrst * gaussian_mask

                before = beat.r - onset
                after = offset - beat.r
                preprend = max(0, rpeak_before-before)
                append = max(0, rpeak_after-after)

                pqrst = np.pad(pqrst, (preprend, append), mode='constant', constant_values=0)
                pqrst = pqrst[:int(record.fs)]

                median_beat[i,:] += pqrst
                i+=1
        
        for i in range(median_beat.shape[0]):
            median_beat /= sum([1 for beat in record.qrs if beat.cluster_id == largest_cluster_id])

        record.median_beat = median_beat

    def extract_median_beat(self, record: Record, time_before_peak=0.4, time_after_peak=0.6, gaussian_time=0.1, preprocess=True):
        print("Analyse record", record.recordname)

        self.segmenter.segment(record, preprocess=preprocess)
        print("self reflect")
        self.reflection.reflect(record)
        print("calculate median")
        self.calculate_median(record, time_before_peak, time_after_peak, gaussian_time)
        
        print("Median beat extracted")

    def extract_median_beat_batch(self, records: list, time_before_peak=0.4, time_after_peak=0.6, gaussian_time=0.1, preprocess=True):
        print("Extract median beat batch", len(records))

        self.segmenter.batch(records, preprocess=preprocess)
        self.reflection.batch(records)

        i=0
        for record in tqdm(records, desc="Extracting median beats"):
            print(i)
            try:
                self.calculate_median(record, time_before_peak, time_after_peak, gaussian_time)
            except Exception as e:
                record.median_beat = np.zeros((len(record.available_lead_names),int(record.fs)))
                print(f"Error processing record {record.recordname}: {e}")
            i+=1
        
        print("Median beat extracted for batch")

    def segment(self, record: Record, preprocess=True):
        print("Segment record", record.recordname)

        st = time.time()
        self.segmenter.segment(record, preprocess=preprocess)
        print("Segmenter", time.time()-st)
        st = time.time()
        self.reflection.reflect(record)
        print("Reflection", time.time()-st)

        return record

    def segment_batch(self, records: list, preprocess=True):
        print("Segment batch", len(records))

        st = time.time()
        self.segmenter.batch(records, preprocess=preprocess)
        print("Segmenter", time.time()-st)
        st = time.time()
        self.reflection.batch(records)
        print("Reflection", time.time()-st)

        return records

    def embed(self, record: Record, preprocess=True):
        self.segmenter.segment(record, preprocess=preprocess)
        self.reflection.reflect(record)
        features = self.embedder.embed(record)

        if self.debug["total"]:
            self.plot(record)
            
        return features

    def analyse_noise(self, record: Record, preprocess=True):
        print("Analyse noise", record.recordname)

        record = self.segmenter.segment(record, preprocess=preprocess)
        record = self.reflection.noise_module.correct(record)
        pc_of_noise = self.embedder.get_percentage_noise(record)

        return record
