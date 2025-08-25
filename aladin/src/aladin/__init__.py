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
    def __init__(self, modelpaths=[], debug={}, usecache=False, embed=False):

        self.debug = {
            "preprocessor": False,
            "segmenter": False,
            "afibdetector": False,
            "reflection": False,
            "total": False,
            "logic": False
        }

        self.init_debug(debug)

        self.segmenter = UNetSegmenter(modelpaths, debug = self.debug["segmenter"], cache = usecache, embed=embed)
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

    def analyse_batch(self, records: list, arrhythmia=None):
        print("Analyse batch", len(records))

        st = time.time()
        self.segmenter.batch(records)
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


    def analyse(self, record: Record):
        print("Analyse record", record.recordname)

        st = time.time()
        self.segmenter.segment(record)
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

    def embed(self, record: Record):
        self.segmenter.segment(record)
        self.reflection.reflect(record)
        features = self.embedder.embed(record)

        if self.debug["total"]:
            self.plot(record)
            
        return features

    def analyse_noise(self, record: Record):
        print("Analyse noise", record.recordname)

        record = self.segmenter.segment(record)
        record = self.reflection.noise_module.correct(record)
        pc_of_noise = self.embedder.get_percentage_noise(record)

        return record
