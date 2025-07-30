import numpy as np
import os

from aladin.selfreflection.base import ReflectionBase

from aladin.utils.helpers import Cluster, Beat, Record, resize_signal, get_regions, MorphologicalClustering
from aladin.utils.morphological import closingcentered, openingcentered
from scipy.signal import butter, lfilter, filtfilt
import scipy.signal as sps
from dtw import *
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

class QRSReflection(ReflectionBase):
    def __init__(self, debug=False, cache=False):
        print("QRSReflection module initialized, " + ("debug mode" if debug else "production mode"))

        self.debug = debug
        self.clusters = []
        self.clusterer = MorphologicalClustering(debug=self.debug, cache=cache)

    def match_t_waves(self, record: Record):
        
        if len(record.beats) < 2:
            return record

        t = get_regions(record.delineation["t"]["binary"])

        for i in range(len(record.beats)-1):
            search = (record.beats[i].qrs[0], record.beats[i+1].qrs[1])

            for j in range(len(t)):
                if t[j][0] > search[0] and t[j][1] < search[1]:
                    record.beats[i].t = t[j]
                    break
        
        #match last T-wave
        search = (record.beats[-1].qrs[0], len(record.filtered_ecg))
        for j in range(len(t)):
            if t[j][0] > search[0] and t[j][1] < search[1]:
                record.beats[-1].t = t[j]
                break
            
        return record

    def set_rr(self, record: Record):

        beats = record.get_beats()
        for i in range(len(beats)):
            beats[i].rr = np.nan
            beats[i].rr_raw = np.nan

        for i in range(1,len(beats)):
            if beats[i].abnormal is False:# and beats[i-1].abnormal is False:
                rr = (beats[i].get_r_wave()-beats[i-1].get_r_wave())/record.fs
            else:
                rr = np.nan

            last_good_rr = np.nan
            k = i
            while np.isnan(last_good_rr) and k > 0:
                k -= 1
                last_good_rr = beats[k].rr

            beats[i].rr = rr
            beats[i].rr_raw = (beats[i].get_r_wave()-beats[i-1].get_r_wave())/record.fs

        if len(beats) > 1 and not np.isnan(beats[1].rr):
            beats[0].rr = beats[1].rr

        if len(beats) > 1:  
            beats[0].rr_raw = beats[1].rr_raw

        for i in range(len(beats)):
            beats[i].rr_smooth = beats[i].rr

        #smooth RR intervals by averaging with neighbors
        for i in range(1,len(beats)-1):
            #if current RR is NaN, but neighbors are not
            if np.isnan(beats[i].rr_smooth) and not np.isnan(beats[i-1].rr_smooth) and not np.isnan(beats[i+1].rr_smooth):
                beats[i].rr_smooth = (beats[i-1].rr_smooth + beats[i+1].rr_smooth)/2
            #if current RR is not NaN
            elif not np.isnan(beats[i].rr_smooth):
                if np.isnan(beats[i-1].rr_smooth) and np.isnan(beats[i+1].rr_smooth):
                    beats[i].rr_smooth = beats[i].rr_smooth
                #if only left neighbor is NaN
                elif np.isnan(beats[i-1].rr_smooth) and not np.isnan(beats[i+1].rr_smooth):
                    beats[i].rr_smooth = (beats[i+1].rr_smooth + beats[i].rr_smooth)/2
                #if only right neighbor is NaN
                elif np.isnan(beats[i+1].rr_smooth) and not np.isnan(beats[i-1].rr_smooth):
                    beats[i].rr_smooth = (beats[i-1].rr_smooth + beats[i].rr_smooth)/2
                else:
                    beats[i].rr_smooth = (beats[i-1].rr_smooth + beats[i+1].rr_smooth + beats[i].rr_smooth)/3
            else:
                beats[i].rr_smooth = beats[i].rr_smooth
        
            

        return record

    def reflect(self, record):
        print("Reflecting on QRS complexes in record", record.recordname)

        st = time.time()

        self.clusters, unmatchedbeats = self.clusterer.cluster(record)
        print("Clustering", time.time()-st)
        st = time.time()

        for i, cluster in enumerate(self.clusters):
            cluster.set_width(record.bandpass_signal)
            otherclusters = self.clusters[:i]
            otherclusters = np.concatenate((otherclusters,self.clusters[i+1:]))
            cluster.identify_abnormality(record, otherclusters)

        print("Identifying abnormality", time.time()-st)
        st = time.time()

        beats = self.clusterer.get_beats_from_clusters(self.clusters) + unmatchedbeats

        beats = sorted(beats, key=lambda x: x.get_r_wave())
        record.beats = beats
        record = self.match_t_waves(record)
        record = self.set_rr(record)

        print("Setting RR intervals", time.time()-st)

        if self.debug:
            self.clusterer.plot(record)

        return record