import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import aladin._main as cpp_backend
from aladin.selfreflection.pwave import PWaveReflection

def reflect_on_record(record, debug=False):
    """
    Reflect on a record and return the record with reflection results.
    """
    reflection = Reflection(debug=debug)
    reflection.initialize(record)
    reflection.reflect_on_noise()
    reflection.reflect_on_qrs()
    reflection.reflect_on_afib()
    reflection.reflect_on_p_waves()
    
    return record

class Reflection():
    def __init__(self, debug=False, cache=False):
        print("Reflection module initialized")

        self.cpp_reflection = cpp_backend.Reflection()
        self.pwave_reflection = PWaveReflection(debug=debug)
        self.record = None
        self.debug = debug
        self.plot = False
        pass
    
    def initialize(self, record):
        if self.debug: print("Initialize reflection module")
        st = time.time()
        self.cpp_reflection.initialize(record.cpp_record)
        self.record = record
        if self.debug: print("Initialize", time.time()-st)

    def reflect_on_noise(self):
        st = time.time()
        #find noise
        self.cpp_reflection.reflect_on_noise()
        if self.debug: print("Noise", time.time()-st)

    def reflect_on_qrs(self):
        st = time.time()
        #find and cluster QRS complexes
        self.cpp_reflection.reflect_on_qrs()
        if self.debug: print("QRS", time.time()-st)

        if self.debug and self.plot:
            #plot QRS beats
            nbeats = self.cpp_reflection.number_of_qrs_beats
            if nbeats == 0:
                print("No QRS beats found")
                return

            nrow = int(np.floor(np.sqrt(nbeats)))
            ncol = int(np.ceil(nbeats/nrow))

            fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4), sharex=True, dpi=200)
            ax = ax.flatten()
            for i in range(nbeats):
                ax[i].plot(self.cpp_reflection.get_qrs_beat(i).ecg, label="Cluster " + str(i))
                for j in range(self.cpp_reflection.get_qrs_beat(i).number_of_dominant_points):
                    midpoint = self.cpp_reflection.get_qrs_beat(i).get_dominant_point(j).j
                    support = self.cpp_reflection.get_qrs_beat(i).get_dominant_point(j).support
                    xs = np.array([support[0], midpoint, support[1]])
                    ax[i].plot(xs, self.cpp_reflection.get_qrs_beat(i).ecg[xs], color='green', label="Dominant Point")
                    ax[i].scatter(xs, self.cpp_reflection.get_qrs_beat(i).ecg[xs], color='green', label="Dominant Point")
                peak = self.cpp_reflection.get_qrs_beat(i).peak
                ax[i].scatter(peak, self.cpp_reflection.get_qrs_beat(i).ecg[peak], color='red', label="Peak")
                # ax[i].scatter([sup_start, sup_end], [self.cpp_reflection.get_qrs_cluster(i).template.ecg[int(sup_start)], self.cpp_reflection.get_qrs_cluster(i).template.ecg[int(sup_end)]], color='red', label="Support Region")
                # ax[i].set_title("Cluster " + str((sup_end-sup_start)/self.record.fs) + "samples")
                # ax[i].set_xlabel("Time (s)")
                # ax[i].set_ylabel("Amplitude (mV)")
            plt.show()
            plt.savefig("beats.png")

            nbeats = self.cpp_reflection.number_of_qrs_beats
            fig, ax = plt.subplots(2, 1, figsize=(20, 4), dpi=200)
            ax[0].plot(self.record.filtered_ecg, label="Filtered ECG")
            ax[0].set_xlim(0, len(self.record.filtered_ecg))
            for i in range(nbeats):
                ax[1].plot(self.cpp_reflection.get_qrs_beat(i).r, self.cpp_reflection.get_qrs_beat(i).rr, 'ro')
            ax[1].set_xlim(0, len(self.record.filtered_ecg))

            plt.show()
            plt.savefig("rr.png")

            #plot QRS clusters
            nclusters = self.cpp_reflection.number_of_qrs_clusters
            print("Number of clusters", nclusters)
            fig, ax = plt.subplots(1, max(2,nclusters), figsize=(nclusters*4, 4), sharex=True, dpi=200)
            for i in range(nclusters):
                ax[i].plot(self.cpp_reflection.get_qrs_cluster(i).template.ecg, label="Cluster " + str(i))
                sup_start = self.cpp_reflection.get_qrs_cluster(i).wave_onset
                sup_end = self.cpp_reflection.get_qrs_cluster(i).wave_offset
                for j in range(self.cpp_reflection.get_qrs_cluster(i).template.number_of_dominant_points):
                    midpoint = self.cpp_reflection.get_qrs_cluster(i).template.get_dominant_point(j).j
                    support = self.cpp_reflection.get_qrs_cluster(i).template.get_dominant_point(j).support
                    xs = np.array([support[0], midpoint, support[1]])
                    ax[i].plot(xs, self.cpp_reflection.get_qrs_cluster(i).template.ecg[xs], color='green', label="Dominant Point")
                    ax[i].scatter(xs, self.cpp_reflection.get_qrs_cluster(i).template.ecg[xs], color='green', label="Dominant Point")
            plt.show()
            plt.savefig("clusters.png")

            fig, ax = plt.subplots(1, 1,figsize=(20, 4), dpi=200)
            #ax.plot(self.record.filtered_ecg, label="Filtered ECG")
            ax.plot(self.record.ecg_no_qrst, label="ECG without QRST")
            ax.set_xlim(0, len(self.record.filtered_ecg))
            plt.show()
            plt.savefig("ecg_no_qrst.png")

    def reflect_on_afib(self):
        st = time.time()
        self.cpp_reflection.reflect_on_afib()
        if self.debug: print("AFIB", time.time()-st)

    def reflect_on_p_waves(self):
        st = time.time()
        self.cpp_reflection.reflect_on_p_waves()
        #self.pwave_reflection.reflect(self.record)
        if self.debug: print("PWave", time.time()-st)

        if self.debug and self.plot:
            #plot P waves
            nbeats = self.cpp_reflection.number_of_p_beats
            if nbeats == 0:
                print("No P beats found")
                return
            nrow = int(np.floor(np.sqrt(nbeats)))
            ncol = int(np.ceil(nbeats/nrow))

            fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4), sharex=True, dpi=200)
            ax = ax.flatten()
            for i in range(nbeats):
                ax[i].plot(self.cpp_reflection.get_p_beat(i).ecg, label="Cluster " + str(i))
                for j in range(self.cpp_reflection.get_p_beat(i).number_of_dominant_points):
                    midpoint = self.cpp_reflection.get_p_beat(i).get_dominant_point(j).j
                    support = self.cpp_reflection.get_p_beat(i).get_dominant_point(j).support
                    xs = np.array([support[0], midpoint, support[1]])
                    ax[i].plot(xs, self.cpp_reflection.get_p_beat(i).ecg[xs], color='green', label="Dominant Point")
                    ax[i].scatter(xs, self.cpp_reflection.get_p_beat(i).ecg[xs], color='green', label="Dominant Point")
                peak = self.cpp_reflection.get_p_beat(i).peak
                ax[i].scatter(peak, self.cpp_reflection.get_p_beat(i).ecg[peak], color='red', label="Peak")
                # ax[i].scatter([sup_start, sup_end], [self.cpp_reflection.get_qrs_cluster(i).template.ecg[int(sup_start)], self.cpp_reflection.get_qrs_cluster(i).template.ecg[int(sup_end)]], color='red', label="Support Region")
                # ax[i].set_title("Cluster " + str((sup_end-sup_start)/self.record.fs) + "samples")
                # ax[i].set_xlabel("Time (s)")
                # ax[i].set_ylabel("Amplitude (mV)")
            plt.show()
            plt.savefig("p_waves.png")

            #plot P clusters
            nclusters = self.cpp_reflection.number_of_p_clusters
            #print("Number of P clusters", nclusters)
            fig, ax = plt.subplots(1, max(2,nclusters), figsize=(nclusters*4, 4), sharex=True, dpi=200)
            for i in range(nclusters):
                ax[i].plot(self.cpp_reflection.get_p_cluster(i).template.ecg, label="Cluster " + str(i))
                sup_start = self.cpp_reflection.get_p_cluster(i).wave_onset
                sup_end = self.cpp_reflection.get_p_cluster(i).wave_offset
                for j in range(self.cpp_reflection.get_p_cluster(i).template.number_of_dominant_points):
                    midpoint = self.cpp_reflection.get_p_cluster(i).template.get_dominant_point(j).j
                    support = self.cpp_reflection.get_p_cluster(i).template.get_dominant_point(j).support
                    xs = np.array([support[0], midpoint, support[1]])
                    ax[i].plot(xs, self.cpp_reflection.get_p_cluster(i).template.ecg[xs], color='green', label="Dominant Point")
                    ax[i].scatter(xs, self.cpp_reflection.get_p_cluster(i).template.ecg[xs], color='green', label="Dominant Point")
            plt.show()
            plt.savefig("pclusters.png")
    
    def reset(self):
        if self.debug: print("Reset reflection module")
        self.cpp_reflection.reset()
        self.record = None

    def reflect(self, record):
        if self.debug: print("Self-reflect on record", record.recordname)

        print(record.recordname)
        t0 = time.time()
        self.initialize(record)
        st = time.time()
        if self.debug: print("Initialize time", time.time()-t0)
        self.reflect_on_noise()
        st = time.time()
        if self.debug: print("Noise time", time.time()-st)
        self.reflect_on_qrs()
        st = time.time()
        if self.debug: print("QRS time", time.time()-st)
        self.reflect_on_afib()
        st = time.time()
        if self.debug: print("AFIB time", time.time()-st)
        self.reflect_on_p_waves()

        if self.debug: print("Reflection time", time.time()-t0)

        self.cpp_reflection.reset()

        return record

    def batch(self, records):
        #print("Self-reflect on batch", len(records))
        
        def update_progress(progress):
            return

        batch_reflection = cpp_backend.BatchReflection()
        batch_reflection.reflect_on_batch([record.cpp_record for record in records], update_progress)


        # for record in tqdm(records):
        #     self.reflect(record)

        # with ProcessPool(nodes=4) as export_pool:
        #     # map preserves order and is equivalent to list comprehension
        #     ret = list(tqdm(export_pool.map(reflect_on_record, records), total=len(records)))

        # with multiprocessing.get_context("spawn").Pool(32) as export_pool:
        #     r = []
        #     for record in tqdm(records):
        #         r.append(
        #             export_pool.starmap_async(
        #                 reflect_on_record,
        #                 ((record),)
        #             )
        #         )
            

        #     ret = [i.get()[0] for i in r]

        return records