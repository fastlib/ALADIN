import numpy as np
import matplotlib.pyplot as plt
import pickle

from aladin.selfreflection.base import ReflectionBase
from aladin.selfreflection.qrscomplex import Beat

from aladin.utils.helpers import Record, resize_signal, get_regions, regions_to_binary
from aladin.utils.morphological import closingcentered, openingcentered, dilationcentered
import re

class AfibReflection(ReflectionBase):
    def __init__(self, debug=False):
        print("AfibReflection module initialized")
        self.debug = debug

    def get_number_of_qrs_waves(self, record, region):
        
        assert "delineation" in record.__dict__, "No delineation data found in record"

        qrs_normal_binary = record.delineation["qrs"]["binary"]
        qrs_normal_binary = closingcentered(qrs_normal_binary, np.ones(int(0.15 * record.fs)))
        qrs_normal_binary = qrs_normal_binary[region[0]:region[1]]
        qrs = get_regions(qrs_normal_binary)
        n_qrs = len(qrs)

        return n_qrs

    def extend_if_possible(self, record):

        hasp = []
        haspx = []
        beats = record.get_beats()
        for i in range(len(beats)):
            hasp.append(beats[i].p != None)
            haspx.append(beats[i].qrs[0])

        if len(hasp) < 5:
            return

        hasp = np.array(hasp, dtype=int)
        hasp = np.convolve(hasp, np.ones(5), mode='same')
        no_p = hasp < 3
        no_p_binary = np.zeros(len(record.ecg))
        no_p_regions = get_regions(no_p)
        for region in no_p_regions:
            #print("Region has no P waves, adding region from", region[0], "to", region[1])
            no_p_binary[haspx[region[0]]:haspx[region[1]-1]] = 1

        afib = record.afib["binary"]
        if np.any(afib==1) == 0:
            return
        
        fps = record.afib["false_positive"]
        no_p_binary[afib==1] = 0
        no_p_binary[fps==1] = 0

        no_p_binary = openingcentered(no_p_binary, np.ones(int(1 * record.fs)))
        no_p_regions = get_regions(no_p_binary)
        for region in no_p_regions:
            #print("Region has no P waves, but has QRS waves: ", self.get_number_of_qrs_waves(record, region))
            if self.get_number_of_qrs_waves(record, region) < 3:
                no_p_binary[region[0]:region[1]] = 0

        original_afib = record.afib["binary"].copy()
        possible_afib = no_p_binary

        uncertain_regions = record.afib["uncertainty"] > 0.9
        possible_afib_regions = get_regions(np.logical_and(possible_afib, uncertain_regions))


        mean_hr_afib = 60 / np.median([beats[i].rr for i in range(len(beats)) if not np.isnan(beats[i].rr) and original_afib[beats[i].get_r_wave()] == 1])
        hrv_afib = self.cosen([beats[i].rr for i in range(len(beats)) if not np.isnan(beats[i].rr) and original_afib[beats[i].get_r_wave()] == 1])
        #print("Mean HR during AFIB extending analysis:", mean_hr_afib)

        for region in possible_afib_regions:
            #print("Region has no P waves, extending from", region[0], "to", region[1])
            #print("Region has low uncertainty and is likely AFIB")
            beats_in_region = [beats[i] for i in range(len(beats)) if not np.isnan(beats[i].rr) and beats[i].get_r_wave() >= region[0] and beats[i].get_r_wave() < region[1]]
            mean_hr = 60 / np.median([beat.rr for beat in beats_in_region])

            equal_hr = True
            equal_hrv = True
            if np.abs(mean_hr_afib - mean_hr) > mean_hr_afib * 0.1:
                equal_hr = False

            if len(beats_in_region) > 8:
                cosen_afib = self.cosen([beats[i].rr for i in range(len(beats)) if not np.isnan(beats[i].rr) and original_afib[beats[i].get_r_wave()] == 1])
                cosen = self.cosen([beat.rr for beat in beats_in_region])
                if cosen_afib > -1.4 and cosen > -1.4:
                    equal_hrv = True
                elif cosen_afib < -1.4 and cosen < -1.4:
                    equal_hrv = True
                else:
                    equal_hrv = False
                #print("Cosen AFIB:", cosen_afib, "of new region Cosen:", cosen, "Equal HR:", equal_hr, "Equal HRV:", equal_hrv)

            else:
                cv_afib = np.std([beats[i].rr for i in range(len(beats)) if not np.isnan(beats[i].rr) and original_afib[beats[i].get_r_wave()] == 1])/np.mean([beats[i].rr for i in range(len(beats)) if not np.isnan(beats[i].rr) and original_afib[beats[i].get_r_wave()] == 1])
                cv = np.std([beat.rr for beat in beats_in_region])/np.mean([beat.rr for beat in beats_in_region])
                if cv_afib < 0.12 and cv < 0.12:
                    equal_hrv = True
                elif cv_afib > 0.12 and cv > 0.12:
                    equal_hrv = True
                else:
                    equal_hrv = False

                #print("CV AFIB:", cv_afib, "of new region CV:", cv, "Equal HR:", equal_hr, "Equal HRV:", equal_hrv)

            if equal_hr and equal_hrv:
                #print("Region has similar HR and HRV to AFIB, extending")
                record.afib["binary"][region[0]:region[1]] = 1
            #else:
                #print("Region has different HR or HRV to AFIB, skipping")

        record.afib["binary"] = closingcentered(record.afib["binary"], np.ones(int(1 * record.fs)))


        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(20, 6), dpi=200)
            ax.plot(record.normalized_ecg)
            ax.plot(haspx, hasp)
            ax.set_title("P-wave detection")
            plt.savefig('extending_afib.png')

    def correct_for_ivr_or_vt(self, record, region):

        beats = record.get_beats()
        beattypes = ["V" if b.abnormal else "N" for b in beats]
        beatstr = "".join(beattypes)
        #print("BEATSTR", beatstr)

        record.afib["false_positive"] = np.zeros(len(record.ecg))

        # i) Find multiple, consecutive V's
        vt_ivr_matches = re.finditer(r'V{3,}', beatstr)
        for match in vt_ivr_matches:
            start = match.start()
            end = match.end()
            #print("Region has VT or IVR, correcting from" , beats[start].get_r_wave(), "to", beats[end-1].get_r_wave(), "with", end-start, "beats", start, end)
            record.afib["false_positive"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 1
            record.afib["binary"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 0

        # ii) Find repeating NV patterns of at least 3 repeats
        big_matches = re.finditer(r'(NV){3,}', beatstr)
        for match in big_matches:
            start = match.start()
            end = match.end()
            #print("Region has BIG, correcting from" , beats[start].get_r_wave(), "to", beats[end-1].get_r_wave(), "with", end-start, "beats", start, end)
            record.afib["false_positive"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 1
            record.afib["binary"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 0

        # iii) Find repeating NNV patterns of at least 3 repeats
        tri_matches = re.finditer(r'(NNV){3,}', beatstr)
        for match in tri_matches:
            start = match.start()
            end = match.end()
            #print("Region has TRI, correcting from" , beats[start].get_r_wave(), "to", beats[end-1].get_r_wave(), "with", end-start, "beats", start, end)
            record.afib["false_positive"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 1
            record.afib["binary"][beats[start].get_r_wave():beats[end-1].get_r_wave()] = 0

    def correct_for_no_p(self, record, region):

        p_waves = get_regions(record.delineation["p"]["logits"] > 0.25)
        beats = record.get_beats()

        for i in range(len(beats)):
            p_search_start = max(0,beats[i].qrs[0] - int(0.5*record.fs)) if i == 0 else (beats[i-1].t[1] if beats[i-1].t != None else beats[i-1].qrs[1])
            p_search_end = beats[i].qrs[0] - int(0.075*record.fs)
            candidates = [p for p in p_waves if p[0] > p_search_start and p[0] < p_search_end]
            record.beats[i].p = None
            if len(candidates) == 0:
                continue
            
            record.beats[i].p = candidates[-1]
            #print("Beat", i, "has P wave at", candidates[-1])

        hasp = []
        haspx = []
        beats = record.get_beats()
        for i in range(len(beats)):
            hasp.append(beats[i].p != None)
            haspx.append(beats[i].qrs[0])

        if len(hasp) < 5:
            return

        hasp = np.array(hasp, dtype=int)
        hasp = np.convolve(hasp, np.ones(5), mode='same')
        normalp = hasp >= 3
        normalp_regions = get_regions(normalp)
        for region in normalp_regions:
            st = haspx[region[0]]
            end = haspx[region[1]-1]
            #print("Region has P waves, correcting from", st, "to", end)
            record.afib["binary"][st:end] = 0


        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
            ax.plot(record.normalized_ecg)
            ax.plot(haspx, hasp)
            ax.set_title("P-wave detection")
            plt.savefig('pwave_detection.png')
        
    def check_hrv(self, record):
            
        assert "beats" in record.__dict__, "No beat data found in record"

        beats = record.get_beats()

        regions = get_regions(record.afib["binary"])

        for region in regions:
            rr = [beats[i].rr for i in range(len(beats)) if beats[i].abnormal == 0 and beats[i].get_r_wave() >= region[0] and beats[i].get_r_wave() < region[1]]

            iqr = np.percentile(rr, 75) - np.percentile(rr, 25)
            med = np.median(rr)
            l_thres = med - 1.5*iqr
            u_thres = med + 1.5*iqr
            #rr = [r for r in rr if r > l_thres and r < u_thres]

            too_regular = False
            if len(rr) > 8:
                #print("COSen:", self.cosen(rr))
                if self.cosen(rr) < -1.4:
                    too_regular = True
                
            # elif len(rr) > 3:
            #     cv = np.std(rr)/np.mean(rr)
            #     if cv < 0.12:
            #         too_regular = True

            if too_regular:
                #print("Region has too regular RR intervals, correcting from", region[0], "to", region[1])
                record.afib["regularity"][region[0]:region[1]] = 1

        return 0

    def cosen(self, ibi_series, m=1, r=None):
        """
        Calculate the Coefficient of Sample Entropy (COSEn) for a time series.
        
        Parameters:
        ibi_series (array-like): Interbeat intervals (time series).
        m (int): Embedding dimension (default is 2).
        r (float): Tolerance for similarity, default is 0.2 * std of ibi_series.
        
        Returns:
        float: COSEn value.
        """
        ibi_series = np.asarray(ibi_series)
        N = len(ibi_series)
        
        def _count_matches(m, r):
            # Create embedding vectors of length m
            X = np.array([ibi_series[i : i + m] for i in range(N - m + 1)])
            count = 0
            for i in range(len(X)):
                for j in range(len(X)):
                    if i != j:
                        d = np.max([np.abs(X[i][k] - X[j][k]) for k in range(m)])
                        # Calculate max absolute difference
                        if d <= r:
                            count += 1
            return count
        
        r = 0.025 #0.2 * np.std(ibi_series) if r is None else r
        A = 0
        B = 0
        while A < 5 and r < 0.5:
            r += 0.005
            # Compute B (matches for dimension m) and A (matches for dimension m+1)
            B = _count_matches(m, r)
            A = _count_matches(m + 1, r)

        if B == 0:
            return np.inf  # To avoid division by zero

        #print(-np.log(A / B), -np.log(2*r), -np.log(np.nanmean(ibi_series)))
        
        res = -np.log(A / B) if A != 0 else 0
        res += np.log(2*r)
        res -= np.log(np.nanmean(ibi_series))
        
        return res

    def entropyaf(self, ibi_series, m=1, r=None):


        ibi_series = np.asarray(ibi_series)
        N = len(ibi_series)

        def dist(x, y):
            eps = 1e-5
            maximum = np.max([np.abs(x[i] - y[i]) for i in range(len(x))])
            minimum = np.min([np.abs(x[i] - y[i]) for i in range(len(x))])
            return (maximum - minimum) / (maximum + minimum + eps)

        def sim_degree(x, y, n, r):
            d = dist(x, y)
            return np.exp(-(d**n) / r)
        
        def _count_matches(m, n, r):
            
            # Create embedding vectors of length m
            X = np.array([ibi_series[i : i + m] for i in range(N - m + 1)])
            n = len(X)
            tot_avg = 0
            for i in range(len(X)):
                avg = 0
                for j in range(len(X)):
                    avg += sim_degree(X[i], X[j], n, r)
                avg /= n
                tot_avg += avg
            tot_avg /= n

            return tot_avg

        r = 0.05
        n = 0.125
        A = _count_matches(m + 1, n, r)
        B = _count_matches(m, n, r)

        entropy = -np.log(A / B) if A != 0 else 0
        entropy += np.log(2*r)
        entropy -= 0.8*np.log(np.nanmean(ibi_series))

        return entropy

    def reflect(self, record):

        assert "afib" in record.__dict__, "No atrial fibrillation data found in record"

        # First, we need to find regions of possible atrial fibrillation
        possible_afib = record.afib["logits"][1] > 0.25
        possible_afib = closingcentered(possible_afib, np.ones(int(3 * record.fs)))
        possible_afib = openingcentered(possible_afib, np.ones(int(3 * record.fs)))
        regions_of_focus = get_regions(possible_afib)

        # If no regions are found, we can safely assume that there is no atrial fibrillation
        if len(regions_of_focus) == 0:
            record.afib["binary"] = np.zeros(len(record.ecg))

        # Check for ventricular rhythms. These can introduce false positives
        for i, region in enumerate(regions_of_focus):
            self.correct_for_ivr_or_vt(record, region)
        regions_of_focus = get_regions(record.afib["binary"])

        # Check for regions with no P waves. Using this, we self-correct the AFib model using the delineation model
        for i, region in enumerate(regions_of_focus):
            self.correct_for_no_p(record, region)

        # Noisy regions can also introduce false positives
        record.afib["binary"][record.noise_mask==1] = 0

        # Smooth out the binary signal
        record.afib["binary"] = closingcentered(record.afib["binary"], np.ones(int(1 * record.fs)))
        regions_of_focus = get_regions(record.afib["binary"])

        # The resulting region cannot be too short (less than 3 QRS complexes)
        for i, region in enumerate(regions_of_focus):
            if self.get_number_of_qrs_waves(record, region) < 3:
                record.afib["binary"][region[0]:region[1]] = 0

        # Smooth out the binary signal
        record.afib["binary"] = openingcentered(record.afib["binary"], np.ones(int(1 * record.fs)))
        record.afib["binary"] = closingcentered(record.afib["binary"], np.ones(int(1 * record.fs)))
        regions_of_focus = get_regions(record.afib["binary"])

        # Remove short regions where the AFib model is uncertain
        for i, region in enumerate(regions_of_focus):
            mean_uncertainty = np.mean(np.log(record.afib["uncertainty"][region[0]:region[1]]))

            if mean_uncertainty < -5:
                #print("Region has low uncertainty, skipping")
                continue

            if region[1] - region[0] < 3 * record.fs:
                #print("Region is too short to count as atrial fibrillation")
                record.afib["binary"][region[0]:region[1]] = 0

        #self.check_hrv(record)

        # See if we can extend the AFib mask at the beginning and ends
        self.extend_if_possible(record)

        return record