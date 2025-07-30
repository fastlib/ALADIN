import numpy as np
from aladin.utils.helpers import Record
import re
import scipy

from aladin.utils.helpers import Cluster, Beat, Record, resize_signal, get_regions, closingcentered, openingcentered

class Embedder():
    def __init__(self, debug=False):
        print("LogicEngine initialized")
        pass

    def get_percentage_noise(self, record):
        noise_mask = record.noise_mask
        afib = record.afib["binary"]
        noise_mask[afib == 1] = 0

        return {"pc_of_noise": np.sum(noise_mask) / len(noise_mask)}

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

    def get_hr_features(self, record):
        beats = record.get_beats()

        if len(beats) <= 3:
            return {
                "raw_hr_mean": 0,
                "raw_hr_std": 0,
                "raw_hr_min": 0,
                "raw_hr_max": 0,
                "raw_hr_kurtosis": 0,
                "raw_hr_skewness": 0,
                "filt_hr_mean": 0,
                "filt_hr_std": 0,
                "filt_hr_min": 0,
                "filt_hr_max": 0,
                "filt_hr_kurtosis": 0,
                "filt_hr_skewness": 0,
                "cosen_filt": 0,
                "entropy_filt": 0,
                "cv_filt": 0
            }
        
        rawibis = [beats[i].rr_raw for i in range(1,len(beats))]
        raw_hr_mean = 60/np.nanmean(rawibis)
        raw_hr_std = np.nanstd(rawibis)
        raw_hr_min = 60/np.nanmax(rawibis)
        raw_hr_max = 60/np.nanmin(rawibis)
        raw_hr_kurtosis = scipy.stats.kurtosis(rawibis)
        raw_hr_skewness = scipy.stats.skew(rawibis)

        filtibis = [beats[i].rr for i in range(1,len(beats))]
        filt_hr_mean = 60/np.nanmean(filtibis)
        filt_hr_std = np.nanstd(filtibis)
        filt_hr_min = 60/np.nanmax(filtibis)
        filt_hr_max = 60/np.nanmin(filtibis)
        filt_hr_kurtosis = scipy.stats.kurtosis(filtibis)
        filt_hr_skewness = scipy.stats.skew(filtibis)

        cosen_filt = self.cosen(filtibis)
        entropy_filt = self.entropyaf(filtibis)
        cv_filt = np.std(filtibis) / np.mean(filtibis)

        return {
            "raw_hr_mean": raw_hr_mean,
            "raw_hr_std": raw_hr_std,
            "raw_hr_min": raw_hr_min,
            "raw_hr_max": raw_hr_max,
            "raw_hr_kurtosis": raw_hr_kurtosis,
            "raw_hr_skewness": raw_hr_skewness,
            "filt_hr_mean": filt_hr_mean,
            "filt_hr_std": filt_hr_std,
            "filt_hr_min": filt_hr_min,
            "filt_hr_max": filt_hr_max,
            "filt_hr_kurtosis": filt_hr_kurtosis,
            "filt_hr_skewness": filt_hr_skewness,
            "cosen_filt": cosen_filt,
            "entropy_filt": entropy_filt,
            "cv_filt": cv_filt
        }

    def get_qrs_features(self, record):
        beats = record.get_beats()

        if len(beats) < 2:
            return {
                "qrsw_mean": 0,
                "qrsw_std": 0,
                "qrs_abnorm": 0,
                "qrs_snr": 0
            }
        
        qrs_lens = [(beats[i].qrs[1] - beats[i].qrs[0])/record.fs for i in range(len(beats))]
        qrs_mean = np.mean(qrs_lens)
        qrs_std = np.std(qrs_lens)
        qrs_snr = np.mean([beats[i].snr for i in range(len(beats))])

        isabnorm = [beats[i].abnormal for i in range(len(beats))]

        return {
            "qrsw_mean": qrs_mean,
            "qrsw_std": qrs_std,
            "qrs_abnorm": np.sum(isabnorm) / len(isabnorm),
            "qrs_snr": qrs_snr
        }

    def get_rhythm_patterns(self, record):
        beats = record.get_beats()

        beattypes = ["V" if b.abnormal else "N" for b in beats]
        beatstr = "".join(beattypes)

        beatstr = beatstr.replace("VVVNVVV", "VVVVVVV")
        beatstr = beatstr.replace("VVVNNVVV", "VVVVVVVV")
        print("BEATSTR", beatstr)

        vt_ivr_matches = re.finditer(r'V{3,}', beatstr)
        big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beatstr)
        tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beatstr)
        quad_matches = re.finditer(r'((VNNN){3,}|(NNNV){3,})', beatstr)

        return {
            "has_vt": len(list(vt_ivr_matches)) > 0,
            "has_bigeminy": len(list(big_matches)) > 0,
            "has_trigeminy": len(list(tri_matches)) > 0,
            "has_quadrigeminy": len(list(quad_matches)) > 0
        }


    def get_pwave_features(self, record):
        beats = record.get_beats()

        if len(beats) < 2:
            return {
                "pwave_count": 0,
                "pwave_power": 0,
                "pwave_polarity": 0,
                "pwave_pr_mean": 0,
                "pwave_pr_std": 0,
                "pwave_pr_kurtosis": 0,
                "pwave_pr_skewness": 0,
                "pwave_behind": 0,
                "dangling_pwaves": 0
            }
        
        p_waves = get_regions(record.delineation["p"]["binary"])
        pn = 0
        polarities = []
        prs = []
        behinds = 0
        avgpower = 0

        for i in range(len(beats)):
            beats[i].prev_beat = beats[i-1].get_r_wave() if i > 0 else 0

        for i in range(len(beats)):
            beats[i].p = None
            p_search_start = beats[i].prev_beat+int(0.2*record.fs)
            p_search_end = beats[i].qrs[0]
            candidates = [p for p in p_waves if p[0] > p_search_start and p[0] < p_search_end]
            if len(candidates) == 0:
                continue

            lastcandidate = candidates[-1]
            dist = np.abs(beats[i].qrs[0] - lastcandidate[0])
            if dist >= int(0.075*record.fs) and lastcandidate[1] - lastcandidate[0] >= 0.025*record.fs:
                beats[i].p = lastcandidate
                beats[i].pr = dist/record.fs
                beats[i].p_polarity = np.round(np.mean(record.p_wave_polarity[lastcandidate[0]:lastcandidate[1]]))
                beats[i].p_not_matched = np.round(np.mean(record.p_wave_unmatched[lastcandidate[0]:lastcandidate[1]]))
                beats[i].p_group = np.round(np.mean(record.p_wave_group[lastcandidate[0]:lastcandidate[1]]))
                polarities.append(beats[i].p_polarity)
                prs.append(beats[i].pr)
                pn += 1
            else:
                print(f"[{i}] P-wave found too close to QRS or too short")
            
            pwave = record.ecg[lastcandidate[0]:lastcandidate[1]]
            power = np.sum(pwave**2)/len(pwave)
            avgpower += power

            if beats[i].p is None:
                p_search_start = beats[i].qrs[0]
                p_search_end = beats[i].qrs[1] + int(0.2*record.fs)
                candidates = [p for p in p_waves if p[0] > p_search_start and p[0] < p_search_end]
                if len(candidates) == 0:
                    continue

                lastcandidate = candidates[-1]
                if lastcandidate[1] - lastcandidate[0] >= 0.025*record.fs:
                    print(f"[{i}] P-wave found behind QRS")
                    beats[i].p = lastcandidate
                    beats[i].pr = (beats[i].qrs[0] - lastcandidate[0])/record.fs
                    beats[i].p_polarity = np.round(np.mean(record.p_wave_polarity[lastcandidate[0]:lastcandidate[1]]))
                    beats[i].p_not_matched = np.round(np.mean(record.p_wave_unmatched[lastcandidate[0]:lastcandidate[1]]))
                    beats[i].p_group = np.round(np.mean(record.p_wave_group[lastcandidate[0]:lastcandidate[1]]))
                    behinds += 1
                else:
                    print(f"[{i}] P-wave found too short")

        if avgpower == 0:
            avgpower = 0
        else:
            avgpower /= pn
        dangling_p_waves = record.unmatched_p

        return {
            "pwave_count": pn/len(beats),
            "pwave_power": avgpower,
            "pwave_polarity": np.mean(polarities),
            "pwave_pr_mean": np.mean(prs),
            "pwave_pr_std": np.std(prs),
            "pwave_pr_kurtosis": scipy.stats.kurtosis(prs),
            "pwave_pr_skewness": scipy.stats.skew(prs),
            "pwave_behind": behinds/len(beats),
            "dangling_pwaves": len(dangling_p_waves)/len(beats)
        }

    def get_twave_features(self, record):
        beats = record.get_beats()

        if len(beats) < 2:
            return {
                "twave_tp_mean": 0,
                "twave_tp_std": 0
            }

        tp_intervals = []
        num_p = 0
        for i in range(1,len(beats)):
            if beats[i].p is not None and beats[i-1].t is not None:
                tp_intervals.append((beats[i].p[0] - beats[i-1].t[1])/record.fs)
                num_p += 1
            else:
                tp_intervals.append(np.nan)

        return {
            "twave_tp_mean": np.nanmean(tp_intervals),
            "twave_tp_std": np.nanstd(tp_intervals),
        }

    def embed(self, record):
        noise_features = self.get_percentage_noise(record)
        hr_features = self.get_hr_features(record)
        qrs_features = self.get_qrs_features(record)
        rhythm_patterns = self.get_rhythm_patterns(record)
        pwave_features = self.get_pwave_features(record)
        twave_features = self.get_twave_features(record)

        return {
            **noise_features,
            **hr_features,
            **qrs_features,
            **rhythm_patterns,
            **pwave_features,
            **twave_features
        }