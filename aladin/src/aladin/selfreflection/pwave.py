import numpy as np

from aladin.selfreflection.base import ReflectionBase

from aladin.utils.helpers import Cluster, Beat, Record, resize_signal, get_regions, closingcentered, openingcentered, MorphologicalClustering
import scipy.signal as sps
from scipy.signal import butter, filtfilt
from dtw import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate, correlation_lags

from tqdm import tqdm
import time
import os
import multiprocessing


class PWaveReflection(ReflectionBase):
    def __init__(self, debug=False):
        self.debug = debug
        print("PWaveReflection module initialized")
    
    def plot_p_waves(self, record: Record, searchlocations, initial, post_1, post_2, sup):
        
        ecg = record.filtered_ecg
        uncertainty = record.delineation["p"]["uncertainty_processed"]
        fig, ax = plt.subplots(3, 1, figsize=(20, 8), dpi=200, sharex=True)
        ax[0].plot(ecg)
        ax[0].scatter(post_1, ecg[post_1], color="red", zorder=10)
        ax[0].scatter(post_2, ecg[post_2], color="blue", zorder=10)
        for s in sup:
            ax[0].axvspan(s[0], s[1], color="green", alpha=0.5, zorder=0)
        #ax[0].scatter(gt, np.ones(len(gt)), color="green", zorder=10)

        ax[1].plot(ecg)
        ax[1].scatter(initial, ecg[initial], color="green", zorder=10)

        ax[2].plot(uncertainty)
        #print(searchlocations)
        if len(searchlocations) > 0:
            ax[2].scatter(searchlocations, ecg[searchlocations], color="red", zorder=10)
        plt.savefig("p_waves.png")
        plt.show()

    def shift_wave(self, wave, lag):
        if lag > 0:
            wave = np.pad(wave, (lag, 0))
        else:
            wave = wave[-lag:]

        return wave

    def make_equal_length(self, src, dst):
        lendst = len(dst)
        if len(src) < lendst:
            src = np.pad(src, (0, lendst - len(src)))
        else:
            src = src[:lendst]

        return src

    def find_best_offset(self, array1, array2):
        min_rmse = float('inf')
        max_corr = 0
        best_offset = 0
        
        # Ensure array1 is the longer array to allow for shifts of array2
        switched = False
        if len(array2) > len(array1):
            array1, array2 = array2, array1
            switched = True

        for offset in range(len(array1) - len(array2) + 1):
            shifted_array1 = array1[offset:offset + len(array2)]
            corr = np.corrcoef(array2, shifted_array1)

            if corr[0,1] > max_corr:
                max_corr = corr[0,1]
                best_offset = offset

        if switched:
            best_offset = -best_offset

        return best_offset, max_corr

    def minimum_mean_squared_error(self, array1, array2):
        array1 = np.array(array1.copy())
        array2 = np.array(array2.copy())
        
        # Calculate the optimal vertical shift (offset)
        c = np.mean(array1 - array2)
        
        # Apply the offset to the second array
        aligned_array2 = array2 + c
        
        # Compute the MSE
        mse = np.sqrt(np.mean((array1 - aligned_array2) ** 2))
        
        return mse

    def find_best_pair(self, waves, alreadyaligned):
        correlation_scores = np.zeros((len(waves), len(waves)))
        correlation_offsets = np.zeros((len(waves), len(waves)))

        notaligned_inds = np.where(alreadyaligned==-1)[0]

        for i in notaligned_inds:
            for j in notaligned_inds:
                if j <= i:  
                    continue
                correlation = correlate(waves[i], waves[j], mode='full') / (np.linalg.norm(waves[i]) * np.linalg.norm(waves[j]))
                maxscore = np.max(correlation)  # max cross-correlation as similarity measure
                correlation_offsets[i, j] = np.argmax(correlation)
                lag = -correlation_lags(len(waves[i]), len(waves[j]), mode='full')[np.argmax(correlation)]
                tmp1 = self.shift_wave(waves[i].copy(), lag)
                tmp1 = self.make_equal_length(tmp1, waves[j])
                rmse = np.sqrt(mean_squared_error(tmp1, waves[j]))/np.max(np.abs(waves[j]))
                rmse += np.sqrt(mean_squared_error(waves[j], tmp1))/np.max(np.abs(tmp1))
                rmse /= 2
                correlation_scores[i, j] = rmse
                #print(f"Score between {i} and {j}: {rmse}")


        maxscore = 0
        minscore = float('inf')
        bestpair = (0, 0)
        for i in notaligned_inds:
            for j in notaligned_inds:
                if j <= i:  
                    continue

                if correlation_scores[i, j] < minscore:
                    minscore = correlation_scores[i, j]
                    bestpair = (i, j)

        if minscore > 0.2:
            #print("No good pair found")
            return (), 0, 0

        lags = correlation_lags(len(waves[bestpair[0]]), len(waves[bestpair[1]]), mode='full')
        offset = int(correlation_offsets[bestpair[0], bestpair[1]])
        lag = -lags[offset]

        return bestpair, lag, maxscore
        
    def recursively_align_waves(self, waves, alreadyaligned, avg_wave=None, iteration=0, group=0, debugax=None):

        aligned_inds = np.where(alreadyaligned>-1)[0]
        notaligned_inds = np.where(alreadyaligned==-1)[0]

        if np.all(alreadyaligned>-1):
            #print("All waves aligned")
            return avg_wave, alreadyaligned

        if iteration == 0:
            first_best_pair, lag, maxscore = self.find_best_pair(waves, alreadyaligned)
            if len(first_best_pair) == 0:
                #print("No good pair found")
                return avg_wave, alreadyaligned
            
            waves[first_best_pair[0]] = self.shift_wave(waves[first_best_pair[0]], lag)
            waves[first_best_pair[0]] = self.make_equal_length(waves[first_best_pair[0]], waves[first_best_pair[1]])
            alreadyaligned[first_best_pair[0]] = group
            alreadyaligned[first_best_pair[1]] = group
            avg_wave = np.mean(np.array([waves[first_best_pair[0]],waves[first_best_pair[1]]]), axis=0)

        else:
            correlation_scores = np.zeros((len(waves)))
            correlation_offsets = np.zeros((len(waves)))

            for i in notaligned_inds:
                correlation = correlate(waves[i], avg_wave, mode='full') / (np.linalg.norm(waves[i]) * np.linalg.norm(avg_wave))
                if np.all(np.isnan(correlation)):
                    #print("All nans in correlation")
                    continue
                maxscore = np.nanmax(correlation)  # max cross-correlation as similarity measure
                correlation_offsets[i] = np.nanargmax(correlation)
                lag = -correlation_lags(len(waves[i]), len(avg_wave), mode='full')[np.argmax(correlation)]
                tmp1 = self.shift_wave(waves[i].copy(), lag)
                tmp1 = self.make_equal_length(tmp1, avg_wave)
                rmse = np.sqrt(mean_squared_error(tmp1, avg_wave))/np.max(np.abs(avg_wave))
                rmse += np.sqrt(mean_squared_error(avg_wave, tmp1))/np.max(np.abs(tmp1))
                rmse /= 2
                correlation_scores[i] = rmse

            maxscore = 0
            minscore = float('inf')
            bestmatch = 0
            for i in notaligned_inds:
                if correlation_scores[i] < minscore:
                    minscore = correlation_scores[i]
                    bestmatch = i

            if minscore > 0.2:
                #print("No good match found")
                return avg_wave, alreadyaligned

            offset = int(correlation_offsets[bestmatch])

            lag = correlation_lags(len(waves[bestmatch]), len(avg_wave), mode='full')
            lag = -lag[offset]

            waves[bestmatch] = self.shift_wave(waves[bestmatch], lag)
            waves[bestmatch] = self.make_equal_length(waves[bestmatch], avg_wave)
            alreadyaligned[bestmatch] = group
            aligned_inds = np.where(alreadyaligned==group)[0]
            npwaves = np.array([waves[i] for i in aligned_inds])

            avg_wave = np.mean(npwaves, axis=0)

        if debugax is not None:
            for i in np.where(alreadyaligned)[0]:
                debugax[iteration].plot(waves[i], label=f"p_wave_{i}", lw=0.5)
            debugax[iteration].plot(avg_wave, label="avg_wave", lw=1, color="black")
            debugax[iteration].set_ylim(-1, 1)
            debugax[iteration].set_title(f"Iteration {iteration}")
            debugax[iteration].text(0, -0.1, f"Score: {maxscore:.2f}")

        return self.recursively_align_waves(waves, alreadyaligned, avg_wave=avg_wave, iteration=iteration+1, group=group, debugax=debugax)

    def recursively_group_and_align_waves(self, waves, alreadyaligned, groups=[], avg_waves=[], iteration=0, debug=False):

        if np.all(alreadyaligned>-1):
            #print("All waves aligned")
            return avg_waves, groups, []

        if type(waves[0]) == list:
            waves = [np.array(w) for w in waves]

        if iteration == 0:
            avg_waves = []
            groups = []

        ax = None
        if debug:
            nrows = int(np.round(np.sqrt(len(waves))))
            ncols = int(np.ceil(len(waves) / nrows))
            fig, ax = plt.subplots(nrows, ncols, figsize=(nrows*4, ncols*4), dpi=200)
            #flatten the multidimensional ax numpy array
            if nrows*ncols > 1:
                ax = ax.flatten()

        avg_wave, alreadyaligned = self.recursively_align_waves(waves, alreadyaligned, debugax=ax, group=iteration)

        if debug:
            plt.show()

        if avg_wave is None:
            #print("No good match found")
            return avg_waves, groups, np.where(alreadyaligned==-1)[0]
        
        x0 = 0
        x1 = len(avg_wave)
        y0 = avg_wave[0]
        y1 = avg_wave[-1]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m*x0
        for j in range(len(avg_wave)):
            avg_wave[j] = avg_wave[j] - (m*j + b)
        
        if len(avg_waves) > 0:
            correlation_scores = np.zeros((len(avg_waves)))
            correlation_offsets = np.zeros((len(avg_waves)))

            for i in range(len(avg_waves)):

                correlation = correlate(avg_waves[i], avg_wave, mode='full') / (np.linalg.norm(avg_waves[i]) * np.linalg.norm(avg_wave))
                correlation_offsets[i] = np.nanargmax(correlation)
                lag = -correlation_lags(len(avg_waves[i]), len(avg_wave), mode='full')[np.argmax(correlation)]
                tmp1 = self.shift_wave(avg_waves[i].copy(), lag)
                tmp1 = self.make_equal_length(tmp1, avg_wave)
                maxrmse = self.minimum_mean_squared_error(avg_wave, -avg_wave)
                rmse = self.minimum_mean_squared_error(tmp1, avg_wave)/maxrmse
                correlation_scores[i] = rmse
                #print(f"Score between avg_wave {i} and current avg_wave in {iteration}: {rmse}")

            minscore = float('inf')
            bestmatch = 0
            for i in range(len(avg_waves)):
                if correlation_scores[i] < minscore:
                    minscore = correlation_scores[i]
                    bestmatch = i

            if minscore < 0.3 and False:
                #print("Merge groups")
                groups[bestmatch] = np.concatenate((groups[bestmatch], np.where(alreadyaligned==iteration)[0]))
            else:
                avg_waves.append(avg_wave)
                groups.append(np.where(alreadyaligned==iteration)[0])
        else:
            avg_waves.append(avg_wave)
            groups.append(np.where(alreadyaligned==iteration)[0])

        
        if len(np.where(alreadyaligned==-1)[0]) < 3:
            #print("No more waves to align")
            return avg_waves, groups, np.where(alreadyaligned==-1)[0]

        return self.recursively_group_and_align_waves(waves, alreadyaligned, groups, avg_waves, iteration=iteration+1, debug=debug)

    def get_match_scores(self, record: Record, possiblepeaks, avg_wave):

        ecg = record.filtered_ecg
        p_search_window_width = int(0.1*record.fs)
        p_search_windows = [(max(0,p-p_search_window_width), min(len(ecg),p+p_search_window_width)) for p in possiblepeaks]
        p_waves = [ecg[p_search_windows[i][0]:p_search_windows[i][1]] for i in range(len(p_search_windows))]

        to_remove = []
        factor = 1/np.max(np.abs(avg_wave))
        avg_wave = avg_wave * factor

        for i in range(len(p_search_windows)):

            ponset = p_search_windows[i][0]
            poffset = p_search_windows[i][1]
            
            p_waves[i] = ecg[ponset:poffset]

            p_waves[i] = p_waves[i] * factor

            #remove linear trend
            x0 = 0
            x1 = len(p_waves[i])
            y0 = p_waves[i][0]
            y1 = p_waves[i][-1]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m*x0
            for j in range(len(p_waves[i])):
                p_waves[i][j] = p_waves[i][j] - (m*j + b)

            #filter using a savgol filter
            p_waves[i] = sps.savgol_filter(p_waves[i], 17, 3)

            #remove p waves that are too small
            if np.sum(np.abs(p_waves[i])) < 1:
                to_remove.append(i)

            correlations = correlate(p_waves[i], avg_wave, mode='full') / (np.linalg.norm(p_waves[i]) * np.linalg.norm(avg_wave))
            best_corr = np.max(correlations)
            lags = correlation_lags(len(p_waves[i]), len(avg_wave), mode='full')
            best_lag = -lags[np.argmax(correlations)]
            p_waves[i] = self.shift_wave(p_waves[i], best_lag)
            p_waves[i] = self.make_equal_length(p_waves[i], avg_wave)
            min_rmse = 0
            zero_rmse = np.sqrt(mean_squared_error(avg_wave, np.zeros(len(avg_wave))))
            rmse = np.sqrt(mean_squared_error(p_waves[i], avg_wave))
            ratio = zero_rmse/rmse
            #print(f"Best lag: {best_lag}, Best Corr: {best_corr}, RMSE ratio: {ratio}")

            if (best_corr < 0.8) or (best_corr > 0.8 and ratio < 1):
                #print("-> Not a good match!", i)
                to_remove.append(i)
        
        p_waves = [p_waves[i] for i in range(len(p_waves)) if i not in to_remove]
        removed_possiblepeaks = [possiblepeaks[i] for i in range(len(possiblepeaks)) if i in to_remove]
        possiblepeaks = [possiblepeaks[i] for i in range(len(possiblepeaks)) if i not in to_remove]

        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
            for i in range(len(p_waves)):
                ax.plot(p_waves[i], label=f"p_wave_{i}", lw=0.5)
            ax.plot(avg_wave, label="avg_wave", lw=1, color="black")
            #ax.set_ylim(-1, 1)
            plt.savefig("all_p_waves.png")
            plt.show()

        return possiblepeaks, removed_possiblepeaks
    
    def get_avg_p_wave(self, record: Record):
        

        beats = record.get_beats()
        medianrange = np.median([np.max(record.filtered_ecg[beat.qrs[0]:beat.qrs[1]]) - np.min(record.filtered_ecg[beat.qrs[0]:beat.qrs[1]]) for beat in beats])

        ecg = record.filtered_ecg
        ps = get_regions(record.delineation["p"]["logits"]>0.1)
        p_centers = [p[0] + (p[1] - p[0]) // 2 for p in ps]

        p_search_window_width = int(0.1*record.fs)
        p_search_windows = [(max(0,p-p_search_window_width), min(len(ecg),p+p_search_window_width)) for p in p_centers]
        p_waves = [ecg[p_search_windows[i][0]:p_search_windows[i][1]].copy() for i in range(len(p_search_windows))]

        unmatched_sig = np.zeros(len(record.ecg))
        to_remove = []
        #find inital p waves and remove linear trend and filter
        for i in range(len(ps)):

            ponset = ps[i][0]
            poffset = ps[i][1]
            
            p_waves[i] = ecg[ponset:poffset]

            #remove linear trend
            x0 = 0
            x1 = len(p_waves[i])
            y0 = p_waves[i][0]
            y1 = p_waves[i][-1]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m*x0
            for j in range(len(p_waves[i])):
               p_waves[i][j] = p_waves[i][j] - (m*j + b)

            #remove p waves that are too small
            if np.sum(np.abs(p_waves[i])) < 1 or np.all(np.isnan(p_waves[i])) or poffset-ponset < 0.05*record.fs:
                #print("Too small or too short", p_waves[i])
                to_remove.append(i)

        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
            for i in range(len(p_waves)):
                ax.plot(p_waves[i], label=f"p_wave_{i}", lw=0.5)
            plt.savefig("all_p_waves.png")
            plt.show()

        for i in to_remove:
            unmatched_sig[p_search_windows[i][0]:p_search_windows[i][1]] = 1

        p_waves = [p_waves[i] for i in range(len(p_waves)) if i not in to_remove]
        p_centers = [p_centers[i] for i in range(len(p_centers)) if i not in to_remove]
        p_search_windows = [p_search_windows[i] for i in range(len(p_search_windows)) if i not in to_remove]

        p_wave_avgs, groups, remaining = self.recursively_group_and_align_waves(p_waves, -np.ones(len(p_waves), dtype=int), iteration=0, debug=self.debug)
        #print(remaining)

        if len(p_wave_avgs) == 0:
            #print("No p waves found")
            return [], [], [], [], []
        #find p-wave peak


        for i in remaining:
            unmatched_sig[p_search_windows[i][0]:p_search_windows[i][1]] = 1

        p_wave_groups = []

        p_wave_avg_peakpos = []
        polarity_sig = np.ones(len(record.ecg))
        group_sig = -np.ones(len(record.ecg))
        for i, p_wave_avg in enumerate(p_wave_avgs):

            p_wave_avgs[i] /= medianrange
            #remove linear trend
            x0 = 0
            x1 = len(p_wave_avgs[i])
            y0 = p_wave_avgs[i][0]
            y1 = p_wave_avgs[i][-1]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m*x0
            for j in range(len(p_wave_avgs[i])):
                p_wave_avgs[i][j] = p_wave_avgs[i][j] - (m*j + b)

            #find p-wave peak
            p_wave_pos = p_wave_avg.copy()
            p_wave_pos[p_wave_pos < 0] = 0
            p_wave_neg = p_wave_avg.copy()
            p_wave_neg[p_wave_neg > 0] = 0
            polarity = 1

            maximum = np.max(p_wave_avg)
            minimum = np.min(p_wave_avg)
            #print("Max: ", maximum, "Min: ", minimum, "delta: ", maximum - minimum)

            if maximum-minimum < 0.02:
                #print("-> Flat P Wave!")
                p_wave_avg_peakpos.append(np.argmax(p_wave_avg))
                polarity = 1
            elif maximum > -minimum*2:
                #print("-> Positive P Wave!", np.max(p_wave_avg), -np.min(p_wave_avg))
                p_wave_avg_peakpos.append(np.argmax(p_wave_avg))
                polarity = 1
            elif -minimum > maximum*2:
                #print("-> Negative P Wave!", np.max(p_wave_avg), -np.min(p_wave_avg))
                p_wave_avg_peakpos.append(np.argmin(p_wave_avg))
                polarity = -1
            else:
                #print("-> Biphasic P Wave!")
                p_wave_avg_peakpos.append(np.argmax(p_wave_pos))
                p_wave_avg_peakpos.append(np.argmin(p_wave_neg))
                polarity = 1

            p_grp = []
            for j in range(len(groups[i])):
                p_grp.append({"onset": p_search_windows[groups[i][j]][0], "offset": p_search_windows[groups[i][j]][1], "peak": p_wave_avg_peakpos[i], "polarity": polarity})
                polarity_sig[p_search_windows[groups[i][j]][0]:p_search_windows[groups[i][j]][1]] = polarity
                group_sig[p_search_windows[groups[i][j]][0]:p_search_windows[groups[i][j]][1]] = i
            p_wave_groups.append(p_grp)

        for i in remaining:
            p_grp = []
            wave = p_waves[i]/medianrange
            polarity = 1

            maximum = np.max(wave)
            minimum = np.min(wave)
            #print("Max: ", maximum, "Min: ", minimum, "delta: ", maximum - minimum)

            if maximum-minimum < 0.02:
                #print("-> Remaining wave is Flat P Wave!")
                polarity = 1
            elif maximum > -minimum*2:
                #print("-> Remaining wave is Positive P Wave!", np.max(wave), -np.min(wave))
                polarity = 1
            elif -minimum > maximum*2:
                #print("-> Remaining wave is Negative P Wave!", np.max(wave), -np.min(wave))
                polarity = -1
            else:
                #print("-> Remaining wave is Biphasic P Wave!")
                polarity = 1

            polarity_sig[p_search_windows[i][0]:p_search_windows[i][1]] = polarity


        record.p_wave_groups = p_wave_groups
        record.p_wave_polarity = polarity_sig
        record.p_wave_unmatched = unmatched_sig
        record.p_wave_group = group_sig
        record.average_p_waves = p_wave_avgs

        if self.debug:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
            for i in range(len(p_wave_avgs)):
                ax.plot(p_wave_avgs[i], label=f"p_wave_avg_{i}", lw=1)
                #ax.scatter(p_wave_avg_peakpos[i], p_wave_avgs[i][p_wave_avg_peakpos[i]], color="red")

            ax.legend()
            plt.savefig("average_p_waves.png")
            plt.show()

        return p_wave_avgs, groups, p_waves, p_centers, p_wave_avg_peakpos
    
    def get_positive_peaks(self, record: Record):
        
        original_p_regions = get_regions(record.cpp_record.delineations.p.binary)
        original_p_centers = [p[0] + (p[1] - p[0]) // 2 for p in original_p_regions]
        uncertainty = record.cpp_record.delineations.p.uncertainty
        peaks = sps.find_peaks(uncertainty, height=-2.5, distance=int(0.2*record.fs), prominence=3)
        peaks = peaks[0]

        #if peak is closer than 0.1s to an existing prediction, remove it
        toremove = []
        for peak in peaks:
            for prediction in original_p_centers:
                if abs(peak - prediction) < 0.2*record.fs:
                    toremove.append(peak)

        peaks = np.array([peak for peak in peaks if peak not in toremove])

        #remove peaks that are inside afib regions
        toremove = []
        for peak in peaks:
            if record.cpp_record.delineations.afib.binary[peak] == 1:
                toremove.append(peak)

        peaks = np.array([peak for peak in peaks if peak not in toremove])

        return peaks

    def remove_qrs_regions(self, record: Record):

        beats = record.qrs

        for i in range(len(beats)):
            record.cpp_record.delineations.p.uncertainty[beats[i].onset:beats[i].offset] = -10
            if beats[i].t is not None:
                record.cpp_record.delineations.p.uncertainty[beats[i].t.onset:beats[i].t.offset] = -10

    def process_uncertainty(self, record: Record):
        
        uncertainty = np.log(record.cpp_record.delineations.p.uncertainty)
        uncertainty = sps.medfilt(uncertainty, kernel_size=21)
        uncertainty = sps.savgol_filter(uncertainty, 21, 3)
        record.cpp_record.delineations.p.set_uncertainty(uncertainty)

    def preprocess(self, record: Record):

        fs = record.fs

        #fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=200)

        signal = record.ecg.copy()
        assert len(signal.shape) == 1


        highcut = 30
        lowcut = 0.5
        nyquist = 0.5 * fs
        high = highcut / nyquist
        low = lowcut / nyquist
        b, a = butter(4, [low,high], btype='band')
        signal = filtfilt(b, a, signal)

        record.bandpass_ecg = signal
        record.bandpass_ecg_no_qrst, record.noise_ecg_no_qrst = self.remove_qrst_from_signal(record)


        return record
    

    def remove_qrst_from_signal(self, record, debug=False):

        fs = record.fs

        beats = record.get_beats()

        if len(beats) == 0:
            return record.filtered_ecg, record.filtered_ecg

        ecg = record.filtered_ecg.copy()

        for i in range(len(beats)):
            
            st = beats[i].qrs[0]# - buffer
            en = beats[i].t[1] if beats[i].t is not None else beats[i].qrs[1]# + buffer

            x0 = st
            x1 = en
            y0 = ecg[st]
            y1 = ecg[en]
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m*x0

            for k in range(st, en):
                ecg[k] = m*k + b

        ecg_without_qrst = ecg.copy()
        ecg_noise = ecg.copy()
        
        highcut = 10
        nyquist = 0.5 * fs
        high = highcut / nyquist
        b, a = butter(4, high, btype='high')
        for i in range(len(beats)-1):
            if i == 0:
                st = 0
            else:
                st = beats[i-1].t[1] if beats[i-1].t is not None else beats[i-1].qrs[1]
            en = beats[i].qrs[0]

            if en-st > 15:
                ecg_noise[st:en] = filtfilt(b, a, ecg_noise[st:en])

        return ecg_without_qrst, ecg_noise

    def calculate_overlap(self, region1, region2):
        """
        Calculate the overlap percentage between two regions.
        
        Args:
            region1: Tuple (start1, end1) defining the first region.
            region2: Tuple (start2, end2) defining the second region.
            
        Returns:
            float: Overlap percentage relative to the smaller of the two regions.
        """
        # Unpack the regions
        start1, end1 = region1
        start2, end2 = region2
        
        # Calculate the overlap region
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        # If there's no overlap, return 0
        if overlap_start >= overlap_end:
            return 0.0
        
        # Calculate overlap length
        overlap_length = overlap_end - overlap_start
        
        # Calculate the lengths of both regions
        region1_length = end1 - start1
        region2_length = end2 - start2
        
        # Calculate the overlap percentage relative to the smaller region
        smaller_region_length = region2_length
        overlap_percentage = (overlap_length / smaller_region_length) * 100
        
        return overlap_percentage

    def filter_binary_signal(self, record: Record, threshold=0.5):

        p_binary = np.array(record.cpp_record.delineations.p.logits > threshold, dtype=int)
        p_binary = openingcentered(p_binary, np.ones(int(0.01 * record.fs)))
        p_binary = closingcentered(p_binary, np.ones(int(0.05 * record.fs)))
        isafib = record.cpp_record.delineations.afib.binary
        isnoise = record.cpp_record.delineations.noise.binary
        t_binary = np.array(record.cpp_record.delineations.t.logits > 0.5, dtype=int)
        p_binary[isafib==1] = 0
        p_binary[isnoise==1] = 0

        p_waves = get_regions(p_binary)
        p_centers = [(p[0] + p[1]) // 2 for p in p_waves]
        
        beats = record.qrs
        qrs_centers = [beat.r for beat in beats]
        qrs_median = np.median(np.diff(qrs_centers))

        ibis = np.diff(p_centers)
        start, end = self.get_starting_point(ibis, qrs_median)

        if len(beats) == 0:
            p_binary = np.zeros(len(p_binary))
        else:
            beat_to_remove = []
            for wave in p_waves:
                if np.all(t_binary[wave[0]:wave[1]] == 1):
                    p_binary[wave[0]:wave[1]] = 0

                #get closest beat 
                idx = np.argmin([abs((beat.onset+beat.offset)/2 - (wave[0] + wave[1])/2) for beat in beats])
                overlap = self.calculate_overlap((wave[0], wave[1]), (beats[idx].onset, beats[idx].offset))
                #print("Closest beat: ", idx, "distance: ", abs((beats[idx].qrs[0]+beats[idx].qrs[1])/2 - (wave[0] + wave[1])/2), "overlap: ", overlap)
                if overlap > 75:
                    beat_to_remove.append(idx)

            #remove beats that are too close to p waves
            #record.beats = [beats[i] for i in range(len(beats)) if i not in beat_to_remove]
        

        record.cpp_record.delineations.p.set_binary(p_binary)

        print(record.cpp_record.delineations.p.binary)

        record = self.match_p_waves(record)

        return record

    def get_snr(self, fore, back):

        background_power = np.sum(back**2)/len(back) #np.sum(np.abs(back))
        #get signal power
        signal_power = np.sum(fore**2)/len(fore)
        #get snr
        snr = 10*np.log10(signal_power/background_power)
        #snr = signal_power/background_power

        return snr

    def get_ratio(self, fore, back):
        
        #remove nan
        back = back[~np.isnan(back)]
        fore = fore[~np.isnan(fore)]
        #remove linear trend

        background_power = np.sum(np.abs(back))/len(back) #np.sum(np.abs(back))
        #get signal power
        signal_power = np.sum(np.abs(fore))/len(fore)
        #get snr
        snr = signal_power/background_power

        print("Signal power: ", signal_power, "Background power: ", background_power)

        return snr

    def match_p_waves(self, record: Record):

        p_waves = get_regions(record.cpp_record.delineations.p.binary)
        beats = record.qrs
        unmatched_p = []
        ecgfilt = record.filtered_ecg
        ecgp = record.ecg_no_qrst
        ecgn = record.ecg_noise
        self.get_avg_p_wave(record)

        pwaves = []
        #ecgp = sps.savgol_filter(ecgp, 21, 3)
        

        for i in range(len(beats)):
            p_search_start = 0 if i == 0 else beats[i-1].offset + int(0.2*record.fs)
            p_noise_win_start = 0 if i == 0 else beats[i-1].t.offset if beats[i-1].t is not None else beats[i-1].offset
            p_search_end = beats[i].onset
            p_noise_win_end = p_search_end

            qrs_start = beats[i].onset
            qrs_end = beats[i].offset
            fore = ecgfilt[max(0,qrs_start):min(len(ecgp),qrs_end)].copy()
            fore -= np.median(fore)

            back = ecgn[max(0,p_noise_win_start):min(len(ecgp),p_noise_win_end)].copy()
            snr = self.get_snr(fore,back)
            record.qrs[i].snr = snr

            candidates = [p for p in p_waves if p[0] > p_search_start and p[0] < p_search_end]
            record.qrs[i].pratio = 0
            if len(candidates) == 0:
                continue
            lastcandidate = candidates[-1]

            pwave = ecgp[lastcandidate[0]:lastcandidate[1]].copy()
            pwave -= np.median(pwave)
            pratio = self.get_ratio(fore,pwave)
            record.qrs[i].pratio = pratio

            if len(candidates) == 1:
                if np.abs(qrs_start - lastcandidate[0]) > 0.5*record.fs and not np.any(record.noise_mask[lastcandidate[0]:qrs_start]):
                    unmatched_p.append(candidates[0])

            record.qrs[i].p = candidates[-1]
            if len(candidates) > 1:
                for candidate in candidates[:-1]:
                    if not np.any(record.noise_mask[candidate[0]:qrs_start]):
                        unmatched_p.append(candidate)

        record.unmatched_p = unmatched_p

        return record

    def find_p_rhythm(self, record: Record):

        p_binary = record.delineation["p"]["binary"]
        p_binary = openingcentered(p_binary, np.ones(int(0.01 * record.fs)))
        p_binary = closingcentered(p_binary, np.ones(int(0.05 * record.fs)))

        p_waves = get_regions(p_binary)
        p_centers = [(p[1] + p[0]) // 2 for p in p_waves]

        start = 1
        steadyrhythms = []
        while start < len(p_centers)-1:
            end = start

            error = False
            while end < len(p_centers)-1 and not error:
                end += 1
                for i in range(start, end):
                    if (p_centers[i] - p_centers[i-1])*1.5 < (p_centers[i+1] - p_centers[i]):
                        error = True
                        break
                    if (p_centers[i] - p_centers[i-1])*0.66 > (p_centers[i+1] - p_centers[i]):
                        error = True
                        break

            if end - start > 3:
                #print("Found a rhythm", p_centers[start], p_centers[end])
                return p_centers[start:end]

            start = end

    def draw_poincare(self, record: Record):

        p_binary = record.delineation["p"]["binary"]
        p_binary = openingcentered(p_binary, np.ones(int(0.01 * record.fs)))
        p_binary = closingcentered(p_binary, np.ones(int(0.05 * record.fs)))

        p_waves = get_regions(p_binary)
        p_centers = [(p[1] + p[0]) // 2 for p in p_waves]

        ibis = np.diff(p_centers)
        iqrs = np.quantile(ibis, [0.25, 0.75])
        median = np.median(ibis)
        upperthreshold = median + 2*(iqrs[1] - iqrs[0])
        #ibis = ibis[ibis < upperthreshold]

        x = ibis[:-1]
        y = ibis[1:]

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
        ax.plot(x, y, color="black")
        ax.scatter(x, y, color="red")
        plt.savefig("poincare.png")
        plt.show()

    def get_starting_point(self, ibis, qrs_median, before=None):

        median = np.median([ibi for ibi in ibis])
        before = before if before is not None else len(ibis)
        #print("Median: ", median)

        start = 0
        end = 0
        streaks = []
        while start < len(ibis):
            end = start
            for i in range(start, len(ibis)):
                end = i
                if ibis[i] > median*1.25 or ibis[i] < median*0.75:
                    break

            #print(start, end)
            streaks.append((start, end))
            start += 1

        #print(streaks)
        longest = 0
        longeststreak = (0, 0)
        for streak in streaks:
            if streak[1] - streak[0] >= 3:
                longest = streak[1] - streak[0]
                longeststreak = streak
                break

        return longeststreak

    def get_pr(self, record: Record, pos):

        beats = record.get_beats()
        #get closest beat after pos
        i = 0
        for i in range(len(beats)):
            if beats[i].qrs[0] > pos:
                break

        if i == 0:
            return 0

        pr = beats[i].qrs[0] - pos

        return pr

    def remove_false_positives(self, record: Record, p_binary, preds, p_centers):

        fp = []
        for p in p_centers:
            found = False
            for pred in preds:
                if abs(pred - p) < int(0.1*record.fs):
                    found = True
                    break
            if not found:
                fp.append(p)

        if len(fp) <= 3:
            for f in fp:
                pr = self.get_pr(record, f)
                if pr > 0.2*record.fs:
                    #print("Found false positive")
                    p_binary[(f-int(0.1*record.fs)):(f+int(0.1*record.fs))] = 0

        return p_binary

    def analyse_predictions(self, record: Record, preds, p_centers):

        tp = 0
        fp = 0
        fn = 0

        afib = record.afib["binary"]
        isnoise = record.noise_mask

        for p in p_centers:
            found = False
            for pred in preds:
                if abs(pred - p) < int(0.1*record.fs) and afib[pred] == 0 and isnoise[pred] == 0:
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1

        return tp, fp, fn

    def analyse_threshold(self, record: Record, possible_peaks=[], threshold=0.5):

        p_binary = record.cpp_record.delineations.p.logits > threshold
        p_binary = openingcentered(p_binary, np.ones(int(0.01 * record.fs)))
        p_binary = closingcentered(p_binary, np.ones(int(0.05 * record.fs)))
        afib = record.cpp_record.delineations.afib.binary
        isnoise = record.cpp_record.delineations.noise.binary

        p_waves = get_regions(p_binary)
        p_centers = [(p[0] + p[1]) // 2 for p in p_waves]# if afib[p[0]:p[1]].max() == 0 and isnoise[p[0]:p[1]].max() == 0]
        
        beats = record.qrs
        qrs_centers = [beat.r for beat in beats]
        qrs_median = np.median(np.diff(qrs_centers))

        ibis = np.diff(p_centers)
        start, end = self.get_starting_point(ibis, qrs_median)
        #print("Starting point", start, end)

        preds = []
        tp = 0
        fp = 0
        fn = 0
        if end - start >= 3:
            preds = self.get_predicted_locations(p_centers, (start, end))

            tp, fp, fn = self.analyse_predictions(record, preds, p_centers)
        else:
            tp = len([p for p in p_waves if afib[p[0]:p[1]].max() == 0 and isnoise[p[0]:p[1]].max() == 0])
            fp = len([p for p in p_waves if afib[p[0]:p[1]].max() == 1 or isnoise[p[0]:p[1]].max() == 1])

        for i in range(len(possible_peaks)):
            matched = False
            for p in range(len(p_centers)):
                if abs(possible_peaks[i] - p_centers[p]) < int(0.1*record.fs):
                    matched = True
                    break
            if not matched:
                fn += 1

        if tp == 0:
            return tp, fp, fn, 0

        se = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f1 = 2 * (se * ppv) / (se + ppv)

        return tp, fp, fn, f1


    def check_hypothesis(self, prediction, actual, std, two_sided=False):
        gauss = np.random.normal(0, std, 1000)
        delta = actual - prediction
        if two_sided:
            delta = np.abs(actual - prediction)
            return (np.sum(gauss > delta) + np.sum(gauss < -delta)) / 1000
        
        delta = prediction - actual
        return np.sum(gauss > delta) / 1000

    def get_predicted_locations(self, p_centers, longeststreak):

        predictions = []
        #print(p_centers, longeststreak)
        running_median = np.median(np.diff(p_centers[longeststreak[0]:longeststreak[1]]))
        running_std = running_median//8
        alpha = 0.33

        for k in range(longeststreak[0], longeststreak[1]):
            predictions.append(p_centers[k])

        i = longeststreak[1]

        while i < len(p_centers):
            pred = int(p_centers[i-1] + running_median)
            actual = p_centers[i]
            std = running_median//8
            p = self.check_hypothesis(pred, actual, running_std, two_sided=True)

            #print(p)
            if p < 0.05: #significantly different -> too late or too early
                #print("Significantly different")
                if pred > actual:
                    #print("This beat is too early, so we will search for the next beat which is at a multiple of the median interval")
                    
                    m = 1
                    g = 0
                    s = running_std
                    while m < 4:
                        pred = int(p_centers[i-1] + m*running_median)
                        for k in range(i, len(p_centers)):
                            p = self.check_hypothesis(pred, p_centers[k], s, two_sided=True)
                            if p > 0.05:
                                g = k
                                break
                        if g == 0:
                            s *= 1.5
                            m += 1
                        else:
                            break

                    if m == 4:
                        #print("No suitable beat found")
                        predictions.append(actual)
                        running_std = running_median//8
                        i += 1
                        continue

                    #print(f"Found a suitable beat at {m} times the median interval")
                    predictions.append(p_centers[g])
                    i += 1
                else:
                    #print("This beat is too late, so we will search for the next beat which is at a multiple of the median interval")
                    m = 2
                    s = running_std
                    while m < 4:
                        pred = int(p_centers[i-1] + m*running_median)
                        s *= 1.5
                        p = self.check_hypothesis(pred, actual, s, two_sided=True)
                        if p > 0.05:
                            break
                        m += 1

                    if m == 4:
                        #print("No suitable beat found")
                        predictions.append(actual)
                        running_std = running_median//8
                        i += 1
                        continue

                    #print(f"Found a suitable beat at {m} times the median interval")
                    for j in range(m):
                        pred = int(p_centers[i-1] + (j+1)*running_median)
                        predictions.append(pred)
                    i += 1
            else:
                running_median = (1-alpha)*running_median + alpha*(p_centers[i] - p_centers[i-1])
                predictions.append(actual)
                running_std = running_median//8
                i += 1

        predictions = [p for p in predictions if p < p_centers[-1]]

        return predictions     

    def correct_by_rhythm(self, record: Record):

        #self.find_p_rhythm(record)
        self.draw_poincare(record)

    def reflect(self, record):
        
        st = time.time()

        self.process_uncertainty(record)
        self.remove_qrs_regions(record)
        possible_peaks = self.get_positive_peaks(record)

        print("Finished finding peaks in ", time.time()-st)
        st = time.time()

        thresholds = {}
        best_f1 = 0
        best_threshold = 0.25
        for t in np.linspace(0.5, 0.1, 10):
            tp, fp, fn, f1 = self.analyse_threshold(record, possible_peaks, threshold=t)
            thresholds[t] = (tp, fp, fn, f1)
            print(f"Threshold: {t}, TP: {tp}, FP: {fp}, FN: {fn}, F1: {f1}")

        print("Finished analysing thresholds in ", time.time()-st)
        st = time.time()

        #select threshold with highest tp and lowest fp
        best_threshold = 0.5
        best_tp = 0
        best_f1 = 0
        best_fn = float('inf')
        for t in thresholds:
            tp, fp, fn, f1 = thresholds[t]
            if f1 >= best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"Best threshold: {best_threshold}")

        record = self.filter_binary_signal(record, threshold=best_threshold)

        print("Finished filtering binary signal in ", time.time()-st)


        return record