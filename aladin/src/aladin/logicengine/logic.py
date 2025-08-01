import numpy as np
from aladin.utils.helpers import Record
import re
import time
import ruptures as rpt

import aladin._main as cpp_backend
from aladin.utils.helpers import Cluster, Beat, Record, resize_signal, get_regions, closingcentered, openingcentered



class LogicEngine():
    def __init__(self, debug=False, customarrhythmia=None):
        #print("LogicEngine initialized")
        self.debug = debug
        self.arrhythmia = {
            "AFIB": True,
            "NOISE": True,
            "AVB": True,
            "CHB": True,
            "SVT": True,
            "EAR": True,
            "NSR": True,
            "Wenckebach" : True,
            "VT": True,
            "TRIGEMINY": True,
            "BIGEMINY": True,
            "IVR": True,
            "PAC_PVC": True,
        }
        if customarrhythmia is not None:
            for key in customarrhythmia:
                if key in self.arrhythmia:
                    self.arrhythmia[key] = customarrhythmia[key]
                else:
                    print("Warning: Custom arrhythmia " + key + " not recognized, skipping.")
        pass

    def get_hr_in_range(self, record: Record, start: int, end: int):
        beats = record.qrs
        beats = [b for b in beats if b.r >= start and b.r <= end]

        if np.all([beat.abnormal for beat in beats]):
            ibis = [beats[i].rr_raw for i in range(1, len(beats))]
        else:
            ibis = [beats[i].rr_raw for i in range(1, len(beats))]

        hr = 60/np.nanmean(ibis)
        return hr

    def check_paroxysmal_tachycardia(self, record: Record):
        
        beats = record.qrs
        afib = record.delineations.afib.logits
        afib_uncertainty = record.delineations.afib.uncertainty

        #if total number of beats is less than 5, skip
        if len(beats) < 5:
            return

        #get the RR intervals
        rrs = [60/beats[i].rr_smooth if beats[i].diagnosis != "AFIB" and beats[i].rr_smooth > 0 else np.nan for i in range(1,len(beats))]
        # print(rrs)

        # algo = rpt.Pelt(model="rbf").fit(np.array(rrs))
        # change_points = algo.predict(pen=10)  # Adjust penalty to be more or less sensitive
        #print("Change points detected at indices:", change_points)

        #get regions of tachycardia
        tachycardia = np.array(rrs) > 100
        tachycardia = openingcentered(tachycardia, np.ones(5))
        tachycardia = closingcentered(tachycardia, np.ones(2))
        tachycardia = get_regions(tachycardia)
        #print("Regions of tachycardia", tachycardia)
        


        for region in tachycardia:

            #if all RR intervals are NaN, or if the region is too small, skip
            if np.all([np.isnan(beats[i].rr) for i in range(region[0],region[1]-1)]) or region[1] - region[0] < 3:
                continue
            
            #get estimate of HR
            hr = 60/np.nanmean([beats[i].rr_smooth for i in range(region[0],region[1]-1)])

            afib_in_region = np.mean(afib[beats[region[0]].onset:beats[region[1]].offset])
            
            #get an estimate of the average TP interval and the ratio between P and QRS complexes
            tp_intervals = []
            num_p = 0
            for i in range(max(region[0],1),region[1]-1):
                if beats[i].p is not None and beats[i-1].t is not None:
                    tp_intervals.append((beats[i].p.onset - beats[i-1].t.offset)/record.fs)
                    num_p += 1
                else:
                    tp_intervals.append(np.nan)
            pqrs_ratio = num_p/(region[1]-region[0])
            if len(tp_intervals) > 0 and not np.all(np.isnan(tp_intervals)):
                tp = np.nanmean(tp_intervals)
            else:
                tp = 0

            #Determine if the region is tachycardia or not
            if pqrs_ratio >= 0.5:
                if tp <= 0.05:
                    record.add_diagnosis(
                        "SVT",
                        "Supraventricular tachycardia detected as the ECG shows typical (narrow) QRS complexes at a heart rate of " + str(np.round(hr)) + " which is more than 100 bpm. Although the rhythm does show P waves, these are too close to the previous T-waves to be considered normal.",
                        beats[region[0]].r,
                        beats[region[1]-1].r
                    )
                else:
                    record.add_subdiagnosis(
                        "TACHYCARDIA",
                        "Normal sinus rhythm with tachycardia detected. The ECG shows normal (narrow) QRS complexes at a heart rate of " + str(np.round(hr)) + " which is more than 100 bpm. The rhythm shows P waves, which are normal and those are not too close to the previous T-waves.",
                        beats[region[0]].r,
                        beats[region[1]-1].r
                    )
            elif afib_in_region < 0.25:
            #else:
                record.add_diagnosis(
                    "SVT",
                    "Supraventricular tachycardia detected as the ECG shows typical (narrow) QRS complexes at a heart rate of " + str(np.round(hr)) + " which is more than 100 bpm. The rhythm does not show P waves, which is typical for SVT. ",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )
            else: 
                record.add_diagnosis(
                    "AFIB",
                    "Supraventricular tachycardia detected as the ECG shows typical (narrow) QRS complexes at a heart rate of " + str(np.round(hr)) + " which is more than 100 bpm. The rhythm does not show P waves, which is typical for SVT. ",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )


    def check_ear(self, record: Record):

        invp = []
        unmatchedp = []
        beats = record.qrs
        for i in range(len(beats)):
            invp.append(beats[i].p is not None and beats[i].p.inverted and beats[i].junctional == False and not beats[i].diagnosis == "SUDDEN_BRADY" and not beats[i].diagnosis == "SVT")

        if len(invp) < 5:
            return record

        invp = np.array(invp, dtype=int)
        #print(invp)
        invp = np.pad(invp, (1,1), 'constant', constant_values=(0,0))
        #print(invp)
        invp = openingcentered(invp, np.ones(4))
        #print(invp)
        invp = invp[1:-1]
        #print(invp)

        # #invp = np.convolve(invp, np.ones(5), mode='valid')
        # inverted_regions = invp >= 3
        inverted_regions = get_regions(invp)

        for region in inverted_regions:
            #print("Inverted region, check p-waves", region[0], region[1])
            record.add_diagnosis(
                "EAR",
                "EAR detected as the ECG shows inverted P-waves while the QRS and T-waves are upright. This is typical for EAR.",
                beats[region[0]].r,
                beats[region[1]-1].r
            )

        #Check if P waves were clustered at all
        if np.all([p.cluster == None for p in record.p]):
            return
        
        group_sig = [beat.p.cluster.id if beat.p is not None and beat.p.cluster is not None else -1 for beat in record.qrs]
        groups = np.unique(group_sig)
        groups = [g for g in groups if g != -1]

        #print(group_sig)
        #print(groups)

        proper_groups = []
        n_groups = 0

        for group in groups:
            proper_groups.append([])

            #Get a binary mask of all p waves in the group
            group_level = group_sig == group

            #Only keep the p waves of beats that are not yet diagnosed 
            group_level = np.array([group_level[i] if beats[i].diagnosis == "" else 0 for i in range(len(beats))], dtype=int)

            #Do a morphological opening to remove streaks shorter than 6
            group_level = np.pad(group_level, (1,1), 'constant', constant_values=(0,0))
            group_level = openingcentered(group_level, np.ones(5))
            group_level = group_level[1:-1]

            #Get the remaining streaks
            group_level_regions = get_regions(group_level)
            if len(group_level_regions) > 0:
                n_groups += 1
                proper_groups[-1] = group_level_regions

        #Get the group ids of the p wave groups that are longer than 6 long where no beat was diagnosed
        groupinds = [i for i in range(len(proper_groups)) if len(proper_groups[i]) > 0]
        #print("Group ids", groupinds)

        #If no streak is left, skip
        if len(groupinds) == 0:
            return

        #Only do something if we have more than 2 large p wave streaks
        if n_groups > 1:

            #search the group with the tallest p wave
            heights = []
            for ind in groupinds:
                #print("Group id", ind)
                clst = [cl for cl in record.p_clusters if cl.id == groups[ind]][0]
                heights.append(np.max(clst.template.ecg))

            group_with_tallest_p_wave = groupinds[np.argmax(heights)]

            #Find other groups that are not the tallest
            for ind, group in enumerate(proper_groups):

                for region in group:
                    #print("Group region", region[0], region[1])
                    if ind == group_with_tallest_p_wave:
                        continue
                    
                    record.add_diagnosis(
                        "EAR",
                        "EAR detected as the ECG shows a consistent streak of abnormal P-waves when compared to the rest of the recording.",
                        beats[region[0]].r,
                        beats[region[1]-1].r
                    )

    def check_avb(self, record: Record):

        dangling_p_waves = [p for p in record.p if p.unmatched]
        beats = record.qrs
        allpwaves = record.p
        dangling_indices = [i for i in range(len(allpwaves)) if allpwaves[i].unmatched]

        #check for consecutive dangling P-waves
        consecutive_dangling = []
        for st in dangling_indices:
            alldangling = True
            en = st+1
            for i in range(st+1, len(allpwaves)):
                if allpwaves[i].unmatched == False:
                    en = i
                    alldangling = False
                    break
                num_beats = len([b for b in beats if b.r > allpwaves[st].onset and b.r < allpwaves[i].onset])
                if num_beats > 0:
                    alldangling = False
                    en = i
                    break
                hasnoise = np.any(record.delineations.noise.binary[allpwaves[st].onset:allpwaves[i].offset])
                if hasnoise:
                    alldangling = False
                    en = i
                    break
                
            if alldangling:
                en = len(allpwaves)
            if en - st > 1:
                consecutive_dangling.append((st,en))
        
        #print("Consecutive dangling", consecutive_dangling)

        to_remove = []
        removed_ids = []
        for streak in consecutive_dangling:
            last_beat = None
            for i in range(len(beats)):
                if beats[i].r < allpwaves[streak[0]].onset:
                    last_beat = i
                else:
                    break
            next_beat = last_beat + 1 if last_beat is not None else None

            if last_beat is None or next_beat >= len(beats):
                #print("No last or next beats found, skip")
                continue

            if beats[last_beat].p is None or beats[next_beat].p is None:
                #print("No P-waves found, skip")
                continue
            
            #Check if the P-wave intervals are stable
            qrsibi = beats[next_beat].rr_raw
            pibi = [allpwaves[i].onset - allpwaves[i-1].onset for i in range(streak[0]+1,streak[1])]
            pibi.append(beats[next_beat].p.onset - allpwaves[streak[1]-1].onset)
            maxdiff = np.max(np.abs(np.diff(pibi)))

            if maxdiff > 0.25*np.mean(pibi):
                #print("P-wave intervals are not stable, skip")
                continue



            pibi = np.median(pibi)/record.fs
            #print("QRS ibi", qrsibi)
            #print("P ibi", pibi)
            #print("Pibi * streak", pibi*(streak[1]-streak[0]+1))

            if np.abs(qrsibi - pibi*(streak[1]-streak[0]+1)) > 0.1*qrsibi:
                #print("QRS interval is not a multiple of the PP interval", np.abs(qrsibi - pibi*(streak[1]-streak[0]+1)), 0.1*qrsibi)
                continue

            if (beats[last_beat].diagnosis == "" or beats[last_beat].diagnosis == "AVB_TYPE2") and beats[next_beat].diagnosis == "":
                beats[last_beat].diagnosis = "AVB_TYPE2"
                beats[next_beat].diagnosis = "AVB_TYPE2"
                to_remove.append(list(range(streak[0], streak[1]+1)))


        removed_ids = [dangling_p_waves[i].id for i in range(len(dangling_p_waves)) if i in to_remove]
        dangling_p_waves = [dangling_p_waves[i] for i in range(len(dangling_p_waves)) if i not in to_remove]

        for i, wave in enumerate(dangling_p_waves):
            #print("Checking Dangling P-wave", wave)
            last_beat = None
            for j in range(len(beats)):
                if beats[j].r < wave.onset:
                    last_beat = j
                else:
                    break

            next_beat = last_beat + 1 if last_beat is not None else None

            if last_beat is None or next_beat >= len(beats):
                #print("No last or next beats found, skip")
                continue

            pwaves_between = [p for p in allpwaves if p.onset > beats[last_beat].r and p.onset < beats[next_beat].r]
            hasnoise = np.any(record.delineations.noise.binary[beats[last_beat].onset:beats[next_beat].offset])
            if len(pwaves_between) != 2:
                #print("No two P-waves between beats, skip")
                continue
            
            if hasnoise:
                #print("Noise in between beats, skip")
                continue

            last_pr = beats[last_beat].pr
            next_pr = beats[next_beat].pr

            if np.isnan(last_pr) or np.isnan(next_pr):
                #print("PR interval is NaN, skip")
                continue

            #print("Last PR", last_pr, "Next PR", next_pr)
            #print("Last beat", last_beat, "Next beat", next_beat)
            qrsibi = beats[next_beat].rr_raw
            pibi = ((beats[next_beat].p.onset - wave.onset))/(record.fs)
            #print("QRS ibi", qrsibi)
            #print("P ibi", pibi)

            if np.abs(qrsibi - 2*pibi) > 0.1*qrsibi:
                #print("QRS interval is not a multiple of the PP interval -> not an AVB 2:1 block")
                continue

            if last_pr - next_pr > 0.03:
                #print("PR interval changes significantly, this could be wenckebach")
                continue

            if (beats[last_beat].diagnosis == "" or beats[last_beat].diagnosis == "AVB_TYPE2") and beats[next_beat].diagnosis == "":
                beats[last_beat].diagnosis = "AVB_TYPE2"
                beats[next_beat].diagnosis = "AVB_TYPE2"
                to_remove.append(i)

        #print(to_remove)
        removed_ids.append([dangling_p_waves[i].id for i in range(len(dangling_p_waves)) if i in to_remove])
        removed_ids = [item for sublist in removed_ids for item in sublist]

        for wave in record.p:
            if wave.id in removed_ids:
                wave.unmatched = False

        is_avb = np.array([beat.diagnosis == "AVB_TYPE2" for beat in beats])
        is_avb = np.pad(is_avb, (1,1), 'constant', constant_values=(0,0))
        is_avb = is_avb[1:-1]

        for i in range(len(beats)):
            if beats[i].diagnosis == "AVB_TYPE2":
                beats[i].diagnosis = ""

        avb_regions = get_regions(is_avb)

        for region in avb_regions:
            record.add_diagnosis(
                "AVB_TYPE2",
                "",
                beats[region[0]].r,
                beats[region[1]-1].r
            )

        return record
        
    def check_wenckebach(self, record: Record):

        dangling_p_waves = [p for p in record.p if p.unmatched]
        beats = record.qrs
        allpwaves = record.p

        #print("Remaining dangling P-waves", len(dangling_p_waves))

        for i, wave in enumerate(dangling_p_waves):
            #print("Checking Dangling P-wave", wave)
            last_beat = None
            for i in range(len(beats)):
                if beats[i].r < wave.onset:
                    last_beat = i
                else:
                    break

            next_beat = last_beat + 1 if last_beat is not None else None

            if last_beat is None or next_beat >= len(beats):
                #print("No last or next beats found, skip")
                continue

            last_pr = beats[last_beat].pr
            next_pr = beats[next_beat].pr if beats[next_beat].pr < 0.5 else np.nan

            if np.isnan(last_pr):
                #print("PR interval is NaN, skip")
                continue

            #print("Last PR", last_pr, "Next PR", next_pr)
            #print("Last beat", last_beat, "Next beat", next_beat)

            if not np.isnan(next_pr) and last_pr - next_pr <= 0.03:
                #print("PR interval does not change significantly, this cannot be wenckebach")
                continue

            no_next = False
            if np.isnan(next_pr):
                no_next = True

            last_rr = beats[last_beat].rr_raw
            next_rr = beats[next_beat].rr_raw
            #print("Last RR", last_rr, "Next RR", next_rr, last_rr*2, np.abs(last_rr*2 - next_rr), 0.1*next_rr)
            if not (np.abs(last_rr*2 - next_rr) < 0.25*next_rr or (np.abs(last_rr - next_rr) < 0.1*next_rr)):
                #print("RR interval is not a multiple of the PP interval and it is not a 2:1 wenckebach -> not a wenckebach")
                continue

            hasnoise = np.any(record.delineations.noise.binary[beats[last_beat].onset:beats[next_beat].offset])
            if hasnoise:
                #print("Noise in between beats, skip")
                continue

            st = last_beat
            prevpr = beats[st].pr
            streak = []
            beatinds = [st]
            ps = [beats[st].p.onset, wave.onset]
            p_wave_status = []
            if beats[next_beat].p is not None and beats[next_beat].pr < 0.5:
                ps.append(beats[next_beat].p.onset)

            if last_beat == 0:
                st -= 1
            else:
                while st > 0:
                    st -= 1
                    #print("Checking beat", st, "PR", beats[st].pr)
                    if beats[st].p is None:
                        #print("No P-wave found at beat", st)
                        break
                    if beats[st].pr >= prevpr:
                        #print("PR interval does not decrease at beat", st)
                        break
                    if beats[st].pr > 0.5:
                        break
                    if beats[st].junctional:
                        #print("Junctional beat found at beat", st)
                        break
                    if len([p for p in allpwaves if p.onset > beats[st].r and p.onset < beats[st+1].r]) > 1:
                        #print("Too many P-waves between beats", st, st+1)
                        break

                    streak.append(prevpr)
                    beatinds.append(st)
                    prevpr = beats[st].pr
                    ps.append(beats[st].p.onset)

            streak.append(prevpr)
            st += 1

            #sort ps
            ps = np.sort(ps)
            pibis = np.diff(ps)

            #print(streak)
            if np.any(np.abs(np.diff(pibis)) > 0.25*np.median(pibis)):
                #print("P-wave intervals are not stable", ps, pibis, 0.25*np.median(pibis))
                continue
            
            if np.abs(next_pr - streak[0]) > 0.3:
                #print("PR interval change is too large")
                continue

            if len(streak) == 1:
                continue

            if len(streak) == 1 and beats[next_beat].junctional:
                #print("Next beat is junctional but streak is only 1 beat long, not enough information to determine if this is wenckebach")
                continue
            
            if no_next:
                if len(streak) == 1 or np.any([s > 0.5 for s in streak]) or streak[-1] > 0.3 or np.any([beats[i].abnormal for i in beatinds]):
                    #print("We use strict criteria for wenckebach if no next beat found, this cannot be wenckebach")
                    continue

            if (beats[last_beat].diagnosis == "" or beats[last_beat].diagnosis == "WENCKEBACH" or (beats[last_beat].diagnosis == "SUDDEN_BRADY" and len(streak)>2)):
                for i in range(st, next_beat+1):
                    beats[i].diagnosis = "WENCKEBACH"

            


        is_wenck = np.array([beat.diagnosis == "WENCKEBACH" for beat in beats])
        #print("Is wenck", is_wenck)

        for i in range(len(beats)):
            if beats[i].diagnosis == "WENCKEBACH":
                beats[i].diagnosis = ""

        wenck_regions = get_regions(is_wenck)

        for region in wenck_regions:
            record.add_diagnosis(
                "WENCKEBACH",
                "Wenckebach detected because a progressive prolongation of the PR interval was detected before a blocked QRS wave.",
                beats[region[0]].r,
                beats[region[1]-1].r
            )

    def check_chb(self, record: Record):

        beats = record.qrs
        afib = record.delineations.afib.binary
        afib_uncertainty = record.delineations.afib.uncertainty

        #if total number of beats is less than 5, skip
        if len(beats) < 5:
            return

        #get the RR intervals
        rrs = [60/beats[i].rr_raw if (beats[i].diagnosis != "AFIB" and beats[i].rr_raw>0) else np.nan for i in range(len(beats))]

        #get regions of tachycardia
        bradycardia = np.array(rrs) < 45
        bradycardia = np.array([bradycardia[i] and beats[i].diagnosis == "" for i in range(len(beats))])
        bradycardia = openingcentered(bradycardia, np.ones(4))
        bradycardia = get_regions(bradycardia)

        for region in bradycardia:
            region = (max(0, region[0]-1), region[1])
            #print("Bradycardia region", beats[region[0]].r/record.fs, beats[region[1]].r/record.fs)

            pwaves = [p for p in record.p if p.onset >= beats[region[0]].r and p.onset <= beats[region[1]-1].r]
            pibis = [(pwaves[i].onset - pwaves[i-1].onset)/record.fs for i in range(1,len(pwaves))]
            ibis = [beats[i].rr_raw for i in range(region[0],region[1])]
            prs = [beats[i].pr for i in range(region[0],region[1])]

            if len(prs) < 2 or len(pwaves) < 3 or len(ibis) < 2:
                #print("Too few PR intervals")
                continue
            
            if np.all([np.isnan(pr) for pr in prs]):
                #print("No PR intervals found")
                continue

            iqr = np.nanpercentile(prs, 75) - np.nanpercentile(prs, 25)
            prs_raw = prs.copy()
            lower = np.nanpercentile(prs, 25) - 1.5*iqr
            upper = np.nanpercentile(prs, 75) + 1.5*iqr
            prs = [i for i in prs if i >= lower and i <= upper]

            #print("PR intervals", prs)
            prstd = np.nanstd(prs)
            prm = np.nanmedian(prs)
            #print("PR std: ", prstd, "mean: ", prm)
            pibi_median = np.nanmedian(pibis)
            qrsibi_median = np.nanmedian(ibis)
            #print("P-IBI median", pibi_median, "QRS-IBI median", qrsibi_median)
            npwaves = len(pwaves)
            nqrss = len(ibis)

            if prstd < 0.1 or np.all([pr <= 0.3 for pr in prs]) or len([pr for pr in prs if pr > 0.3]) < 2 or npwaves < 3:
                #print("PR interval is too stable or every QRS complex has a PR interval of smaller than 0.3s, this cannot be CHB")
                continue

            if np.abs(qrsibi_median - pibi_median) > 0.1*qrsibi_median or prm > 0.3:
                #print("PR intervals", prs)
                #print("PR std: ", prstd, "mean: ", prm)
                #print("P-IBI median", pibi_median, "QRS-IBI median", qrsibi_median)
                #print("Atrial rhythm is not the same as ventricular rhythm or PR interval is too long -> CHB")
                record.add_diagnosis(
                    "SUDDEN_BRADY",
                    "Sudden bradycardia detected because a sudden drop in HR was detected in combination with an irregular and discordinated atrial rhythm.",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )


    def check_noise(self, record: Record):

        noise_mask = record.delineations.noise.binary
        afib = record.delineations.afib.binary
        noise_mask[afib == 1] = 0
        noise_mask = closingcentered(noise_mask, np.ones(int(record.fs*0.5)))
        noise_mask = openingcentered(noise_mask, np.ones(int(record.fs*0.5)))
        regions = get_regions(noise_mask)

        for region in regions:
            record.add_diagnosis(
                "NOISE", 
                "Severe noise detected in the signal between " + str(region[0]/record.fs) + " and " + str(region[1]/record.fs) + " seconds.",
                region[0],
                region[1]
            )

    def estimate_initial_hr(self, record: Record, step=1):

        beats = record.qrs
        afib = record.delineations.afib.binary
        nn_intervals = []

        for s in range(0,step):
            #print("s: ", step+s)

            if len(nn_intervals) > 2:
                break

            for i in range(step+s,len(beats),step):
                #print("i: ", i, end=" ")

                if afib[beats[i].r]:
                    #print("skipped because of afib or tachy")
                    continue

                #check if we have a p-wave and a qrs complex
                if beats[i].abnormal or beats[i-step].abnormal:
                    #print("skipped because of no p-wave or qrs")
                    continue

                if beats[i].r > len(record.ecg) and len(nn_intervals) > 2:
                    break
                
                #print(beats[i].get_r_wave(), beats[i-step].get_r_wave())
                nn_intervals.append(beats[i].r - beats[i-step].r)

        if len(nn_intervals) == 0:
            if step<4:
                #print("initial hr: ", "nothing found so we will check with skip=", step+1)
                return self.estimate_initial_hr(record, step=step+1)/(step+1)
            return False
        
        #print("initial hr: ", ((60*record.fs)/np.nanmedian(nn_intervals))*(step))
        
        return np.nanmedian(nn_intervals)

     
    def search_first_decent_beat(self, record: Record):
        
        beats = record.qrs
        afib = record.delineations.afib.binary

        for i in range(len(beats)):
            beats[i].startingposition = False

            if afib[beats[i].r] == 0 and beats[i].p is not None and not beats[i].abnormal:
                beats[i].startingposition = True
                return i, record
            
        return np.nan, record
    

    def check_hypothesis(self, prediction, actual, std, two_sided=False):
        gauss = np.random.normal(0, std, 1000)
        delta = actual - prediction
        if two_sided:
            delta = np.abs(actual - prediction)
            return (np.sum(gauss > delta) + np.sum(gauss < -delta)) / 1000
        
        delta = prediction - actual
        return np.sum(gauss > delta) / 1000

    def check_pvc_and_pac(self, record: Record):
        
        beats = record.qrs
        afib = record.delineations.afib.binary

        initialhr = self.estimate_initial_hr(record)
        #print("Mean RR: ", initialhr)
        if not initialhr:
            #print("No initial hr found")
            return record

        startingposition, record = self.search_first_decent_beat(record)
        if np.isnan(startingposition) or (len(beats) - startingposition) < 3:
            #print("No decent beat found")
            return record
    
        #print("Starting position: ", startingposition)

        still_to_check = []
        insert_phantoms = []
        skipnext = False

        for i in range(len(beats)):
            beats[i].isearly = False
            beats[i].prediction_m = 0
            beats[i].prediction_std = 0
            beats[i].uncertain = 0


        #loop over all qrs complexes
        for i in range(startingposition,len(beats)):
            beat = beats[i]
            
            if i>startingposition and i<(len(beats)-1):
                
                history = [(beats[j].ibi if j >= startingposition else initialhr) for j in range(i-3,i)]
                weight = [1,2,3]
                if beats[i].p is not None and not beats[i].abnormal:
                    history.append(beats[i].r - beats[i-1].r)
                    weight.append(4)

                history = np.array(history)
                meanhr = np.sum(history*np.array(weight))/np.sum(weight)
                
                #print("beat:", i+startingposition, " meanhr: ", meanhr, " history:", history, " weights:", weight)

                #see how certain we are about the mean ibi
                uncertainty = np.sum([beats[j].uncertain*max(0,(4-(i-j))/(min(5,i))) for j in range(max(1,i-5),i)])

                #calculate std of ibi
                stdhr = meanhr*0.075

                #set the mean and std of the first 2 beats to the initial heart rate
                if i==startingposition+1:
                    beats[startingposition].hr = beats[i].hr = meanhr
                    beats[startingposition].hrv = beats[i].hrv = stdhr

                beats[i].hr = meanhr
                beats[i].hrv = stdhr
                
                last_normal_beat = i-1
                while last_normal_beat > -2 and (beats[last_normal_beat].isearly or beats[last_normal_beat].abnormal):
                    last_normal_beat -= 1

                #check if we are allowed to handle this beat
                if not skipnext:
                    
                    #print_debug("Handle beat ", i, "cur_ibi = ",pqrsts[i]["hr"], "(",np.round((pqrsts[i]["hr"]/fs),3),"s)")
                    #make prediction based on previous beat and average heart rate
                    predictedlocation_cur_mean = beats[i-1].r + beats[i].hr
                    predictedlocation_cur_std = beats[i-1].hrv*(1+uncertainty)
                    cur_too_early = self.check_hypothesis(predictedlocation_cur_mean, beats[i].r, predictedlocation_cur_std, two_sided=False)
                    cur_too_late = self.check_hypothesis(-predictedlocation_cur_mean, -beats[i].r, predictedlocation_cur_std, two_sided=False)

                    #make second order prediction of next beat based on current beat and average heart rate
                    predictedlocation_next_mean = beats[i].r + beats[i].hr
                    next_too_late = self.check_hypothesis(-predictedlocation_next_mean, -beats[i+1].r, predictedlocation_cur_std, two_sided=False)
                    next_on_time = self.check_hypothesis(predictedlocation_next_mean, beats[i+1].r, predictedlocation_cur_std, two_sided=True)

                    #make prediction based on 2RR rule and average heart rate
                    predictedlocation_next_2RR = beats[i-1].r + 2*beats[i].hr
                    next_2RR = self.check_hypothesis(-predictedlocation_next_2RR, -beats[i+1].r, predictedlocation_cur_std, two_sided=False)

                    predictedlocation_next_1RR = beats[i-1].r + beats[i].hr
                    next_1RR = self.check_hypothesis(predictedlocation_next_1RR, beats[i+1].r, predictedlocation_cur_std*2, two_sided=True)

                    #check if the current beat is too early. When its a ventricular beat, we can be more certain about it
                    if (cur_too_early < 0.05 and beats[i].abnormal) or (cur_too_early < 0.01):

                        beats[i].isearly = True
                        curibi = beats[i].r - beats[i-1].r
                        beats[i].ibi = curibi
                        #print("Too early normal")

                        beats[i].prediction_m = predictedlocation_cur_mean
                        beats[i].prediction_std = predictedlocation_cur_std

                        #check if the following pause follows the full compensatory pause rule (2RR)
                        #use one-sided t-test
                        #print("Next too late: ", next_too_late)
                        if next_1RR > 0.05:
                            #print("Half compensatory pause -> hint: PVC")
                            beats[i+1].isearly = False
                            beats[i].ibi = beats[i-1].hr
                            beats[i+1].ibi = beats[i-1].hr
                            beats[i+1].uncertain = 0.5
                            beats[i+1].prediction_m = predictedlocation_next_1RR
                            beats[i+1].prediction_std = predictedlocation_cur_std
                            # if beats[i].abnormal and (beats[i].qrs[1] - beats[i].qrs[0]) > 0.12*record.fs:
                            #     beats[i].part_of = "PVC"
                            #beats[i].estimated_diagnosis"] = "PVC"
                            skipnext = True

                        elif next_2RR < 0.05:
                            #print("Full compensatory pause -> hint: PVC")
                            beats[i+1].isearly = False
                            beats[i+1].ibi = beats[i-1].hr
                            beats[i].ibi = beats[i-1].hr
                            beats[i+1].uncertain = 0.5
                            beats[i+1].prediction_m = predictedlocation_next_2RR
                            beats[i+1].prediction_std = predictedlocation_cur_std
                            if beats[i].abnormal and (beats[i].width) > 0.12*record.fs and beats[i].diagnosis not in ["VT", "IVR", "BIGEMINY", "TRIGEMINY"]:
                                beats[i].diagnosis = "PVC"
                            #beats[i]["estimated_diagnosis"] = "PVC"
                            skipnext = True

                        #if the following pause does not follow the full compensatory pause rule, check if the sinus node has been reset 
                        #if that happens than we have a post-extrasystolic pause
                        elif next_too_late < 0.1:
                            #print("Post-extrasystolic pause -> hint: PAC")
                            beats[i+1].isearly = False
                            beats[i+1].ibi = beats[i-1].hr
                            beats[i].ibi = beats[i-1].hr
                            beats[i+1].uncertain = 0.5
                            beats[i+1].prediction_m = predictedlocation_next_2RR
                            beats[i+1].prediction_std = predictedlocation_cur_std
                            beats[i].diagnosis = "PAC" if not beats[i].abnormal else "PVC" if beats[i].diagnosis not in ["VT", "IVR", "BIGEMINY", "TRIGEMINY"] else beats[i].diagnosis
                            #beats[i].estimated_diagnosis = "PAC"
                            skipnext = True
                        elif next_on_time > 0.05:
                            #print("Normal pause after early beat -> PAC")
                            beats[i].diagnosis = "PAC" if not beats[i].abnormal else "PVC" if beats[i].diagnosis not in ["VT", "IVR", "BIGEMINY", "TRIGEMINY"] else beats[i].diagnosis
                            beats[i].ibi = beats[i-1].hr
                            skipnext = False

                        
                        #print("=====================================")
                        #print("")
                    else:
                        #TODO: check current beat on PVC likeleyhood. It cannot be an atrial premature beat, but it can still be a ventricular premature beat

                        #check if we are a bit too early but with a normal beat. Then, if we are significantly late on the next beat, we could have dealt with a PAC
                        
                        if cur_too_early < 0.1 and not beats[i].abnormal and next_too_late < 0.05: 
                            #print("PAC")
                            beats[i].isearly = True
                            #use previous normal ibi to maintain heart rate
                            beats[i].ibi = beats[i-1].hr

                            beats[i].prediction_m = predictedlocation_cur_mean
                            beats[i].prediction_std = predictedlocation_cur_std
                            beats[i].diagnosis = "PAC"
                            #beats[i].estimated_diagnosis = "PAC"

                            if next_2RR < 0.05:
                                #print("Full compensatory pause -> hint: PVC")
                                beats[i+1].isearly = False
                                beats[i+1].ibi = beats[i-1].hr
                                beats[i+1].uncertain = 0.5
                                beats[i+1].prediction_m = predictedlocation_next_2RR
                                beats[i+1].prediction_std = predictedlocation_cur_std
                                #print("Already existing diagnosis: ", beats[i].diagnosis)
                                if beats[i].abnormal and (beats[i].width) > 0.12*record.fs and beats[i].diagnosis not in ["VT", "IVR", "BIGEMINY", "TRIGEMINY"]:
                                    beats[i].diagnosis = "PVC"
                                skipnext = True

                            #if the following pause does not follow the full compensatory pause rule, check if the sinus node has been reset 
                            #if that happens than we have a post-extrasystolic pause
                            elif next_too_late < 0.05:
                                #print("Post-extrasystolic pause -> hint: PAC")
                                beats[i+1].isearly = False
                                beats[i+1].ibi = beats[i-1].hr
                                beats[i+1].uncertain = 0.5
                                beats[i+1].prediction_m = predictedlocation_next_2RR
                                beats[i+1].prediction_std = predictedlocation_cur_std
                                beats[i].diagnosis = "PAC"
                                skipnext = True

                        #check if we are dealing with a late beat
                        elif cur_too_late < 0.05:
                            #print("late beat at ", beats[i].r/record.fs)
                            beats[i].isearly = False
                            curibis = beats[i].r - beats[i-1].r

                            #print("We found a late beat at ", beats[i].r/record.fs, " s")
                            hypothesis = self.check_hypothesis(beats[i-1].r+beats[i].hr, (beats[i-1].r+beats[i].r)/2, beats[i].hrv, two_sided=True)
                            #print("Hypothesis: ", hypothesis)
                            #print("Previous beat normality: ", beats[i-1].abnormal)
                            #print("Previous ibi: ", beats[i-1].ibi)

                            #if the beat is far too late, we cannot use the current ibi to maintain heart rate. Instead we copy the previous ibi
                            if curibis > beats[i-1].ibi*2:
                                beats[i].ibi = beats[i-1].ibi
                                beats[i+1].uncertain = 0.5

                            #if the beat is not too late, we can use the average of the current and previous ibi's to adjust but maintain heart rate
                            else:
                                beats[i].ibi = beats[i-1].ibi
                                #pqrsts[i]["ibi"] = curibis
                                beats[i+1].uncertain = 0.5

                            beats[i].prediction_m = predictedlocation_cur_mean
                            beats[i].prediction_std = predictedlocation_cur_std

                        else:
                            beats[i].isearly = False
                            beats[i].prediction_m = predictedlocation_cur_mean
                            beats[i].prediction_std = predictedlocation_cur_std
                            beats[i].ibi = beats[i].r - beats[i-1].r
                else:
                    #print("Beat ", i, " is not handled, skipped")
                    #print("Ibi: ", beats[i].ibi)
                    skipnext = False
                    # if beats[i].qrs is None:
                    #     beats[i].isearly = False
                    #     beats[i].prediction_m = beats[i].r
                    #     beats[i].prediction_std = 0
                    #     beats[i].ibi = beats[i].r - beats[i-1].r
            else:
                beats[i].isearly = False
                beats[i].ibi = initialhr if i < startingposition+1 else beats[i-1].ibi
                beats[i].hrv = record.fs*0.1
                beats[i].hr = beats[startingposition].ibi

            #print("=====================================")

        for beat in beats:
            if beat.diagnosis == "PAC":
                record.add_subdiagnosis(
                    "PAC",
                    "A single premature atrial contraction (PAC) detected.",
                    beat.r,
                    beat.r+1
                )
            elif beat.diagnosis == "PVC":
                record.add_subdiagnosis(
                    "PVC",
                    "A single premature ventricular contraction (PVC) detected.",
                    beat.r,
                    beat.r+1
                )


        return record

    def check_nsr(self, record: Record):

        beats = record.qrs
        p_waves = record.p

        hasp = []
        for i in range(len(beats)):
            hasp.append(beats[i].junctional == False and beats[i].diagnosis == "" and beats[i].p is not None and beats[i].double_p == False and beats[i].pr < 0.5)

        if len(hasp) < 5:
            return record

        hasp = np.array(hasp, dtype=int)
        hasp = np.pad(hasp, (2,2), 'constant', constant_values=(1,1))
        hasp = np.convolve(hasp, np.ones(5), mode='valid')
        normal_regions = hasp >= 4
        normal_regions = get_regions(normal_regions)

        for region in normal_regions:
            #print("Normal region, check p-waves", region[0], region[1])
            prs = []
            if region[1] - region[0] < 2:
                continue

            for i in range(region[0],region[1]):
                prs.append(beats[i].pr)

            #Get HR in the normal region
            hr = self.get_hr_in_range(record, beats[region[0]].r, beats[region[1]-1].r)

            #Get mean beat widths
            widths = [beats[i].width/record.fs for i in range(region[0], region[1])]
            meanwidth = np.mean(widths)

            #Get initial HR
            mean_rr = self.estimate_initial_hr(record)
            if not mean_rr:
                initialhr = hr
            else:
                initialhr = 60/(mean_rr/record.fs)

            #print("HR", initialhr)
            #print("Widths", meanwidth)
            #print("Mean stdiff", meanstdiff)
            #print("PR intervals", prs)

            if np.all([np.isnan(pr) for pr in prs]) or len(prs) < 2:
                #print("No PR intervals found")
                continue

            prsstd = np.nanstd(prs)
            prsm = np.round(np.nanmean(prs)*100)/100

            #print("PR interval mean: ", prsm, "std: ", prsstd, "threshold: ", 0.1*prsm, 0.2)
            if initialhr < 50:
                record.add_subdiagnosis(
                    "BRADYCARDIA",
                    "Bradycardia detected because the heart rate is below 50 BPM.",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )

            if meanwidth >= 0.14:
                record.add_subdiagnosis(
                    "IVB",
                    "Normal sinus rhythm with an widened QRS complex detected. This could indicate an intraventricular block.",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )

            
            if prsstd < 0.1 and prsm < 0.2:
                record.add_diagnosis(
                    "NSR",
                    "Normal sinus rhythm with regular PR interval detected, typical QRS width, and a normal HR of " + str(np.round(hr))+ " which is between 50 and 100 BPM.",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )
            elif prsstd < 0.1 and prsm < 0.5:
                record.add_diagnosis(
                    "AVB_TYPE1",
                    "Normal sinus rhythm with a prolonged PR interval detected. The atrial rhythm is regular and corresponds to the ventricular rhyhtm. This indicates a first degree AV block.",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )
            elif prsm < 0.2:
                record.add_diagnosis( 
                    "NSR",
                    "Normal sinus rhythm detected, typical QRS width, and a normal HR of " + str(np.round(hr))+ " which is between 50 and 100 BPM. However, we could not detect a regular PR interval. ",
                    beats[region[0]].r,
                    beats[region[1]-1].r
                )

        isnoisy = []
        for i in range(len(beats)):
            isnoisy.append(beats[i].snr < 20 and beats[i].diagnosis == "" and beats[i].p is None and 60/beats[i].rr_smooth < 100 and not np.isnan(beats[i].rr))

        isnoisy = np.array(isnoisy, dtype=int)
        isnoisy = np.pad(isnoisy, (1,1), 'constant', constant_values=(0,0))
        isnoisy = openingcentered(isnoisy, np.ones(2))
        isnoisy = closingcentered(isnoisy, np.ones(3))
        isnoisy = isnoisy[1:-1]

        noisy_regions = get_regions(isnoisy)

        for region in noisy_regions:
            record.add_diagnosis(
                "NOISE",
                "Severe noise detected.",
                beats[region[0]].r,
                beats[region[1]-1].r
            )

    def check_rhythms(self, record: Record):

        beats = record.qrs

        #Get the beat types as a string
        beattypes = ["V" if b.abnormal else "N" for b in beats]
        beatstr = "".join(beattypes)
        #print("Beat string: ", beatstr)
        afib = record.delineations.afib.binary
        afibregions = get_regions(afib)
        
        if self.arrhythmia["AFIB"]:
            #Check for AFIB
            for region in afibregions:
                beats_inside_region = [b for b in beats if b.r >= region[0] and b.r <= region[1]]
                
                record.add_diagnosis(
                    "AFIB", 
                    "Atrial fibrillation detected due to abnormal atrial activity and irregular ventricular rhythm.",
                    region[0],
                    region[1]
                )

        st = time.time()
        #Correct errors when many ventricular beats are present
        beatstr = beatstr.replace("VVVNVVV", "VVVVVVV")
        beatstr = beatstr.replace("VVVNNVVV", "VVVVVVVV")

        if self.arrhythmia["VT"] or self.arrhythmia["IVR"] or self.arrhythmia["SVT"]:
            # i) Find multiple, consecutive V's
            vt_ivr_matches = re.finditer(r'V{3,}', beatstr)
            for match in vt_ivr_matches:
                #get HR of the streak
                hr = self.get_hr_in_range(record, beats[match.start()].r, beats[match.end()-1].r)
                period = (60 / hr)*record.fs if hr > 0 else 0

                #print("Ventricular match: ", match.start(), match.end(), "HR: ", hr)

                #get median width
                medianwidth = np.mean([(beats[i].width) for i in range(match.start(), match.end())])

                #print("Median width: ", medianwidth/record.fs)

                #check if any are part of AFIB
                isafib = np.any([beat.diagnosis == "AFIB" for beat in beats[match.start()+1:match.end()-1]])

                if isafib:
                    continue
                
                #check PR intervals
                prs = [beats[i].pr for i in range(match.start(), match.end())]
                if not np.all(np.isnan(prs)) and len(prs) > 2:
                    prsstd = np.nanstd(prs)
                    prsm = np.nanmedian(prs)

                    #check if we are dealing with CHB that is causing IVR
                    if prsstd > 0.1 or prsm > 0.5:
                        #print("There are P waves and those signal a CHB causing IVR")
                        continue
                
                if hr < 100 and medianwidth >= 0.12*record.fs:
                    #If HR is low but QRS is wide, we have IVR
                    record.add_diagnosis(
                        "IVR",
                        "Detected IVR with a heart rate of " + str(hr) + " bpm across " + str(match.end()-match.start()) + " beats. The QRS complexes are widened and show an abnormal shape while the ventricular rhythm is regular and the HR is below 100 bpm.",
                        beats[match.start()].r,
                        beats[match.end()-1].r
                    )
                elif hr > 100 and medianwidth/period >= 0.25:
                    # If HR is high and QRS is wide, we have VT
                    record.add_diagnosis(
                        "VT",
                        "Detected VT with a heart rate of " + str(hr) + " bpm across " + str(match.end()-match.start()) + " beats. The QRS complexes are widened and show an abnormal shape while the ventricular rhythm is regular and the HR is above 100 bpm.",
                        beats[match.start()].r,
                        beats[match.end()-1].r
                    )
                elif hr > 140 and medianwidth >= 0.12*record.fs:
                    # If HR is high and QRS is wide, we have VT
                    record.add_diagnosis(
                        "VT",
                        "Detected VT with a heart rate of " + str(hr) + " bpm across " + str(match.end()-match.start()) + " beats. The QRS complexes are widened and show an abnormal shape while the ventricular rhythm is regular and the HR is above 100 bpm.",
                        beats[match.start()].r,
                        beats[match.end()-1].r
                    )
                elif hr > 100 and medianwidth <= 0.12*record.fs:
                    # If HR is high and QRS is narrow, we have SVT
                    record.add_diagnosis(
                        "SVT",
                        "Detected SVT with a heart rate of " + str(hr) + " bpm across " + str(match.end()-match.start()) + " beats. Although beats show an abnormal shape, beats have a median QRS width of " + str(medianwidth/record.fs) + " seconds, which is smaller than 120ms. At the same time, the HR is above 100 BPM. As such, this is likely SVT.",
                        beats[match.start()].r,
                        beats[match.end()-1].r
                    )


        if self.arrhythmia["BIGEMINY"]:
            # ii) Find repeating NV patterns of at least 3 repeats
            big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beatstr)
            for match in big_matches:
                record.add_diagnosis(
                    "BIGEMINY",
                    "Detected BIGEMINY across " + str(match.end()-match.start()) + " beats. A normal beat is followed by a ventricular beat in a repeating pattern.",
                    beats[match.start()].r,
                    beats[match.end()-1].r
                )

        if self.arrhythmia["TRIGEMINY"]:
            # iii) Find repeating NNV patterns of at least 3 repeats
            tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beatstr)
            for match in tri_matches:
                record.add_diagnosis(
                    "TRIGEMINY",
                    "Detected TRIGEMINY across " + str(match.end()-match.start()) + " beats. Two normal beats are followed by a ventricular beat in a repeating pattern.",
                    beats[match.start()].r,
                    beats[match.end()-1].r
                )

        if self.arrhythmia["TRIGEMINY"]:
            # iv) Find repeating NNNV patterns of at least 3 repeats
            tri_matches = re.finditer(r'((VNNN){3,}|(NNNV){3,})', beatstr)
            for match in tri_matches:
                record.add_diagnosis(
                    "QUADRIGEMINY",
                    "Detected QUADRIGEMINY across " + str(match.end()-match.start()) + " beats. Three normal beats are followed by a ventricular beat in a repeating pattern.",
                    beats[match.start()].r,
                    beats[match.end()-1].r
                )

        
        #Now check for junctional rhythms
        beattypes = ["J" if b.junctional else "V" if b.abnormal else "N" for b in beats]
        beatstr = "".join(beattypes)
        beatstr = beatstr.replace("JJJVJJJ", "JJJJJJJ")
        beatstr = beatstr.replace("JJJNJJJ", "JJJJJJJ")

        j_matches = re.finditer(r'J{3,}', beatstr)
        for match in j_matches:
            #get HR of the streak
            hr = self.get_hr_in_range(record, beats[match.start()].r, beats[match.end()-1].r)
            #print("Junctional match: ", match.start(), match.end(), "HR: ", hr)

            #if HR is lower than 100
            if hr < 100:
                #print("Junctional rhythm detected")
                record.add_diagnosis(
                    "JUNCTIONAL",
                    "Detected JUNCTIONAL rhythm as no atrial activity was observed while the ventricular beat shows a narrow QRS complex. Furthermore, the heart rate is below 100 BPM and the ventricular rhythm is regular.",
                    beats[match.start()].r,
                    beats[match.end()-1].r
                )

        #print("V Rhythm check took: ", time.time() - st, " seconds")

        st = time.time()
        #Check for SVT
        if self.arrhythmia["SVT"]:
            self.check_paroxysmal_tachycardia(record)
            #print("SVT check took: ", time.time() - st, " seconds")
        st = time.time()
        
        #Check for AVB
        if self.arrhythmia["AVB"]:
            self.check_avb(record)
            #print("AVB check took: ", time.time() - st, " seconds")
        st = time.time()

        #Check for CHB
        if self.arrhythmia["CHB"]:
            self.check_chb(record)
            #print("CHB check took: ", time.time() - st, " seconds")
        st = time.time()

        #Check for Wenckebach
        if self.arrhythmia["Wenckebach"]:
            self.check_wenckebach(record)
            #print("Wenckebach check took: ", time.time() - st, " seconds")
        st = time.time()

        #Check for Ectopic Atrial Rhythms
        if self.arrhythmia["EAR"]:
            self.check_ear(record)
            #print("Ectopic Atrial Rhythm check took: ", time.time() - st, " seconds")
        st = time.time()

        #Check for normal sinus rhythm
        if self.arrhythmia["NSR"]:
            self.check_nsr(record)
            #print("NSR check took: ", time.time() - st, " seconds")
        

        return record

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

    def junctional_beats(self, record: Record):

        beats = record.qrs
        afib = record.delineations.afib.binary

        for i in range(len(beats)):
            # if (beats[i].p is not None):
            #     print(f"[{i}] Pos:", beats[i].onset/record.fs ," Width: ", beats[i].width, "Abnormal: ", beats[i].abnormal, "SNR: ", beats[i].snr, "P-wave: (", beats[i].p.onset, beats[i].p.offset, ") PRatio: ", beats[i].pratio)
            # else:
            #     print(f"[{i}] Pos:", beats[i].onset/record.fs ," Width: ", beats[i].width, "Abnormal: ", beats[i].abnormal, "SNR: ", beats[i].snr, "P-wave: () PRatio: ", beats[i].pratio)
                
            if (beats[i].abnormal and beats[i].width > 0.12 * record.fs) or afib[beats[i].onset:beats[i].offset].any() or beats[i].snr < 30:
                beats[i].junctional = False
                continue

            if beats[i].p is None:
                beats[i].junctional = True
            elif beats[i].p is not None and beats[i].onset < beats[i].p.onset:
                beats[i].junctional = True
            elif beats[i].p is not None and (beats[i].onset - beats[i].p.onset) < 0.075*record.fs:
                beats[i].junctional = True
            elif beats[i].p is not None and beats[i].pratio > 50:
                beats[i].junctional = True
            else:
                beats[i].junctional = False
    
    def print_diagnoses(self, record: Record):
        print("PREDICTED DIAGNOSES ======================")
        print("Main diagnoses:")
        for diagnosis in record.diagnosis:
            print(diagnosis.name, "between", diagnosis.onset/record.fs, "and", diagnosis.offset/record.fs, "seconds, duration: ", (diagnosis.offset-diagnosis.onset)/record.fs, "seconds")
        # print("Subdiagnoses:")
        # for diagnosis in record.subdiagnosis:
        #     print(diagnosis.name, "between", diagnosis.onset/record.fs, "and", diagnosis.offset/record.fs, "seconds")
        print("=========================================")
        # print("EXPLANATION ==============================")
        # print(record.explanation)
        # print("=========================================")

    def get_full_explanation(self, record: Record):
        explanation = ""
        for diagnosis in record.diagnosis:
            explanation += diagnosis.explanation + "\n"
        for diagnosis in record.subdiagnosis:
            explanation += diagnosis.explanation + "\n"
        return explanation

    def diagnose(self, record: Record):
        #print("Diagnose record", record.recordname)
        
        st = time.time()
        self.check_noise(record)
        #print("Noise check took", time.time() - st, "seconds")
        st = time.time()

        self.junctional_beats(record)
        #print("Junctional beats took", time.time() - st, "seconds")
        st = time.time()

        self.check_rhythms(record)
        #print("Rhythm check took", time.time() - st, "seconds")
        st = time.time()

        if self.arrhythmia["PAC_PVC"]:
            self.check_pvc_and_pac(record)
            #print("PVC and PAC check took", time.time() - st, "seconds")
        st = time.time()

        #self.print_diagnoses(record)

        return record
