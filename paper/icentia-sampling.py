
#search random files
#search for AFIB, VT, IVR, BIG, TRI, Pauses where both humans and algorithm have results (both sensitivity and specificity)
#search for AVB2, Wenckebach, CHB, SVT, where only the algorithm has results (only specificity)

import numpy as np
import pickle
import os
import wfdb
import matplotlib.pyplot as plt
import re
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import gzip
from tqdm import tqdm
import json
import time


class StratifiedSampler:
    def __init__(self, root, num_samples=250):

        self.root = root
        self.dataroot = '/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA'
        self.paths = []
        self.patients = []
        self.all_paths_per_patient = {}
        self.samples_human_per_patient_per_arrhythmia = {}
        self.samples_aladin_per_patient_per_arrhythmia = {}
        self.samples_human_per_arrhythmia = {}
        self.samples_aladin_per_arrhythmia = {}
        self.num_samples = num_samples
        self.num_human_samples = 0
        self.num_aladin_samples = 0
        self.num_human_samples_per_arrhythmia = {}
        self.num_aladin_samples_per_arrhythmia = {}
        self.num_samples_searched_per_arrhythmia = {}
        self.samples = {}
        self.all_data = []
        self.all_data_per_patient = {}
        self.mapping = {}
        self.aladin_corrected = {}

        self.find_paths()

    def find_paths(self):
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.endswith('.pkl.gz'):
                    path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(path, self.root)
                    self.paths.append(relative_path[:-7])
                    patient = relative_path.split("/")[1]
                    self.patients.append(patient)
                    self.samples_human_per_patient_per_arrhythmia[patient] = {}
                    self.samples_aladin_per_patient_per_arrhythmia[patient] = {}

    def calculate_overlap(self, onset1, offset1, onset2, offset2):
        """
        Calculate the overlap between two time intervals.
        """
        start = max(onset1, onset2)
        end = min(offset1, offset2)
        overlap = max(0, end - start)
        return overlap   

    def analyse_file(self, path):

        # get human diagnoses
        anns = wfdb.rdann(os.path.join(self.dataroot, path), 'atr')
        human_diagnoses = self.get_diagnosis_from_human(anns)

        patient = path.split("/")[1]

        # get Aladin diagnoses
        with gzip.open(os.path.join(self.root, path + ".pkl.gz"), 'rb') as f:
            data = pickle.load(f)

        path = data["path"]

        aladin_diagnoses = self.get_diagnosis_from_aladin(data["diagnosis"])

        unique_human = list(set(hd[0] for hd in human_diagnoses))
        unique_aladin = list(set(ad["type"] for ad in aladin_diagnoses))

        not_interested = ["TACHYCARDIA", "BRADYCARDIA", "SVT"]
        unique_human = [d for d in unique_human if d not in not_interested]
        unique_aladin = [d for d in unique_aladin if d not in not_interested]

        selected_arrhythmias_human = []
        selected_arrhythmias_aladin = []

        for arrhythmia in list(self.num_samples_searched_per_arrhythmia.keys()):
            if arrhythmia not in self.samples:
                self.num_samples_searched_per_arrhythmia[arrhythmia] += 1
            elif len(self.samples[arrhythmia]) < self.num_samples:
                self.num_samples_searched_per_arrhythmia[arrhythmia] += 1

        for arrhythmia in unique_human:
            if arrhythmia not in self.samples_human_per_patient_per_arrhythmia[patient]:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 0
            else:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] += 1
                #continue

            if arrhythmia not in self.samples_human_per_arrhythmia:
                self.samples_human_per_arrhythmia[arrhythmia] = 0

            if arrhythmia not in self.samples:
                self.samples[arrhythmia] = []

            # if  len(self.samples[arrhythmia]) >= self.num_samples:
            #     continue
            
            self.samples_human_per_arrhythmia[arrhythmia] += 1
             
            human_arrhythmias = [{"type": a[0], "onset": int(a[1]), "offset": int(a[2]), "human": True, "aladin": False, "path": path, "patient": patient} for a in human_diagnoses if a[0] == arrhythmia]
            human_arrhythmia = np.random.choice(human_arrhythmias, 1)[0]
            aladin_arrhythmias = [{"type": a["type"], "onset": int(a["onset"]), "offset": int(a["offset"])} for a in aladin_diagnoses if a["type"] == human_arrhythmia["type"]]
             
            for arr in aladin_arrhythmias:
                overlap = self.calculate_overlap(arr["onset"], arr["offset"], human_arrhythmia["onset"], human_arrhythmia["offset"])
                if overlap > 0:
                    human_arrhythmia["aladin"] = True
                    break

            selected_arrhythmias_human.append(human_arrhythmia)


        for arrhythmia in unique_aladin:
            if arrhythmia not in self.samples_aladin_per_patient_per_arrhythmia[patient]:
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 0
            else:
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] += 1
                #continue

            if arrhythmia not in self.samples_aladin_per_arrhythmia:
                self.samples_aladin_per_arrhythmia[arrhythmia] = 0

            if arrhythmia not in self.samples:
                self.samples[arrhythmia] = []
            # if len(self.samples[arrhythmia]) >= self.num_samples:
            #     continue

            self.samples_aladin_per_arrhythmia[arrhythmia] += 1
            
            aladin_arrhythmias = [{"type": a["type"], "onset": int(a["onset"]), "offset": int(a["offset"]), "human": False, "aladin": True, "path": path, "patient": patient} for a in aladin_diagnoses if a["type"] == arrhythmia]
            aladin_arrhythmia = np.random.choice(aladin_arrhythmias, 1)[0]
            human_arrhythmias = [{"type": a[0], "onset": int(a[1]), "offset": int(a[2])} for a in human_diagnoses if a[0] == aladin_arrhythmia["type"]]

            for arr in human_arrhythmias:
                overlap = self.calculate_overlap(arr["onset"], arr["offset"], aladin_arrhythmia["onset"], aladin_arrhythmia["offset"])
                if overlap > 0:
                    aladin_arrhythmia["human"] = True
                    break
            
            selected_arrhythmias_aladin.append(aladin_arrhythmia)

        for selected_arrhythmia_human in selected_arrhythmias_human:
            # if len(self.samples[selected_arrhythmia_human["type"]]) >= self.num_samples:
            #     continue

            if selected_arrhythmia_human["type"] not in self.samples:
                self.samples[selected_arrhythmia_human["type"]] = []
            if selected_arrhythmia_human["type"] not in self.num_human_samples_per_arrhythmia:
                self.num_human_samples_per_arrhythmia[selected_arrhythmia_human["type"]] = 0

            self.samples[selected_arrhythmia_human["type"]].append(selected_arrhythmia_human)
            self.num_human_samples_per_arrhythmia[selected_arrhythmia_human["type"]] += 1
            self.num_human_samples += 1

        for selected_arrhythmia_aladin in selected_arrhythmias_aladin:
            # if len(self.samples[selected_arrhythmia_aladin["type"]]) >= self.num_samples:
            #     continue

            if selected_arrhythmia_aladin["type"] not in self.samples:
                self.samples[selected_arrhythmia_aladin["type"]] = []
            if selected_arrhythmia_aladin["type"] not in self.num_aladin_samples_per_arrhythmia:
                self.num_aladin_samples_per_arrhythmia[selected_arrhythmia_aladin["type"]] = 0

            self.samples[selected_arrhythmia_aladin["type"]].append(selected_arrhythmia_aladin)
            self.num_aladin_samples_per_arrhythmia[selected_arrhythmia_aladin["type"]] += 1
            self.num_aladin_samples += 1

    def analyse_file_from_json(self, path, objs_per_path):

        human_diagnoses = []
        for obj in objs_per_path:
            if obj["human"]:
                human_diagnoses.append([obj["type"], obj["onset"], obj["offset"]])

        patient = path.split("/")[1]

        aladin_diagnoses = []
        for obj in objs_per_path:
            if obj["aladin"]:
                aladin_diagnoses.append({"type": obj["type"], "onset": int(obj["onset"]), "offset": int(obj["offset"])})

        unique_human = list(set(hd[0] for hd in human_diagnoses))
        unique_aladin = list(set(ad["type"] for ad in aladin_diagnoses))

        not_interested = ["TACHYCARDIA", "BRADYCARDIA"]
        unique_human = [d for d in unique_human if d not in not_interested]
        unique_aladin = [d for d in unique_aladin if d not in not_interested]

        selected_arrhythmias_human = []
        selected_arrhythmias_aladin = []

        # for arrhythmia in list(self.num_samples_searched_per_arrhythmia.keys()):
        #     if arrhythmia not in self.samples:
        #         self.num_samples_searched_per_arrhythmia[arrhythmia] += 1
        #     elif len(self.samples[arrhythmia]) < self.num_samples:
        #         self.num_samples_searched_per_arrhythmia[arrhythmia] += 1

        for arrhythmia in unique_human:
            if arrhythmia not in self.samples_human_per_patient_per_arrhythmia[patient]:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 0
            else:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] += 1
                continue

            if arrhythmia not in self.samples_human_per_arrhythmia:
                self.samples_human_per_arrhythmia[arrhythmia] = 0

            if arrhythmia not in self.samples:
                self.samples[arrhythmia] = []

            if  len(self.samples[arrhythmia]) >= self.num_samples:
                continue
            
            self.samples_human_per_arrhythmia[arrhythmia] += 1
             
            human_arrhythmias = [{"type": a[0], "onset": int(a[1]), "offset": int(a[2]), "human": True, "aladin": False, "path": path, "patient": patient} for a in human_diagnoses if a[0] == arrhythmia]
            human_arrhythmia = np.random.choice(human_arrhythmias, 1)[0]
            aladin_arrhythmias = [{"type": a["type"], "onset": int(a["onset"]), "offset": int(a["offset"])} for a in aladin_diagnoses if a["type"] == human_arrhythmia["type"]]
             
            for arr in aladin_arrhythmias:
                overlap = self.calculate_overlap(arr["onset"], arr["offset"], human_arrhythmia["onset"], human_arrhythmia["offset"])
                if overlap > 0:
                    human_arrhythmia["aladin"] = True
                    break

            selected_arrhythmias_human.append(human_arrhythmia)


        for arrhythmia in unique_aladin:
            if arrhythmia not in self.samples_aladin_per_patient_per_arrhythmia[patient]:
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 0
            else:
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] += 1
                continue

            if arrhythmia not in self.samples_aladin_per_arrhythmia:
                self.samples_aladin_per_arrhythmia[arrhythmia] = 0

            if arrhythmia not in self.samples:
                self.samples[arrhythmia] = []
            if len(self.samples[arrhythmia]) >= self.num_samples:
                continue

            self.samples_aladin_per_arrhythmia[arrhythmia] += 1
            
            aladin_arrhythmias = [{"type": a["type"], "onset": int(a["onset"]), "offset": int(a["offset"]), "human": False, "aladin": True, "path": path, "patient": patient} for a in aladin_diagnoses if a["type"] == arrhythmia]
            aladin_arrhythmia = np.random.choice(aladin_arrhythmias, 1)[0]
            human_arrhythmias = [{"type": a[0], "onset": int(a[1]), "offset": int(a[2])} for a in human_diagnoses if a[0] == aladin_arrhythmia["type"]]

            for arr in human_arrhythmias:
                overlap = self.calculate_overlap(arr["onset"], arr["offset"], aladin_arrhythmia["onset"], aladin_arrhythmia["offset"])
                if overlap > 0:
                    aladin_arrhythmia["human"] = True
                    break
            
            selected_arrhythmias_aladin.append(aladin_arrhythmia)

        for selected_arrhythmia_human in selected_arrhythmias_human:
            if len(self.samples[selected_arrhythmia_human["type"]]) >= self.num_samples:
                continue

            if selected_arrhythmia_human["type"] not in self.samples:
                self.samples[selected_arrhythmia_human["type"]] = []
            if selected_arrhythmia_human["type"] not in self.num_human_samples_per_arrhythmia:
                self.num_human_samples_per_arrhythmia[selected_arrhythmia_human["type"]] = 0

            self.samples[selected_arrhythmia_human["type"]].append(selected_arrhythmia_human)
            self.num_human_samples_per_arrhythmia[selected_arrhythmia_human["type"]] += 1
            self.num_human_samples += 1

        for selected_arrhythmia_aladin in selected_arrhythmias_aladin:
            if len(self.samples[selected_arrhythmia_aladin["type"]]) >= self.num_samples:
                continue

            if selected_arrhythmia_aladin["type"] not in self.samples:
                self.samples[selected_arrhythmia_aladin["type"]] = []
            if selected_arrhythmia_aladin["type"] not in self.num_aladin_samples_per_arrhythmia:
                self.num_aladin_samples_per_arrhythmia[selected_arrhythmia_aladin["type"]] = 0

            self.samples[selected_arrhythmia_aladin["type"]].append(selected_arrhythmia_aladin)
            self.num_aladin_samples_per_arrhythmia[selected_arrhythmia_aladin["type"]] += 1
            self.num_aladin_samples += 1

    def analyse_obj(self, obj):

        # {
        #     "type": "AFIB",
        #     "onset": 229690,
        #     "offset": 238959,
        #     "human": true,
        #     "aladin": false,
        #     "path": "p03/p03827/p03827_s18",
        #     "patient": "p03827"
        # },

        included_arrhythmia = ["NSR", "NOISE", "VT>10s", "VT<10s", "IVR", "TRIGEMINY", "BIGEMINY"]

        #     "NSR": 0,
        #     "NOISE": 0,
        #     "AFIB": 0,
        #     "VT>10s": 0,
        #     "VT<10s": 0,
        #     "SVT>30s": 0,
        #     "IVR": 0,
        #     "TRIGEMINY": 0,
        #     "BIGEMINY": 0,
        #     "WENCKEBACH": 0,
        #     "AVB_TYPE2": 0,
        #     "SUDDEN_BRADY": 0

        arrhythmia = obj["type"]
        onset = obj["onset"]
        offset = obj["offset"]
        human = obj["human"]
        aladin = obj["aladin"]
        path = obj["path"]
        patient = obj["patient"]

        # if arrhythmia not in included_arrhythmia:
        #     return

        if human and aladin:
            if arrhythmia not in self.samples_human_per_patient_per_arrhythmia[patient] or \
                arrhythmia not in self.samples_aladin_per_patient_per_arrhythmia[patient]:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 0
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 0

                if arrhythmia not in self.samples_human_per_arrhythmia:
                    self.samples_human_per_arrhythmia[arrhythmia] = 0
                if arrhythmia not in self.samples_aladin_per_arrhythmia:
                    self.samples_aladin_per_arrhythmia[arrhythmia] = 0

                if arrhythmia not in self.samples:
                    self.samples[arrhythmia] = []

                if arrhythmia not in self.num_human_samples_per_arrhythmia:
                    self.num_human_samples_per_arrhythmia[arrhythmia] = 0
                if arrhythmia not in self.num_aladin_samples_per_arrhythmia:
                    self.num_aladin_samples_per_arrhythmia[arrhythmia] = 0

                if len(self.samples[arrhythmia]) < self.num_samples:
                    self.samples[arrhythmia].append(obj)
                    self.num_human_samples_per_arrhythmia[arrhythmia] += 1
                    self.num_human_samples += 1
                    self.num_aladin_samples_per_arrhythmia[arrhythmia] += 1
                    self.num_aladin_samples += 1

            else:
                self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 1
                self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 1
        else:
            if human:
                if arrhythmia not in self.samples_human_per_patient_per_arrhythmia[patient]:
                    self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 0

                    if arrhythmia not in self.samples_human_per_arrhythmia:
                        self.samples_human_per_arrhythmia[arrhythmia] = 0

                    if arrhythmia not in self.samples:
                        self.samples[arrhythmia] = []

                    if arrhythmia not in self.num_human_samples_per_arrhythmia:
                        self.num_human_samples_per_arrhythmia[arrhythmia] = 0

                    if len(self.samples[arrhythmia]) < self.num_samples:
                        self.samples[arrhythmia].append(obj)
                        self.num_human_samples_per_arrhythmia[arrhythmia] += 1
                        self.num_human_samples += 1

                else:
                    self.samples_human_per_patient_per_arrhythmia[patient][arrhythmia] = 1

            if aladin:
                if arrhythmia not in self.samples_aladin_per_patient_per_arrhythmia[patient]:
                    self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 0

                    if arrhythmia not in self.samples_aladin_per_arrhythmia:
                        self.samples_aladin_per_arrhythmia[arrhythmia] = 0

                    if arrhythmia not in self.samples:
                        self.samples[arrhythmia] = []

                    if arrhythmia not in self.num_aladin_samples_per_arrhythmia:
                        self.num_aladin_samples_per_arrhythmia[arrhythmia] = 0

                    if len(self.samples[arrhythmia]) < self.num_samples:
                        self.samples[arrhythmia].append(obj)
                        self.num_aladin_samples_per_arrhythmia[arrhythmia] += 1
                        self.num_aladin_samples += 1

                else:
                    self.samples_aladin_per_patient_per_arrhythmia[patient][arrhythmia] = 1

        
    def get_diagnosis_from_human(self, anns, fs=250):
        diagnoses = []
        beattypes = [a for a in anns.symbol if a != '+']
        beatpos = [anns.sample[i] for i, a in enumerate(anns.symbol) if a != '+']
        beattypes = "".join(beattypes)
        beattypes = beattypes.replace('S', 'N')

        big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
        for big_match in big_matches:
            start = beatpos[big_match.start()]
            end = beatpos[big_match.end()-1]
            #print("BIG pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
            #if (end-start)/fs > 10:
            diagnoses.append(["BIGEMINY", start, end])

        tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
        for tri_match in tri_matches:
            start = beatpos[tri_match.start()]
            end = beatpos[tri_match.end()-1]
            #print("TRI pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
            #if (end-start)/fs > 10:
            diagnoses.append(["TRIGEMINY", start, end])

        vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
        for vt_ivr_match in vt_ivr_matches:
            start = beatpos[vt_ivr_match.start()]
            end = beatpos[vt_ivr_match.end()-1]
            nbeats = vt_ivr_match.end() - vt_ivr_match.start()
            avg_ibi = (end - start) / (nbeats-1)
            hr = 60 / (avg_ibi / fs)
            #print("VT/IVR pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
            if hr > 100:
                if (end-start)/fs >= 10:
                    diagnoses.append(["VT>10s", start, end])
                else:
                    diagnoses.append(["VT<10s", start, end])
            else:
                diagnoses.append(["IVR", start, end])

        #search for VVV pattern

        low_signal_quality_matches = re.finditer(r'Q{3,}', beattypes)
        for low_signal_quality_match in low_signal_quality_matches:
            start = beatpos[low_signal_quality_match.start()]
            end = beatpos[low_signal_quality_match.end()-1]
            #print("Low signal quality pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (low_signal_quality_match.end()-low_signal_quality_match.start()), "beats")
            if (end-start)/fs > 30:
                diagnoses.append(["NOISE", start, end])

        rhythm_type = ""
        rhythm_start = 0
        #foundafib = False

        for idx, beat in enumerate(anns.symbol):
            if beat == 'N':
                continue
            # if beat == 'V':
            #     start = anns.sample[idx] - fs*0.15
            #     end = anns.sample[idx] + fs*0.15
            #     #labelregions.append(["PVC", start, end])
            # elif beat == 'S':
            #     start = anns.sample[idx] - fs*0.15
            #     end = anns.sample[idx] + fs*0.15
            #     #labelregions.append(["SVPB", start, end])
            if beat == '+':
                if anns.aux_note[idx] == '(N':
                    rhythm_type = "NSR"
                    rhythm_start = anns.sample[idx]
                elif anns.aux_note[idx] == '(AFIB':
                    rhythm_type = "AFIB"
                    rhythm_start = anns.sample[idx]
                    foundafib = True
                    #print("AFIB/AFL", rhythm_start)
                elif anns.aux_note[idx] == '(AFL':
                    rhythm_type = "AFIB"
                    rhythm_start = anns.sample[idx]
                    foundafib = True
                elif anns.aux_note[idx] == ')':
                    if (anns.sample[idx]-rhythm_start)/fs > 30:
                        #if rhythm_type != "NSR":
                        diagnoses.append([rhythm_type, rhythm_start, anns.sample[idx]])
                        # if rhythm_type != "NSR":
                        #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")
                    rhythm_type = ""
                    rhythm_start = 0

        if rhythm_type and rhythm_start:
            #if rhythm_type != "NSR":
            diagnoses.append([rhythm_type, rhythm_start, anns.sample[-1]])
            # if rhythm_type != "NSR":
            #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[-1]-rhythm_start)/fs, "s")

        return diagnoses

    def get_diagnosis_from_aladin(self, diagnoses, fs=250):

        filtered_diagnoses = []

        for diagnosis in diagnoses:
            onset = diagnosis.get("onset", 0)
            offset = diagnosis.get("offset", 0)
            duration = (offset - onset) / 250  # Convert to seconds

            if diagnosis["type"] == "AFIB" and duration >= 30: #min 90% overlap
                filtered_diagnoses.append({"type": "AFIB", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "VT" and duration >= 10:
                filtered_diagnoses.append({"type": "VT>10s", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "VT" and duration < 10:
                filtered_diagnoses.append({"type": "VT<10s", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "SVT" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "SVT>30s", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "SVT" and duration < 30: #min 90% overlap
                filtered_diagnoses.append({"type": "SVT<30s", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "IVR":
                filtered_diagnoses.append({"type": "IVR", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "TRIGEMINY":
                filtered_diagnoses.append({"type": "TRIGEMINY", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "BIGEMINY":
                filtered_diagnoses.append({"type": "BIGEMINY", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "NOISE" and duration >= 30: #min 90% overlap
                filtered_diagnoses.append({"type": "NOISE", "onset": onset, "offset": offset})
            elif diagnosis["type"] == "WENCKEBACH" or \
                    diagnosis["type"] == "AVB" or \
                    diagnosis["type"] == "IVR" or \
                    diagnosis["type"] == "AVB_TYPE2" or \
                    diagnosis["type"] == "SUDDEN_BRADY":
                filtered_diagnoses.append({"type": diagnosis["type"], "onset": onset, "offset": offset})

        #find regions without diagnoses
        if len(filtered_diagnoses) > 0:
            start = 0
            for diagnosis in filtered_diagnoses:
                if diagnosis["onset"] > start:
                    filtered_diagnoses.append({"type": "NSR", "onset": start, "offset": diagnosis["onset"]})
                start = max(start, diagnosis["offset"])
            if start < 250 * 60:
                filtered_diagnoses.append({"type": "NSR", "onset": start, "offset": 250 * 60})

        return filtered_diagnoses

    def sample_from_json(self, json_file="samples_checked.json"):

        raw_samples_file = "/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA/samples_xxl.json"
        #checked_samples_file = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/icentia2_diagnoses2025-08-06_21-41-14.json"
        checked_samples_file = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/icentia2_diagnoses2025-08-09_13-59-36.json"

        with open(raw_samples_file, "r") as f:
            rawdata_per_arrhythmia = json.load(f)

        with open(checked_samples_file, "r") as f:
            checkeddata = json.load(f)

        rawdata = []
        for arrhythmia, samples in rawdata_per_arrhythmia.items():
            rawdata.extend(samples)

        self.all_data = []
        pool = []

        for raw_sample in tqdm(rawdata):
            if raw_sample["type"] in ["NSR", "NOISE"]:
                if raw_sample["offset"] - raw_sample["onset"] >= 60 * 250 and raw_sample["human"] and raw_sample["aladin"]:
                    self.all_data.append(raw_sample)
            elif not raw_sample["aladin"]:
                self.all_data.append(raw_sample)
            else:
                pool.append(raw_sample)

        # for raw_sample in tqdm(rawdata):
        #     if raw_sample["type"] in ["NSR", "NOISE"]:
        #         if raw_sample["offset"] - raw_sample["onset"] >= 60 * 250 and raw_sample["human"] and raw_sample["aladin"]:
        #             all_objs.append(raw_sample)
        #     else:
        #         all_objs.append(raw_sample)

        print(f"Found {len(pool)} samples in the pool.")

        #match checked samples with raw samples
        for checked_sample in tqdm(checkeddata):
            raw_sample = None
            for sample in pool:
                if sample["path"] == checked_sample["path"] and \
                    sample["onset"] == checked_sample["segment"][0] and \
                    sample["offset"] == checked_sample["segment"][1]:
                    raw_sample = sample
                    pool.remove(sample)
                    break
            check_diagnoses = self.get_diagnosis_from_aladin(checked_sample["raw"])

            if raw_sample is not None:
                if raw_sample["type"] in [d["type"] for d in check_diagnoses]:
                    #print(f"Added checked sample")
                    raw_sample["aladin"] = True
                    self.all_data.append(raw_sample)
            else:
                print("ALADIN removed a FP, but no raw sample found", checked_sample["path"], checked_sample["segment"])


        objs_per_path = {}
        for obj in self.all_data:
            if obj["path"] not in objs_per_path:
                objs_per_path[obj["path"]] = []
            objs_per_path[obj["path"]].append(obj)

            if obj["patient"] not in self.all_data_per_patient:
                self.all_data_per_patient[obj["patient"]] = []
            self.all_data_per_patient[obj["patient"]].append(obj)
        #exit()
        # for arrhythmia objs in data.items():
        #     self.all_data.extend(objs)

        print(f"Found {len(self.all_data)} samples in total.")
        print(f"Found {len(self.all_data_per_patient)} patients in total.")

        #self.find_paths()
        n_files = len(self.paths)

        for path in self.paths:
            patient = path.split("/")[1]
            if patient not in self.all_paths_per_patient:
                self.all_paths_per_patient[patient] = []
            self.all_paths_per_patient[patient].append(path)

        print(f"Sampling {n_files} files from {len(self.paths)} total files.")
        print(f"Found {len(self.all_paths_per_patient)} patients with paths.")

        selected_paths = np.random.choice(self.paths, n_files, replace=False)
        self.num_samples_searched_per_arrhythmia = {
            "NSR": 0,
            "NOISE": 0,
            "AFIB": 0,
            "VT>10s": 0,
            "VT<10s": 0,
            "SVT>30s": 0,
            "IVR": 0,
            "TRIGEMINY": 0,
            "BIGEMINY": 0,
            "WENCKEBACH": 0,
            "AVB_TYPE2": 0,
            "SUDDEN_BRADY": 0
        }
        class_mapper = {}
        class_mapper["AFIB"] = "CRITICAL"
        class_mapper["SUDDEN_BRADY"] = "CRITICAL"
        class_mapper["SVT>30s"] = "CRITICAL"
        class_mapper["VT>10s"] = "CRITICAL"
        class_mapper["VT<10s"] = "NONCRITICAL"
        class_mapper["AVB_TYPE2"] = "NONCRITICAL"
        class_mapper["WENCKEBACH"] = "NONCRITICAL"
        class_mapper["BIGEMINY"] = "NONCRITICAL"
        class_mapper["TRIGEMINY"] = "NONCRITICAL"
        class_mapper["IVR"] = "NONCRITICAL"
        class_mapper["NSR"] = "NORMAL"
        class_mapper["NOISE"] = "NORMAL"

        self.num_samples_searched_per_arrhythmiaclass = {
            "NORMAL": 0,
            "NONCRITICAL": 0,
            "CRITICAL": 0,
        }
        patients = list(self.all_paths_per_patient.keys())
        patients = np.random.choice(patients, len(patients), replace=False)

        t0 = time.time()
        for i, patient in enumerate(patients):
            paths = self.all_paths_per_patient[patient]
            if len(paths) == 0:
                continue

            paths = np.random.choice(paths, len(paths), replace=False)

            for arrhythmia in list(self.num_samples_searched_per_arrhythmia.keys()):
                if arrhythmia not in self.samples:
                    self.num_samples_searched_per_arrhythmia[arrhythmia] = 1
                    #self.num_samples_searched_per_arrhythmiaclass[class_mapper[arrhythmia]] += 1
                elif len(self.samples[arrhythmia]) < self.num_samples:
                    self.num_samples_searched_per_arrhythmia[arrhythmia] += 1
                    #self.num_samples_searched_per_arrhythmiaclass[class_mapper[arrhythmia]] += 1

            for arrhythmiaclass in list(self.num_samples_searched_per_arrhythmiaclass.keys()):
                arrhythmias = [a for a in self.num_samples_searched_per_arrhythmia.keys() if class_mapper[a] == arrhythmiaclass]
                finished = True
                for arrhythmia in arrhythmias:
                    if self.num_samples_searched_per_arrhythmia[arrhythmia] < self.num_samples:
                        finished = False
                        break

                if not finished:
                    self.num_samples_searched_per_arrhythmiaclass[arrhythmiaclass] += 1

            for path in paths:
                objs = objs_per_path.get(path, [])
                self.analyse_file_from_json(path, objs)

            t1 = time.time()
            elapsed = t1 - t0
            estimated_time = (n_files - i - 1) * elapsed / (i + 1)
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            estimated_time_str = time.strftime("%M:%S", time.gmtime(estimated_time))
            # remaining_time = estimated_time - elapsed
            # remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            print(f"[{i+1}/{n_files} | {elapsed_str}<{estimated_time_str}] ", end="")
            for arrhythmia in list(self.samples.keys()):
                print(f"{arrhythmia}: {len(self.samples[arrhythmia])}", end=", ")
                if len(self.samples[arrhythmia]) < self.num_samples:
                    foundall = False
            print(end="\r")


        # t0 = time.time()
        # for i, path in enumerate(selected_paths):
        #     objs = objs_per_path.get(path, [])
        #     self.analyse_file_from_json(path, objs)
        #     foundall = True
        #     t1 = time.time()
        #     elapsed = t1 - t0
        #     estimated_time = (n_files - i - 1) * elapsed / (i + 1)
        #     elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
        #     estimated_time_str = time.strftime("%M:%S", time.gmtime(estimated_time))
        #     # remaining_time = estimated_time - elapsed
        #     # remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
        #     print(f"[{i+1}/{n_files} | {elapsed_str}<{estimated_time_str}] ", end="")
        #     for arrhythmia in list(self.samples.keys()):
        #         print(f"{arrhythmia}: {len(self.samples[arrhythmia])}", end=", ")
        #         if len(self.samples[arrhythmia]) < self.num_samples:
        #             foundall = False
        #     print(end="\r")
        #     if foundall:
        #         print("Found all samples for all arrhythmias.")
        #         break

        print("N samples human:", self.num_human_samples)
        print("N samples Aladin:", self.num_aladin_samples)

        for arrhythmia, n_samples in self.num_human_samples_per_arrhythmia.items():
            print(f"N samples human for {arrhythmia}: {n_samples}")
        for arrhythmia, n_samples in self.num_aladin_samples_per_arrhythmia.items():
            print(f"N samples Aladin for {arrhythmia}: {n_samples}")

        for arrhythmia, n_samples in self.num_samples_searched_per_arrhythmia.items():
            print(f"Patients searched for {arrhythmia}: {n_samples}")

        for arrhythmia, n_samples in self.num_samples_searched_per_arrhythmiaclass.items():
            print(f"Patients searched for {arrhythmia}: {n_samples}")

        # print("N samples human per patient per arrhythmia:")
        # for patient, arrhythmias in self.samples_human_per_patient_per_arrhythmia.items():
        #     print(f"{patient}: {arrhythmias}")

        # print("N samples Aladin per patient per arrhythmia:")
        # for patient, arrhythmias in self.samples_aladin_per_patient_per_arrhythmia.items():
        #     print(f"{patient}: {arrhythmias}")


    def sample(self, n_files=None):

        if n_files == None:
            n_files = len(self.paths)

        selected_paths = np.random.choice(self.paths, n_files, replace=False)
        self.num_samples_searched_per_arrhythmia = {
            "NSR": 0,
            "NOISE": 0,
            "AFIB": 0,
            "VT>10s": 0,
            "VT<10s": 0,
            "SVT>30s": 0,
            "IVR": 0,
            "TRIGEMINY": 0,
            "BIGEMINY": 0,
            "WENCKEBACH": 0,
            "AVB_TYPE2": 0,
            "SUDDEN_BRADY": 0
        }

        t0 = time.time()
        for i, path in enumerate(selected_paths):
            self.analyse_file(path)
            foundall = True
            t1 = time.time()
            elapsed = t1 - t0
            estimated_time = (n_files - i - 1) * elapsed / (i + 1)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
            # remaining_time = estimated_time - elapsed
            # remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            print(f"[{i+1}/{n_files} | {elapsed_str}<{estimated_time_str}] ", end="")
            for arrhythmia in list(self.samples.keys()):
                print(f"{arrhythmia}: {len(self.samples[arrhythmia])}", end=", ")
                if len(self.samples[arrhythmia]) < self.num_samples:
                    foundall = False
            print(end="\r")
            # if foundall:
            #     print("Found all samples for all arrhythmias.")
            #     break

        print("N samples human:", self.num_human_samples)
        print("N samples Aladin:", self.num_aladin_samples)

        for arrhythmia, n_samples in self.num_human_samples_per_arrhythmia.items():
            print(f"N samples human for {arrhythmia}: {n_samples}")
        for arrhythmia, n_samples in self.num_aladin_samples_per_arrhythmia.items():
            print(f"N samples Aladin for {arrhythmia}: {n_samples}")

        for arrhythmia, n_samples in self.num_samples_searched_per_arrhythmia.items():
            print(f"Patients searched for {arrhythmia}: {n_samples}")

        # print("N samples human per patient per arrhythmia:")
        # for patient, arrhythmias in self.samples_human_per_patient_per_arrhythmia.items():
        #     print(f"{patient}: {arrhythmias}")

        # print("N samples Aladin per patient per arrhythmia:")
        # for patient, arrhythmias in self.samples_aladin_per_patient_per_arrhythmia.items():
        #     print(f"{patient}: {arrhythmias}")

    def get_performance(self, arrhythmia):

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for sample in self.samples.get(arrhythmia, []):
            if sample["human"] and sample["aladin"]:
                tp += 1
            elif not sample["human"] and sample["aladin"]:
                fp += 1
            elif sample["human"] and not sample["aladin"]:
                tn += 1

            patient = sample["patient"]
            patient_samples = [s for s in self.all_data_per_patient.get(patient, []) if s["type"] == arrhythmia]
            found = any(s["aladin"] for s in patient_samples)
            if sample["human"] and not found:
                fn += 1
                #print(f"False negative for {arrhythmia} in patient {patient}: {sample}")

        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (ppv * se) / (ppv + se) if (ppv + se) > 0 else 0

        print(f"Performance for {arrhythmia}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"  Sensitivity: {se:.4f}, Specificity: {sp:.4f}, Accuracy: {acc:.4f}, NPV: {npv:.4f}, PPV: {ppv:.4f}, F1 Score: {f1:.4f}")

        return tp, fp, fn, tn

    def export_all_samples(self, result_path):

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        num_print_per_arrhythmia = 100
        num_samples_per_page = 4

        all_selected_samples = []

        for arrhythmia, samples in self.samples.items():
            #if arrhythmia in ["NSR", "NOISE", "WENCKEBACH", "AVB_TYPE2", "SUDDEN_BRADY", "SVT>30s", "AFIB"]:
            #if arrhythmia in ["NSR", "NOISE", "VT>10s", "VT<10s", "IVR", "TRIGEMINY", "BIGEMINY"]:
            #num_print_per_arrhythmia = 50 if arrhythmia == "NOISE" or arrhythmia == "NSR" else 100
            selected_samples = np.random.choice(samples, size=min(num_print_per_arrhythmia, len(samples)), replace=False)
            all_selected_samples.extend(selected_samples)

        print(f"Total samples selected: {len(all_selected_samples)} from {len(self.samples)} arrhythmias")

        all_selected_samples = list(np.random.choice(all_selected_samples, len(all_selected_samples), replace=False))  # Shuffle the samples
        
        #find doubles using for loop
        doubles = []
        for i in range(len(all_selected_samples)):
            ndoubles = 0
            for j in range(i+1, len(all_selected_samples)):
                if all_selected_samples[i]["patient"] == all_selected_samples[j]["patient"] and all_selected_samples[i]["type"] == all_selected_samples[j]["type"] and all_selected_samples[i]["human"] == all_selected_samples[i]["aladin"]:
                    ndoubles += 1
                    doubles.append((all_selected_samples[i], all_selected_samples[j], all_selected_samples[i]["type"], all_selected_samples[j]["type"]))
                    all_selected_samples.pop(j)
                    break  # Break to avoid index error after removing an element

        print(f"Total samples selected: {len(all_selected_samples)} from {len(self.samples)} arrhythmias")

        for i in range(0, len(all_selected_samples), num_samples_per_page):
            self.export_page(all_selected_samples[i:i + num_samples_per_page], result_path, page=i//4)

        with open(os.path.join(result_path, "mapping.json"), 'w') as f:
            json.dump(self.mapping, f, indent=4)

    def export_page(self, samples, result_path, page=0):

        margin_cm = 1              # Margin on each side
        subplot_margin_cm = 1.5
        total_width = 34 + 2 * margin_cm # Total width of the plot in cm
        total_subplot_height = 4 # Height of each subplot in cm
        total_height = (total_subplot_height+subplot_margin_cm)*len(samples) + margin_cm # Total height of the plot in cm
        plot_width = total_width - 2 * margin_cm        # 
        plot_height = total_height - 2 * margin_cm
        dpi = 300   

        plot_width_in = plot_width / 2.54
        margin_in = margin_cm / 2.54
        subplot_margin_in = subplot_margin_cm / 2.54
        total_width_in = plot_width_in + 2 * margin_in

        plot_height_in = plot_height / 2.54
        total_height_in = plot_height_in + margin_in * 2
        subplot_height_in = total_subplot_height / 2.54

        # Create figure with total size
        fig = plt.figure(figsize=(total_width_in, total_height_in), dpi=dpi)

        axes = []
        y = 0

        for i in range(len(samples)):

            # Create axes with 1 cm margin
            left = margin_in / total_width_in
            bottom = ((y * (subplot_height_in + subplot_margin_in)) + margin_in) / total_height_in
            right = 1 - left
            top = ((y * (subplot_height_in + subplot_margin_in)) + subplot_height_in  + margin_in) / total_height_in

            # print(i, "left:", left, "bottom:", bottom, "right:", right, "top:", top)
            # print((top-bottom)*total_height_in*2.54, "cm height of subplot")
            # print((right-left)*total_width_in*2.54, "cm width of subplot")
            # print(34/4)
            ax = fig.add_axes([left, bottom, right - left, top - bottom])
            axes.append(ax)

            y += 1

        for i, sample in enumerate(samples):
            self.export_sample(axes[i], sample, idx=page * 4 + (3 - i))

        filename = f"trace_page_{page}.pdf"
        filename = filename.replace("/", "_")
        plt.savefig(os.path.join(result_path, filename))
        plt.close(fig)

    def save_to_json(self):
        with open("samples.json", 'w') as f:
            json.dump(self.samples, f, indent=4)

        with open("num_samples_searched_per_arrhythmia.json", 'w') as f:
            json.dump(self.num_samples_searched_per_arrhythmia, f, indent=4)

        # with open("samples_aladin_per_patient_per_arrhythmia.json", 'w') as f:
        #     json.dump(self.samples_aladin_per_patient_per_arrhythmia, f, indent=4)

    def set_sample_from_files(self, files):

        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                for idx, sample in data.items():
                    arr = sample["type"]
                    patient = sample["path"].split("/")[1]
                    sample["patient"] = patient
                    if arr not in self.samples:
                        self.samples[arr] = []
                    self.samples[arr].append(sample)

    def export_sample(self, ax, sample, idx=0):
        
        dataroot = '/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA'

        path = sample["path"]
        region = (sample["onset"], sample["offset"])
        midpoint = int((region[0] + region[1]) // 2)
        midpoint = int(np.round(midpoint / 250) * 250)  # Round to nearest 250 ms
        window = 250 * 34  # 34 seconds window

        rec = wfdb.rdheader(os.path.join(dataroot, path))
        size = rec.sig_len

        start = max(0, min(size-window, midpoint - int(window // 2)))
        end = min(size, start + window)
        #print(f"Exporting sample from {start} to {end} for patient {sample['patient']} with type {sample['type']}")

        rec = wfdb.rdrecord(os.path.join(dataroot, path), sampto=end, sampfrom=start)
        ecg_segment = rec.p_signal[:, 0]
        time = np.arange(start, end) / rec.fs

        for xline in np.arange(np.round(start/rec.fs)*rec.fs, np.round(end/rec.fs)*rec.fs, 50):
            xline = np.round(xline, 1)
            if xline % rec.fs == 0:
                ax.axvline(x=xline/rec.fs, color='#E88776', linewidth=1, alpha=1)
            else:
                ax.axvline(x=xline/rec.fs, color='#E88776', linewidth=0.5, alpha=0.75)

            # for xline in np.arange(np.round(start/rec.fs)*rec.fs, np.round(end/rec.fs)*rec.fs, 10):
            #     ax.axvline(x=xline/rec.fs, color='#E88776', linewidth=0.25, alpha=0.5)

        for yline in np.arange(-2, 2, 0.5):
            yline = np.round(yline, 1)
            if yline == 0:
                ax.axhline(y=yline, color='#E88776', linewidth=1.5, alpha=0.75)
            else:
                ax.axhline(y=yline, color='#E88776', linewidth=0.5, alpha=0.75)
        
        for yline in np.arange(-2, 2, 0.1):
            ax.axhline(y=yline, color='#E88776', linewidth=0.25, alpha=0.5)

        ax.axhline(y=2, color='#E88776', linewidth=0.5, alpha=0.75)
        ax.axvline(x=end/rec.fs, color='#E88776', linewidth=0.5, alpha=0.75)

        ax.plot(time, ecg_segment, label='ECG Signal', color='black', linewidth=0.5)
        ax.set_axis_off()
        ax.set_ylim(-2, 2)
        ax.set_xlim(start/rec.fs, end/rec.fs)
        ax.set_xticks([])
        ax.set_yticks([])

        x0 = np.round(start/rec.fs) + 1
        x1 = x0 + 1
        ax.plot([x0, x1], [1.5, 1.5], color='black', linewidth=1)
        ax.text(x0 + 0.5, 1.55, f"1s", fontsize=6, ha='center', va='bottom', color='black')

        ax.plot([x0, x0], [1, 1.5], color='black', linewidth=1)
        ax.text(x0-0.1, 1.25, f"0.5mV", fontsize=6, ha='right', va='center', color='black')

        #ax.set_title(f"Patient: {sample['path']} | Type: {sample['type']}, ALADIN: {sample['aladin']}, Human: {sample['human']}", fontsize=8)
        #ax.set_title(f"ID: {idx} | Suggested: {sample['type']}", fontsize=8)
        ax.set_title(f"ID: {idx}", fontsize=8)
        self.mapping[str(idx)] = {
            "path": sample["path"],
            "onset": sample["onset"],
            "offset": sample["offset"],
            "type": sample["type"],
            "human": sample["human"],
            "aladin": sample["aladin"]
        }





if __name__ == "__main__":

    root_directory = '/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA-Cleaned'
    result_directory = '/home/lukas/UU/ASRA/ALADINv2/traces'
    num_samples = 100
    sampler = StratifiedSampler(root_directory, num_samples)
    sampler.sample_from_json()
    #sampler.sample()#n_files=2400)
    #sampler.save_to_json()
    #sampler.set_sample_from_files(["/home/lukas/UU/ASRA/ALADINv2/traces_ventricular/mapping.json", "/home/lukas/UU/ASRA/ALADINv2/traces_atrial/mapping.json"])
    #sampler.set_sample_from_files(["/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json"])

    sampler.get_performance("AFIB")
    sampler.get_performance("VT>10s")
    sampler.get_performance("VT<10s")
    sampler.get_performance("IVR")
    sampler.get_performance("TRIGEMINY")
    sampler.get_performance("BIGEMINY")

    #sampler.export_all_samples(result_directory)


# Hours searched for NSR: 147.2
# Hours searched for NOISE: 472.65
# Hours searched for AFIB: 1040.75
# Hours searched for VT>10s: 181134.2
# Hours searched for VT<10s: 4565.5
# Hours searched for SVT>30s: 181134.2
# Hours searched for IVR: 13120.35
# Hours searched for TRIGEMINY: 1396.1
# Hours searched for BIGEMINY: 1266.15
# Hours searched for WENCKEBACH: 84250.15
# Hours searched for AVB_TYPE2: 31506.55
# Hours searched for SUDDEN_BRADY: 181134.2


