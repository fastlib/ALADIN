import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob 
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, Literal, Optional

#plt.rcParams['font.family'] = 'DejaVu Sans'

from aladin.utils.benchmark_utils import Data, Model, DiagnosticBenchmark, ICENTIASAMPLEData

class ALADINVirtual():
    def __init__(self):
        self.name = "ALADIN"
        self.save_output = False

    def process_diagnoses(self, diagnoses):

        filtered_diagnoses = []
        raw_diagnoses = []

        for diagnosis in diagnoses:
            name = diagnosis["type"]
            onset = diagnosis["onset"]
            offset = diagnosis["offset"]
            duration = (offset - onset) / 250  # Convert to seconds

            if name == "AFIB" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "AFIB", "onset": onset, "offset": offset})
            elif name == "VT" and duration >= 10:
                filtered_diagnoses.append({"type": "VT>10s", "onset": onset, "offset": offset})
            elif name == "VT" and duration < 10:
                filtered_diagnoses.append({"type": "VT<10s", "onset": onset, "offset": offset})
            elif name == "SVT" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "SVT>30s", "onset": onset, "offset": offset})
            elif name == "IVR":
                filtered_diagnoses.append({"type": "IVR", "onset": onset, "offset": offset})
            elif name == "TRIGEMINY":
                filtered_diagnoses.append({"type": "TRIGEMINY", "onset": onset, "offset": offset})
            elif name == "BIGEMINY":
                filtered_diagnoses.append({"type": "BIGEMINY", "onset": onset, "offset": offset})
            elif name == "NOISE" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "NOISE", "onset": onset, "offset": offset})
            elif name == "NSR" and duration >= 27: #min 90% overlap
                filtered_diagnoses.append({"type": "NSR", "onset": onset, "offset": offset})
            elif name == "WENCKEBACH" or \
                    name == "AVB" or \
                    name == "IVR" or \
                    name == "AIVR" or \
                    name == "AVB_TYPE2" or \
                    name == "SUDDEN_BRADY":
                filtered_diagnoses.append({"type": name, "onset": onset, "offset": offset})

        return filtered_diagnoses, []


class HumanTechnician():
    def __init__(self, id=0):
        self.name = f"Human Technician"
        self.id = id
        self.save_output = False
        self.predict_file = "/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods"
        self.annotation_data = {}
        self.consensus = {}
        self.get_predictions()

    def get_predictions(self):
        #read excel file
        if not os.path.exists(self.predict_file):
            print("Annotation file not found:", self.predict_file)
            return

        dat = pd.read_excel(self.predict_file, engine='odf', sheet_name=0, skiprows=0, names=["ID","Technician","Consensus"], usecols="A,C,G").to_dict(orient='records')
        columnname = "Technician"

        for ann in dat:
            recordname = "record_" + str(int(ann["ID"]))
            
            if columnname in ann and not pd.isna(ann[columnname]):
                label = ann[columnname]

                if label[:3] == "NOT":
                    self.annotation_data[recordname] = [""]
                else:
                    label = label.replace("NOT ", "")
                    self.annotation_data[recordname] = [label]

            consensus = ann["Consensus"]
            if not pd.isna(consensus) and not consensus == "??":
                consensus = consensus.replace("[","").replace("]","").replace("'", "").strip()
                labels = consensus.split(",")
                self.consensus[recordname] = []
                for l in labels:
                    l = l.strip()
                    self.consensus[recordname].append(l)

        print(len(self.annotation_data), "records with annotations found in the annotation file.")
        

    def predict(self, sig, fs, meta=None, preprocess=False):

        case = meta["record"]
        arrhythmia = meta["arrhythmia"]
        consensus = self.consensus.get(case, [""])
        predicted_episodes = self.annotation_data.get(case, [""])

        #print(case, predicted_episodes)
        #print(case, arrhythmia, "TRUE:", arrhythmia in consensus, "PRED:", arrhythmia in predicted_episodes)


        #change key name in each item
        if predicted_episodes[0] != "":
            return [{"type": label, "onset": 0, "offset": len(sig)} for label in predicted_episodes], {"arrhythmia": arrhythmia}
        else:
            return None, {"arrhythmia": arrhythmia}

class HumanCardiologist():
    def __init__(self, id=0):
        self.name = f"Human Cardiologist {id+1}"
        self.id = id
        self.save_output = False
        self.predict_file = "/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods"
        self.annotation_data = {}
        self.consensus = {}
        self.get_predictions()

    def get_predictions(self):
        #read excel file
        if not os.path.exists(self.predict_file):
            print("Annotation file not found:", self.predict_file)
            return

        dat = pd.read_excel(self.predict_file, engine='odf', sheet_name=0, skiprows=0, names=["ID","Cardiologist1","Cardiologist2","Cardiologist3","Consensus"], usecols="A,D,E,F,G").to_dict(orient='records')
        columnname = "Cardiologist" + str(self.id+1)

        for ann in dat:
            recordname = "record_" + str(int(ann["ID"]))
            
            if columnname in ann and not pd.isna(ann[columnname]):
                label = ann[columnname]
                label = label.replace("[","").replace("]","").replace("'", "").strip()

                self.annotation_data[recordname] = []
                labels = label.split(",")
                for l in labels:
                    l = l.strip()
                    #if l not in self.annotation_data[recordname]:
                    self.annotation_data[recordname].append(l)

            consensus = ann["Consensus"]
            if not pd.isna(consensus) and not consensus == "??":
                consensus = consensus.replace("[","").replace("]","").replace("'", "").strip()
                labels = consensus.split(",")
                self.consensus[recordname] = []
                for l in labels:
                    l = l.strip()
                    self.consensus[recordname].append(l)

        print(len(self.annotation_data), "records with annotations found in the annotation file.")
        

    def predict(self, sig, fs, meta=None, preprocess=False):

        case = meta["record"]
        arrhythmia = meta["arrhythmia"]
        consensus = self.consensus.get(case, [""])
        predicted_episodes = self.annotation_data.get(case, [""])

        #print(case, predicted_episodes)
        #print(case, arrhythmia, "TRUE:", arrhythmia in consensus, "PRED:", arrhythmia in predicted_episodes)

        #do not punish cardiologists who predicted wenckebach with 2:1 as pure wenckebach
        if "WENCKEBACH" in predicted_episodes and "AVB_TYPE2" in consensus and "WENCKEBACH" in consensus:
            predicted_episodes.append("AVB_TYPE2")


        #change key name in each item
        if predicted_episodes[0] != "":
            return [{"type": label, "onset": 0, "offset": len(sig)} for label in predicted_episodes], {"arrhythmia": arrhythmia}
        else:
            return None, {"arrhythmia": arrhythmia}


def bootstrap_diff_test(
    f1_a: np.ndarray,
    f1_b: np.ndarray,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    ci_method: Literal["percentile", "basic"] = "percentile",
) -> Dict[str, object]:
    """
    Bootstrap-based hypothesis test for the difference of dataset-level F1 scores.
    
    Parameters
    ----------
    f1_a : np.ndarray
        Array of shape (B,) with bootstrap F1 scores for method A (same replicates order as f1_b).
    f1_b : np.ndarray
        Array of shape (B,) with bootstrap F1 scores for method B.
    alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 -> 95% CI).
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        - "two-sided": H0: mean(A-B) = 0
        - "greater":   H0: mean(A-B) <= 0, H1: > 0
        - "less":      H0: mean(A-B) >= 0, H1: < 0
    ci_method : {"percentile", "basic"}, default="percentile"
        Method for the bootstrap CI on the difference (A - B).
        - "percentile": [q_alpha/2, q_1-alpha/2] of the bootstrap differences
        - "basic":      2*mean_diff - [q_1-alpha/2, q_alpha/2]
    
    Returns
    -------
    result : dict
        {
          "mean_A": float,
          "mean_B": float,
          "mean_diff": float,          # mean of (A - B) over bootstrap replicates
          "ci": (low, high),           # bootstrap CI for the difference
          "alpha": float,
          "p_value": float,            # empirical bootstrap p-value
          "alternative": str,
          "diffs": np.ndarray          # the bootstrap differences (A - B), shape (B,)
        }
    
    Notes
    -----
    - This uses the *paired* bootstrap replicates you already have. Increasing B
      refines precision but does not inflate the evidence (no fake big-n).
    - For the two-sided p-value we use the standard "twice the tail probability in the
      opposite direction of the observed mean difference" approach.
    """
    f1_a = np.asarray(f1_a).ravel()
    f1_b = np.asarray(f1_b).ravel()
    if f1_a.shape != f1_b.shape:
        raise ValueError("f1_a and f1_b must have the same shape.")
    if f1_a.ndim != 1:
        raise ValueError("Inputs must be 1D arrays of bootstrap replicates.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    
    diffs = f1_a - f1_b
    B = diffs.size
    mean_A = float(np.mean(f1_a))
    mean_B = float(np.mean(f1_b))
    mean_diff = float(np.mean(diffs))
    
    # Confidence interval
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    q_low, q_high = np.percentile(diffs, [lower_q, upper_q])
    if ci_method == "percentile":
        ci = (float(q_low), float(q_high))
    elif ci_method == "basic":
        # "basic" CI: 2*theta_hat - [q_high, q_low]
        ci = (float(2 * mean_diff - q_high), float(2 * mean_diff - q_low))
    else:
        raise ValueError("ci_method must be 'percentile' or 'basic'.")
    
    # Empirical bootstrap p-values with a tiny continuity correction
    # to avoid 0 when all diffs are on one side.
    eps = 1e-12
    B = diffs.size
    if alternative == "two-sided":
        p_lower = (np.sum(diffs <= 0) + 1) / (B + 1)
        p_upper = (np.sum(diffs >= 0) + 1) / (B + 1)
        p_value = float(2 * min(p_lower, p_upper))
        p_value = min(1.0, p_value)
    elif alternative == "greater":
        # H1: mean_diff > 0  -> tail at <= 0
        p_value = float((np.sum(diffs <= 0) + 1) / (B + 1))
    elif alternative == "less":
        # H1: mean_diff < 0 -> tail at >= 0
        p_value = float((np.sum(diffs >= 0) + 1) / (B + 1))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    
    return {
        "mean_A": mean_A,
        "mean_B": mean_B,
        "mean_diff": mean_diff,
        "ci": ci,
        "alpha": alpha,
        "p_value": p_value,
        "alternative": alternative,
        "diffs": diffs,
    }

def pairwise_differences(data):
    models = data['Model'].unique()
    differences = []
    arrhythmia = data['Arrhythmia'].values[0]  # Assuming all rows have the same arrhythmia
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]
            scores1 = data[data['Model'] == model1]['Score'].values
            scores2 = data[data['Model'] == model2]['Score'].values

            print(f"Arrhythmia: {arrhythmia}, {len(scores1)} vs {len(scores2)}")
            res = bootstrap_diff_test(scores1, scores2)
            p = res['p_value']
            print(f"Arrhythmia: {arrhythmia}, {len(scores1)} vs {len(scores2)}: p={p:.3f}")

            differences.append({'arrhythmia': arrhythmia, 'model_1': model1, 'model_2': model2, 'p': p})

    return pd.DataFrame(differences)


def add_significance(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom')

def p_to_stars(p):
    if p >= 0.05:
        return "ns"     # not significant
    elif p >= 0.01:
        return "*"
    elif p >= 0.001:
        return "**"
    else:
        return "***"

def get_average_cardiologist_metrics():

    average_cardiologist = {}

    human1 = HumanCardiologist(0)
    human2 = HumanCardiologist(1)
    human3 = HumanCardiologist(2)

    icentia = ICENTIASAMPLEData("ICENTIA", 
        sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", 
        annfile="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods", 
        allfile="/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA/samples_xxl.json",
        asynchronous=True)
    experiment = DiagnosticBenchmark(icentia, [human1, human2, human3])
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True, sampled=True)
    experiment.report()

    return distributions

def get_cardiologist_metrics(i):

    cardiologist = {}

    human = HumanCardiologist(i)
    icentia = ICENTIASAMPLEData("ICENTIA", sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", annfile="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods",asynchronous=True)
    experiment = DiagnosticBenchmark(icentia, human)
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True, sampled=True)

    return distributions


def get_technician_metrics():

    technician = {}

    human = HumanTechnician()
    icentia = ICENTIASAMPLEData("ICENTIA", sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", annfile="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods",asynchronous=True)
    experiment = DiagnosticBenchmark(icentia, human)
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True, sampled=True)

    return distributions

def get_bootstrapped_performance(df, n_iterations=1000):

    tps = {}
    fps = {}
    fns = {}
    f1s = {}

    patients_per_arrhythmia = {
        "NSR": 55,
        "NOISE": 74,
        "AFIB": 395,
        "VT>10s": 4735,
        "VT<10s": 399,
        "SVT>30s": 277,
        "IVR": 1009,
        "TRIGEMINY": 264,
        "BIGEMINY": 268,
        "WENCKEBACH": 3750,
        "AVB_TYPE2": 1387,
        "SUDDEN_BRADY": 4835
    }

    # AVB_TYPE2: TP=15.0 +-(2.1), FP=12.7 +-(3.5), FN=0.9 +-(0.9), F1=0.687
    # BIGEMINY: TP=132.5 +-(18.7), FP=34.3 +-(5.7), FN=0.0 +-(0.0), F1=0.883
    # SVT>30s: TP=83.8 +-(13.4), FP=65.0 +-(18.2), FN=7.2 +-(4.6), F1=0.701
    # AFIB: TP=100.3 +-(10.7), FP=5.6 +-(3.7), FN=3.0 +-(3.4), F1=0.960
    # SUDDEN_BRADY: TP=0.2 +-(0.1), FP=0.0 +-(0.0), FN=0.0 +-(0.0), F1=0.900
    # IVR: TP=4.5 +-(2.1), FP=32.3 +-(5.2), FN=1.7 +-(1.2), F1=0.205
    # VT<10s: TP=73.7 +-(9.0), FP=17.0 +-(7.3), FN=5.8 +-(2.8), F1=0.865
    # TRIGEMINY: TP=119.3 +-(16.2), FP=48.9 +-(14.2), FN=5.3 +-(3.9), F1=0.814
    # VT>10s: TP=1.6 +-(0.8), FP=0.3 +-(0.3), FN=0.8 +-(0.4), F1=0.723
    # WENCKEBACH: TP=6.5 +-(1.5), FP=2.4 +-(0.8), FN=0.8 +-(0.4), F1=0.799

    not_interested = ["NOISE", "NSR"]

    for i in range(n_iterations):
        # Sample with replacement
        sample = df.sample(n=len(df), replace=True)

        arrhythmias = sample['type'].unique()
        #arrhythmias = [arr for arr in arrhythmias if arr not in not_interested]

        for arrhythmia in arrhythmias:
            # Filter the sample for the current arrhythmia
            sample_arrhythmia = sample[sample['type'] == arrhythmia]

            # Calculate the number of patients for this arrhythmia
            num_patients = patients_per_arrhythmia.get(arrhythmia, 0)

            # If there are no patients for this arrhythmia, skip it
            if num_patients == 0:
                continue


            # Calculate true positives, false positives, and false negatives
            tp = sample_arrhythmia[sample_arrhythmia['gt'] & sample_arrhythmia['pred']].shape[0]
            fp = sample_arrhythmia[~sample_arrhythmia['gt'] & sample_arrhythmia['pred']].shape[0]
            fn = sample_arrhythmia[sample_arrhythmia['gt'] & ~sample_arrhythmia['pred']].shape[0]

            # Store the results
            if arrhythmia not in tps:
                tps[arrhythmia] = []
                fps[arrhythmia] = []
                fns[arrhythmia] = []
                f1s[arrhythmia] = []

            tps[arrhythmia].append(tp)
            fps[arrhythmia].append(fp)
            fns[arrhythmia].append(fn)

            # Calculate F1 score

            if (tp + fp + fn) > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
            else:
                f1 = 0

            f1s[arrhythmia].append(f1)

    for arrhythmia in tps.keys():
        tps[arrhythmia] = np.array(tps[arrhythmia], dtype=np.float32)
        fps[arrhythmia] = np.array(fps[arrhythmia], dtype=np.float32)
        fns[arrhythmia] = np.array(fns[arrhythmia], dtype=np.float32)
        tps[arrhythmia] *= 1000 / patients_per_arrhythmia[arrhythmia]
        fps[arrhythmia] *= 1000 / patients_per_arrhythmia[arrhythmia]
        fns[arrhythmia] *= 1000 / patients_per_arrhythmia[arrhythmia]

    per_arrhythmia = {}

    for arrhythmia in tps.keys():
        per_arrhythmia[arrhythmia] = {
            'TP': tps[arrhythmia],
            'FP': fps[arrhythmia],
            'FN': fns[arrhythmia],
            'F1': f1s[arrhythmia]
        }

    return per_arrhythmia

def get_labels(samplefile, annfile):

    sample_data = json.load(open(samplefile, 'r'))

    samples = {}
    for key in list(sample_data.keys()):
        newkey = "rec_"+str(key)
        samples[newkey] = sample_data[key]
        samples[newkey]["ID"] = samples[newkey]["type"]+"_"+samples[newkey]["path"]

    anns = {}

    mapper = {
        "NSR": "NSR",
        "NOISE": "NOISE",
        "Noise": "NOISE",
        "AFIB": "AFIB",
        "AFL": "AFIB",
        "VT>10s": "VT>10s",
        "VT<10s": "VT<10s",
        "IVR": "IVR",
        "TRI": "TRIGEMINY",
        "BIG": "BIGEMINY",
        "SVT": "SVT>30s",
        "AVB2": "AVB_TYPE2",
        "Wenckebach": "WENCKEBACH",
        "CHB": "SUDDEN_BRADY"
    }

    dat = pd.read_excel(annfile, engine='openpyxl', sheet_name=0, skiprows=3, names=["ID","Label1","Label2","Label3","Comment"], usecols="A,B,C,D").to_dict(orient='records')
    for d in dat:
        newkey = "rec_"+str(d["ID"])
        #print(newkey, d["Label1"], d["Label2"], d["Label3"])
        labels = []
        if not pd.isna(d["Label1"]) and d["Label1"] in mapper:
            labels.append(mapper[d["Label1"]])
        if not pd.isna(d["Label2"]) and d["Label2"] in mapper:
            labels.append(mapper[d["Label2"]])
        if not pd.isna(d["Label3"]) and d["Label3"] in mapper:
            labels.append(mapper[d["Label3"]])

        if len(labels) == 0:
            continue

        matched_sample = samples[newkey]
        anns[matched_sample["ID"]] = (np.any([x in matched_sample['type'] for x in labels]), matched_sample['type'])

        #anns[matched_sample["ID"]] = labels

    return anns, samples

def get_diagnosis_from_aladin(diagnoses, fs=250):
    filtered_diagnoses = []

    for diagnosis in diagnoses:
        onset = diagnosis.get("onset", 0)
        offset = diagnosis.get("offset", 0)
        duration = (offset - onset) / 250  # Convert to seconds

        if diagnosis["type"] == "AFIB" and duration >= 27:
            filtered_diagnoses.append({"type": "AFIB", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "VT" and duration >= 10:
            filtered_diagnoses.append({"type": "VT>10s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "VT" and duration < 10:
            filtered_diagnoses.append({"type": "VT<10s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "SVT" and duration >= 27:
            filtered_diagnoses.append({"type": "SVT>30s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "IVR":
            filtered_diagnoses.append({"type": "IVR", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "TRIGEMINY":
            filtered_diagnoses.append({"type": "TRIGEMINY", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "BIGEMINY":
            filtered_diagnoses.append({"type": "BIGEMINY", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "WENCKEBACH" or \
                diagnosis["type"] == "AVB" or \
                diagnosis["type"] == "AVB_TYPE2" or \
                diagnosis["type"] == "SUDDEN_BRADY":
            filtered_diagnoses.append({"type": diagnosis["type"], "onset": onset, "offset": offset})

    if np.all([d["type"] == "NOISE" for d in diagnoses]):
        filtered_diagnoses.append({"type": "NOISE", "onset": 0, "offset": 30*250})

    if np.all([d["type"] == "NSR" or d["type"] == "AVB_TYPE1" or d["type"] == "TACHYCARDIA" or d["type"] == "BRADYCARDIA" or d["type"] == "IVB" for d in diagnoses]):
        filtered_diagnoses.append({"type": "NSR", "onset": 0, "offset": 30*250})

    return filtered_diagnoses

def get_labels_consensus(samplefile, annfile):

    sample_data = json.load(open(samplefile, 'r'))

    samples = {}
    for key in list(sample_data.keys()):
        newkey = "rec_"+str(key)
        samples[newkey] = sample_data[key]
        samples[newkey]["ID"] = samples[newkey]["type"]+"_"+samples[newkey]["path"]

    anns = {}

    mapper = {
        "NSR": "NSR",
        "NOISE": "NOISE",
        "Noise": "NOISE",
        "AFIB": "AFIB",
        "AFL": "AFIB",
        "VT>10s": "VT>10s",
        "VT<10s": "VT<10s",
        "IVR": "IVR",
        "TRI": "TRIGEMINY",
        "BIG": "BIGEMINY",
        "SVT": "SVT>30s",
        "AVB2": "AVB_TYPE2",
        "Wenckebach": "WENCKEBACH",
        "CHB": "SUDDEN_BRADY"
    }

    dat = pd.read_excel(annfile, engine='odf', sheet_name=0, skiprows=1, names=["ID","Consensus"], usecols="A,G").to_dict(orient='records')
    for d in dat:
        newkey = "rec_"+str(d["ID"])
        #print(newkey, d["Label1"], d["Label2"], d["Label3"])
        labels = []
        consensus = str(d["Consensus"])
        raw_labels = consensus.split(",")
        for label in raw_labels:
            if label in mapper:
                labels.append(mapper[label])

        if len(labels) == 0:
            continue

        matched_sample = samples[newkey]
        anns[matched_sample["ID"]] = (np.any([x in matched_sample['type'] for x in labels]), matched_sample['type'])

        #anns[matched_sample["ID"]] = labels

    return anns, samples

def get_disagreement():
    sample_gabor="/home/lukas/UU/ASRA/ALADINv2/traces_gabor/mapping.json"
    ann_gabor="/home/lukas/UU/ASRA/ALADINv2/traces_gabor/ECG_annotation_final.xlsx" 

    gabor_ann_matched, _ = get_labels(sample_gabor, ann_gabor)

    sample_matt = "/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json"
    ann_lukas="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_lukas.xlsx"
    ann_naren="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_naren.xlsx"
    ann_matt = "/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_matt.xlsx"
    ann_consensus = "/home/lukas/UU/ASRA/ALADINv2/paper/disagreement_consensus.ods"

    lukas_ann_matched, _ = get_labels(sample_matt, ann_lukas)
    naren_ann_matched, recs = get_labels(sample_matt, ann_naren)
    matt_ann_matched, _ = get_labels(sample_matt, ann_matt)
    consensus_ann_matched, _ = get_labels_consensus(sample_matt, ann_consensus)

    print(consensus_ann_matched)


    aligned = list(set(gabor_ann_matched.keys()).intersection(set(naren_ann_matched.keys())))
    consensus_labels = [gabor_ann_matched[k] for k in aligned]
    naren_labels = [naren_ann_matched[k] for k in aligned]
    print(consensus_labels)

    # disagreement = 0
    # disagreement_types = []
    # for i in range(len(matt_labels)):
    #     if not np.any([x in matt_labels[i] for x in naren_labels[i]]):
    #         disagreement += 1
    #         disagreement_types.append((aligned[i], matt_labels[i], naren_labels[i]))

    # print(f"Disagreement between Matt and Naren: {disagreement} out of {len(matt_labels)} ({disagreement/len(matt_labels)*100:.2f}%)")

    # print("Disagreement types:")
    # for dt in disagreement_types:
    #     print(f"Type: {dt[0]}, Matt: {dt[1]}, Naren: {dt[2]}")

    objs = {}

    for k_str in list(recs.keys()):
        k = int(k_str.split('_')[-1])
        obj = {}
        ID = recs[k_str]["ID"]
        obj["ID"] = int(k)
        obj["path"] = recs[k_str]["ID"]
        obj["Human"] = recs[k_str]["type"] if recs[k_str]["human"] else "NOT "+ recs[k_str]["type"]
        #obj["ALADIN"] = recs[k]["type"] if recs[k]["aladin"] else "-"
        obj["Annotator"] = gabor_ann_matched.get(ID, [])
        obj["Matt"] = matt_ann_matched.get(ID, [])
        obj["Naren"] = naren_ann_matched.get(ID, [])
        obj["Lukas"] = lukas_ann_matched.get(ID, [])
        objs[str(k)] = obj

    df = pd.DataFrame.from_dict(objs, orient='index')
    df.sort_values(by='ID', inplace=True)

    df.to_csv("paper/all_disagreement.csv", index=False)

    # print("Matt labels:", matt_labels)

    print(len(consensus_labels), len(naren_labels))
    kappa = cohen_kappa_score([ml[0] for ml in consensus_labels], [nl[0] for nl in naren_labels])

    print(f"Cohen's Kappa between Consensus and Naren: {kappa}")


def get_results(aladin_file):
    sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json"
    consensus="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods"
    #cardiologist1 = "/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_matt.xlsx"
    cardiologist2 = "/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/ECG_annotation_matt.xlsx"
    #cardiologist3 = "/home/lukas/UU/ASRA/ALADINv2/traces_gabor/ECG_annotation_gabor.xlsx"

    sample_data = json.load(open(sample, 'r'))
    aladin_data = json.load(open(aladin_file, 'r'))
    aladin_results = aladin_data["results"][0]["results"]
    aladin_results_per_record = {}
    for rec in aladin_results:
        rec_id = rec["record"]
        if rec_id not in aladin_results_per_record:
            aladin_results_per_record[rec_id] = {}
        aladin_results_per_record[rec_id]["raw"] = rec["raw"]

    

    samples = {}
    for key in list(sample_data.keys()):
        newkey = "record_"+str(key)
        samples[newkey] = sample_data[key]
        

    cardiologist = {}
    mapper = {
        "NSR": "NSR",
        "NOISE": "NOISE",
        "AFIB": "AFIB",
        "AFL": "AFIB",
        "VT>10s": "VT>10s",
        "VT<10s": "VT<10s",
        "IVR": "IVR",
        "TRI": "TRIGEMINY",
        "BIG": "BIGEMINY",
        "SVT": "SVT>30s",
        "SVT>30s": "SVT>30s",
        "AVB2": "AVB_TYPE2",
        "AVB_TYPE2": "AVB_TYPE2",
        "Wenckebach": "WENCKEBACH",
        "WENCKEBACH": "WENCKEBACH",
        "CHB": "SUDDEN_BRADY",
        "SUDDEN_BRADY": "SUDDEN_BRADY",
        "BIGEMINY": "BIGEMINY",
        "TRIGEMINY": "TRIGEMINY",
    }
    
    dat = pd.read_excel(consensus, engine='odf', sheet_name=0, skiprows=1, names=["ID","Ann1","Ann2","Ann3","Consensus","Ann4"],usecols="A,D,E,F,G,H").to_dict(orient='records')
    #cardiologist2_data = pd.read_excel(cardiologist2, engine='openpyxl', sheet_name=0, skiprows=3, names=["ID","Label1","Label2","Label3","Comment"], usecols="A,B,C,D").to_dict(orient='records')
    #cardiologist2_data = pd.DataFrame(cardiologist2_data)
    #cardiologist2_data["ID"] = cardiologist2_data["ID"].apply(lambda x: "record_"+str(x))

    filtered_samples = {}
    uncertain_samples = {}

    print(len(dat))

    for d in dat:
        newkey = "record_"+str(d["ID"])
        if newkey not in samples:
            continue

        arrhythmia = samples[newkey]['type']
        consensus = str(d["Consensus"])
        raw_labels = consensus.split(",")
        labels = []
        #print(newkey)
        for label in raw_labels:
            label = label.replace(" ", "")
            if label in mapper:
                labels.append(mapper[label])
            elif label == "??":
                otheranns = [d["Ann1"], d["Ann2"], d["Ann3"], d["Ann4"]]
                randomorder = np.random.permutation(otheranns)
                uncertain_samples[newkey] = {
                    "ID": newkey[7:],
                    "Annotator A": randomorder[0],
                    "Annotator B": randomorder[1],
                    "Annotator C": randomorder[2],
                    "Annotator D": randomorder[3],
                }
            else:
                print("Unknown label:", label)

        if len(labels) == 0:
            continue

        aladin_predictions = aladin_results_per_record.get(newkey, {}).get("raw", [])
        aladin_labels = get_diagnosis_from_aladin(aladin_predictions)
        aladin_label_types = list(set([x["type"] for x in aladin_labels]))
        #print("ALADIN labels for", newkey, ":", aladin_label_types)

        if arrhythmia in labels:
            samples[newkey]["gt"] = True
        else:
            samples[newkey]["gt"] = False

        if len(aladin_label_types) == 0:
            samples[newkey]["aladin"] = False
        elif arrhythmia in aladin_label_types:
            samples[newkey]["aladin"] = True
        else:
            samples[newkey]["aladin"] = False

        #samples[newkey]["human"] = True if arrhythmia in labels else False

        raw_cardiologist2_labels = cardiologist2_data[cardiologist2_data["ID"] == newkey]
        if not pd.isna(raw_cardiologist2_labels["Label1"].values[0]):
            cardiologist2_labels = []
            if raw_cardiologist2_labels["Label1"].values[0] in mapper:
                cardiologist2_labels.append(mapper[raw_cardiologist2_labels["Label1"].values[0]])
            if not pd.isna(raw_cardiologist2_labels["Label2"].values[0]) and raw_cardiologist2_labels["Label2"].values[0] in mapper:
                cardiologist2_labels.append(mapper[raw_cardiologist2_labels["Label2"].values[0]])
            if not pd.isna(raw_cardiologist2_labels["Label3"].values[0]) and raw_cardiologist2_labels["Label3"].values[0] in mapper:
                cardiologist2_labels.append(mapper[raw_cardiologist2_labels["Label3"].values[0]])

            if "WENCKEBACH" in cardiologist2_labels and "AVB_TYPE2" in labels and "WENCKEBACH" in labels:
                cardiologist2_labels.append("AVB_TYPE2")

            # if arrhythmia == "AVB_TYPE2":
            #     samples[newkey]["cardiologist"] = True if ("AVB_TYPE2" in cardiologist2_labels or "WENCKEBACH" in cardiologist2_labels) else False
            # else:
            samples[newkey]["cardiologist"] = True if arrhythmia in cardiologist2_labels else False

        filtered_samples[newkey] = samples[newkey]


    # for k in list(samples.keys()):
    #     print(k, samples[k]['type'], samples[k]['human'], samples[k]['aladin'])

    # Convert the samples to a DataFrame
    df = pd.DataFrame.from_dict(filtered_samples, orient='index')
    df = df.reset_index().rename(columns={'index': 'Record'})
    #print(df)

    print(len(uncertain_samples))
    df_uncertain = pd.DataFrame.from_dict(uncertain_samples, orient='index')
    df_uncertain.to_csv("paper/list_of_all_disagreements.csv", index=False)
    #print(df_uncertain)

    # arrhythmias = df['type'].unique()
    # arrhythmias = ["IVR"]

    # for arrhythmia in arrhythmias:
    #     afib = df[df['type'] == arrhythmia]

    #     for i, row in afib.iterrows():
    #         if not row["gt"] and row["aladin"]:
    #             print("False positive:", row["Record"])

    #     # Convert the 'human' and 'aladin' columns to boolean
    #     # afib['human'] = afib['human'].apply(lambda x: True if x else False)
    #     # afib['aladin'] = afib['aladin'].apply(lambda x: True if x else False)

    #     # # Calculate the F1 score
    #     # afib['human'] = afib['human'].astype(int)
    #     # afib['aladin'] = afib['aladin'].astype(int)

    #     human_vals = afib['human'].values
    #     aladin_vals = afib['aladin'].values

    #     tp = np.sum((human_vals == 1) & (aladin_vals == 1))
    #     fp = np.sum((human_vals == 0) & (aladin_vals == 1))
    #     fn = np.sum((human_vals == 1) & (aladin_vals == 0))
    #     tn = np.sum((human_vals == 0) & (aladin_vals == 0))

    #     se = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    #     f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    #     print(arrhythmia, "TP:", tp, "FP:", fp, "FN:", fn, "SE:", se, "SP:", sp, "F1:", f1)

    return df

    #f1 = f1_score(afib['human'], afib['aladin'], average='macro')

def make_piechart():

    #     Patients searched for NSR: 52
    # Patients searched for NOISE: 70
    # Patients searched for AFIB: 323
    # Patients searched for VT>10s: 4823
    # Patients searched for VT<10s: 376
    # Patients searched for SVT>30s: 212
    # Patients searched for IVR: 976
    # Patients searched for TRIGEMINY: 209
    # Patients searched for BIGEMINY: 213
    # Patients searched for WENCKEBACH: 2702
    # Patients searched for AVB_TYPE2: 1432
    # Patients searched for SUDDEN_BRADY: 4300

    data = {
        "AFIB/FL": 323,
        "VT>10s": 4823*4,
        "VT<10s": 376,
        "AVB2": 1432,
        "CHB": 4300*10,
        "BIG": 213,
        "IVR": 976,
        "NOISE": 70,
        "NSR": 52,
        "SVT>30s": 212,
        "TRI": 209,
        "WENCK": 2702
    }
    for key in data.keys():
        data[key] = (4948/data[key])*100

    max_val = max(data.values())
    for key in data.keys():
        data[key] = (data[key] / max_val) * 100
        print(key, data[key])

    fig, ax = plt.subplots(1, 1, figsize=(4, 1.6), dpi=300)
    #use colormap blues

    ax.bar(data.keys(), data.values(), color="#E0DBC9", width=0.5, linewidth=0.5, edgecolor="#BFBA9F", zorder=3)
    ax.set_ylabel("Percentage", fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, rotation=0, color='#95a5a6')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#95a5a6')
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_yticklabels([0,20,40,60,80,100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')


    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.9, bottom=0.2, left=0.12, right=0.99)
    

    #ax.pie(data.values(), labels=data.keys(), autopct='%1.0f%%', colors=colors, startangle=90, textprops={'fontsize': 5})

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig5-icentia-distribution.svg")

def lighter(col_str):
    #get rgb from hex string
    rgb = tuple(int(col_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #add 50 to each value
    rgb = tuple([x + 100 if x + 100 < 255 else 255 for x in rgb])
    #convert back to hex string

    return '#%02x%02x%02x' % rgb

def boxplot(ax, data):
    #get 7 colors of the pastel palette
    colors = sns.color_palette('pastel', 7)
    models = ["ECGFounder", "ResNet", "ALADIN"]

    colors = {
        'ECGFounder': "#2968A4",
        'ResNet': "#5C8FC6",
        'ALADIN': "#B4413D",
    }

    for i, model in enumerate(models):
        sns.boxplot(x='Model', y='Score', data=data[data['Model'] == model], ax=ax, color=lighter(colors[model]), width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)

def barplot(ax, data):
    #get 7 colors of the pastel palette
    models = data['Model'].unique()
    models = [m for m in models if m[:4] != "Card" and m[:4] != "Tech"]  # Exclude these models from the bar plot

    colors = {
        'ECGFounder': "#E2E2EA", #"##5C8FC6",
        'Technicians': "#C2C7D4", #"#2968A4", #"#5C8FC6",
        # 'ECGFounder': "#9099AA",
        # 'Hannun': "#9099AA",
        'ALADIN': "#D63F3D", 
        "Avg. Card.": "#E0DBC9", #"#924D85",
        'Card. 1': "#9099AA",
        'Card. 2': "#9099AA",
        'Card. 3': "#9099AA",
        'Card. 4': "#9099AA",
        'Card. 5': "#9099AA",
        'Card. 6': "#9099AA",
        'Card. 7': "#9099AA",
        'Card. 8': "#9099AA",
        'Card. 9': "#9099AA"
    }
    edgecolors = {
        'ECGFounder': "#C2C7D4",
        'Technicians': "#9099AA",
        'ALADIN': "#B4413D",
        "Avg. Card.": "#E0DBC9"
    }

    for i, model in enumerate(models):

        ax.bar(i, data[data['Model'] == model]['Score'].mean(), 
            color=colors[model],
            edgecolor=edgecolors[model] if model in edgecolors else colors[model],
            linewidth=0.5,
            width=0.8,  # Width of the bars
            zorder=1
        )

        # sns.barplot(
        #     x='Model', 
        #     y='Score', 
        #     data=data[data['Model'] == model], 
        #     ax=ax, 
        #     color=colors[model], 
        #     edgecolor=edgecolors[model] if model in edgecolors else colors[model],
        #     linewidth=0.5,
        #     width=0.8,  
        #     errorbar=None,
        #     zorder=2
        # )
        ax.set_xlim(-1.5, len(models) + 0.5)
        ci_upper = np.percentile(data[data['Model'] == model]['Score'], 97.5)
        ci_lower = np.percentile(data[data['Model'] == model]['Score'], 2.5)
        mean_score = data[data['Model'] == model]['Score'].mean()

        # Draw error bars manually
        ax.errorbar(
            x=[i],
            y=[mean_score],
            yerr=[[mean_score - ci_lower], [ci_upper - mean_score]],
            fmt='none',
            color="black",
            elinewidth=0.5,
            capsize=2,
            capthick=0.5,
            zorder=3
        )
        # ax.text(
        #     i,                                  # x position
        #     ci_upper + 0.025,                    # y position (slightly above upper whisker)
        #     f"{mean_score * 100:.1f}\n({(ci_lower) * 100:.1f}-\n{(ci_upper) * 100:.1f})",           # formatted value in %
        #     ha='center',
        #     va='bottom',
        #     fontsize=5,
        #     clip_on=False,
        # )
    
        # for container in ax.containers:
        #     #check if container has patches
        #     if hasattr(container, 'patches'):
        #         ax.bar_label(container, fmt='%.1f', label_type='edge', padding=5, fontsize=5)

        #sns.boxplot(x='Model', y='Score', data=data[data['Model'] == model], ax=ax, color=lighter(colors[model]), width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)

    modeldata = data[data['Model'].apply(lambda x: x in models)]
    pairwise = pairwise_differences(modeldata)

    return pairwise

def make_barplots(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(2, 6, figsize=(7.08, 3), dpi=300)
    axs = axs.flatten()

    arrhythmia_formatted = {
        "AFIB": "AFIB/AFL > 30s",
        "AVB_TYPE2": "Second-degree AVB",
        "BIGEMINY": "Bigeminy",
        "TRIGEMINY": "Trigeminy",
        "IVR": "IVR",
        "NOISE": "Noise",
        "NSR": "Normal Sinus Rhythm",
        "SUDDEN_BRADY": "Third-degree AVB",
        "SVT>30s": "SVT > 30s",
        "VT>10s": "VT > 10s",
        "VT<10s": "VT < 10s",
        "WENCKEBACH": "Wenckebach"
    }

    pairwise_comparisons_df = []

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[df['Arrhythmia'] == arr]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        pcdf = barplot(ax, subset)
        pairwise_comparisons_df.append(pcdf)
        #ax.set_title(arr, fontsize=6)
        ax.set_xlabel(arrhythmia_formatted[arr], fontsize=6)
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['left'].set_visible(False)
            #xlabels = ["F", "H", "A", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            #ax.set_xticklabels(xlabels, fontsize=5)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', width=0.25, labelsize=3, color='#1C1C1C')
        #else:
        #    ax.tick_params(axis='x', width=0.5, rotation=90, labelsize=5, color='#9099AA')
        if i % 6 == 0:
            #ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            #ax.spines['left'].set_visible(True)
            ax.set_ylabel('F1 Score', fontsize=6)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
            spine.set_color('#1C1C1C')
        # rotate x labels 45 degrees and draw tick lines
        ax.tick_params(axis='y', width=0.25, labelsize=5, color='#1C1C1C')
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=5)
        ax.set_ylim(0, 1)
        #ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#bdc3c7')

    pairwise_comparisons_df = pd.concat(pairwise_comparisons_df, axis=0)

    # pvalues = pairwise_comparisons_df['p'].values
    # reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')
    # pairwise_comparisons_df['p'] = pvals_corrected
    # pairwise_comparisons_df = pairwise_comparisons_df[pairwise_comparisons_df['p'] < 0.05]  # Filter for significant differences

    # for index, row in pairwise_comparisons_df.iterrows():
    #     ax = axs[arrhythmias.index(row['arrhythmia'])]

    #     m1 = row['model_1']
    #     m2 = row['model_2']
    #     p = row['p']
    #     m1_i = models.index(m1)
    #     m2_i = models.index(m2)

    #     # if m1 == "Avg. Card." or m2 == "Avg. Card.":
    #     #     continue

    #     tmp = min(m1_i, m2_i)
    #     m2_i = max(m1_i, m2_i)
    #     m1_i = tmp

    #     m1_y = 1
    #     m2_y = 1
    #     max_y = 1.025  # Add some space above the highest bar
        
    #     if m2_i-m1_i < 2:
    #         m1_i += 0.1
    #         m2_i -= 0.1
    #     else:
    #         max_y += 0.06

    #     ax.plot([m1_i, m1_i, m2_i, m2_i], [m1_y, max_y, max_y, m2_y], lw=0.5, c='black', clip_on=False)
    #     ax.text((m1_i+m2_i)*.5, max_y-0.02, p_to_stars(p), ha='center', va='bottom', fontsize=5, color='black', clip_on=False)

    plt.subplots_adjust(wspace=0.2, hspace=0.4, top=0.95, bottom=0.1, left=0.05, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig5-boxplot-icentia.svg")
    plt.savefig("./paper/images/fig5-boxplot-icentia.png", dpi=300, facecolor="white", edgecolor="white", transparent=False)

def barplot_horizontal(ax, data):
    #get 7 colors of the pastel palette

    models = data['Model'].unique()
    models = [m for m in models if m[:4] != "Card"]  # Exclude these models from the bar plot
    arrhythmias = data['Arrhythmia'].unique()

    colors = {
        'ECGFounder': "#E2E2EA", #"##5C8FC6",
        'Technicians': "#C2C7D4", #"#2968A4", #"#5C8FC6",
        # 'ECGFounder': "#9099AA",
        # 'Hannun': "#9099AA",
        'ALADIN': "#D63F3D", 
        "Avg. Card.": "#E0DBC9", #"#924D85",
        "Cardiologist": "#E0DBC9", #"#924D85",
        'Card. 1': "#9099AA",
        'Card. 2': "#9099AA",
        'Card. 3': "#9099AA",
        'Card. 4': "#9099AA",
        'Card. 5': "#9099AA",
        'Card. 6': "#9099AA",
        'Card. 7': "#9099AA",
        'Card. 8': "#9099AA",
        'Card. 9': "#9099AA"
    }
    edgecolors = {
        'ECGFounder': "#C2C7D4",
        'Technicians': "#9099AA",
        'ALADIN': "#B4413D",
        "Avg. Card.": "#E0DBC9",
        "Cardiologist": "#E0DBC9",
    }

    #agg_data = data.groupby(['Arrhythmia', 'Model'])['Score'].mean().reset_index()
    #print(agg_data)

    order=['AFIB', 'SVT>30s', 'SUDDEN_BRADY', 'VT>10s', 'AVB_TYPE2', 'WENCKEBACH', 'VT<10s', 'IVR', 'TRIGEMINY', 'BIGEMINY']
    print(data['Arrhythmia'].unique())

    order = list(reversed(order))
    models = list(reversed(models))

    for i, arrhythmia in enumerate(order):
        for j, model in enumerate(models):
            
            y = (i*2.5)+0.25+j
            xs = data[(data['Model'] == model) & (data['Arrhythmia'] == arrhythmia)]['Score']

            ax.barh(y, xs.mean(), color=colors[model], edgecolor=edgecolors[model] if model in edgecolors else colors[model], linewidth=0.5, height=0.7, zorder=1)

            print(arrhythmia)
            ci_upper = np.percentile(xs.values, 97.5)
            ci_lower = np.percentile(xs.values, 2.5)
            mean_score = xs.mean()

            # Draw error bars manually
            ax.errorbar(
                x=[mean_score],
                y=[y],
                xerr=[[mean_score - ci_lower], [ci_upper - mean_score]],
                fmt='none',
                color="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=3
            )
            if mean_score < 50:
                ax.text(ci_upper+5, y-0.06, f'{mean_score:.1f} ({ci_lower:.1f}-{ci_upper:.1f})', ha='left', va='center', fontsize=5, color='black', clip_on=False)

    # sns.barplot(
    #     x='Score', 
    #     y='Arrhythmia', 
    #     hue='Model',
    #     data=data, 
    #     orient='h',
    #     ax=ax, 
    #     palette=colors, 
    #     width=0.5, 
    #     linewidth=0.5,
    #     zorder=2,
    #     legend=False
    # )
        # ci_upper = np.percentile(data[data['Model'] == model]['Score'], 97.5)
        # ci_lower = np.percentile(data[data['Model'] == model]['Score'], 2.5)
        # # Draw error bars manually
        # ax.errorbar(
        #     x=[i],
        #     y=[data[data['Model'] == model]['Score'].mean()],
        #     yerr=[[data[data['Model'] == model]['Score'].mean() - ci_lower], [ci_upper - data[data['Model'] == model]['Score'].mean()]],
        #     fmt='none',
        #     color=colors[model],
        #     elinewidth=0.3,
        #     capsize=2,
        #     capthick=0.5,
        #     zorder=3
        # )

def make_ranking_plot(df):

    basefolder = os.environ.get('benchmark_data')
    other_models = pd.read_csv(basefolder+"/results_all_F1_scores_for_each_classification_type.csv", skiprows=2)
    other_models.columns = ['Rank', 'F1n_test', 'F1a_test', 'F1o_test', 'F1p_test', 'F1tot_test', 'F1n_train','F1a_train','F1o_train','F1p_train','F1tot_train','Entry','Closed','Authors']
    other_models["F1_test"] = other_models.apply(lambda x: (x["F1n_test"] + x["F1a_test"] + x["F1o_test"])/3, axis=1)
    print(other_models)

    other_models = other_models.sort_values(by='F1_test', ascending=False)
    other_f1s = other_models['F1_test'].values

    own_f1_dist = []
    class_a = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        own_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)

    own_f1 = np.mean(own_f1_dist)
    print("ALADIN:", own_f1)
    print("Best competitor:", np.max(other_f1s))
    own_f1_low, own_f1_high = np.percentile(own_f1_dist, 2.5), np.percentile(own_f1_dist, 97.5)

    ecgf_f1_dist = []
    class_a = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        ecgf_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    ecgf_f1 = np.mean(ecgf_f1_dist)
    print("ECGFounder:", ecgf_f1)
    ecgf_f1_low, ecgf_f1_high = np.percentile(ecgf_f1_dist, 2.5), np.percentile(ecgf_f1_dist, 97.5)

    hannun_f1_dist = []
    class_a = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        hannun_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    hannun_f1 = np.mean(hannun_f1_dist)
    print("ResNet:", hannun_f1)
    hannun_f1_low, hannun_f1_high = np.percentile(hannun_f1_dist, 2.5), np.percentile(hannun_f1_dist, 97.5)

    all_f1s = []
    for i in range(len(other_f1s)):
        all_f1s.append((other_f1s[i],"#E2E2EA"))
    all_f1s.append((own_f1,"#B4413D"))
    all_f1s.append((hannun_f1,"#5C8FC6"))
    all_f1s.append((ecgf_f1,"#2968A4"))

    all_f1s = sorted(all_f1s, key=lambda x: x[0], reverse=True)
    ranking = np.arange(1, len(all_f1s)+1)

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.5), dpi=300)
    ax.bar(ranking, [x[0] for x in all_f1s], color=[x[1] for x in all_f1s], width=0.7)
    ax.set_xticks([1] + list(np.arange(5, len(all_f1s), 5)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)], fontsize=5)
    ax.set_ylabel("F1 score", fontsize=6)
    ax.set_xlabel("Final competition ranking", fontsize=6)
    ax.set_xlim(0.5, len(all_f1s)+0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    own_pos = [x[0] for x in all_f1s].index(own_f1)+1
    hannun_pos = [x[0] for x in all_f1s].index(hannun_f1) +1
    ecgf_pos = [x[0] for x in all_f1s].index(ecgf_f1)+1
    print(own_pos, hannun_pos, ecgf_pos)

    #draw confidence intervals
    ax.plot([own_pos,own_pos], [own_f1_low, own_f1_high], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_low, own_f1_low], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_high, own_f1_high], color="black", linewidth=0.5)

    ax.plot([hannun_pos,hannun_pos], [hannun_f1_low, hannun_f1_high], color="black", linewidth=0.5)
    ax.plot([hannun_pos-0.2, hannun_pos+0.2], [hannun_f1_low, hannun_f1_low], color="black", linewidth=0.5)
    ax.plot([hannun_pos-0.2, hannun_pos+0.2], [hannun_f1_high, hannun_f1_high], color="black", linewidth=0.5)

    ax.plot([ecgf_pos,ecgf_pos], [ecgf_f1_low, ecgf_f1_high], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_low, ecgf_f1_low], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_high, ecgf_f1_high], color="black", linewidth=0.5)

    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#9099AA')

    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig4-ranking_plot.svg")
    plt.savefig("paper/images/fig4-ranking_plot.png", dpi=300)

def make_barplots_horizontal(df):

    fig, axs = plt.subplots(1, 3, figsize=(7.08, 3), sharey=True, dpi=300)
    axs = axs.flatten()


    patients_per_arrhythmia = {
        "NSR": 55,
        "NOISE": 74,
        "AFIB": 395,
        "VT>10s": 4735,
        "VT<10s": 399,
        "SVT>30s": 277,
        "IVR": 1009,
        "TRIGEMINY": 264,
        "BIGEMINY": 268,
        "WENCKEBACH": 3750,
        "AVB_TYPE2": 1387,
        "SUDDEN_BRADY": 4835
    }

    arrhythmias = df['Arrhythmia'].unique()
    df['Score'] = df['Score'].astype(float)

    for i, row in df.iterrows():
        arrhythmia = row['Arrhythmia']
        df.at[i, 'Score'] *= 1000 / patients_per_arrhythmia[arrhythmia]


    order=['AFIB', 'SVT>30s', 'CHB', 'VT>10s', 'AVB2', 'WENCKEBACH', 'VT<10s', 'IVR', 'TRI', 'BIG']
    order_formatted=['AFIB > 30s', 'SVT > 30s', '3rd deg. AVB', 'VT > 10s', '2nd deg. AVB', 'Wenckebach', 'VT < 10s', 'IVR', 'Trigeminy', 'Bigeminy']
    order = list(reversed(order))
    order_formatted = list(reversed(order_formatted))

    #true positives
    ax = axs[0]
    subset = df[df['Metric'] == "TP"]
    barplot_horizontal(ax, subset)
    #ax.set_title("Correct diagnoses", fontsize=6)
    ax.set_ylabel('')
    ax.set_xlabel('Correct diagnoses per 1,000 patients', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xticklabels(np.arange(0, 301, 50), fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
    ax.set_yticks(np.arange(len(order))*2.5+0.75)
    ax.set_yticklabels(order_formatted)
    ax.set_xlim(0, 350)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='#bdc3c7')

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')

    #false positives
    ax = axs[1]
    subset = df[df['Metric'] == "FP"]
    barplot_horizontal(ax, subset)
    #ax.set_title("False alarms", fontsize=6)
    ax.set_ylabel('')
    ax.set_xlabel('False alarms per 1,000 patients', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xticklabels(np.arange(0, 301, 50), fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
    ax.set_yticks(np.arange(len(order))*2.5+0.75)
    ax.set_yticklabels(order_formatted)
    ax.set_xlim(0, 350)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='#bdc3c7')

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')

    #false negatives
    ax = axs[2]
    subset = df[df['Metric'] == "FN"]
    barplot_horizontal(ax, subset)
    ax.set_ylabel('')
    ax.set_xlabel('Missed cases per 1,000 patients', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xticklabels(np.arange(0, 301, 50), fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
    ax.set_yticks(np.arange(len(order))*2.5+0.75)
    ax.set_yticklabels(order_formatted)
    ax.set_xlim(0, 350)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='#bdc3c7')

    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')

    #  # Plot a boxplot for each arrhythmia in its own subplot
    # for i, arr in enumerate(arrhythmias):
    #     ax = axs[i]
    #     subset = df[df['Arrhythmia'] == arr]
    #     # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
    #     barplot(ax, subset))
    #     ax.set_title(arr, fontsize=6)
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     if i > 0:
    #         xlabels = ["F", "H", "A", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    #         ax.set_xticklabels(xlabels, fontsize=5)
    #         ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    #     else:
    #         ax.tick_params(axis='x', width=0.5, rotation=90, labelsize=5, color='#9099AA')
    #     if i % 6 == 0:
    #         ax.set_ylabel('F1 Score', fontsize=6)
    #     #else:
    #         #ax.spines['left'].set_visible(False)

    #     # make spines linewidth 0.5
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(0.5)
    #         spine.set_color('#9099AA')
    #     # rotate x labels 45 degrees and draw tick lines
    #     ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
    #     ax.set_ylim(0, 1)
    #     ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#bdc3c7')

    
    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.95, bottom=0.2, left=0.1, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig5-icentia-triage.svg")
    plt.savefig("./paper/images/fig5-icentia-triage.png", dpi=300)

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

def best_non_overlapping(intervals, lowerbetter=False):
    """
    Return the non-overlapping interval with the highest mean value.
    If no such interval exists, return None.
    """
    n = len(intervals)
    non_overlapping = []

    for i in range(n):
        lo1, hi1 = intervals[i]
        overlap = False
        for j in range(n):
            if i == j:
                continue
            lo2, hi2 = intervals[j]
            # Check overlap (endpoints count as overlap)
            if not (hi1 < lo2 or hi2 < lo1):
                overlap = True
                break
        if not overlap:
            non_overlapping.append(intervals[i])

    if not non_overlapping:
        return None

    if not lowerbetter:
        for i, (lo, hi) in enumerate(non_overlapping):
            highest = True
            for j in range(n):
                if i == j:
                    continue
                lo2, hi2 = intervals[j]
                v = (lo2 + hi2) / 2

                # Check if the current interval is higher than any other interval
                if v > (lo + hi) / 2:
                    highest = False
                    break

            if highest:
                return i
    else:
        for i, (lo, hi) in enumerate(non_overlapping):
            lowest = True
            for j in range(n):
                if i == j:
                    continue
                lo2, hi2 = intervals[j]
                v = (lo2 + hi2) / 2

                # Check if the current interval is higher than any other interval
                if v < (lo + hi) / 2:
                    lowest = False
                    break

            if lowest:
                return i

    # Pick interval with highest mean
    return -1

def generate_metrics_table(df, percentage=True):

    maxcol = 12

    def format_ci(ci, percentage=True, only_ci=False):
        if percentage:
            if only_ci:
                return "(" + str(np.round(ci[1]*1000)/10) + "-" + str(np.round(ci[2]*1000)/10) + ")"
            else:
                return str(np.round(ci[0]*1000)/10)
        else:
            if only_ci:
                return  "(" + str(np.round(ci[1], 2)) + "-" + str(np.round(ci[2], 2)) + ")"
            else:
                return str(np.round(ci[0], 2))

    print(df)
    metrics = df["Metric"].unique()
    models = df["Model"].unique()
    arrhythmias = df["Arrhythmia"].unique()

    metrics_formatted = {
        "acc": "Accuracy (95\% CI), \%",
        "se": "Sensitivity (95\% CI), \%",
        "sp": "Specificity (95\% CI), \%",
        "npv": "NPV (95\% CI), \%",
        "ppv": "PPV (95\% CI), \%",
        "f1": "F1 Score (95\% CI), \%",
        "TP": "Correct diagnoses (95\% CI)",
        "FP": "False alarms (95\% CI)",
        "FN": "Missed cases (95\% CI)",
    }
    arrhythmias_formatted = {
        "AFIB": "AFIB/AFL > 30s",
        "AVB_TYPE2": "Second-degree AVB",
        "BIGEMINY": "Bigeminy",
        "TRIGEMINY": "Trigeminy",
        "IVR": "IVR",
        "NOISE": "Noise",
        "NSR": "Normal Sinus Rhythm",
        "SUDDEN_BRADY": "Third-degree AVB",
        "SVT>30s": "SVT > 30s",
        "VT>10s": "VT > 10s",
        "VT<10s": "VT < 10s",
        "WENCKEBACH": "Wenckebach"
    }

    maxmetrics = int(maxcol/len(models))

    latex = "\\begin{tabular}{" + "l|"
    for metric in metrics[:maxmetrics]:
        for model in models:
            latex += "r"
        latex += "|" if metric != metrics[-1] else "} \n"

    latex += "Arrhythmia & "
    for metric in metrics[:maxmetrics]:
        latex += "\\multicolumn{"+str(len(models))+"}{c|}{\\textbf{" + metrics_formatted[metric] + "} } & "
    latex = latex[:-2] + " \\\\ \\hline\n"

    latex += " & "
    for metric in metrics[:maxmetrics]:
        for model in models:
            latex += "\\textbf{" + model + "} & "
    latex = latex[:-2] + " \\\\ \\hline\n"

    for arrhythmia in arrhythmias:
        latex += arrhythmias_formatted[arrhythmia] + " & "

        for metric in metrics[:maxmetrics]:
            intervals = []
            for model in models:
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)
                intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)

                if best_interval == i:
                    latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage) + "} & "
                else:
                    latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage) + " & "

        latex = latex[:-2] + " \\\\\n"

        latex +=  " & "
        for metric in metrics[:maxmetrics]:
            for model in models:
                intervals = []
            for model in models:
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)
                intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)

                if best_interval == i:
                    latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + "} & "
                else:
                    latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + " & "

        latex = latex[:-2] + " \\\\ \\hline \n"

    if len(metrics) > maxmetrics:
        latex = latex[:-2] + "\\hline \n"
        latex += " & "
        for metric in metrics[maxmetrics:]:
            latex += "\\multicolumn{"+str(len(models))+"}{c|}{\\textbf{" + metrics_formatted[metric] + "} } & "
        latex = latex[:-2] + " \\\\ \\hline\n"

        latex += " & "
        for metric in metrics[maxmetrics:]:
            for model in models:
                latex += "\\textbf{" + model + "} & "
        latex = latex[:-2] + " \\\\ \\hline\n"

        for arrhythmia in arrhythmias:
            latex += arrhythmias_formatted[arrhythmia] + " & "
            for metric in metrics[maxmetrics:]:
                for model in models:
                    intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)

                    if best_interval == i:
                        latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage) + "} & "
                    else:
                        latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage) + " & "

            latex = latex[:-2] + " \\\\\n"

            latex +=  " & "
            for metric in metrics[maxmetrics:]:
                for model in models:
                    intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)

                    if best_interval == i:
                        latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + "} & "
                    else:
                        latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + " & "

            latex = latex[:-2] + " \\\\ \\hline \n"

    latex = latex[:-1] + " \\hline \n"
    latex += "\\end{tabular}"

    return latex

if __name__ == "__main__":
    
    # Load the data
    icentia = ICENTIASAMPLEData("ICENTIA", 
        sample="/home/lukas/UU/ASRA/ALADINv2/traces_matt_v2/mapping.json", 
        annfile="/home/lukas/UU/ASRA/ALADINv2/paper/consensus.ods",
        allfile="/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA/samples_xxl.json",
        asynchronous=True)
    aladinvirt = ALADINVirtual()
    aladin_experiment = DiagnosticBenchmark(icentia, aladinvirt)

    print("ALADIN results")
    basefolder = os.environ.get('benchmark_results')
    aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_ICENTIA")
    aladin_metrics, aladin_distributions = aladin_experiment.aggregate(aladin_file, bootstrap=True, sampled=True)
    aladin_experiment.report()

    data = {
        "ALADIN": aladin_distributions
    }
    print("get average cardiologist metrics")
    data["Avg. Card."] = get_average_cardiologist_metrics()
    # for i in range(3):
    #     print(f"Card. {i+1}:")
    #     data[f"Card. {i+1}"] = get_cardiologist_metrics(i)
    #data["Technicians"] = get_technician_metrics()


    #diagnosis application
    arrhythmias = list(data["ALADIN"].keys())
    models = list(data.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    metrics = ["acc", "se", "sp", "ppv", "npv", "f1"]
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            for metric in metrics:
                scores = data[model][arr][metric]
                for i, score in enumerate(scores):
                    rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": metric})

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)
    # make_barplots(df)
    #print(generate_metrics_table(df))
    #make_confusion_matrix_aladin(aladin_file)
    # make_confusion_matrix_ecgfounder(ecgfounder_file)
    #make_ranking_plot(df)

    rowdata_triage = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            tp_scores = data[model][arr]['tp']
            for i, score in enumerate(tp_scores):
                rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "TP"})
            fp_scores = data[model][arr]['fp']
            for i, score in enumerate(fp_scores):
                rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "FP"})
            fn_scores = data[model][arr]['fn']
            for i, score in enumerate(fn_scores):
                rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "FN"})

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata_triage)
    print(generate_metrics_table(df, percentage=False))
    #make_barplots_horizontal(df)
