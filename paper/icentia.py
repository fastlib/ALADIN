import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import matplotlib.pyplot as plt

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

basefolder = os.environ.get('benchmark_results')
aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_ICENTIA")
#aladin_file = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_ICENTIA_[2025-07-24_10-56-23].json"
#aladin_file = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_ICENTIA_[2025-07-24_15-57-42].json"

data = json.load(open(aladin_file, 'r'))
results = data["results"][0]["results"]

trues = [list(set(r["true"])) for r in results]
preds = [list(set(r["predicted"])) for r in results]
recs = [r["record"] for r in results]

# patients = np.unique([r["record"].split("_")[0] for r in results])
# patient_trues = {p: [] for p in patients}
# patient_preds = {p: [] for p in patients}

# for i in range(len(trues)):
#     patient = recs[i].split("_")[0]
#     patient_trues[patient].extend(trues[i])
#     if len(rawpreds[i]) > 0:
#         for j in range(len(rawpreds[i])):
#             if rawpreds[i][j]["type"] == "AFIB" and (rawpreds[i][j]["offset"] - rawpreds[i][j]["onset"]) > 30 * 250:
#                 preds[i].append("AFIB/AFL")
#                 patient_preds[patient].append("AFIB/AFL")
#             elif rawpreds[i][j]["type"] == "VT" and (rawpreds[i][j]["offset"] - rawpreds[i][j]["onset"]) > 10 * 250:
#                 preds[i].append("VT")
#                 patient_preds[patient].append("VT")
#             elif rawpreds[i][j]["type"] == "TRIGEMINY" and (rawpreds[i][j]["offset"] - rawpreds[i][j]["onset"]) > 10 * 250:
#                 preds[i].append("TRIGEMINY")
#                 patient_preds[patient].append("TRIGEMINY")
#             elif rawpreds[i][j]["type"] == "BIGEMINY" and (rawpreds[i][j]["offset"] - rawpreds[i][j]["onset"]) > 10 * 250:
#                 preds[i].append("BIGEMINY")
#                 patient_preds[patient].append("BIGEMINY")
#             elif rawpreds[i][j]["type"] == "IVR" and (rawpreds[i][j]["offset"] - rawpreds[i][j]["onset"]) > 10 * 250:
#                 preds[i].append("IVR")
#                 patient_preds[patient].append("IVR")

# Remove duplicates in preds
# for p in list(patient_trues.keys()):
#     patient_trues[p] = list(set(patient_trues[p]))
#     patient_preds[p] = list(set(patient_preds[p]))

# trues = [patient_trues[p] for p in patients]
# preds = [patient_preds[p] for p in patients]

        

arrhythmias = list(set([item for sublist in trues for item in sublist]))
arrhythmias = ["NSR", "AFIB/AFL", "VT", "TRIGEMINY", "BIGEMINY"]
del arrhythmias[arrhythmias.index("NSR")]  # Remove NSR as it is not an arrhythmia
print("Arrhythmias:", arrhythmias)

tp = [0] * len(arrhythmias)
fp = [0] * len(arrhythmias)
tn = [0] * len(arrhythmias)
fn = [0] * len(arrhythmias)
for i, arrhythmia in enumerate(arrhythmias):
    for j in range(len(trues)):
        if arrhythmia in trues[j]:
            if arrhythmia in preds[j]:
                tp[i] += 1
            else:
                fn[i] += 1
                #print(f"False Positive for {arrhythmia} in record {patients[j]}: {preds[j]}")
        elif arrhythmia in preds[j]:
            fp[i] += 1
            #print(f"False Positive for {arrhythmia} in record {patients[j]}: {preds[j]}")
        else:
            tn[i] += 1
            
for i, arrhythmia in enumerate(arrhythmias):
    print(f"{arrhythmia}: TP={tp[i]}, FP={fp[i]}, TN={tn[i]}, FN={fn[i]}")
    precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
    recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"F1 Score for {arrhythmia}: {f1:.4f}")

#print(len(results))

# predictions = np.array([mapper[p] for p in predictions])

# confusion = confusion_matrix(Y, np.round(predictions).astype(int), labels=list(mapper.values()))
# print(confusion)

# f1_scores = []

# for i in range(len(confusion)):
#     TP = confusion[i, i]
#     FP = confusion[:, i].sum() - TP
#     FN = confusion[i, :].sum() - TP
    
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
#     f1_scores.append(f1)
#     print(f1)

# print("F1 Scores:", f1_scores)
# print("Average F1 Score:", np.mean(f1_scores[:3]))


