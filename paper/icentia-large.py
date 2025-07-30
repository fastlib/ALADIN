import os
from tqdm import tqdm
import gzip
import pickle
import wfdb
import re
import json
import datetime


def find_patients(base_path):
    patients = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("p") and len(dir_name) == 6:  # e.g., p00001
                patients.append({"patient": dir_name, "path": os.path.join(root, dir_name)})
    return patients

def get_diagnosis_from_ann(anns, fs=250):
    diagnoses = []
    beattypes = [a for a in anns.symbol if a != '+']
    beattypes = "".join(beattypes)
    beattypes = beattypes.replace('S', 'N')

    big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
    for big_match in big_matches:
        start = anns.sample[big_match.start()]
        end = anns.sample[big_match.end()-1]
        #print("BIG pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
        if (end-start)/fs > 10:
            diagnoses.append(["BIGEMINY", start, end])

    tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
    for tri_match in tri_matches:
        start = anns.sample[tri_match.start()]
        end = anns.sample[tri_match.end()-1]
        #print("TRI pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
        if (end-start)/fs > 10:
            diagnoses.append(["TRIGEMINY", start, end])

    vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
    for vt_ivr_match in vt_ivr_matches:
        start = anns.sample[vt_ivr_match.start()]
        end = anns.sample[vt_ivr_match.end()-1]
        #print("VT/IVR pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
        if (end-start)/fs > 10:
            diagnoses.append(["VT", start, end])
    #search for VVV pattern

    low_signal_quality_matches = re.finditer(r'Q{5,}', beattypes)
    for low_signal_quality_match in low_signal_quality_matches:
        start = anns.sample[low_signal_quality_match.start()]
        end = anns.sample[low_signal_quality_match.end()-1]
        #print("Low signal quality pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (low_signal_quality_match.end()-low_signal_quality_match.start()), "beats")
        diagnoses.append(["LOW_SIGNAL_QUALITY", start, end])

    rhythm_type = ""
    rhythm_start = 0
    #foundafib = False

    for idx, beat in enumerate(anns.symbol):
        if beat == 'N':
            continue
        if beat == 'V':
            start = anns.sample[idx] - fs*0.15
            end = anns.sample[idx] + fs*0.15
            #labelregions.append(["PVC", start, end])
        elif beat == 'S':
            start = anns.sample[idx] - fs*0.15
            end = anns.sample[idx] + fs*0.15
            #labelregions.append(["SVPB", start, end])
        elif beat == '+':
            if anns.aux_note[idx] == '(N':
                rhythm_type = "NSR"
                rhythm_start = anns.sample[idx]
            elif anns.aux_note[idx] == '(AFIB':
                rhythm_type = "AFIB/AFL"
                rhythm_start = anns.sample[idx]
                foundafib = True
                #print("AFIB/AFL", rhythm_start)
            elif anns.aux_note[idx] == '(AFL':
                rhythm_type = "AFIB/AFL"
                rhythm_start = anns.sample[idx]
                foundafib = True
            elif anns.aux_note[idx] == ')':
                if (anns.sample[idx]-rhythm_start)/fs > 30:
                    if rhythm_type != "NSR":
                        diagnoses.append([rhythm_type, rhythm_start, anns.sample[idx]])
                    # if rhythm_type != "NSR":
                    #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[idx]-rhythm_start)/fs, "s")
                rhythm_type = ""
                rhythm_start = 0

    if rhythm_type and rhythm_start:
        if rhythm_type != "NSR":
            diagnoses.append([rhythm_type, rhythm_start, anns.sample[-1]])
        # if rhythm_type != "NSR":
        #     print("Rhythm type:", rhythm_type, "with duration: ", (anns.sample[-1]-rhythm_start)/fs, "s")

    return diagnoses

def get_results_from_folder(folder_path):

    ds = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl.gz'):
            with gzip.open(os.path.join(folder_path, file_name), 'rb') as f:
                data = pickle.load(f)
                diagnoses = data.get("diagnosis", [])
                d = []
                for diagnosis in diagnoses:
                    if diagnosis["type"] == "AFIB" and (diagnosis["offset"] - diagnosis["onset"]) > 30 * 250:
                        d.append("AFIB/AFL")
                    elif diagnosis["type"] == "VT" and (diagnosis["offset"] - diagnosis["onset"]) > 10 * 250:
                        d.append("VT")
                    elif diagnosis["type"] == "SVT" and (diagnosis["offset"] - diagnosis["onset"]) > 10 * 250:
                        d.append("SVT")
                    elif diagnosis["type"] == "TRIGEMINY" and (diagnosis["offset"] - diagnosis["onset"]) > 10 * 250:
                        d.append("TRIGEMINY")
                    elif diagnosis["type"] == "BIGEMINY" and (diagnosis["offset"] - diagnosis["onset"]) > 10 * 250:
                        d.append("BIGEMINY")
                    elif diagnosis["type"] == "IVR" and (diagnosis["offset"] - diagnosis["onset"]) > 10 * 250:
                        d.append("IVR")
                    elif diagnosis["type"] == "WENCKEBACH" or \
                            diagnosis["type"] == "AVB" or \
                            diagnosis["type"] == "AVB_TYPE2" or \
                            diagnosis["type"] == "CHB" or \
                            diagnosis["type"] == "AVB_TYPE1" or \
                            diagnosis["type"] == "IVB" or \
                            diagnosis["type"] == "BRADYCARDIA" or \
                            diagnosis["type"] == "TACHYCARDIA":
                        d.append(diagnosis["type"])
            
            relative_path = os.path.relpath(folder_path, os.environ.get('benchmark_data')+'/ICENTIA-results/ICENTIA')
            recordname = file_name[:-7]
            #patient_id = 
            anns = wfdb.rdann(os.path.join("/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA", relative_path, recordname), 'atr')
            t = get_diagnosis_from_ann(anns)
            ts = [i[0] for i in t]

            ds.append({"record": recordname, "true":ts, "predicted": d})#, "raw": diagnoses})
            
    return ds

def create_results_json(patients):

    results = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "ICENTIA",
        "results": [
            {
                "model": "ALADIN", 
                "modelpaths": ["Dataset200_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"], 
                "results": []
            }
        ]
    }
    for patient in tqdm(patients, desc="Processing patients"):
        patient_path = patient["path"]
        patient_results = get_results_from_folder(patient_path)
        results["results"][0]["results"].extend(patient_results)

    results["results"][0]["results"].sort(key=lambda x: x["record"])

    with open(os.path.join(os.environ.get('benchmark_results'), "diagnosis", "set_level_diagnosis_ALADIN_ICENTIA_[" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "].json"), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    basefolder = os.environ.get('benchmark_data')
    patients = find_patients(basefolder+'/ICENTIA-results/ICENTIA')
    print(f"Found {len(patients)} patient folders.")

    create_results_json(patients)
