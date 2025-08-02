
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
        #if (end-start)/fs > 10:
        diagnoses.append(["BIGEMINY", start, end])

    tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
    for tri_match in tri_matches:
        start = anns.sample[tri_match.start()]
        end = anns.sample[tri_match.end()-1]
        #print("TRI pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
        #if (end-start)/fs > 10:
        diagnoses.append(["TRIGEMINY", start, end])

    vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
    for vt_ivr_match in vt_ivr_matches:
        start = anns.sample[vt_ivr_match.start()]
        end = anns.sample[vt_ivr_match.end()-1]
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
            if (end-start)/fs >= 10:
                diagnoses.append(["IVR", start, end])

    #search for VVV pattern

    low_signal_quality_matches = re.finditer(r'Q{3,}', beattypes)
    for low_signal_quality_match in low_signal_quality_matches:
        start = anns.sample[low_signal_quality_match.start()]
        end = anns.sample[low_signal_quality_match.end()-1]
        #print("Low signal quality pattern found in", recordname, "with duration: ", (end-start)/fs, "s, and ", (low_signal_quality_match.end()-low_signal_quality_match.start()), "beats")
        if (end-start)/fs > 10:
            diagnoses.append(["NOISE", start, end])

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

def get_diagnosis_from_aladin(diagnoses, fs=250):

    filtered_diagnoses = []

    for diagnosis in diagnoses:
        onset = diagnosis.get("onset", 0)
        offset = diagnosis.get("offset", 0)
        duration = (offset - onset) / 250  # Convert to seconds

        if diagnosis["type"] == "AFIB" and duration >= 30:
            filtered_diagnoses.append({"type": "AFIB", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "VT" and duration >= 10:
            filtered_diagnoses.append({"type": "VT>10s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "VT" and duration < 10:
            filtered_diagnoses.append({"type": "VT<10s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "SVT" and duration >= 30:
            filtered_diagnoses.append({"type": "SVT>30s", "onset": onset, "offset": offset})
        # elif diagnosis["type"] == "SVT" and duration > 10:
        #     filtered_diagnoses.append({"type": "SVT<30s", "onset": onset, "offset": offset})
        elif diagnosis["type"] == "IVR" and duration >= 10:
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

    return filtered_diagnoses

def find_arrhythmia_in_record(root, file, tally, botoclient):

    dataroot = "/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA"
    if not os.path.exists(os.path.join(dataroot, file + ".atr")):

        # Define the S3 bucket and the local directory
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        botoclient.download_file(bucket_name, prefix + file+".atr", os.path.join(dataroot, file + ".atr"))
        print(f"Downloaded {file}.atr from S3 bucket {bucket_name} to {dataroot}")

    if not os.path.exists(os.path.join(dataroot, file + ".dat")):

        # Define the S3 bucket and the local directory
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        botoclient.download_file(bucket_name, prefix + file+".dat", os.path.join(dataroot, file + ".dat"))
        print(f"Downloaded {file}.dat from S3 bucket {bucket_name} to {dataroot}")

    if not os.path.exists(os.path.join(dataroot, file + ".hea")):

        # Define the S3 bucket and the local directory
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        botoclient.download_file(bucket_name, prefix + file+".hea", os.path.join(dataroot, file + ".hea"))
        print(f"Downloaded {file}.hea from S3 bucket {bucket_name} to {dataroot}")

    anns = wfdb.rdann(os.path.join(dataroot, file), 'atr')
    human_diagnoses = get_diagnosis_from_ann(anns)

    with gzip.open(os.path.join(root, file + ".pkl.gz"), 'rb') as f:
        data = pickle.load(f)

    aladin_diagnoses = get_diagnosis_from_aladin(data["diagnosis"])

    rec = wfdb.rdrecord(os.path.join(dataroot, file))
    ecg = rec.p_signal[:, 0]  # Assuming the first channel is the ECG signal

    selected_arrhythmias = find_discrepancies(human_diagnoses, tally, aladin_diagnoses)
    make_trace(ecg, data["delineations"], selected_arrhythmias, file)

    return selected_arrhythmias


def calculate_overlap(start1, end1, start2, end2):
    """
    Calculate the overlap between two time intervals.
    Returns the overlap duration if they overlap, otherwise returns 0.
    """
    if start1 < end2 and start2 < end1:
        return min(end1, end2) - max(start1, start2)
    return 0

def find_discrepancies(human_diagnoses, tally, aladin_diagnoses):

    unique_human = list(set(hd[0] for hd in human_diagnoses))
    unique_aladin = list(set(ad["type"] for ad in aladin_diagnoses))
    unique_both = list(set(unique_human + unique_aladin))
    print("Unique human diagnoses:", unique_human)
    print("Unique Aladin diagnoses:", unique_aladin)
    print("Unique diagnoses in both:", unique_both)

    not_interested = ["NOISE"]
    unique_both = [d for d in unique_both if d not in not_interested]

    selected_arrhythmias = []

    for arrhythmia in unique_both:
        if arrhythmia not in tally:
            tally[arrhythmia] = 0

        if tally[arrhythmia] > 250:
            continue
        print(f"Checking arrhythmia: {arrhythmia}")
        human_arrhythmias = [{"type": a[0], "onset": a[1], "offset": a[2], "human": False, "aladin": False} for a in human_diagnoses if a[0] == arrhythmia]
        aladin_arrhythmias = [{"type": a["type"], "onset": a["onset"], "offset": a["offset"], "human": False, "aladin": False} for a in aladin_diagnoses if a["type"] == arrhythmia]
        all_arrhythmias = human_arrhythmias + aladin_arrhythmias
        selected_arrhythmias.append(np.random.choice(all_arrhythmias, 1)[0])

    for arrhythmia in selected_arrhythmias:
        onset = arrhythmia["onset"]
        offset = arrhythmia["offset"]
        arr = arrhythmia["type"]

        if arr not in tally:
            tally[arr] = 0

        tally[arr] += 1

        human_arr = [a for a in human_diagnoses if a[0] == arrhythmia["type"]]
        aladin_arr = [a for a in aladin_diagnoses if a["type"] == arrhythmia["type"]]

        for arr in human_arr:
            overlap = calculate_overlap(arr[1], arr[2], onset, offset)
            if overlap > 0:
                arrhythmia["human"] = True
                break

        for arr in aladin_arr:
            overlap = calculate_overlap(arr["onset"], arr["offset"], onset, offset)
            if overlap > 0:
                arrhythmia["aladin"] = True
                break


    for arrhythmia in selected_arrhythmias:
        if not arrhythmia["human"] and not arrhythmia["aladin"]:
            print(f"Arrhythmia {arrhythmia['type']} has no overlap with human or Aladin diagnoses.")
        elif arrhythmia["human"] and not arrhythmia["aladin"]:
            print(f"Arrhythmia {arrhythmia['type']} is detected by human but not by Aladin.")
        elif not arrhythmia["human"] and arrhythmia["aladin"]:
            print(f"Arrhythmia {arrhythmia['type']} is detected by Aladin but not by human.")
        else:
            print(f"Arrhythmia {arrhythmia['type']} is detected by both human and Aladin.")

    return selected_arrhythmias

def make_trace(ecg, delineation, events, filename):
    
    if not os.path.exists("./traces"):
        os.makedirs("./traces")

    print(delineation.keys())

    for event in events:
        fs = 250
        window = 60
        onset = event["onset"]
        offset = event["offset"]
        middle = (onset + offset) // 2
        start = int(max(0, middle - fs*window//2))  # 30 seconds before the middle
        end = int(min(len(ecg), middle + fs*window//2))  # 30 seconds after the middle

        arrhythmia_type = event["type"]
        ts = np.linspace(0, len(ecg)/fs, len(ecg))

        plt.figure(figsize=(window, 4))
        plt.plot(ts[start:end], ecg[start:end], label='ECG Signal', color='blue')
        plt.axvline(x=onset/fs, color='red', linestyle='--', label='Onset')
        plt.axvline(x=offset/fs, color='green', linestyle='--', label='Offset')
        plt.xlim(ts[start], ts[end-1])
        plt.title(f'ECG Signal with {arrhythmia_type}. Detected by Aladin: {event["aladin"]}, Human: {event["human"]}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        pcolor = '#e74c3c'
        tcolor = '#3498DB'
        qrsabnormcolor = '#f1c40f'
        qrscolor = '#2ecc71'

        for p in delineation["p"]:
            if p[0] >= start and p[0] <= end:
                plt.axvspan((p[0])/fs, (p[1])/fs, color=pcolor, alpha=0.5)

        for qrs in delineation["qrs"]:
            col = qrscolor if not qrs[2] else qrsabnormcolor
            if qrs[0] >= start and qrs[0] <= end:
                plt.axvspan((qrs[0])/fs, (qrs[1])/fs, color=col, alpha=0.5)

        for t in delineation["t"]:
            if t[0] >= start and t[0] <= end:
                plt.axvspan((t[0])/fs, (t[1])/fs, color=tcolor, alpha=0.5)

        for noise in delineation["noise"]:
            if noise[0] >= start and noise[0] <= end:
                plt.axvspan((noise[0])/fs, (noise[1])/fs, color='gray', alpha=0.5)

        filename = filename.replace("/", "_").replace("\\", "_")
        print(f"Saving trace for {arrhythmia_type} in file: ./{filename}_{arrhythmia_type}.png")
        plt.savefig("./traces/" + filename + f'_{arrhythmia_type}.png')
        plt.close()

def find_all_file_paths(root):
    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.pkl.gz'):
                path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(path, root)
                file_paths.append(relative_path[:-7])
    return file_paths

if __name__ == "__main__":

    boto_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    root_directory = '/home/lukas/UU/ASRA/ALADINv2/data/ICENTIA-results/ICENTIA-Dataset301'
    file_paths = find_all_file_paths(root_directory)
    print(f"Found {len(file_paths)} files in {root_directory}")
    np.random.shuffle(file_paths)

    all_arrhythmias = []
    tally = {}
    for file in tqdm(file_paths[:100]):
        #print(f"Processing file: {file}")
        arrhythmias = find_arrhythmia_in_record(root_directory, file, tally, boto_client)
        all_arrhythmias.extend(arrhythmias)

    print("Tally of arrhythmias detected:")
    for arrhythmia, count in tally.items():
        print(f"{arrhythmia}: {count}")

    print(f"Total arrhythmias found: {len(all_arrhythmias)}")


