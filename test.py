import numpy as np
import pandas as pd
import pickle
import json
import time
import os
import re
import glob
import wfdb
from tqdm import tqdm
import ast
import argparse
import matplotlib.pyplot as plt

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import NoCredentialsError

#os.environ["aladin_models"] = "/home/lukas/UU/ASRA/ALADINv2/models"
from aladin import ALADIN
from aladin.core import Record, RecordCollection


def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

def load_case_csv(dir, case):
    file = os.path.join(dir, case + ".csv")
    df = pd.read_csv(file, header=None)
    ecg = df.iloc[:, 1].values
    fs = 500

    record = Record(ecg, fs, "DEMO", case)
    return record

def cusum(signal, k=0.5, h=5):
    s_pos = np.zeros(len(signal))
    s_neg = np.zeros(len(signal))
    change_points = []

    for i in range(1, len(signal)):
        s_pos[i] = max(0, s_pos[i-1] + signal[i] - k)
        s_neg[i] = min(0, s_neg[i-1] + signal[i] + k)

        if s_pos[i] > h:
            print(s_pos[i], s_neg[i])
            change_points.append(i)
            s_pos[i] = 0
            s_neg[i] = 0
        elif s_neg[i] < -h:
            change_points.append(i)
            s_pos[i] = 0
            s_neg[i] = 0

    return change_points

def load_case(dir, case, database="STANFORD"):

    file = os.path.join(dir,case)

    if database=="ICENTIA" and not os.path.exists(file + ".dat"):
        bucket_name = 'physionet-open'
        prefix = 'icentia11k-continuous-ecg/1.0/'
        boto_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        if not os.path.exists(file + ".dat"):
            print("Downloading", file + ".dat")
            print(prefix + file + ".dat")
            os.makedirs(os.path.dirname(file), exist_ok=True)
            boto_client.download_file(bucket_name, prefix + case + ".dat", file + ".dat")
        if not os.path.exists(file + ".hea"):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            boto_client.download_file(bucket_name, prefix + case + ".hea", file + ".hea")
        if not os.path.exists(file + ".atr"):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            boto_client.download_file(bucket_name, prefix + case + ".atr", file + ".atr")

    print(file)
    rec = wfdb.rdrecord(file)
    ecg = rec.p_signal[:,0]
    fs = rec.fs
    #ecg -= np.mean(ecg)  # Center the signal
    #ecg /= np.std(ecg)  # Normalize the signal

    if os.path.exists(file + ".atr"):
        anns = wfdb.rdann(file, 'atr')
        anntypes = anns.symbol
        beattypes = "".join(anntypes)
        beattypes = beattypes.replace('S', 'N')
        diagnoses = []
        print(beattypes)

        big_matches = re.finditer(r'((NV){3,}|(VN){3,})', beattypes)
        for big_match in big_matches:
            start = anns.sample[big_match.start()]
            end = anns.sample[big_match.end()-1]
            print("BIG pattern found in", file, "with duration: ", (end-start)/fs, "s, and ", (big_match.end()-big_match.start()), "beats")
            if (end-start)/fs > 10:
                diagnoses.append(["BIGEMINY", start, end])

        tri_matches = re.finditer(r'((VNN){3,}|(NNV){3,})', beattypes)
        for tri_match in tri_matches:
            start = anns.sample[tri_match.start()]
            end = anns.sample[tri_match.end()-1]
            print(beattypes[tri_match.start():tri_match.end()])
            print("TRI pattern found in", file, "between ", start/fs, " and ", end/fs," with duration: ", (end-start)/fs, "s, and ", (tri_match.end()-tri_match.start()), "beats")
            if (end-start)/fs > 10:
                diagnoses.append(["TRIGEMINY", start, end])

        vt_ivr_matches = re.finditer(r'V{3,}', beattypes)
        for vt_ivr_match in vt_ivr_matches:
            start = anns.sample[vt_ivr_match.start()]
            end = anns.sample[vt_ivr_match.end()-1]
            print("VT/IVR pattern found in", file, "with duration: ", (end-start)/fs, "s, and ", (vt_ivr_match.end()-vt_ivr_match.start()), "beats")
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

        for diagnose in diagnoses:
            print("Diagnosis:", diagnose[0], "from", diagnose[1]/rec.fs, "s to", diagnose[2]/rec.fs, "s")
    #annnotes = ann.aux_note
    #print(annnotes)

    #annpath = dir+"/"+case+'.episodes.json'
    #episodes = json.load(open(glob.glob(annpath)[0]))["episodes"]

    #anntypes = [ann["rhythm_name"] for ann in episodes]

    record = Record(ecg, rec.fs, "DEMO", case)

    # fig, ax = plt.subplots(1, 1, figsize=(20, 4), sharex=True, dpi=200)
    # ax.plot(record.filtered_ecg, label="ECG Signal")
    # plt.savefig("demo.png")

    #record.groundtruth = anntypes
    return record

def load_case_trimmed(dir, case, start, end):
    file = os.path.join(dir, case)
    rec = wfdb.rdrecord(file)
    ecg = rec.p_signal[:, 0]

    fs = rec.fs
    st = int(start * fs)
    en = int(end * fs)

    if st < 0 or en > len(ecg):
        raise ValueError("Start and end times are out of bounds for the ECG signal.")

    ecg_segment = ecg[st:en]

    record = Record(ecg_segment, fs, "DEMO", case)
    return record

def analyse_single_case(record):
    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": True})
    aladin.segmenter.segment(record)
    print(record.diagnoses)

def test_loading():
    try:
        aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": True})
    except Exception as e:
        print(f"Error loading ALADIN: {e}")
        return False
    
def test_record_creation():
    rec = load_case("./data/demo", "STANFORD1")
    assert rec is not None, "Failed to create Record object"
    assert rec.get_fs() == 200, "Sample rate mismatch"

def test_segmenter():
    rec = load_case("./data/demo", "STANFORD1")
    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": True})
    aladin.segmenter.segment(rec)

def compare_outputs(arrhythmia):
    file1 = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_CINC_[2025-03-19_22-20-35].json"
    file2 = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_CINC_[2025-07-23_16-35-33].json"
    # file1 = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_STANFORD_[2025-04-03_01-33-02].json"
    # file2 = "/home/lukas/UU/ASRA/ALADINv2/results/diagnosis/set_level_diagnosis_ALADIN_STANFORD_[2025-06-23_14-40-40].json"

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    res1 = data1["results"][0]["results"]
    res2 = data2["results"][0]["results"]
    # Compare the two dictionaries
    recs = [d["record"] for d in res1]

    for rec in recs:
        row1 = next((item for item in res1 if item["record"] == rec), None)
        row2 = next((item for item in res2 if item["record"] == rec), None)
        isinrow1 = True if arrhythmia in row1["predicted"] else False
        isinrow2 = True if arrhythmia in row2["predicted"] else False
        if isinrow1 != isinrow2:
            print(f"Record {rec} has different arrhythmia predictions: {row1['predicted']} vs {row2['predicted']}, which should be {row1['true']}.")
    #print(recs)

def test_reflection():
    # b6ffad5cb4baa43e80ff8018708bf88a_0002
    # 3271d3a1e20669bd85db5e322de5602b_0002
    # b76ef7bf6f59ae17a7db71c14643e28d_0001
    # cb2b409154577abdc8f461eda56bf93f_0001
    # 33d52a72127a243615430ee6c4fbd037_0001
    # c5daac7461d15864fc9727e13d9d6328_0001
    # b672c71b171342498d52cb9ef64091e9_0002
    # 04dca24ac7ef2bd6d787c2503273259b_0001
    # 04b3b85424b46f8d9637b42d6090e2a4_0005
    # 2f129c9200f90bb7aedd56539d3a378e_0003
    # 7ee047a61367ba3ca3fd97e2dc5e3349_0002

    # EAR
    # d8f20c17f741ebee049a76bdf1c02570_0001
    # 8046165010b6735e86ef45bbe839d374_0005
    # d2e7e092def3cf8bcd070acc4331db04_0006
    # 48613fc0013262a638e33bc132813d98_0001
    # f16dfbd64d61064e681e158eebda625f_0002
    # a9d9501b7dfb21eb142e5df6f3038bd6_0001
    # 691e03fc2eea41dab3f7873eda02c240_0003
    # 22c28207a311e1f54ab009d2269bee2d_0003
    # ca349fb0ef0fa751b0fb20bf5a2e59bd_0004

    #IVR
    # c3f2dcdc7c21930f29f686c5e82b4133_0001
    # e3cfc5a263bd1bc1cc65b0c17f3d3e65_0001
    # 482de87985baff5b29ad2e21af95a920_0003
    # 7ee047a61367ba3ca3fd97e2dc5e3349_0002
    # 3d3e478d6450c89733b7e9de303329f7_0001
    # 2c83b85218ad9451eba4b6491c3d5863_0002
    # f9113eb0786f9bf39c2684a4031f44a6_0009
    # ab04852a19295329c9cbeac417876875_0002
    # c1009c76e2fca07103199c6d3f20479a_0004
    # aaf16bad905b218608ae343969c75beb_0001
    # 0e9148f405e39b98568ad169c1cb1e21_0002

    #48d4e8ce9bb72876e9f5cc98cd52aae3_0003

    rec = load_case("./data/STANFORD", "1ab23e8d8299643d0262ecd7cf51af4e_0001")
    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": True, "total": True})
    aladin.analyse(rec)
    aladin.plot(rec)


def analyse_result_per_diagnosis(diagnosis):


    basefolder = os.environ.get('benchmark_results')
    file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_STANFORD")
    dat = json.load(open(file))

    results = dat["results"][0]["results"]

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(results)):
        isdiagnosed = False
        for d in results[i]["predicted"]:
            if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                #print("AVB")
                isdiagnosed = True
                break
            if d == diagnosis:
                isdiagnosed = True
                break
        isannotated = False
        for d in results[i]["true"]:
            if diagnosis == "AVB" and (d == "SUDDEN_BRADY" or d == "AVB_TYPE2"):
                isannotated = True
                break
            if diagnosis == "AFIB/AFL" and (d == "AFIB" or d == "AFL"):
                isannotated = True
                break
            if d == diagnosis:
                isannotated = True
                break

        if isdiagnosed and isannotated:
            tp += 1
        elif isdiagnosed and not isannotated:
            print("False positive:", results[i]["record"], results[i]["true"])
            fp += 1
        elif not isdiagnosed and isannotated:
            print("False negative:", results[i]["record"])
            fn += 1

    if tp > 0:
        se = tp / (tp + fn)
        pp = tp / (tp + fp)
        f1 = 2 * (se * pp) / (se + pp)

        print("Diagnosis: ", diagnosis)
        print(tp, fp, fn)
        print("F1: ", f1)
        print("Sensitivity: ", se)
        print("Precision: ", pp)
    else:
        print("Diagnosis: ", diagnosis)
        print("No true positives")

def test_rdb():

    recfiles = glob.glob("/home/lukas/UU/ASRA/ALADINv2/data/RDB/dat_csv/*.csv")
    print(f"Found {len(recfiles)} records")

    records = []
    for file in recfiles:
        relative_path = file.split("/")[-1][:-4]
        rec = load_case_csv("/home/lukas/UU/ASRA/ALADINv2/data/RDB/dat_csv", relative_path)
        records.append(rec)

    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": True, "total": True})
    aladin.analyse(records[1])

def test_icentia():
    t0 = time.time()
    basefolder = os.environ.get('benchmark_data')
    #rec = load_case("./data/ICENTIA", "p00/p00153/p00153_s20", "ICENTIA")
    rec = load_case("./data/ICENTIA", "p06/p06226/p06226_s32", "ICENTIA")
    #rec = load_case_trimmed(basefolder+"/ICENTIA", "p07/p07840/p07840_s28", 2000, 2200)
    #rec = load_case_trimmed("/home/lukas/UU/ASRA/Datasets/ICENTIA", "p07/p07840/p07840_s28", 3800, 3950)
    #rec = load_case_trimmed("/home/lukas/UU/ASRA/Datasets/ICENTIA", "p02/p02050/p02050_s48", 2800, 3000)
    #rec = load_case("/home/lukas/UU/ASRA/Datasets/MIT-NORMAL", "16265")

    t1 = time.time()
    print(f"Loading record took {t1-t0:.2f} seconds")
    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": False, "afibdetector": False, "reflection": False, "total": False})
    aladin.analyse(rec)
    aladin.plot(rec, xlim=(380,600))
    t2 = time.time()
    print(f"ALADIN took {t2-t1:.2f} seconds")

def test_batch():
    aladin = ALADIN(modelpaths=["Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": False, "afibdetector": False, "reflection": False, "total": False})
    records = []
    # for file in os.listdir("./data/STANFORD"):
    #     if file.endswith(".dat"):
    #         rec = load_case("./data/STANFORD", file[:-4])
    #         records.append(rec)

    records = [
        #load_case("/home/lukas/UU/ASRA/Datasets/ICENTIA", "p01/p01056/p01056_s13"),
        load_case("/home/lukas/UU/ASRA/Datasets/ICENTIA", "p05/p05205/p05205_s15"),
    ]

    collection = RecordCollection(records)
    collection.preprocess()

    aladin.analyse_batch(records)
    

if __name__ == "__main__":
    test_icentia()
    #test_reflection()
    #test_batch()
    #test_rdb()
    #compare_outputs("A")
    #analyse_result_per_diagnosis("SVT")
    print("All tests passed.")

