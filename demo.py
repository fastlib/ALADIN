import numpy as np
import pandas as pd
import pickle
import json
import os
import glob
import wfdb
from tqdm import tqdm
import ast
import argparse

from aladin import ALADIN
from aladin.core import Record

import matplotlib.pyplot as plt

def load_case(dir, case):

    file = os.path.join(dir,case)
    rec = wfdb.rdrecord(file)
    ecg = { 
        "II": rec.p_signal[:,0]
    }

    annpath = dir+"/"+case+'.episodes.json'
    episodes = json.load(open(glob.glob(annpath)[0]))["episodes"]

    anntypes = [ann["rhythm_name"] for ann in episodes]

    record = Record(ecg, rec.fs, "DEMO", case)
    record.groundtruth = anntypes
    return record

def plot_median_beat(record):
    median_beat = record.median_beat.ecg
    p_onset = record.median_beat.delineations.p.onset
    p_offset = record.median_beat.delineations.p.offset
    qrs_onset = record.median_beat.delineations.qrs.onset
    qrs_offset = record.median_beat.delineations.qrs.offset
    t_onset = record.median_beat.delineations.t.onset
    t_offset = record.median_beat.delineations.t.offset


    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
    
    ax.plot(median_beat[0,:], color='black', label="ECG Signal")
    ax.set_title("Median Beat")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")

    
    pcolor = '#e74c3c'
    tcolor = '#3498DB'
    qrscolor = '#2ecc71'

    ax.axvspan(p_onset, p_offset, color=pcolor, alpha=0.5, label="P wave")
    ax.axvspan(qrs_onset, qrs_offset, color=qrscolor, alpha=0.5, label="QRS complex")
    ax.axvspan(t_onset, t_offset, color=tcolor, alpha=0.5, label="T wave")

    plt.savefig("median_beat.png")

def analyse_single_case(record):

    aladin = ALADIN(modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"],
                    debug={"segmenter": True, "afibdetector": False, "reflection": False, "total": True})

    #segment     
    aladin.segment(record)
    #see record.delineations.[p, qrs, t, abnormal_qrs, noise, afib] for binary masks

    #extract median beat
    aladin.extract_median_beat(record)
    median_beat = record.median_beat.ecg
    #see record.median_beat.delineations.[p, qrs, t].[onset, offset, mask] for median beat delineations
    
    plot_median_beat(record)

    #analyse and diagnose
    aladin.analyse(record)
    #see record.diagnosis.[name, explanation, onset, offset] for the predicted diagnosis, and record.subdiagnosis.[] for the predicted subdiagnosis
    
    #use aladin without preprocessing, so that you can do your own preprocessing
    #NOTE: Performance may be altered if you use your preprocessing
    aladin.analyse(record, preprocess=False)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Load and analyze ECG cases")
    argparser.add_argument("--case", type=str, help="Case name to analyze", required=True)
    args = argparser.parse_args()
    case = args.case

    # Specify the path to the directory containing the .ecg files
    directory_path = "./data/demo"

    #Load the ECG file from disk and create the Record object
    record = load_case(directory_path, case)

    #Segment and analyse the ECG
    analyse_single_case(record)

