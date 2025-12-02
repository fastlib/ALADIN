import numpy as np
import pandas as pd
import json
import os

import argparse

from pymongo import MongoClient

import sys
sys.path.append('./benchmark')
from dataloader import MongoDBDatasetGenerator
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results


def to_one_hot(segmentation):
    #p, qrs, t, nothing, noise, abnormal
    return segmentation[[3,0,1,2,4,5],:]

def create_training_set_files(dl, name, iter=0):

    print(nnUNet_raw)

    base = f"Dataset{iter+100}_{name}_{iter}"
    folder = base
    
    #check if folder exists and delete it if it does
    if os.path.exists(os.path.join(nnUNet_raw, folder)):
        #remove folder even if it is not empty
        os.system("rm -rf "+os.path.join(nnUNet_raw, folder))

    basedir = nnUNet_raw
    if folder != "" and not os.path.exists(os.path.join(basedir, folder)):
        os.makedirs(os.path.join(basedir, folder))

    if not os.path.exists(os.path.join(basedir, folder, 'imagesTr')):
        os.makedirs(os.path.join(basedir, folder, 'imagesTr'))
    if not os.path.exists(os.path.join(basedir, folder, 'labelsTr')):
        os.makedirs(os.path.join(basedir, folder, 'labelsTr'))

    nfiles = 0
    traindf = dl.get_df()
    traindf = traindf.reset_index(drop=True)

    print(traindf.head())

    for i in range(len(traindf)):
        sigII = traindf["xs"][i][0,0,0,:]
        sigV1 = traindf["xs"][i][1,0,0,:]
        sigV6 = traindf["xs"][i][2,0,0,:]

        #set nan to 0
        sigII = np.nan_to_num(sigII)
        sigV1 = np.nan_to_num(sigV1)
        sigV6 = np.nan_to_num(sigV6)

        if np.all(sigII==0) or np.any(np.isnan(sigII)):
            print("Zero signal for record", traindf["records"][i])
            continue
        
        signal = np.stack([sigII, sigV1, sigV6])

        segmentation = to_one_hot(traindf["ys"][i])

        label = [traindf["labels"][i],0]

        nfiles += 1
        recname = str(traindf["records"][i]).replace("/", "_") + "_" + str(traindf["onsets"][i])

        for channel in range(signal.shape[0]):
            channelid = str(channel).zfill(4)
            np.save(os.path.join(basedir, folder, 'imagesTr', f'case_{recname}_{channelid}.npy'), signal[channel,:])

        np.save(os.path.join(basedir, folder, 'labelsTr', f'case_{recname}.npy'), segmentation)
        np.save(os.path.join(basedir, folder, 'labelsTr', f'case_{recname}_cls.npy'), label)

    jsn = {
            "channel_names": {
                "0": "LeadII",
                "1": "LeadV1",
                "2": "LeadV6"
            },
            "labels": {
                "background": 0,
                "p_wave": 1,
                "qrs_wave": 2,
                "t_wave": 3,
                "noise": 4,
                "qrs_abnormal": 5
            },
            "use_for_validation": {
                "background": False, 
                "p_wave": True, 
                "qrs_wave": True, 
                "t_wave": True, 
                "noise": True,
                "qrs_abnormal": True
            },
            "classification_head": True,
            "binary": True,
            "numTraining": nfiles,
            "file_ending": ".npy"
        }

    with open(os.path.join(basedir, folder, 'dataset.json'), 'w') as f:
       json.dump(jsn, f)

def get_nnunet_training_loader():

    #initialize database connection
    client = MongoClient('mongodb://localhost:27018/', username='root', password='ALADIN2025')
    db = client['orion']
    aldb = db['al']
    id = None
    name = "ALADIN"

    #get experiment
    if name is None:
        #get last experiment if no name is provided
        row = aldb.find_one(sort=[("time", -1)])
        if row is not None:
            id = row["_id"]
            name = row["name"]
            iter = len(row["iterations"])
        else:
            #no previous experiment found
            raise Exception("No previous experiment found, and no name provided")
    else:
        #get last experiment with name
        row = aldb.find_one({"name":{"$eq":name}}, sort=[("time", -1)])
        if row is not None:
            id = row["_id"]
            iter = len(row["iterations"])
        else:
            #create new experiment with name
            id = aldb.insert_one({"name":name, "iterations":[]})
            iter = 0

    print("Using experiment with id", id, "and name", name, "and iter", iter)

    dl = MongoDBDatasetGenerator(cachenames="nnunet_train", folder="", split=False, type="train", expid=id, batch_size=32)

    return dl, iter

    

def get_classification_branches_loader():
    dl = MongoDBDatasetGenerator(cachenames="classification_branches", folder="", split=False, type="all")
    return dl, 101 #to create a distance between both

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', type=str, default="nnunet_train")
    args = parser.parse_args()
    datatype = args.datatype

    if datatype == "nnunet_train":
        dl, iter = get_nnunet_training_loader()
        create_training_set_files(dl, "nnunet", iter)

    elif datatype == "classification_branches":
        dl, iter = get_classification_branches_loader()
        create_training_set_files(dl, "all", iter)