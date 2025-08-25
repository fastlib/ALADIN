#!/usr/bin/env bash
set -ex
 
# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
python3.10 --version
pip3.10 --version
 
cd src
pip3.10 install .
cd ..
cd nnUNet
pip3.10 install .
cd ..
 
unzip -o /data/ALADIN-weights.zip -d "/data/models"
unzip -o /data/STANFORD.zip -d "/data"
unzip -o /data/RDB.zip -d "/data"
mkdir /data/VALIDATION
unzip -o /data/VALIDATION.zip -d "/data/VALIDATION"
 
#Set environment variables
export aladin_models=/data/models
export benchmark_results=/results
export benchmark_data=/data
 
#Run demo
python3.10 demo.py --case STANFORD1
python3.10 demo.py --case STANFORD2
python3.10 demo.py --case A01986
python3.10 demo.py --case A08391
 
#Install SoTA models for delineation
git clone https://github.com/guillermo-jimenez/DelineatorSwitchAndCompose.git
cd DelineatorSwitchAndCompose
git clone https://github.com/guillermo-jimenez/sak.git
cd sak
pip3.10 install . --no-deps
cd ..
cd ..
pip3.10 install -r requirements.txt
 
mkdir DelineatorSwitchAndCompose/TrainedModels
unzip -o /data/TrainedModels.zip -d "/code/DelineatorSwitchAndCompose/TrainedModels"
 
#Benchmark delineation
chmod 777 benchmark_delineation.sh
./benchmark_delineation.sh
 
#Benchmark diagnosis on iRhythm dataset
chmod 777 benchmark_diagnosis_STANFORD.sh
./benchmark_diagnosis_STANFORD.sh
 
python3.10 paper/boxplot-stanford.py
 
#Benchmark diagnosis on AliveCor
cd /data/CINC
chmod 777 download.sh
./download.sh
cd ..
cd ..
 
chmod 777 benchmark_diagnosis_CINC.sh
./benchmark_diagnosis_CINC.sh
 
python3.10 paper/boxplot-cinc.py