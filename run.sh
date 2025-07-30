#!/usr/bin/env bash
set -ex

get_installed_cuda_version() {
    # Check if 'nvcc' is available and return its version
    if command -v nvcc &> /dev/null
    then
        # Get the installed CUDA version using nvcc
        installed_version=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
        echo "$installed_version"
    else
        echo "CUDA not installed"
    fi
}


# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
python --version
pip --version
# installed_version=$(get_installed_cuda_version)

# if [ "$installed_version" == "12.8" ]; then
#     pip install torch --index-url https://download.pytorch.org/whl/cu128
# elif [ "$installed_version" == "12.6" ]; then
#     echo "CUDA 12.6 detected, installing normal pytorch version."
#     pip install torch
# fi

#python -m venv ALADIN
#source ALADIN/bin/activate

pip install ./aladin
pip install ./nnUNet

pip install boto3
#python data/ICENTIA/download.py

#check if tar.gz file exists
if [ ! -f "google-cloud-cli-linux-x86_64.tar.gz" ]; then
    echo "google-cloud-cli-linux-x86_64.tar.gz not found, downloading..."
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh --bash-completion=false --path-update=false --usage-reporting=false
    source ./google-cloud-sdk/path.bash.inc
    source ./google-cloud-sdk/completion.bash.inc
else
    source ./google-cloud-sdk/path.bash.inc
    source ./google-cloud-sdk/completion.bash.inc
    echo "google-cloud-cli-linux-x86_64.tar.gz already exists, skipping download."
fi

export GOOGLE_APPLICATION_CREDENTIALS=./data/aladin-466917-e056430d6165.json
gcloud auth activate-service-account --key-file ./data/aladin-466917-e056430d6165.json

mkdir -p ./data/models
gsutil -m cp -rn gs://arts-aladin/Dataset200_all_101 ./data/models

# mkdir -p ./data/CINC/training
# aws s3 sync --no-sign-request s3://physionet-open/challenge-2017/1.0.0/training ./data/CINC/training

#Set environment variables
export aladin_models=./data/models
export benchmark_results=./results
export benchmark_data=./data

#Run demo
# python3.10 demo.py --case STANFORD1
# python3.10 demo.py --case STANFORD2
# python3.10 demo.py --case A01986
# python3.10 demo.py --case A08391

#Run STANFORD benchmark
#time python benchmark_diagnosis.py --method ALADIN --dataset STANFORD --overwrite

#Run ICENTIA benchmark
time python benchmark_diagnosis.py --method ALADIN --dataset ICENTIA --overwrite

# #Run CINC benchmark
# time python3.10 benchmark_diagnosis.py --method ALADIN --dataset CINC --overwrite