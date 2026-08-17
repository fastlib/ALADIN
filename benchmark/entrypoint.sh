#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Please set the HF_TOKEN environment variable (docker run -e HF_TOKEN=...)" >&2
    exit 1
fi

HF_BASE_URL="https://huggingface.co/datasets/fastlib/ALADIN-benchmarks/resolve/main"

download_and_unzip() {
    local filename="$1"
    local destdir="$2"
    local tmpzip

    tmpzip="$(mktemp -t "${filename}.XXXXXX")"
    curl -fL -H "Authorization: Bearer ${HF_TOKEN}" -o "${tmpzip}" "${HF_BASE_URL}/${filename}"
    mkdir -p "${destdir}"
    unzip -o "${tmpzip}" -d "${destdir}"
    rm -f "${tmpzip}"
}

download() {
    local filename="$1"
    local destdir="$2"

    mkdir -p "${destdir}"
    curl -fL -H "Authorization: Bearer ${HF_TOKEN}" -o "${destdir}/${filename}" "${HF_BASE_URL}/${filename}"
}

download_and_unzip "VALIDATION.zip" "/data/VALIDATION"

download_and_unzip "STANFORD.zip" "/data"

download "stanford2.pkl" "/data/STANFORD"

download_and_unzip "TrainedModels.zip" "/app/DelineatorSwitchAndCompose/TrainedModels"

download_and_unzip "competitor_diagnostic_models.zip" "/data/benchmark" 

download_and_unzip "RDB.zip" "/data/RDB"

./data/CINC/download.sh

#9. Make results folder
mkdir -p /results

# #10. Set environment variables
export benchmark_data=/data
export benchmark_results=/results
export HF_HOME=/models

#11. Run delineation benchmark (~ 20 min)
./benchmark_delineation.sh

#12. Create latex table of delineation performance on validation set
python paper/generate_results_tables.py --dataset VAL
python paper/generate_results_tables.py --dataset RDB

#13. Run diagnostic benchmark on Stanford (~ 1.5 hour)
./benchmark_diagnosis_STANFORD.sh

#14. Create boxplot figures of performance on Stanford
python paper/boxplot-stanford.py

#15. Run diagnostic benchmark on CinC competition dataset (~ 3 hours)
./benchmark_diagnosis_CINC.sh

#16. Create boxplot figures of performance on CinC competition set
python paper/boxplot-cinc.py
