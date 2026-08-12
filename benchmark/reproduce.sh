#!/usr/bin/env bash
set -euo pipefail

read -rsp "Enter your reviewer Hugging Face token: " HF_TOKEN
echo

echo HF_TOKEN

mkdir -p results
mkdir -p models

docker build -f benchmark/Dockerfile -t aladin-benchmark .
docker run --rm \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$PWD/data:/data" \
    -v "$PWD/results:/results" \
    -v "$PWD/models:/models" \
    aladin-benchmark 

 
