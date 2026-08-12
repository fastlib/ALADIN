FROM python:3.10-slim


WORKDIR /app

COPY aladin /app/aladin
COPY nnUNet /app/nnUNet 
COPY pyproject.toml /app
COPY CMakeLists.txt /app
COPY README.md /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install ".[test,benchmark]"

COPY benchmark /app/benchmark
COPY demo.py /app
COPY utils /app/utils 
COPY paper /app/paper 
COPY data/demo /app/data/demo
COPY data/nnUNet_plans /app/data/nnUNet_plans 
COPY data/STANFORD /app/data/STANFORD 
COPY data/CINC /app/data/CINC 
COPY benchmark_delineation.py /app
COPY benchmark_delineation.sh /app
COPY benchmark_diagnosis.py /app
COPY benchmark_diagnosis_STANFORD.sh /app
COPY benchmark_diagnosis_CINC.sh /app
COPY requirements.txt /app

#RUN python -m pytest aladin/tests

#### Prepare competitors
#3. Download the CinC dataset (already provided via `data/CINC` build context; use `data/CINC/download.sh`
#   for the full PhysioNet training set, or download it manually from
#   [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) (1.4Gb) and extract it to
#   `/data/CINC` such that you have `/data/CINC/training/...`.
#4/5. Clone Jimenez-Perez github repository and its `sak` dependency
#6. Install Jimenez-Perez dependencies
RUN git clone https://github.com/guillermo-jimenez/DelineatorSwitchAndCompose.git && \
    git clone https://github.com/guillermo-jimenez/sak.git DelineatorSwitchAndCompose/sak && \
    pip install ./DelineatorSwitchAndCompose/sak --no-deps

RUN chmod +x /app/benchmark/entrypoint.sh /app/benchmark_delineation.sh \
    /app/benchmark_diagnosis_STANFORD.sh /app/benchmark_diagnosis_CINC.sh

#### Prepare and run benchmark (steps 1, 2, 6, 8-16) — see benchmark/entrypoint.sh
#Downloads VALIDATION.zip, STANFORD.zip and TrainedModels.zip from the Hugging Face repo
#fastlib/ALADIN-benchmarks using the HF_TOKEN environment variable (passed at `docker run`
#time, e.g. via reproduce.sh), unzips them per the comments above, then runs the full
#benchmark suite (steps 8 onward).
ENTRYPOINT ["/app/benchmark/entrypoint.sh"]
