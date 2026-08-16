![ALADIN](data/resources/aladin_logo_2.png)

[![Cross-platform build](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml/badge.svg)](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml)
[![PyPI](https://img.shields.io/pypi/v/aladin-ecg.svg)](https://pypi.org/project/aladin-ecg/)
[![Python versions](https://img.shields.io/pypi/pyversions/aladin-ecg.svg)](https://pypi.org/project/aladin-ecg/)
[![License](https://img.shields.io/pypi/l/aladin-ecg.svg)](https://pypi.org/project/aladin-ecg/)
[![Downloads](https://img.shields.io/pypi/dm/aladin-ecg.svg)](https://pypi.org/project/aladin-ecg/)

ALADIN is a neuro-symbolic AI model that preprocesses, segments, and diagnoses single- and multi-lead ECG signals. It has been validated extensively on three diverse patient cohorts with a combined size of 13,000 patients. ALADIN can handle any ECG recording from clinical MUSE recordings to handheld KardiaMobile measurements ranging from 6 seconds to 24 hours.

## Installation

```bash
pip install aladin-ecg
```

Model weights are downloaded automatically and anonymously from the [Hugging Face repo `fastlib/ALADIN`](https://huggingface.co/fastlib/ALADIN) the first time they're needed, and cached in `~/.cache/huggingface/hub` (or `HF_HOME`/`HF_HUB_CACHE` if set). To skip the download, point the `aladin_models` environment variable at a local folder that already contains the weights.

### System requirements

- Linux Ubuntu ≥20.04, macOS ≥13.6, or Windows ≥10
- Python 3.11–3.14
- 20GB free disk space
- A modern GPU with ≥12GB VRAM is recommended for training and inference; CPU-only inference is supported but slower

### Installing from source

```bash
git clone https://github.com/fastlib/ALADIN.git
cd ALADIN
python -m venv VENV
source VENV/bin/activate

pip install scikit-build-core pybind11 ninja cmake  # build tooling
pip install torch                                    # required before building the C++ extension
pip install ./nnUNet
pip install ./aladin --no-build-isolation
```

## Usage

```python
import numpy as np
from aladin import ALADIN
from aladin.core import Record

# ecg is a dict keyed by lead name
fs = 250  # Hz
ecg = {"II": np.random.rand(fs * 10)}
record = Record(ecg, fs)

# modelpaths="auto" picks the pretrained 1-lead or 3-lead model based on which
# leads are available (lead II is required; the 3-lead model additionally needs
# V1 and V6, otherwise ALADIN falls back to the 1-lead model)
aladin = ALADIN(modelpaths="auto")

aladin.segment(record)              # segmentation
aladin.analyse(record)              # diagnosis
aladin.extract_median_beat(record)  # median beat extraction

median_beat = record.median_beat.ecg
```

A runnable demo is available via:

```bash
python demo.py --case=[recording]
```

where `[recording]` is one of `STANFORD1`, `STANFORD2`, `A01986`, `A08391`.

## Benchmarks

See [benchmark.md](benchmark.md) for details on reproducing the published benchmarks.

## Changelog

**1.1.2** (2026-08-08) — Added PyPI support

**1.1.1** (2026-07-15) — Cross-platform GitHub Actions, unit tests, automatic model weight downloads from Hugging Face

**1.1.0** (2026-03-16) — Support for 1-, 3-, and 12-lead ECG; median beat extraction and beat-median segmentations
