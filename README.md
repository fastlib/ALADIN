![ALADIN](data/resources/aladin_logo_2.png)

[![Cross-platform build](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml/badge.svg)](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml)
 
ALADIN is a neuro-symbolic AI model that preprocesses, segments, and diagnoses single- and multi-lead ECG signals. It has been validated extensively on three diverse patient cohorts with a combined size of 13,000 patients. ALADIN can handle any ECG recording from clinical MUSE recordings to handheld KardiaMobile measurements ranging from 6 seconds to 24 hours. 
 
## Changelog:

💡Version 1.1.0 16/03/2026:
- Added ability to handle 1, 3, and 12-lead ECG
- Added median beat extraction and corresponding beat median segmentations based on segmentation and QRS clusters
 
## System requirements:
- Linux Ubuntu >= 20.04 or MacOSX 13.6.x or Microsoft Windows >=10
- Python >3.10 
- 20Gb free diskspace
- Modern GPU with >12Gb VRAM is recommended for training and inference
 
Tested on Linux Ubuntu 20.04, MacOSX 13.6.x, and Microsoft Windows 10 and 11 Home. CPU-only support is available, but a modern GPU is required for training and will speed up inference substantially. Tested on NVidia GeForce 3090. 
 
## Installation

When on MacOS, install OpenMP via brew to enable native multithreading:
```bash
brew install libomp
```

Next, clone and install ALADIN:
```bash
git clone https://github.com/fastlib/ALADIN.git
cd ALADIN
python -m venv VENV #create virtual environment
source VENV/bin/activate #activate environment
pip install scikit-build-core pybind11 ninja cmake #build tooling, see macOS note below
pip install torch #must be installed before building aladin's C++ extension
pip install ./nnUNet
pip install ./aladin --no-build-isolation
mkdir models
```

### macOS build notes

ALADIN's C++ extension (`aladin._main`) uses OpenMP for native multithreading.
On macOS, Apple Clang doesn't ship OpenMP itself, so the extension is built
using Homebrew's `libomp` for headers — but it links against **PyTorch's own
bundled `libomp.dylib`** at runtime rather than Homebrew's copy. This is
necessary because loading two independent OpenMP runtimes into the same
Python process (one from Homebrew via this extension, one bundled with
PyTorch) causes macOS to crash under concurrent load, with segfaults inside
`libomp`'s `__kmp_suspend_64`. Sharing a single runtime with PyTorch avoids
that.

This has two consequences for how you install/rebuild ALADIN on macOS:

- **`torch` must already be installed before building `aladin`.** The build
  step needs to `import torch` to locate its bundled `libomp.dylib`. If torch
  isn't importable at build time, the build falls back to Homebrew's
  `libomp` with a `WARNING`, which reintroduces the crash risk above.
- **Always pass `--no-build-isolation` when installing/rebuilding `aladin`.**
  By default, `pip install` builds packages inside a temporary, isolated
  environment that can't see `torch` (or anything else) installed in your
  venv, even though it reuses the same Python interpreter — which defeats
  the point above. `--no-build-isolation` builds directly against your venv
  instead, so make sure `scikit-build-core`, `pybind11`, `ninja`, `cmake`,
  and `torch` are installed in the venv first (as in the install steps
  above). This applies every time you rebuild the extension, e.g. after
  `pip install -e ./aladin` for development.

Even with the extension linked correctly, `scikit-learn` (a direct ALADIN
dependency) ships its own separate, vendored `libomp.dylib` that's outside
this project's control. As a safety net against the same class of crash, we
recommend always running ALADIN on macOS with:
```bash
export OMP_NUM_THREADS=1
export nnUNet_def_n_proc=1
```
This pins OpenMP/nnU-Net internals to single-threaded execution, avoiding
concurrent-thread crashes at some cost to CPU inference speed.

ALADIN will automatically download the model weights from the private Hugging Face repo `AUMC/ALADIN` into
`~/.cache/aladin/models` the first time they're needed. However, this requires Hugging Face account access 
to the `AUMC/ALADIN` repo, and being authenticated locally via `huggingface-cli login` or 
the `HF_TOKEN` environment variable.

## Example usage
```python
import numpy as np
from aladin import ALADIN
from aladin.core import Record

#ecg should be a dictionary with the keys being lead names
fs = 250 #hz
ecg = {"II": np.random.rand(fs*10)}

#create record object
record = Record(ecg, fs)

#create ALADIN object
aladin = ALADIN(modelpaths=["ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding"])

#perform segmentation
aladin.segment(record)

#perform diagnosis
aladin.analyse(record)

#perform median beat extraction
aladin.extract_median_beat(record)
median_beat = record.median_beat.ecg
```

## Example code
```bash
python demo.py --case=[recording]
```
Where `[recording]` can be one of `(STANFORD1, STANFORD2, A01986, A08391)`.


 
### Expected output
- 'aladin_result.png', a figure showing the delineation and diagnosis with on- and offsets
- 'aladin_explanation.txt', a textfile containing a detailed explanation behind the diagnosis
 
\newpage
## Reproduce benchmark using model checkpoints
 
NOTE: Use Python 3.10.x as Jimenez-Perez et al. does not work for >3.10.x
 
### Prepare data
1. Download VALIDATION.zip from [FigShare](https://figshare.com/s/a9cb763a786e0b259cb8) (12Mb) and extract to `/data` such that you have `/data/VALIDATION/...`
1. Download STANFORD.zip from [FigShare](https://figshare.com/s/b470ddc0845aa1a0ea8f) (6Mb) and extract to `/data` such that you have `/data/STANFORD/...`
2. Download the CinC dataset using `wget`
```bash
cd data/CINC
./download.sh #change extension to .bat for Windows
cd ../..
```
or by downloading the zip from [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) (1.4Gb) and extract it to `/data/CINC` such that you have `/data/CINC/training/...`.
 
### Prepare competitors
3. Clone Jimenez-Perez github repository 
```bash
git clone https://github.com/guillermo-jimenez/DelineatorSwitchAndCompose.git
```
4. Setup Jimenez-Perez dependencies 
```bash
cd DelineatorSwitchAndCompose
git clone https://github.com/guillermo-jimenez/sak.git
cd ..
```
5. Install Jimenez-Perez dependencies 
```bash
cd DelineatorSwitchAndCompose/sak
pip install . --no-deps 
cd ../..
```
6. Download trained model from [FigShare](https://figshare.com/s/efceb02a728bb17bd61d) (251Mb) and unzip in `/DelineatorSwitchAndCompose` such that you have `/DelineatorSwitchAndCompose/TrainedModels/...`
7. Install additional dependencies 
```bash
pip install -r requirements.txt
```
8. Download pretrained weights of the diagnostic models from [FigShare](https://figshare.com/s/3906debcd0b1bfbcedb2) (580Mb) and unzip in `/benchmark/weights`
 
### Prepare benchmark
9. Make results folder 
```bash
mkdir results
```
10. Set environment variables 
```bash
export benchmark_data=[absolute path to /data folder]
export benchmark_results=[absolute path to /results folder] #Linux and MacOSX
 
$Env:benchmark_data = "[absolute path to /data folder]"
$Env:benchmark_results = "[absolute path to /results folder]" #Windows
```
 
### Run benchmark
11. Run delineation benchmark (~ 20 min)
```bash
./benchmark_delineation.sh #change extension to .bat for Windows
```
12. Create latex table of delineation performance on validation set 
```bash
python paper/generate_results_tables.py
```
13. Run diagnostic benchmark on Stanford (~ 1.5 hour)
```bash
./benchmark_diagnosis_STANFORD.sh #change extension to .bat for Windows
```
14. Create boxplot figures of performance on Stanford
```bash
python paper/boxplot-stanford.py
```
15. Run diagnostic benchmark on CinC competition datset (~ 30 hours)
```bash
./benchmark_diagnosis_CINC.sh #change extension to .bat for Windows
```
16. Create boxplot figures of performance on CinC competition set 
```bash
python paper/boxplot-cinc.py
```
 
### Expected results
- A delineation performance table in raw latex code corresponding to Supplementary Table 1.
- Two boxplot figures showing the benchmark performance of ALADIN, ECGFounder, ResNet, and the average cardiologist on the iRhythm and AliveCor data sets. The figures are located inside `/paper/images`
 
\newpage
## Resources:
- Main code ZIP file
- Pretrained weights of ALADIN ([FigShare](https://figshare.com/s/550ffac7d873ea677824), 8.1Gb)
- Pretrained weights of Resnet and finetuned ECGFounder ([FigShare](https://figshare.com/s/3906debcd0b1bfbcedb2), 580Mb)
- Jimenez-Perez model weights ([FigShare](https://figshare.com/s/efceb02a728bb17bd61d), 251Mb)
- Stanford dataset ([FigShare](https://figshare.com/s/b470ddc0845aa1a0ea8f), 6Mb)
- Validation dataset ([FigShare](https://figshare.com/s/a9cb763a786e0b259cb8), 12Mb)
- MongoDB ([FigShare](https://figshare.com/s/241b86301e9a4cffcf64), 5.3Gb)