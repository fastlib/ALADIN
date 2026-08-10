![ALADIN](data/resources/aladin_logo_2.png)

[![Cross-platform build](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml/badge.svg)](https://github.com/fastlib/ALADIN/actions/workflows/build.yaml)
 
ALADIN is a neuro-symbolic AI model that preprocesses, segments, and diagnoses single- and multi-lead ECG signals. It has been validated extensively on three diverse patient cohorts with a combined size of 13,000 patients. ALADIN can handle any ECG recording from clinical MUSE recordings to handheld KardiaMobile measurements ranging from 6 seconds to 24 hours. 

```
pip install aladin-ecg
```
 
## Changelog:

💡Version 1.1.2 08/08/2026:
- Added PyPi support

💡Version 1.1.1 15/07/2026:
- Added Github Actions to verify cross-platform compatibality
- Added unit tests
- Downloads missing model weights automatically from Hugging Face (access required)

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

When on MacOS, a homebrew install of OpenMP is required:
```bash
brew install libomp
```

Next, clone and install ALADIN:
```bash
git clone https://github.com/fastlib/ALADIN.git
cd ALADIN
python -m venv VENV #create virtual environment
source VENV/bin/activate #activate environment

pip install scikit-build-core pybind11 ninja cmake #build tooling, must be installed before building aladin
pip install torch #must be installed before building aladin's C++ extension
pip install ./nnUNet
pip install ./aladin --no-build-isolation
mkdir models
```

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
#modelpaths="auto" picks between the pretrained 1-lead and 3-lead models based on which leads
#the record has available (lead II must always be present; the 3-lead model is only used if
#leads II, V1 and V6 are all present, otherwise ALADIN falls back to the 1-lead model)
aladin = ALADIN(modelpaths="auto")

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

## Reproduce benchmark

See [Benchmark.md](benchmark.md) for details on benchmark reproduction.

## macOS build notes

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

ALADIN will automatically download the model weights from the public Hugging Face repo `fastlib/ALADIN`
into huggingface_hub's default cache (`~/.cache/huggingface/hub`, or wherever `HF_HOME`/`HF_HUB_CACHE`
point) the first time they're needed. No Hugging Face account or login is required -- the download is
anonymous. Alternatively, set the `aladin_models` environment variable to a local folder that already
contains the weights, to skip the Hugging Face download entirely.

