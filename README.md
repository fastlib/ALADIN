# ALADIN documentation
 
ALADIN is a neuro-symbolic AI model that preprocesses, segments, and diagnoses single- and multi-lead ECG signals. It has been validated extensively on three diverse patient cohorts with a combined size of 13,000 patients. ALADIN can handle any ECG recording from clinical MUSE recordings to handheld KardiaMobile measurements ranging from 6 seconds to 24 hours. 
 
## Changelog:

💡Version 1.1.0 16/03/2026:
- Added ability to handle 12-lead ECG
- Added median beat extraction based on segmentation and QRS clusters
 
## System requirements:
- Linux Ubuntu >= 20.04 or MacOSX 13.6.x or Microsoft Windows >=10
- Python 3.10 
- >20Gb free diskspace
- Modern GPU with >12Gb VRAM is recommended for training
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) for training
 
Tested on Linux Ubuntu 20.04, MacOSX 13.6.x, and Microsoft Windows 10 and 11 Home. CPU-only support is available, but a modern GPU is required for training and will speed up inference substantially. Tested on NVidia GeForce 3090. ALADIN works on newer versions of Python; but, the algorithm of Jimenez-Perez et al. crashes if Python >3.10.
 
Windows users are advised to use Powershell. Run in administrator mode and see [this](https://stackoverflow.com/a/60210792) StackOverflow question when you experience difficulty activating the virtual python environment (step 2). Also, when using Windows, please change the shell script extensions from `.sh` to `.bat`.
 
## Installation guide (local deployment)
 
Out-of-the-box ALADIN installation evaluated on a few demo recordings. We advise to use a clean virtual environment to prevent issues with library versions.
 
### Preparations (~10 min)
1. Create local python environment 
```{.bash}
python -m venv ALADIN
```
2. Activate python environment 
```{.bash}
source ALADIN/bin/activate #Linux and MacOSX
.\ALADIN\Scripts\activate #Windows
```
3. Install ALADIN 
```{.bash}
pip install ./aladin
```
4. Install nnUNet 
```{.bash}
pip install ./nnUNet
```
5. Create models folder 
```{.bash}
mkdir models
```
6. Download model weights from [FigShare](https://figshare.com/s/550ffac7d873ea677824) (8.1Gb) and unzip in `/models` folder such that you have `/models/Dataset301_all_0/ClassificationTrainer__nnUNetWithClassificationPlans__1d_decoding/...`
7. Set environment variable 
```{.bash}
export aladin_models=[absolute path to root]/models #Linux and MacOSX
$Env:aladin_models="[absolute path to root]\models" #Windows
```
 
### Run demo (~5 min)
8. Run 
```{.bash}
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
```{.bash}
cd data/CINC
./download.sh #change extension to .bat for Windows
cd ../..
```
or by downloading the zip from [PhysioNet](https://physionet.org/content/challenge-2017/1.0.0/) (1.4Gb) and extract it to `/data/CINC` such that you have `/data/CINC/training/...`.
 
### Prepare competitors
3. Clone Jimenez-Perez github repository 
```{.bash}
git clone https://github.com/guillermo-jimenez/DelineatorSwitchAndCompose.git
```
4. Setup Jimenez-Perez dependencies 
```{.bash}
cd DelineatorSwitchAndCompose
git clone https://github.com/guillermo-jimenez/sak.git
cd ..
```
5. Install Jimenez-Perez dependencies 
```{.bash}
cd DelineatorSwitchAndCompose/sak
pip install . --no-deps 
cd ../..
```
6. Download trained model from [FigShare](https://figshare.com/s/efceb02a728bb17bd61d) (251Mb) and unzip in `/DelineatorSwitchAndCompose` such that you have `/DelineatorSwitchAndCompose/TrainedModels/...`
7. Install additional dependencies 
```{.bash}
pip install -r requirements.txt
```
8. Download pretrained weights of the diagnostic models from [FigShare](https://figshare.com/s/3906debcd0b1bfbcedb2) (580Mb) and unzip in `/benchmark/weights`
 
### Prepare benchmark
9. Make results folder 
```{.bash}
mkdir results
```
10. Set environment variables 
```{.bash}
export benchmark_data=[absolute path to /data folder]
export benchmark_results=[absolute path to /results folder] #Linux and MacOSX
 
$Env:benchmark_data = "[absolute path to /data folder]"
$Env:benchmark_results = "[absolute path to /results folder]" #Windows
```
 
### Run benchmark
11. Run delineation benchmark (~ 20 min)
```{.bash}
./benchmark_delineation.sh #change extension to .bat for Windows
```
12. Create latex table of delineation performance on validation set 
```{.bash}
python paper/generate_results_tables.py
```
13. Run diagnostic benchmark on Stanford (~ 1.5 hour)
```{.bash}
./benchmark_diagnosis_STANFORD.sh #change extension to .bat for Windows
```
14. Create boxplot figures of performance on Stanford
```{.bash}
python paper/boxplot-stanford.py
```
15. Run diagnostic benchmark on CinC competition datset (~ 30 hours)
```{.bash}
./benchmark_diagnosis_CINC.sh #change extension to .bat for Windows
```
16. Create boxplot figures of performance on CinC competition set 
```{.bash}
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