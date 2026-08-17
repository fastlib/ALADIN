# Model Card: ALADIN

## Model summary

ALADIN is a neuro-symbolic AI model for ECG preprocessing, delineation (segmentation), and diagnosis. It combines a deep-learning segmentation backbone (nnU-Net based, C++/PyTorch hybrid backend) with a symbolic logic engine that reasons over the segmented waveform to produce rhythm diagnoses and explanations. It supports 1-, 3-, and 12-lead ECG input and has been validated on ECGs ranging from 6-second clinical recordings to 24-hour ambulatory sessions.

- **Developer:** Lukas P.A. Arts
- **Repository:** [github.com/fastlib/ALADIN](https://github.com/fastlib/ALADIN)
- **Model weights:** [huggingface.co/fastlib/ALADIN](https://huggingface.co/fastlib/ALADIN)
- **License:** Apache 2.0
- **PyPI package:** [`aladin-ecg`](https://pypi.org/project/aladin-ecg/)
- **Citation:** No published paper or preprint yet (manuscript in preparation)

## Model architecture

ALADIN is a pipeline of four stages (see `aladin/src/aladin/__init__.py`):

1. **Preprocessing** — C++ backend (`aladin/src/*.cpp`) filtering and signal conditioning.
2. **Segmentation** (`UNetSegmenter`, `aladin/src/aladin/backend`) — an nnU-Net-based model (built on the vendored `nnUNet`/`nnunetv2` and `dynamic-network-architectures`) that delineates P, QRS, and T waves, and flags abnormal QRS, atrial fibrillation, and noise regions. Two pretrained checkpoints are shipped:
   - **1-lead model** — used when only lead II is available.
   - **3-lead model** — used when leads II, V1, and V6 are all available (falls back to the 1-lead model otherwise). Rhythm/logic reasoning always uses lead II regardless of which segmentation model is selected (`aladin/src/aladin/configuration.py`).
3. **Self-reflection** (`Reflection`, `aladin/src/aladin/selfreflection`) — beat clustering and morphology-based correction of the raw segmentation output.
4. **Symbolic diagnosis** (`LogicEngine`, `aladin/src/aladin/logicengine/logic.py`) — a rule-based logic engine that reasons over the corrected delineation to produce rhythm diagnoses with human-readable explanations (see `aladin_explanation.txt` for example outputs).

Diagnoses currently supported by the logic engine: normal sinus rhythm (NSR), atrial fibrillation (AFIB), first/second-degree AV block including Wenckebach (AVB), complete heart block (CHB), supraventricular tachycardia (SVT), ventricular tachycardia (VT), idioventricular rhythm (IVR), ectopic atrial rhythm (EAR), junctional rhythm, bigeminy/trigeminy, premature atrial/ventricular contractions (PAC/PVC), and noise detection. Each is individually togglable via the `customarrhythmia` argument.

## Intended use

- **Primary use case:** Research use for automated ECG delineation (P/QRS/T segmentation, quality assessment, beat classification, beat clustering, feature extraction, median beat extraction, etc) and rhythm-level diagnosis support across clinical (MUSE), ambulatory (ZioPatch), and handheld (KardiaMobile) ECG recordings.
- **Primary users:** Researchers and developers working on ECG signal processing, arrhythmia detection, and cardiology AI tooling.
- **Out of scope:** ALADIN is a research tool and **has not been cleared or approved as a medical device** (e.g. no FDA/CE clearance). It is not intended for standalone clinical diagnosis or treatment decisions, and outputs should not be used as the sole basis for patient care without review by a qualified clinician.

## Training data

The segmentation model was trained using an active-learning selection process across nine public ECG datasets, with separate internal and external hold-out sets used for delineation validation, and three additional cohorts used for diagnosis validation. 

### Training set (𝒰)

| Dataset | Records | Patients | Duration | Frequency (Hz) | Leads | Type | Selected (records) |
|---|---|---|---|---|---|---|---|
| BUT-PDB | 50 | 50 | 2 min | 360 | 2 | Ambulatory | 61 |
| CHAPMAN | 10,247 | 10,247 | 10 sec | 500 | 12 | Clinical | 392 |
| CPSC2018 | 6,877 | 6,877 | 6–60 sec | 500 | 12 | Clinical | 398 |
| CPSC2018 extra | 3,453 | 3,453 | 10 sec | 500 | 12 | Clinical | 341 |
| GEORGIA | 10,344 | 10,344 | 10 sec | 500 | 12 | Clinical | 371 |
| INCART | 72 | 32 | 30 min | 257 | 2 | Clinical | 90 |
| LANCET | 828 | 828 | 10 sec | 500 | 12 | Both | 622 |
| NINGBO | 34,905 | 34,905 | 10 sec | 500 | 12 | Clinical | 150 |
| PTB-XL | 22,353 | 18,869 | 6–60 sec | 500 | 12 | Clinical | 637 |
| **Subtotal** | **89,129** | **85,605** | | | | | **3,962** |

## Validation data 

### Internal delineation validation (𝒱<sub>del,intern</sub>)

Held-out subsets (no overlap with the training set) from CPSC2018 (210 records), CPSC2018 extra (186), ICENTIA (81, 24h ambulatory, 1-lead, 250 Hz), LANCET (72), and NINGBO (248).

### External delineation validation (𝒱<sub>del,extern</sub>)

| Dataset | Records | Patients | Duration | Frequency | Leads | Type |
|---|---|---|---|---|---|---|
| RDB | 2,399 | 2,399 | 10 sec | 500 | 12 | Clinical |

Combined internal + external delineation validation subtotal: 3,196 patients.

### Internal diagnosis validation 
Not applicable (no training was involved, all validation is external)

### External diagnosis validation (𝒱<sub>diag</sub>)

| Dataset | Records | Patients | Duration | Frequency | Leads | Type |
|---|---|---|---|---|---|---|
| iRhythm Zio™ Monitor | 328 | 328 | 30 sec | 200 | 1 | Ambulatory |
| AliveCor KardiaMobile (CinC 2017 Challenge) | 8,528 | 8,528 | 6–60 sec | 300 | 1 | Ambulatory |
| CardioSTAT Long-term ECG (ICENTIA, cleaned) | 157,508 | 4,924 | 70 min | 250 | 1 | Ambulatory |

Diagnosis validation total: **13,780 patients**.

## Reproduction

Benchmark reproduction instructions and scripts are in [`benchmark.md`](benchmark.md) and `paper/`. Please request a Huggingface token from the authors to run the benchmark reproduction as the reproduction
uses datasets that are not openly available anymore. Published comparisons include:

- **Delineation performance** against Jimenez-Perez et al.'s `DelineatorSwitchAndCompose` model on the RDB and internal validation sets.
- **Diagnosis performance** against ECGFounder, a ResNet baseline, and average cardiologist performance, on the Stanford (iRhythm) and CinC (AliveCor) diagnosis validation cohorts (see `paper/boxplot-stanford.py`, `paper/boxplot-cinc.py`).

Note: the `data/STANFORD` and `data/CINC` folders present in this working copy (used by `benchmark_diagnosis_STANFORD.sh` / `benchmark_diagnosis_CINC.sh`) are evaluation/benchmark-reproduction data, not training data.

## Limitations

- Rhythm/logic diagnosis always reasons over lead II only; providing more leads improves the segmentation model choice (1-lead vs 3-lead) but not which lead drives diagnosis.
- The 3-lead model requires leads II, V1, and V6 specifically; any other lead combination beyond lead II alone falls back to the 1-lead model with a warning.
- Diagnosis validation cohorts are single-lead, ambulatory-leaning (Zio, KardiaMobile, ICENTIA); performance on multi-lead clinical-only rhythms outside the validated diagnosis set is less characterized.

## Ethical considerations & clinical status

ALADIN is a research tool and is **not a certified medical device** — it has not received FDA, CE, or other regulatory clearance. It is intended for research and clinical decision-support exploration, not for standalone diagnosis or treatment decisions. Any clinical use must involve review by a qualified healthcare professional.

## How to use

See [README.md](README.md) for installation and usage instructions.

