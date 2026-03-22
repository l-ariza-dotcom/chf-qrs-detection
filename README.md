# Beat-level CHF Morphology Detection from QRS Complex

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PhysioNet](https://img.shields.io/badge/Dataset-PhysioNet%20BIDMC--CHF-lightblue)](https://physionet.org/content/chfdb/1.0.0/)
[![Status](https://img.shields.io/badge/Status-Research%20v12-orange)]()

> **Associated paper:** *Comparative Analysis of Machine Learning Models for Congestive Heart Failure Detection from QRS Complex* — Ing. Adriel Lariza Lozada Romero

---

## 📋 Overview

This repository provides the full reproducible pipeline (`v12`) for **beat-level binary classification** of Congestive Heart Failure (CHF) morphology from single-lead ECG signals. Individual cardiac beats are segmented around R-peaks, handcrafted morphological features are extracted, and five classifiers are benchmarked under identical conditions.

**Normal** (annotation `N`) vs **CHF-associated morphology** (any annotation ≠ `N`)

### Models evaluated

| Model | Notes |
|---|---|
| Random Forest | `n_estimators=200`, balanced class weights |
| Gradient Boosting | `n_estimators=200`, `lr=0.08` |
| XGBoost | `tree_method=hist`, `n_estimators=250` |
| MiniRocket + RidgeCV | Time-series kernels — requires `sktime` |
| 1D-CNN | `Conv1D → BN → GAP → Dense` — requires `tensorflow` |

> MiniRocket and CNN are **optional** — the pipeline runs without them if the libraries are not installed.

---

## 🗂️ Repository Structure

```
chf-qrs-detection/
│
├── chf_detection.py        # Full pipeline — data loading, features, training, evaluation
├── setup.py                # Dependency installer (pip + conda fallback)
│
├── data/                   # ← place BIDMC CHF records here (not included)
│   └── bidmc-chf-database/
│       ├── chf01.hea
│       ├── chf01.dat
│       └── ...
│
└── results/                # Generated automatically at runtime
    └── version_12/
        ├── models/             # Saved .pkl model files
        ├── plots/              # Confusion matrices, ROC curves
        ├── signal_analysis/    # Per-record signal plots
        ├── reports/
        │   ├── paper_metrics_v12.csv        # All metrics + 95% CI
        │   └── reproducibility_config.json  # Full hyperparameter log
        └── cache_records/      # Cached beat features (speeds up re-runs)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/l-ariza-dotcom/chf-qrs-detection.git
cd chf-qrs-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Run the setup script

```bash
python setup.py
```

This will automatically detect whether `pip` or `conda` is available and install all required packages. At the end, it prints download instructions for the dataset.

> ⚠️ **Restart your kernel/environment** after setup before running `chf_detection.py`.

### Dependencies installed

| Package | Purpose |
|---|---|
| `numpy`, `pandas`, `scipy` | Numerical processing |
| `matplotlib` | Visualization |
| `scikit-learn` | ML models, preprocessing, metrics |
| `imbalanced-learn` | BorderlineSMOTE + RandomUnderSampler |
| `xgboost` | XGBoost classifier |
| `wfdb` | Read PhysioNet ECG records |
| `tensorflow` *(optional)* | 1D-CNN model |
| `sktime` *(optional)* | MiniRocket classifier |
| `psutil` | RAM usage monitoring |

---

## 📦 Dataset

This project uses the **BIDMC Congestive Heart Failure Database**, freely available on PhysioNet.

**Download via `wfdb`** (recommended):

```python
import wfdb
wfdb.dl_database('chfdb', dl_dir='./data/bidmc-chf-database')
```

Or download manually from: [https://physionet.org/content/chfdb/1.0.0/](https://physionet.org/content/chfdb/1.0.0/)

---

## 🚀 Running the Pipeline

### 1. Configure paths

Open `chf_detection.py` and update the two paths at the bottom of the file:

```python
DATA_PATH   = Path(r"./data/bidmc-chf-database")   # ← your dataset folder
PROJECT_DIR = Path(r"./results")                    # ← where outputs are saved
```

### 2. Run

```bash
python chf_detection.py
```

An **interactive menu** will appear:

```
──────────────────────────────────────────────────────────────
  MENU v12 — Beat-level CHF Morphology Detection
──────────────────────────────────────────────────────────────
  1)   Quick Test            (2 records,  5k beats)
  2)   Partial Validation    (3 records, 15k beats)
  3)   Full                  (all records, 50k beats, no LOSO)
  4)   Full + LOSO
  5)   Exit
──────────────────────────────────────────────────────────────
```

> 💡 Start with **option 1** to verify your installation before running the full pipeline.

---

## 🔬 Signal Processing & Feature Extraction

Each ECG record is processed as follows:

1. **Bandpass filter** — Butterworth 4th order, 0.5–40 Hz
2. **R-peak detection** — via `wfdb` annotations
3. **Beat segmentation** — 300-sample window (600 ms at 250 Hz) centered on each R-peak
4. **Normalization** — min-max to [−1, 1] per segment
5. **Feature extraction** — morphological and statistical features per beat

---

## ⚖️ Class Balancing

Applied **only to the training set** (no data leakage):

1. `RandomUnderSampler` — caps the majority class at 6,000 beats
2. `BorderlineSMOTE(k_neighbors=3)` — oversamples the minority class

---

## 📊 Outputs

| File | Description |
|---|---|
| `reports/paper_metrics_v12.csv` | Accuracy, F1, Precision, Recall, Specificity, NPV, MCC, AUC-ROC + 95% bootstrap CI |
| `reports/reproducibility_config.json` | All hyperparameters, split strategy, balancing config |
| `plots/*.png` | Confusion matrices and ROC curves per model |
| `models/*.pkl` | Serialized trained models |

### Metrics reported (with 95% Bootstrap CI, n=1,000)

- Accuracy, Precision, Recall (Sensitivity), F1-score
- Specificity, NPV (Negative Predictive Value)
- MCC (Matthews Correlation Coefficient)
- AUC-ROC

---

## 🔁 Reproducibility

All stochastic components use `random_state=42`. The full configuration is logged automatically to `reports/reproducibility_config.json` at the end of each run.

Split strategy: **stratified beat-level split** (80/20), `random_state=42`.

LOSO (Leave-One-Subject-Out) cross-validation is available via **option 4** of the menu.

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{lariza2025chf,
  title   = {Comparative Analysis of Machine Learning Models for
             Congestive Heart Failure Detection from QRS Complex},
  author  = {Lariza Lozada Romero, Adriel},
  journal = {[Journal / Conference Name]},
  year    = {2025},
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contact

**Ing. Adriel Lariza Lozada Romero** — [@l-ariza-dotcom](https://github.com/l-ariza-dotcom)

*Master's student | Biomedical Signal Processing & Machine Learning*
