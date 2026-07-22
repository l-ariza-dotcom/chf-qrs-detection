# Beat-Level ECG Morphology Classification for CHF Detection

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PhysioNet](https://img.shields.io/badge/Dataset-PhysioNet%20BIDMC--CHF-lightblue)](https://physionet.org/content/chfdb/1.0.0/)
[![Status](https://img.shields.io/badge/Status-Research%20v12-orange)](https://github.com/l-ariza-dotcom/chf-qrs-detection)

> **Associated paper:**  
> *Beat-Level ECG Morphology Classification for Congestive Heart Failure Detection:
> A Comparative Study of Machine Learning Approaches*  


---

##  Overview

This repository provides the full reproducible pipeline (`v12`) for **beat-level binary classification** of Congestive Heart Failure (CHF) morphology from single-lead ECG signals.
Individual cardiac beats are segmented around R-peaks using the Pan–Tompkins algorithm, handcrafted morphological/temporal/spectral features are extracted, and five classifiers
are benchmarked under identical experimental conditions.


**Classification task:**  
`Normal` (WFDB annotation `N`) vs `CHF-associated morphology` (any annotation ≠ `N`, i.e., `V`, `S`, `Q`, `r`)

---

## Models Evaluated

| Model | Key Hyperparameters | Framework |
|---|---|---|
| Random Forest | `n_estimators=200`, `max_depth=10`, `criterion=gini` | `scikit-learn` |
| Gradient Boosting | `n_estimators=300`, `learning_rate=0.05`, `max_depth=5` | `scikit-learn` |
| XGBoost | `n_estimators=500`, `η=0.1`, `λ=1.0`, `α=0.5`, `subsample=0.8` | `xgboost` |
| 1D CNN | 3 conv blocks (32→64→128 filters), `dropout=0.3`, 50 epochs, Adam `lr=0.001` | `tensorflow/keras` |
| MiniRocket + Ridge | `num_kernels=10000`, Ridge `α=1.0` | `sktime` |

> **Note:** MiniRocket and CNN are **optional** — the pipeline runs with only the ensemble models if those libraries are not installed.

All hyperparameters were selected via **stratified 5-fold cross-validation** on the balanced training set, optimizing for mean F1-score (`random_state=42`).

---

## Repository Structure

```
chf-qrs-detection/
│
├── chf_detection.py        # Full pipeline — loading, features, training, evaluation
├── setup.py                # Dependency installer (pip + conda fallback)
├── requirements.txt        # Pinned library versions (paper-exact)
│
├── data/                   # ← place BIDMC CHF records here (not included)
│   └── bidmc-chf-database/
│       ├── chf01.hea
│       ├── chf01.dat
│       └── ...
│
└── results/                # Generated automatically at runtime
    └── version_12/
        ├── models/             # Saved .pkl / .keras model files
        ├── plots/              # Confusion matrices, ROC curves (.eps + .png)
        ├── signal_analysis/    # Per-record signal plots
        ├── reports/
        │   ├── paper_metrics_v12.csv        # All metrics + 95% CI (n=1,000 bootstrap)
        │   ├── loso_fold_details.csv        # Per-subject LOSO results
        │   └── reproducibility_config.json  # Full hyperparameter log
        └── cache_records/      # Cached beat features (speeds up re-runs)
```

---

## Installation

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

### 3a. Install from requirements.txt (exact paper versions)

```bash
pip install -r requirements.txt
```

### 3b. Or run the setup script (auto-detects pip / conda)

```bash
python setup.py
```

> **Restart your kernel/environment** after installation before running
> `chf_detection.py`.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.2.5 | Numerical computation |
| `scipy` | 1.15.3 | Signal processing, filtering |
| `scikit-learn` | 1.6.1 | ML models, CV, metrics |
| `imbalanced-learn` | 0.13.0 | BorderlineSMOTE + RandomUnderSampler |
| `xgboost` | 2.1.4 | XGBoost classifier |
| `shap` | 0.47.2 | Post-hoc explainability |
| `wfdb` | ≥4.1.0 | PhysioNet ECG record reader |
| `tensorflow` *(optional)* | 2.19.0 | 1D CNN model |
| `sktime` *(optional)* | 0.37.0 | MiniRocket transform |
| `psutil` | ≥5.9.0 | RAM usage monitoring |

---

## Dataset

This project uses the **BIDMC Congestive Heart Failure Database**, freely available on PhysioNet (open access, no registration required).

**Download via `wfdb`** (recommended):

```python
import wfdb
wfdb.dl_database('chfdb', dl_dir='./data/bidmc-chf-database')
```

Or download manually from: https://physionet.org/content/chfdb/1.0.0/

**Cite the dataset:**
> Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
> *Circulation*, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

---

## Running the Pipeline

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

> Start with **option 1** to verify your installation before running the full pipeline.

---

## Signal Processing & Feature Extraction

Each ECG record is processed through six stages:

1. **Bandpass filter** — Butterworth 4th order, 0.5–40 Hz
2. **R-peak detection** — via `wfdb` annotations (Pan–Tompkins)
3. **Beat segmentation** — 300-sample window (≈1.2 s at 250 Hz) centered on R-peak
4. **Normalization** — min-max to [−1, 1] per segment
5. **Feature extraction** — 18 descriptors: 7 temporal, 6 morphological, 5 spectral
6. **Quality filtering** — segments with SNR < 25 dB are discarded

---

## Class Balancing

Applied **only to the training set** after subject-level splitting (no data leakage):

1. `RandomUnderSampler` — caps the majority class at 6,000 Normal beats
2. `BorderlineSMOTE(k_neighbors=3)` — oversamples the minority class to 6,000 beats

**Result:** balanced training set of 12,000 beats (1:1 ratio), test set preserves the original ≈82:1 imbalance for unbiased evaluation.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Random seed | `random_state=42` (all stochastic operations) |
| Dataset split | Stratified beat-level 80/20 split |
| CV strategy | Stratified 5-fold (hyperparameter tuning only) |
| Primary metric | F1-score (optimization criterion) |
| Bootstrap CI | 95%, n=1,000 stratified resamples |
| LOSO | Leave-One-Subject-Out (15 folds, XGBoost) |
| Hardware | AMD Ryzen 7 8840HS, 16 GB DDR4, no GPU |

---

## Outputs

| File | Description |
|---|---|
| `reports/paper_metrics_v12.csv` | Accuracy, F1, Precision, Recall, Specificity, NPV, MCC, AUC-ROC + 95% CI |
| `reports/loso_fold_details.csv` | Per-subject LOSO accuracy, F1, AUC-ROC |
| `reports/reproducibility_config.json` | All hyperparameters, split strategy, balancing config |
| `plots/confusion_matrices.eps` | Normalized confusion matrices per model |
| `plots/roc_curves.eps` | ROC curves for all five classifiers |
| `models/*.pkl` / `*.keras` | Serialized trained models |

---

## Key Results (Independent Test Set)

| Model | Accuracy | F1 | AUC-ROC | MCC |
|---|---|---|---|---|
| **XGBoost** | **0.9805** | **0.9843** | **0.9863** | **0.5548** |
| Gradient Boosting | 0.9712 | 0.9785 | 0.9801 | 0.4893 |
| Random Forest | 0.9688 | 0.9769 | 0.9764 | 0.4570 |
| MiniRocket | 0.9610 | 0.9723 | 0.9571 | 0.4208 |
| 1D CNN | 0.9012 | 0.9381 | 0.9346 | 0.2637 |

**LOSO cross-validation (XGBoost, 15 folds):**
Mean AUC-ROC = 0.738 ± 0.181 · Mean Accuracy = 0.683 ± 0.389

---

## Reproducibility

All stochastic components use `random_state=42`. The full configuration is logged automatically to `reports/reproducibility_config.json` at the end of each run.

The complete experimental environment is pinned in `requirements.txt`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lozada2026chf,
  title   = {Beat-Level {ECG} Morphology Classification for Congestive Heart Failure
             Detection: A Comparative Study of Machine Learning Approaches},
  author  = {Lozada-Romero, A.L. and Ram{\'i}rez-Villalobos, R. and
             Trujillo, L. and C{\'a}rdenas-Valdez, J.R.},
  journal = {Information Sciences},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file
for details.

---

## Contact

**Ing. Adriel Lariza Lozada Romero** — [@l-ariza-dotcom](https://github.com/l-ariza-dotcom)  
adriel.lr@tectijuana.edu.mx  
*Master's student | Biomedical Signal Processing & Machine Learning*  
Tecnológico Nacional de México / IT de Tijuana

