"""
setup.py — Instalación de dependencias
Beat-level CHF Morphology Detection (v12)
Autor: Ing. Adriel Lariza Lozada Romero

Usage:
    python setup.py

This script checks and isntalls all libraries required to run
chf_detection.py, incluyendo CNN (TensorFlow) and MiniRocket (sktime).

Compatibility:
    - Python 3.9 or higher
    - works with pip O conda (uses conda as fallback if pip is unavailable)
    - Tested on Spyder (conda)
"""

import sys
import subprocess
import importlib


# ══════════════════════════════════════════════════════════════════════════════
#  ALL REQUIRED PACKAGES
#  format: (importa_name, pip_name, conda_name, conda_channel)
# ══════════════════════════════════════════════════════════════════════════════
PACKAGES = [
    # core
    ("numpy",       "numpy",            "numpy",            "conda-forge"),
    ("pandas",      "pandas",           "pandas",           "conda-forge"),
    ("matplotlib",  "matplotlib",       "matplotlib",       "conda-forge"),
    ("scipy",       "scipy",            "scipy",            "conda-forge"),
    # machine learning
    ("sklearn",     "scikit-learn",     "scikit-learn",     "conda-forge"),
    ("imblearn",    "imbalanced-learn", "imbalanced-learn", "conda-forge"),
    ("xgboost",     "xgboost",          "xgboost",          "conda-forge"),
    # ECG
    ("wfdb",        "wfdb",             "wfdb",             "conda-forge"),
    ("tensorflow",  "tensorflow",       "tensorflow",       "conda-forge"),
    ("sktime",      "sktime",           "sktime",           "conda-forge"),
    # utilities
    ("psutil",      "psutil",           "psutil",           "conda-forge"),
    
    
]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _check(import_name: str) -> bool:
    """Returns True if the package is already import"""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _pip_available() -> bool:
    """Checks wheter pip is available in the current enviroment"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True
    )
    return result.returncode == 0


def _conda_available() -> bool:
    """Checks whether conda is available in the path"""
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _install_pip(pip_name: str) -> bool:
    """Attemps to install using pip.  Retruns true if successful"""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
        capture_output=True
    )
    return result.returncode == 0


def _install_conda(conda_name: str, channel: str) -> bool:
    """Attemps to install using conda. Retruns true if successful"""
    try:
        result = subprocess.run(
            ["conda", "install", "-c", channel, conda_name, "-y", "--quiet"],
            capture_output=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _install(pip_name: str, conda_name: str, channel: str,
             use_pip: bool, use_conda: bool) -> bool:
    """
    Installs a package using pip first, if it fails or pip is unavailable,
    it falls back to conda.
    """
    if use_pip:
        print(f"   pip install {pip_name}...", end="", flush=True)
        if _install_pip(pip_name):
            print(" ok")
            return True
        else:
            print(" ❌  → attempting...")

    if use_conda:
        print(f"   conda install {conda_name}...", end="", flush=True)
        if _install_conda(conda_name, channel):
            print(" ok")
            return True
        else:
            print(" ❌")

    return False


def _section(title: str):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#   PYTHON check
# ══════════════════════════════════════════════════════════════════════════════
def check_python():
    major, minor = sys.version_info[:2]
    print(f"\n  Python {major}.{minor} detected", end="")
    if major < 3 or (major == 3 and minor < 9):
        print(" ❌  —  Python 3.9 or superior higher is required.")
        sys.exit(1)
    print(" ✅")


# ══════════════════════════════════════════════════════════════════════════════
#  INSTALLER DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_installers():
    use_pip   = _pip_available()
    use_conda = _conda_available()

    _section("Installers detected")
    print(f"  pip   : {'✅ availbale' if use_pip   else '❌ not availbale — trying with conda'}")
    print(f"  conda : {'✅ availbale' if use_conda else '❌ not availbale'}")

    if not use_pip and not use_conda:
        print("\n  ❌   pip or conda not availbale")
        print("     Install the packages manually from your environment manager.")
        sys.exit(1)

    return use_pip, use_conda


# ══════════════════════════════════════════════════════════════════════════════
#  INSTALATION
# ══════════════════════════════════════════════════════════════════════════════
def install_all(use_pip: bool, use_conda: bool):
    _section("Installing")
    failed = []

    for import_name, pip_name, conda_name, channel in PACKAGES:
        if _check(import_name):
            print(f"  ✔  {pip_name:<25} already instaled")
            continue

        ok = _install(pip_name, conda_name, channel, use_pip, use_conda)
        if not ok:
            failed.append((pip_name, conda_name, channel))

    return failed


# ══════════════════════════════════════════════════════════════════════════════
#   FINAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def verify_all():
    _section(" Final Verification")
    all_ok = True
    for import_name, pip_name, _, __ in PACKAGES:
        ok = _check(import_name)
        status = "✅" if ok else "❌  missing"
        print(f"  {status}  {pip_name}")
        if not ok:
            all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
#   DATASET INSTRUCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def print_dataset_instructions():
    _section("Dataset — BIDMC CHF Database")
    print("""
  This project uses the BIDMC Congestive Heart Failure Database,
publicly available on PhysioNet (free and open access).

Steps to download it:
───────────────────────────────────────────────────────────────
1. Go to: https://physionet.org/content/chfdb/1.0.0/
2. Download all files (.hea, .dat, .ecg) into a local folder.
   You can use the “Download ZIP” button or download via wfdb:

       import wfdb
       wfdb.dl_database('chfdb', dl_dir='./bidmc-chf-database')

3. Open chf_detection.py and update the paths at the end of the file:

       DATA_PATH   = Path(r"PATH_TO_YOUR_FOLDER/bidmc-chf-database")
       PROJECT_DIR = Path(r"PATH_WHERE_RESULTS_WILL_BE_SAVED")

  ─────────────────────────────────────────────────────────────────
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  CHF Detection v12 — Setup ")
    print("═"*60)

    check_python()
    use_pip, use_conda = detect_installers()
    failed = install_all(use_pip, use_conda)
    all_ok = verify_all()
    print_dataset_instructions()

    print("═"*60)
    if all_ok:
        print("  ✅  Setup complete.")
        print("  ⚠️   Restart the kernel before running chf_detection.py")
        print("        so that the newly installed packages are recognized")
        print("\n  Then Run:")
        print("        python chf_detection.py")
    else:
        print("  ❌  Some packages could not be installed automatically")
        print("      Install them manually from Anaconda/conda:\n")
        for pip_name, conda_name, channel in failed:
            print(f"        conda install -c {channel} {conda_name}")
    print("═"*60 + "\n")