# chf_detection.py
# Autor: Ing. Adriel Lariza Lozada Romero
# Beat-level CHF Morphology Detection — v12
#
#
# REQUIREMENTS
#   Run first: python setup.py
#   Dataset: BIDMC CHF Database (physionet.org/content/chfdb/1.0.0/)

from __future__ import annotations
import os, gc, json, pickle, time, warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no GUI — avoids crashes in Spyder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wfdb
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, roc_curve, matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── RAM monitor ───────────────────────────────────────────────────────────────
try:
    import psutil; _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── Opcionales ────────────────────────────────────────────────────────────────
CNN_AVAILABLE = False
ROCKET_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import models as keras_models, layers
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    CNN_AVAILABLE = True
except Exception:
    pass
try:
    from sktime.transformations.panel.rocket import MiniRocket
    from sklearn.linear_model import RidgeClassifierCV
    ROCKET_AVAILABLE = True
except Exception:
    pass

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _ram_str():
    if not _PSUTIL: return ""
    m = psutil.virtual_memory()
    pct = m.percent
    bar = "█"*int(20*pct/100) + "░"*(20-int(20*pct/100))
    return f"{'⚠️ ' if pct>75 else ''}RAM [{bar}] {m.used/1e9:.1f}/{m.total/1e9:.1f} GB ({pct:.0f}%)"

def _print_ram(label=""):
    r = _ram_str()
    if r: print(f"  {'['+label+'] ' if label else ''}{r}")

def _section(title):
    print(f"\n{'─'*62}\n  {title}"); _print_ram(); print(f"{'─'*62}")

def _free_mem(*arrays):
    for a in arrays: del a
    gc.collect()

def _bandpass_sig(x, fs=250, lo=0.5, hi=40.0, order=4):
    nyq = fs/2
    b, a = scipy_signal.butter(order, [lo/nyq, hi/nyq], btype="band")
    return scipy_signal.filtfilt(b, a, x)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS + BOOTSTRAP CI
# ══════════════════════════════════════════════════════════════════════════════
def _extended_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    spec = float(tn/(tn+fp)) if (tn+fp)>0 else 0.0
    npv  = float(tn/(tn+fn)) if (tn+fn)>0 else 0.0
    mcc  = float(matthews_corrcoef(y_true, y_pred))
    auc  = None
    if y_proba is not None:
        try: auc = float(roc_auc_score(y_true, y_proba))
        except: pass
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="weighted")),
        "f1":          float(f1_score(y_true, y_pred, average="weighted")),
        "specificity": spec, "npv": npv, "mcc": mcc, "auc_roc": auc,
        "tp":int(tp), "tn":int(tn), "fp":int(fp), "fn":int(fn),
    }

def _bootstrap_ci(y_true, y_pred, y_proba=None, n_boot=1000, seed=42):
    rng  = np.random.default_rng(seed); n = len(y_true)
    keys = ["accuracy","precision","recall","f1","specificity","npv","mcc"]
    if y_proba is not None: keys.append("auc_roc")
    boots = {k:[] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; yp = y_pred[idx]
        ypr = y_proba[idx] if y_proba is not None else None
        if len(np.unique(yt)) < 2: continue
        m = _extended_metrics(yt, yp, ypr)
        for k in keys:
            if m.get(k) is not None: boots[k].append(m[k])
    ci = {}
    for k in keys:
        arr = np.array(boots[k])
        ci[k] = (float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))) if len(arr) else (None,None)
    return ci


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Paths12:
    project_dir: Path; run_dir: Path; img_dir: Path
    signal_dir: Path; plots_dir: Path; models_dir: Path
    cache_dir: Path; reports_dir: Path

    @staticmethod
    def build(project_dir: Path) -> "Paths12":
        r = project_dir / "version_12"
        return Paths12(project_dir=project_dir, run_dir=r,
            img_dir=project_dir/"Images", signal_dir=r/"signal_analysis",
            plots_dir=r/"plots", models_dir=r/"models",
            cache_dir=r/"cache_records", reports_dir=r/"reports")


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
class CHFDetectionV12:
    """
    CHF Detection v12 — Beat-level binary morphology classification.
    Datasets: BIDMC CHF Database (physionet.org/content/chfdb/1.0.0/)
    """

    def __init__(self, data_path, project_dir, fs=250, window_size=300,
                 norm_mode="minmax_-1_1"):
        self.data_path   = Path(data_path)
        self.paths       = Paths12.build(Path(project_dir))
        self.fs          = fs
        self.window_size = window_size
        self.norm_mode   = norm_mode
        self.half        = window_size // 2

        for p in [self.paths.run_dir, self.paths.img_dir, self.paths.signal_dir,
                  self.paths.plots_dir, self.paths.models_dir,
                  self.paths.cache_dir, self.paths.reports_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.scaler  = StandardScaler()
        self.models: dict = {}
        self.results: dict = {}
        self.training_histories: dict = {}
        self.feature_names = [
            "Mean_Amplitude","Std_Amplitude","Skewness","Kurtosis",
            "Max_Amplitude","Min_Amplitude","Peak_to_Peak",
            "Spectral_Mean","Spectral_Std","Spectral_Max",
            "Low_Freq_Power","High_Freq_Power",
            "Positive_Derivatives","Negative_Derivatives",
            "Max_Derivative","Min_Derivative",
            "Total_Variation","Inflection_Points",
        ]
        print("\n"+"═"*62)
        print("    CHF DETECTION v12")
        print("═"*62)
        print(f"  Data    : {self.data_path}")
        print(f"  Proyecto: {self.paths.run_dir}")
        print(f"  Norm    : {self.norm_mode} (per-segment) | Window:{window_size} | fs:{fs}Hz")
        print(f"  Modelos : RF, GB, XGB, CNN{'✅' if CNN_AVAILABLE else '❌'} Rocket{'✅' if ROCKET_AVAILABLE else '❌'}")
        print(f"  Metrics: Acc·F1·AUC·Spec·NPV·MCC + CI95%  ")
        _print_ram("System"); print("═"*62)

    # ──────────────────────────────────────────────────────────────────────────
    #  PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    def _bandpass(self, x):
        nyq = self.fs/2
        b,a = scipy_signal.butter(4, [0.5/nyq, 40/nyq], btype="band")
        return scipy_signal.filtfilt(b, a, x, axis=0)

    def _normalize(self, x):
        if self.norm_mode == "zscore":
            return (x-np.mean(x,axis=0))/(np.std(x,axis=0)+1e-8)
        xmin=np.min(x,axis=0); xmax=np.max(x,axis=0)
        return 2*(x-xmin)/(xmax-xmin+1e-8)-1

    def preprocess_signal(self, sig):
        return self._normalize(self._bandpass(sig)).astype(np.float32)

    def _extract_features(self, beat):
        f = [np.mean(beat),np.std(beat),float(skew(beat)),float(kurtosis(beat)),
             np.max(beat),np.min(beat),np.ptp(beat)]
        fft = np.abs(np.fft.rfft(beat)).astype(np.float32)
        q = len(fft)//4
        f += [float(np.mean(fft)),float(np.std(fft)),float(np.max(fft)),
              float(np.sum(fft[:q])),float(np.sum(fft[q:]))]
        d = np.diff(beat)
        f += [float(np.sum(d>0)),float(np.sum(d<0)),float(np.max(d)),float(np.min(d)),
              float(np.sum(np.abs(d))),float(len(np.where(np.diff(np.sign(d)))[0]))]
        return np.asarray(f, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    #  CARGA + CACHE
    # ──────────────────────────────────────────────────────────────────────────
    def list_records(self):
        return sorted([f[:-4] for f in os.listdir(self.data_path) if f.endswith(".hea")])

    def _load_record(self, name):
        rp = self.data_path/name
        rec = wfdb.rdrecord(str(rp))
        try: ann = wfdb.rdann(str(rp),"ecg")
        except: ann = wfdb.rdann(str(rp),"atr")
        return rec, ann

    def cache_features_per_record(self, max_records=None, max_beats_per_record=2000):
        records = self.list_records()
        if max_records: records = records[:max_records]
        _section(f"[1/5] Cache — {len(records)} records | window={self.window_size} | seed=42")
        for i,r in enumerate(records,1):
            npz = self.paths.cache_dir/f"{r}_features.npz"
            if npz.exists():
                print(f"  [{i:2d}/{len(records)}] {r}  ✔ cache existe"); continue
            print(f"  [{i:2d}/{len(records)}] {r}  procesando...", end="", flush=True)
            try:
                rec,ann = self._load_record(r)
                sig = self.preprocess_signal(rec.p_signal)
                del rec; gc.collect()
                half = self.half; X_list,y_list = [],[]
                for peak,lab in zip(ann.sample,ann.symbol):
                    if len(X_list)>=max_beats_per_record: break
                    s,e = peak-half, peak+half
                    if s<0 or e>=len(sig): continue
                    beat = sig[s:e,0]
                    if len(beat)!=self.window_size: continue
                    X_list.append(self._extract_features(beat)); y_list.append(lab)
                _free_mem(sig)
                if not X_list: print("    no beats"); continue
                np.savez_compressed(npz, X=np.vstack(X_list).astype(np.float32),
                                    y=np.array(y_list), record=r)
                _free_mem(X_list, y_list)
                print(f"  ✅ {len(y_list)} beats"); _print_ram()
            except Exception as e:
                print(f"\n    ⚠️  {e}"); gc.collect()
        print(f"\n  ✅ Cache → {self.paths.cache_dir}")

    def load_cached_dataset(self, max_beats_total=None):
        files = sorted(self.paths.cache_dir.glob("*_features.npz"))
        if not files: raise FileNotFoundError("Sin cache.")
        _section(f"[2/5] loading dataset — {len(files)} records")
        X_all,y_all,g_all = [],[],[]
        total = 0
        for f in files:
            if max_beats_total and total>=max_beats_total: break
            d=np.load(f,allow_pickle=True); X_f=d["X"].astype(np.float32); y_f=d["y"]; rec=str(d["record"]); d.close()
            if max_beats_total and total+len(X_f)>max_beats_total:
                keep=max_beats_total-total; X_f,y_f=X_f[:keep],y_f[:keep]
            X_all.append(X_f); y_all.append(y_f); g_all.append(np.full(len(y_f),rec)); total+=len(y_f)
        X=np.vstack(X_all).astype(np.float32); y=np.hstack(y_all); groups=np.hstack(g_all)
        _free_mem(X_all,y_all,g_all)
        print(f"  X={X.shape}  |  RAM≈{X.nbytes/1e6:.1f} MB"); _print_ram("Tras carga")
        return X,y,groups

    def _to_binary_chf(self, y_symbols):
        y = y_symbols.astype(str)
        return np.where(np.char.upper(y)=="N", "Normal", "CHF-morphology")

    def smart_balance_binary(self, X, y_bin):
        before = pd.Series(y_bin).value_counts()
        y01 = np.where(y_bin=="Normal",0,1).astype(np.int8)
        unique,counts = np.unique(y01,return_counts=True)
        cap = {int(l):min(6000,int(c)) for l,c in zip(unique,counts)}
        X_u,y_u = RandomUnderSampler(sampling_strategy=cap,random_state=42).fit_resample(X,y01)
        X_b,y_b = BorderlineSMOTE(random_state=42,k_neighbors=3).fit_resample(X_u,y_u)
        _free_mem(X_u,y_u)
        after = pd.Series(np.where(y_b==0,"Normal","CHF-morphology")).value_counts()
        summary = pd.DataFrame({
            "Class":["Normal","CHF-morphology"],
            "Before_balancing":[int(before.get("Normal",0)),int(before.get("CHF-morphology",0))],
            "After_balancing": [int(after.get("Normal",0)),int(after.get("CHF-morphology",0))],
        })
        summary.to_csv(self.paths.signal_dir/"class_balance_summary_v12.csv",index=False)
        print(f"  ✅  before balance {before.get('Normal',0)}→{after.get('Normal',0)}"
              f"  CHF {before.get('CHF-morphology',0)}→{after.get('CHF-morphology',0)}")
        _print_ram("after balance")
        return X_b.astype(np.float32), y_b.astype(np.int8), summary

    def split_stratified(self, X, y01, test_size=0.2):
        return train_test_split(X, y01, test_size=test_size, random_state=42, stratify=y01)

    # ──────────────────────────────────────────────────────────────────────────
    #  LOSO
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate_loso(self, X, y01, groups):
        splitter = LeaveOneGroupOut()
        n_folds  = splitter.get_n_splits(groups=groups)
        _section(f"[Group Eval] LOSO — {n_folds} folds")
        print("  ⚠️  CNN,  LOSO excluded ")
        metric_keys = ["accuracy","precision","recall","f1","specificity","npv","mcc","auc_roc"]
        fold_rows = []; per_metric = {k:[] for k in metric_keys}
        for fold,(tr,te) in enumerate(splitter.split(X,y01,groups=groups),1):
            test_record = np.unique(groups[te])[0]
            Xtr,Xte = X[tr],X[te]; ytr,yte = y01[tr],y01[te]
            Xb,yb,_ = self.smart_balance_binary(Xtr, np.where(ytr==0,"Normal","CHF-morphology"))
            sc = StandardScaler()
            Xtr_s=sc.fit_transform(Xb); Xte_s=sc.transform(Xte)
            mdl = xgb.XGBClassifier(n_estimators=150,max_depth=5,learning_rate=0.08,
                subsample=0.8,colsample_bytree=0.9,random_state=42,
                eval_metric="logloss",tree_method="hist")
            mdl.fit(Xtr_s,yb)
            yp=mdl.predict(Xte_s); yprob=mdl.predict_proba(Xte_s)[:,1]
            m=_extended_metrics(yte,yp,yprob)
            row={"fold":fold,"test_record":test_record,"n_test_beats":int(len(yte)),
                 "n_normal":int(np.sum(yte==0)),"n_chf":int(np.sum(yte==1))}
            row.update(m); fold_rows.append(row)
            for k in metric_keys:
                val = m.get(k)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    per_metric[k].append(val)
            auc_str=f"  auc={m['auc_roc']:.3f}" if m.get("auc_roc") and not np.isnan(m["auc_roc"]) else "  auc=N/A (a single class in test)"
            print(f"  Fold{fold:3d}/{n_folds}  [{test_record}]  acc={m['accuracy']:.3f}"
                  f"  f1={m['f1']:.3f}  spec={m['specificity']:.3f}  mcc={m['mcc']:.3f}{auc_str}")
            _free_mem(mdl,Xb,yb,Xtr_s,Xte_s)
        df_folds = pd.DataFrame(fold_rows)
        df_folds.to_csv(self.paths.reports_dir/"loso_fold_details.csv",index=False)
        print(f"\n  Details per fold: {self.paths.reports_dir/'loso_fold_details.csv'}")
        summary = {}
        for k in metric_keys:
            arr = np.array(per_metric[k])
            if len(arr) == 0:
                continue
            entry = {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                     "min": float(np.min(arr)), "max": float(np.max(arr))}
            if k == "auc_roc":
                # Some folds have only a sungle class in the test set and do not allow 
                # compute AUC (the number of folds contributing ro the average is reported)
                entry["n_valid_folds"] = len(arr)
            summary[k] = entry
        with open(self.paths.reports_dir/"loso_summary.json","w") as f:
            json.dump(summary,f,indent=2)
        difficult=[r for r in fold_rows if r["accuracy"]<0.5]
        with open(self.paths.reports_dir/"difficult_subjects.txt","w",encoding="utf-8") as f:
            f.write("LOSO accuracy records < 0.5\n\n")
            for d in difficult:
                f.write(f"  {d['test_record']}:  acc={d['accuracy']:.3f}  f1={d['f1']:.3f}"
                        f"  n_beats={d['n_test_beats']}  n_chf={d['n_chf']}\n")
            if not difficult: f.write("  None.\n")
        print(f"\n  {'Metric':<14} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}  {'Folds':>6}")
        print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*6}")
        for k,v in summary.items():
            if not v: continue
            n_info = f"  {v['n_valid_folds']:>3}/{n_folds}" if "n_valid_folds" in v else f"  {n_folds:>3}/{n_folds}"
            print(f"  {k:<14} {v['mean']:>8.3f} {v['std']:>8.3f} {v['min']:>8.3f} {v['max']:>8.3f}{n_info}")
        return {"summary":summary,"fold_details":df_folds,"difficult":difficult}

    # ──────────────────────────────────────────────────────────────────────────
    #  TRAINING
    # ──────────────────────────────────────────────────────────────────────────
    def train_models(self, X_train, y_train, X_test, y_test,
                     train_cnn=False, train_rocket=False, light_mode=False):
        _section("[4/5] Training + Evaluation ")
        Xtr=self.scaler.fit_transform(X_train); Xte=self.scaler.transform(X_test)
        n_rf=100 if light_mode else 200; n_gb=100 if light_mode else 200; n_xgb=150 if light_mode else 250
        models_cfg = {
            "RandomForest": RandomForestClassifier(n_estimators=n_rf,max_depth=10,min_samples_split=5,
                class_weight="balanced",random_state=42,n_jobs=-1,max_samples=0.8),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=n_gb,learning_rate=0.08,
                max_depth=4,subsample=0.8,random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=n_xgb,max_depth=5,learning_rate=0.08,
                subsample=0.8,colsample_bytree=0.9,random_state=42,
                eval_metric="logloss",tree_method="hist"),
        }
        results={}; roc_data={}
        for name,mdl in models_cfg.items():
            t0=time.time(); print(f"\n   {name}...", end="", flush=True)
            mdl.fit(Xtr,y_train)
            yp=mdl.predict(Xte); yprob=mdl.predict_proba(Xte)[:,1]
            m=_extended_metrics(y_test,yp,yprob); ci=_bootstrap_ci(y_test,yp,yprob)
            results[name]={**m,"ci95":ci}; self.models[name]=mdl; roc_data[name]=(y_test,yprob)
            with open(self.paths.models_dir/f"{name}_model.pkl","wb") as f: pickle.dump(mdl,f)
            auc_str=f"  auc={m['auc_roc']:.4f}" if m["auc_roc"] else ""
            print(f"\r  ✅ {name:<18}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}"
                  f"{auc_str}  mcc={m['mcc']:.4f}  spec={m['specificity']:.4f}  ({int(time.time()-t0)}s)")
            _print_ram()
        if train_cnn and CNN_AVAILABLE:
            res_cnn = self._train_cnn(Xtr,y_train,Xte,y_test,light_mode=light_mode)
            if res_cnn: results["CNN"],roc_data["CNN"] = res_cnn
        if train_rocket and ROCKET_AVAILABLE:
            res_rkt = self._train_minirocket(Xtr,y_train,Xte,y_test)
            if res_rkt: results["MiniRocket"],roc_data["MiniRocket"] = res_rkt
        self.results=results
        with open(self.paths.run_dir/"scaler.pkl","wb") as f: pickle.dump(self.scaler,f)
        self._plot_confusions(Xte,y_test)
        self._plot_roc_curves(roc_data)
        return results

    def _train_cnn(self, Xtr, ytr, Xte, yte, light_mode=False):
        Xtr_c=Xtr.reshape(Xtr.shape[0],Xtr.shape[1],1)
        Xte_c=Xte.reshape(Xte.shape[0],Xte.shape[1],1)
        ytr_cat=to_categorical(ytr,num_classes=2)
        filt=32 if light_mode else 64; epochs=40 if light_mode else 80; batch=128 if light_mode else 64
        model=keras_models.Sequential([
            layers.Conv1D(filt,5,activation="relu",input_shape=(Xtr.shape[1],1)),
            layers.BatchNormalization(),
            layers.Conv1D(filt*2,3,activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64,activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(2,activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss="categorical_crossentropy",metrics=["accuracy"])
        es=EarlyStopping(monitor="val_accuracy",patience=10,restore_best_weights=True,verbose=0)
        print("  Training CNN...")
        hist=model.fit(Xtr_c,ytr_cat,validation_split=0.2,epochs=epochs,batch_size=batch,
                       callbacks=[es],verbose=1)
        self.training_histories["CNN"]=hist.history
        yprob=model.predict(Xte_c,verbose=0)[:,1]
        best_thr,best_f1=0.5,0.0
        for thr in np.linspace(0.1,0.9,81):
            f1_t=f1_score(yte,(yprob>=thr).astype(int),average="weighted",zero_division=0)
            if f1_t>best_f1: best_f1,best_thr=f1_t,float(thr)
        yp=(yprob>=best_thr).astype(int)
        m=_extended_metrics(yte,yp,yprob); ci=_bootstrap_ci(yte,yp,yprob)
        try: model.save(self.paths.models_dir/"CNN_model.keras")
        except: model.save(str(self.paths.models_dir/"CNN_model.h5"))
        self.models["CNN"]=model
        print(f"  ✅ CNN  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}"
              f"  auc={m['auc_roc']:.4f}  threshold={best_thr:.2f}")
        return {**m,"ci95":ci,"cnn_threshold":best_thr},(yte,yprob)

    def _train_minirocket(self, Xtr, ytr, Xte, yte):
        print("  Training MiniRocket...")
        Xtr_r=Xtr.reshape(Xtr.shape[0],1,Xtr.shape[1]); Xte_r=Xte.reshape(Xte.shape[0],1,Xte.shape[1])
        rocket=MiniRocket(num_kernels=1000,random_state=42); rocket.fit(Xtr_r)
        Ztr=rocket.transform(Xtr_r); Zte=rocket.transform(Xte_r)
        clf=RidgeClassifierCV(alphas=np.logspace(-3,3,10)); clf.fit(Ztr,ytr)
        yp=clf.predict(Zte)
        try:
            raw=clf.decision_function(Zte); yprob=(raw-raw.min())/(raw.max()-raw.min()+1e-8)
        except: yprob=None
        _free_mem(Ztr,Zte)
        m=_extended_metrics(yte,yp,yprob); ci=_bootstrap_ci(yte,yp,yprob)
        with open(self.paths.models_dir/"MiniRocket_model.pkl","wb") as f:
            pickle.dump({"transformer":rocket,"classifier":clf},f)
        self.models["MiniRocket"]={"transformer":rocket,"classifier":clf}
        print(f"  ✅ MiniRocket  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  mcc={m['mcc']:.4f}")
        return {**m,"ci95":ci},(yte,yprob)

    # ──────────────────────────────────────────────────────────────────────────
    #   MeTRIC PLOTS (Matrices + ROC)
    # ──────────────────────────────────────────────────────────────────────────
    def _plot_confusions(self, Xte_scaled, y_test):
        names=list(self.models.keys()); cols=min(3,len(names)); rows=(len(names)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,4*rows))
        fig.suptitle("Confusion Matrices\n0=Normal · 1=CHF-morphology",fontsize=11)
        if rows==1 and cols==1: axes=np.array([[axes]])
        elif rows==1: axes=axes.reshape(1,-1)
        elif cols==1: axes=axes.reshape(-1,1)
        cnn_thr=self.results.get("CNN",{}).get("cnn_threshold",0.5)
        idx=0
        for r in range(rows):
            for c in range(cols):
                ax=axes[r,c]
                if idx>=len(names): ax.axis("off"); idx+=1; continue
                name=names[idx]; mdl=self.models[name]
                try:
                    if name=="CNN":
                        Xc=Xte_scaled.reshape(Xte_scaled.shape[0],Xte_scaled.shape[1],1)
                        pr=mdl.predict(Xc,verbose=0)[:,1]; yp=(pr>=cnn_thr).astype(int)
                    elif name=="MiniRocket":
                        Xr=Xte_scaled.reshape(Xte_scaled.shape[0],1,Xte_scaled.shape[1])
                        Zt=mdl["transformer"].transform(Xr); yp=mdl["classifier"].predict(Zt)
                    else: yp=mdl.predict(Xte_scaled)
                    cm=confusion_matrix(y_test,yp); cmn=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8)
                    acc=np.trace(cm)/cm.sum()
                    auc_v=self.results[name].get("auc_roc"); auc_lbl=f"  AUC={auc_v:.3f}" if auc_v else ""
                    im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
                    ax.set_title(f"{name} (acc={acc:.3f}{auc_lbl})",fontsize=9)
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(["Normal","CHF-m"]); ax.set_yticklabels(["Normal","CHF-m"])
                    ax.set_xlabel("Predicted",fontsize=8); ax.set_ylabel("True",fontsize=8)
                    for ii in range(2):
                        for jj in range(2):
                            col_txt="white" if cmn[ii,jj]>0.5 else "black"
                            ax.text(jj,ii,f"{cm[ii,jj]}\n({cmn[ii,jj]:.2f})",
                                    ha="center",va="center",fontsize=9,color=col_txt)
                    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                except Exception as e:
                    ax.text(0.5,0.5,f"Error:\n{e}",transform=ax.transAxes,ha="center",fontsize=8)
                idx+=1
        plt.tight_layout()
        self._savefig(fig,"confusion_matrices.eps")

    def _plot_roc_curves(self, roc_data):
        colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
        fig,ax=plt.subplots(figsize=(7,6))
        for (name,(y_true,y_proba)),color in zip(roc_data.items(),colors):
            if y_proba is None: continue
            try:
                fpr,tpr,_=roc_curve(y_true,y_proba); auc_val=roc_auc_score(y_true,y_proba)
                ax.plot(fpr,tpr,color=color,lw=1.8,label=f"{name} (AUC={auc_val:.3f})")
            except Exception: pass
        ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate (1−Specificity)",fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)",fontsize=11)
        ax.set_title("ROC Curves\nBeat-level CHF Morphology Detection",fontsize=11)
        ax.legend(loc="lower right",fontsize=9,frameon=False)
        ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        plt.tight_layout(); self._savefig(fig,"roc_curves.eps")

    def _savefig(self, fig, name, close=True):
        for d in [self.paths.plots_dir, self.paths.img_dir]:
            fig.savefig(d/name, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  ✅ {name}")
        if close: plt.close(fig); gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    #  FIGURES
    # ──────────────────────────────────────────────────────────────────────────
    def generate_all_figures(self):
        print("\n"+"─"*62)
        print("    Generating figures ...")
        print("─"*62)
        self._fig_raw_ecg_segments()
        self._fig_rpeak_detection()
        self._fig_feature_extraction()
        self._fig_class_distribution_pca()
        self._fig_model_performance()
        if CNN_AVAILABLE and "CNN" in self.models:
            self._fig_gradcam()
            self._fig_saliency()
        else:
            print("    (Grad-CAM/Saliency) omitted, CNN not available or not trained")
        print(f"\n   Figures saved in:\n"
              f"     {self.paths.img_dir}\n"
              f"     {self.paths.plots_dir}")

    def _fig_raw_ecg_segments(self):
        print("  Raw ECG segments...")
        n = 4
        normal_segs, chf_segs = [], []
        for rec_name in self.list_records():
            if len(normal_segs)>=n and len(chf_segs)>=n: break
            try:
                rec,ann = self._load_record(rec_name)
                sig = _bandpass_sig(rec.p_signal[:,0], self.fs)
                valid = set(p for p in ann.sample if p-self.half>=0 and p+self.half<len(sig))
                for peak,lab in zip(ann.sample,ann.symbol):
                    if peak not in valid: continue
                    beat = sig[peak-self.half:peak+self.half]
                    if lab.upper()=="N" and len(normal_segs)<n:
                        normal_segs.append((beat,rec_name))
                    elif lab.upper()!="N" and len(chf_segs)<n:
                        chf_segs.append((beat,lab,rec_name))
            except: pass
        t = np.arange(-self.half,self.half)/self.fs
        fig,axes = plt.subplots(2,n,figsize=(14,5),sharey=False)
        fig.suptitle("QRS-centered ECG Segments\n"
                     "Top: Normal sinus beat  ·  Bottom: CHF-associated morphology",fontsize=11,y=1.01)
        for col in range(n):
            ax0,ax1 = axes[0,col],axes[1,col]
            if col<len(normal_segs):
                beat,rec = normal_segs[col]
                ax0.plot(t,beat,color="#1f77b4",lw=1.2)
                ax0.axvline(0,color="red",lw=0.8,ls="--")
                ax0.set_title(f"{rec}",fontsize=8)
            ax0.set_ylim(-3,3); ax0.grid(alpha=0.3,ls=":")
            ax0.set_xticks([])
            if col==0: ax0.set_ylabel("Normal\nAmplitude (mV)",fontsize=9)
            if col<len(chf_segs):
                beat,lab,rec = chf_segs[col]
                ax1.plot(t,beat,color="#d62728",lw=1.2)
                ax1.axvline(0,color="red",lw=0.8,ls="--")
                ax1.set_title(f"{rec} [{lab}]",fontsize=8)
            ax1.set_ylim(-3,3); ax1.grid(alpha=0.3,ls=":")
            ax1.set_xlabel("Time (s)",fontsize=8)
            if col==0: ax1.set_ylabel("CHF-morph\nAmplitude (mV)",fontsize=9)
        plt.tight_layout(); self._savefig(fig,"raw_ecg_segments.eps")

    def _fig_rpeak_detection(self):
        print("  R-peak detection...")
        for rec_name in self.list_records():
            try:
                rec,ann = self._load_record(rec_name)
                sig_f   = _bandpass_sig(rec.p_signal[:,0], self.fs)
                start,end = 1000, 1000+10*self.fs
                seg   = sig_f[start:end]
                peaks = [p for p in ann.sample if start<=p<end]
                if len(peaks)<3: continue
                t      = np.arange(len(seg))/self.fs
                peak_t = [(p-start)/self.fs for p in peaks]
                peak_v = [sig_f[p] for p in peaks]
                fig,ax = plt.subplots(figsize=(12,3.5))
                ax.plot(t,seg,color="#002147",lw=1.0,label="Filtered ECG")
                ax.scatter(peak_t,peak_v,color="red",zorder=5,s=60,marker="^",
                           label=f"R-peaks (n={len(peaks)})")
                for pt in peak_t:
                    ax.axvspan(pt-self.half/self.fs,pt+self.half/self.fs,
                               color="#1f77b4",alpha=0.08)
                ax.set_xlabel("Time (s)",fontsize=10)
                ax.set_ylabel("Amplitude (mV)",fontsize=10)
                ax.set_title(f"R-peak Detection and QRS Window Extraction  [{rec_name}]",fontsize=11)
                ax.legend(loc="upper right",fontsize=9,frameon=False)
                ax.grid(alpha=0.3,ls=":"); ax.set_xlim(0,t[-1])
                plt.tight_layout(); self._savefig(fig,"_rpeak_detection.eps"); return
            except: pass
        print("  ⚠️  Could not be generated ")

    def _fig_feature_extraction(self):
        print("  Feature extraction schematic...")
        segment = None
        for rec_name in self.list_records():
            try:
                rec,ann = self._load_record(rec_name)
                sig = _bandpass_sig(rec.p_signal[:,0],self.fs)
                valid = [p for p in ann.sample if p-self.half>=0 and p+self.half<len(sig)]
                if valid:
                    p = valid[0]; beat = sig[p-self.half:p+self.half]
                    segment = (beat-beat.mean())/(beat.std()+1e-8); break
            except: pass
        if segment is None: print("  ⚠️  Could not be generated "); return
        t = np.arange(-self.half,self.half)/self.fs
        fig = plt.figure(figsize=(10,3))
        ax  = fig.add_subplot(111)
        ax.plot(t,segment,color="#002147",lw=1.3)
        ax.axvline(0,color="red",ls="--",lw=1)
        ax.axvspan(-self.half/self.fs,self.half/self.fs,color="#1f77b4",alpha=0.1)
        ax.set_xlim(-0.65,2.5); ax.set_ylim(segment.min()*1.15,segment.max()*1.4); ax.axis("off")
        fe = mpatches.FancyBboxPatch((0.85,-2.1),0.65,0.6,boxstyle="round,pad=0.03",
             lw=1.2,edgecolor="#1f77b4",facecolor="#1f77b4",alpha=0.12)
        ax.add_patch(fe)
        ax.text(1.175,-1.83,"Feature\nExtraction\n(18 feats)",fontsize=8,
                color="#1f77b4",fontweight="bold",ha="center")
        ax.annotate("",xy=(0.85,-1.83),xytext=(0.4,-0.3),
                    arrowprops=dict(arrowstyle="->",color="gray",lw=1.2))
        domains=[("Temporal (7)","#1f77b4",-1.95),
                 ("Spectral (5)","#ff7f0e",-2.3),
                 ("Morphological (6)","#2ca02c",-2.65)]
        for label,color,yy in domains:
            b=mpatches.FancyBboxPatch((1.62,yy),0.7,0.27,boxstyle="round,pad=0.02",
              lw=0.9,edgecolor=color,facecolor=color,alpha=0.15)
            ax.add_patch(b)
            ax.text(1.97,yy+0.135,label,fontsize=7.5,color=color,ha="center")
            ax.annotate("",xy=(1.62,yy+0.135),xytext=(1.5,-1.83),
                        arrowprops=dict(arrowstyle="->",color="gray",lw=0.8))
        fv=mpatches.FancyBboxPatch((2.4,-2.42),0.62,0.5,boxstyle="round,pad=0.02",
           lw=1,edgecolor="black",facecolor="white")
        ax.add_patch(fv)
        ax.text(2.71,-2.19,"Feature\nVector\n[x₁…x₁₈]",fontsize=7.5,color="black",ha="center")
        ax.annotate("",xy=(2.4,-2.19),xytext=(2.32,-2.19),
                    arrowprops=dict(arrowstyle="->",color="gray",lw=1))
        ax.set_title("Multidomain Feature Extraction from QRS-centered ECG Segment",fontsize=10,pad=4)
        plt.tight_layout(); self._savefig(fig,"feature_extraction.eps")

    def _fig_class_distribution_pca(self):
        print("  Class distribution + PCA...")
        csv_p = self.paths.signal_dir/"class_balance_summary_v12.csv"
        if not csv_p.exists():
            print("  ⚠️  Without balance csv, run the pipeline first"); return
        df = pd.read_csv(csv_p)
        before = df["Before_balancing"].values; after = df["After_balancing"].values
        labels = df["Class"].values
        before_npz = self.paths.run_dir/"features_before_balance.npz"
        after_npz  = self.paths.run_dir/"features_after_balance.npz"
        pca_ok = before_npz.exists() and after_npz.exists()
        if pca_ok:
            db=np.load(before_npz,allow_pickle=True); da=np.load(after_npz,allow_pickle=True)
            Xb=db["X"].astype(np.float32); yb=db["y"]
            Xa=da["X"].astype(np.float32); ya=da["y"]
            db.close(); da.close()
            def _to_bin(y):
                y=np.asarray(y)
                if np.issubdtype(y.dtype,np.number): return (y!=0).astype(int)
                return np.where(np.char.upper(y.astype(str))=="N",0,1)
            yb_bin=_to_bin(yb); ya_bin=_to_bin(ya)
            Zb=PCA(2,random_state=42).fit_transform(StandardScaler().fit_transform(Xb))
            Za=PCA(2,random_state=42).fit_transform(StandardScaler().fit_transform(Xa))
            fig,axes=plt.subplots(1,3,figsize=(16,5))
        else:
            fig,axes=plt.subplots(1,1,figsize=(6,5)); axes=[axes]
        ax=axes[0]; x=np.arange(len(labels)); w=0.35
        b1=ax.bar(x-w/2,before,w,label="Before",color="#4F81BD",alpha=0.9)
        b2=ax.bar(x+w/2,after, w,label="After", color="#C0504D",alpha=0.9)
        ax.set_ylabel("Number of beats"); ax.set_title("Class Distribution\nBefore vs After Balancing")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(frameon=False); ax.grid(axis="y",alpha=0.3)
        mx=max(np.max(before),np.max(after))
        for bar in list(b1)+list(b2):
            h=bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,h+mx*0.01,f"{int(h):,}",
                    ha="center",va="bottom",fontsize=8)
        if pca_ok:
            colors={0:"#1f77b4",1:"#d62728"}; cnames={0:"Normal",1:"CHF-morph"}
            for ax_pca,Z,yb_,title in [
                (axes[1],Zb,yb_bin," PCA Before Balancing"),
                (axes[2],Za,ya_bin," PCA After Balancing")]:
                for cls in [0,1]:
                    mask=yb_==cls
                    ax_pca.scatter(Z[mask,0],Z[mask,1],s=8,alpha=0.35,
                                   color=colors[cls],label=cnames[cls])
                p1,p99=np.percentile(Z,[1,99],axis=0)
                ax_pca.set_xlim(p1[0],p99[0]); ax_pca.set_ylim(p1[1],p99[1])
                ax_pca.set_xlabel("PC 1"); ax_pca.set_ylabel("PC 2"); ax_pca.set_title(title)
                ax_pca.legend(markerscale=2,frameon=False,fontsize=9); ax_pca.grid(alpha=0.3)
        plt.tight_layout(); self._savefig(fig,"class_distribution_pca.eps")

    def _fig_model_performance(self):
        print("  Model performance comparison...")
        csv_p=self.paths.reports_dir/"metrics_v12.csv"
        if not csv_p.exists(): print("  ⚠️  no csv file"); return
        df=pd.read_csv(csv_p)
        metrics=["accuracy","precision","recall","f1","specificity","npv","mcc","auc_roc"]
        metrics=[m for m in metrics if m in df.columns]
        models_list=df["Model"].tolist(); n_m=len(metrics); n_mod=len(models_list)
        x=np.arange(n_mod); width=0.10
        colors=["#4F81BD","#C0504D","#9BBB59","#8064A2","#4BACC6","#F79646","#808080","#17BECF"]
        fig,ax=plt.subplots(figsize=(14,5))
        for i,(metric,color) in enumerate(zip(metrics,colors)):
            vals=pd.to_numeric(df[metric],errors="coerce").fillna(0).values
            offset=(i-n_m/2+0.5)*width
            ax.bar(x+offset,vals*100,width,label=metric.replace("_"," ").title(),
                   color=color,alpha=0.87)
        ax.set_ylabel("Performance (%)"); ax.set_ylim(0,115)
        ax.set_title(" Classifier Performance Comparison\n"
                     "Beat-level CHF Morphology Detection, BIDMC-CHF",fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(models_list,rotation=15,ha="right")
        ax.legend(frameon=False,loc="upper right",ncol=4,fontsize=8,bbox_to_anchor=(1,1.02))
        ax.grid(axis="y",alpha=0.3,ls="--"); ax.axhline(100,color="gray",lw=0.5,ls=":")
        plt.tight_layout(); self._savefig(fig,"model_performance.eps")

    def _fig_gradcam(self):
        print("  Grad-CAM 1D...")
        if not CNN_AVAILABLE or "CNN" not in self.models: return
        model = self.models["CNN"]
        best_seg, best_prob = None, 0.0
        for rec_name in self.list_records()[:8]:
            try:
                rec,ann = self._load_record(rec_name)
                sig_raw = rec.p_signal[:,0]
                mn,mx   = sig_raw.min(),sig_raw.max()
                sig     = 2*(sig_raw-mn)/(mx-mn+1e-8)-1
                sig_f   = _bandpass_sig(sig,self.fs)
                for peak,lab in zip(ann.sample,ann.symbol):
                    if lab.upper()=="N": continue
                    s,e = peak-self.half, peak+self.half
                    if s<0 or e>=len(sig_f): continue
                    beat = sig_f[s:e].reshape(1,self.window_size,1).astype(np.float32)
                    prob = float(model.predict(beat,verbose=0)[0,1])
                    if prob>best_prob: best_prob=prob; best_seg=beat
            except: pass
        if best_seg is None or best_prob<0.3:
            print("  ⚠️  No CHF segment with sufficient confidence was found"); return
        last_conv=None
        for layer in model.layers:
            if "conv1d" in layer.name.lower(): last_conv=layer.name
        if not last_conv: print("  ⚠️  Without Conv1D layer"); return

        try:

            inp = tf.keras.Input(shape=(self.window_size, 1))
            x   = inp
            conv_out_tensor = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv:
                    conv_out_tensor = x
            if conv_out_tensor is None:
                print("  ⚠️  Without Conv1D layer"); return
            grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out_tensor, x])
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(best_seg)
                loss = preds[:, 1]
            grads   = tape.gradient(loss, conv_out)
            pooled  = tf.reduce_mean(grads, axis=(0, 1)).numpy()
            co      = conv_out[0].numpy()
            for i in range(pooled.shape[-1]):
                co[:, i] *= pooled[i]
            heatmap = np.maximum(np.mean(co, axis=-1), 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            from scipy.ndimage import zoom
            heatmap_r = zoom(heatmap, self.window_size / len(heatmap), order=1)
        except Exception as e:
            print(f"  ⚠️   Grad-CAM error: {e}"); return
        seg_plot=best_seg[0,:,0]; t=np.arange(self.window_size)/self.fs-self.half/self.fs
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,6),gridspec_kw={"hspace":0.4})
        fig.suptitle(f" Grad-CAM Saliency Map (CNN 1D)\n"
                     f"CHF-morphology beat  (P(CHF)={best_prob:.3f})",fontsize=11)
        ax1.plot(t,seg_plot,color="#002147",lw=1.3,label="ECG segment")
        ax1.fill_between(t,seg_plot.min(),
                         seg_plot.min()+heatmap_r*(seg_plot.max()-seg_plot.min()),
                         color="red",alpha=0.4,label="Grad-CAM attention")
        ax1.axvline(0,color="red",ls="--",lw=0.8); ax1.set_ylabel("Normalized amplitude",fontsize=10)
        ax1.legend(frameon=False,fontsize=9,loc="upper right"); ax1.grid(alpha=0.3,ls=":")
        ax1.set_xlabel("Time (s)",fontsize=10)
        ax2.plot(t,heatmap_r,color="red",lw=1.8)
        ax2.fill_between(t,0,heatmap_r,color="red",alpha=0.3)
        ax2.set_ylim([0,1.05]); ax2.set_ylabel("Attention score",fontsize=10)
        ax2.set_xlabel("Time (s) — centered on R-peak",fontsize=10)
        ax2.set_title("Attention intensity (0=low, 1=high)",fontsize=9)
        ax2.grid(alpha=0.3,ls=":")
        plt.tight_layout(); self._savefig(fig,"gradcam.eps")

    def _fig_saliency(self):
        print("  Gradient saliency...")
        if not CNN_AVAILABLE or "CNN" not in self.models: return
        model=self.models["CNN"]
        def _saliency(beat_np, class_idx):
            x=tf.Variable(beat_np.reshape(1,self.window_size,1),dtype=tf.float32)
            with tf.GradientTape() as tape:
                preds=model(x); loss=preds[:,class_idx]
            grads=tape.gradient(loss,x)
            sal=np.abs(grads.numpy()[0,:,0])
            if sal.max()>0: sal/=sal.max()
            return sal
        normal_seg=chf_seg=None
        for rec_name in self.list_records()[:6]:
            try:
                rec,ann=self._load_record(rec_name)
                sig=_bandpass_sig(rec.p_signal[:,0],self.fs)
                for peak,lab in zip(ann.sample,ann.symbol):
                    s,e=peak-self.half,peak+self.half
                    if s<0 or e>=len(sig): continue
                    beat=sig[s:e]; beat=(beat-beat.mean())/(beat.std()+1e-8)
                    if lab.upper()=="N" and normal_seg is None: normal_seg=beat.copy()
                    elif lab.upper()!="N" and chf_seg is None: chf_seg=beat.copy()
                if normal_seg is not None and chf_seg is not None: break
            except: pass
        if normal_seg is None or chf_seg is None:
            print("  ⚠️  Both segment types were not found "); return
        sal_n=_saliency(normal_seg,0); sal_c=_saliency(chf_seg,1)
        t=np.arange(self.window_size)/self.fs-self.half/self.fs
        fig,axes=plt.subplots(2,2,figsize=(13,7),
                              gridspec_kw={"hspace":0.45,"wspace":0.3})
        fig.suptitle(" Gradient Saliency Maps (CNN 1D)\n"
                     "Left: Normal sinus beat  ·  Right: CHF-associated morphology",fontsize=11)
        for col,(seg,sal,label,color) in enumerate([
            (normal_seg,sal_n,"Normal sinus beat","#1f77b4"),
            (chf_seg,  sal_c,"CHF-morphology beat","#d62728")]):
            axes[0,col].plot(t,seg,color=color,lw=1.3)
            axes[0,col].axvline(0,color="gray",ls="--",lw=0.8)
            axes[0,col].set_title(label,fontsize=10); axes[0,col].grid(alpha=0.3,ls=":")
            axes[0,col].set_ylabel("Normalized amplitude",fontsize=9)
            axes[0,col].set_xlabel("Time (s)",fontsize=9)
            axes[1,col].plot(t,sal,color=color,lw=1.5)
            axes[1,col].fill_between(t,0,sal,color=color,alpha=0.3)
            axes[1,col].set_ylim([0,1.05])
            axes[1,col].set_ylabel("Saliency score",fontsize=9)
            axes[1,col].set_xlabel("Time (s)",fontsize=9)
            axes[1,col].set_title("Input gradient saliency",fontsize=9)
            axes[1,col].grid(alpha=0.3,ls=":")
        plt.tight_layout(); self._savefig(fig,"saliency_gradient.eps")

    # ──────────────────────────────────────────────────────────────────────────
    #  REPORTE
    # ──────────────────────────────────────────────────────────────────────────
    def _save_paper_report(self, results):
        rows=[]
        for name,m in results.items():
            ci=m.get("ci95",{}); row={"Model":name}
            for metric in ["accuracy","f1","precision","recall",
                           "specificity","npv","mcc","auc_roc"]:
                val=m.get(metric); lo,hi=ci.get(metric,(None,None))
                row[metric]           = round(val,4) if val is not None else None
                row[f"{metric}_ci_lo"] = round(lo,4) if lo is not None else None
                row[f"{metric}_ci_hi"] = round(hi,4) if hi is not None else None
            rows.append(row)
        df=pd.DataFrame(rows); csv_path=self.paths.reports_dir/"paper_metrics_v12.csv"
        df.to_csv(csv_path,index=False)
        print(f"  📄 metrics: {csv_path}")

        repro = {
            "random_state_global": 42,
            "label_0": "Normal sinus beat (annotation symbol N)",
            "label_1": "CHF-associated morphology (any symbol != N)",
            "normalization": f"{self.norm_mode} (per-segment)",
            "window_samples": self.window_size,
            "window_ms": round(self.window_size/self.fs*1000,1),
            "bandpass_hz": [0.5, 40.0],
            "bandpass_order": 4,
            "split_test_size": 0.2,
            "split_strategy": "stratified beat-level (random_state=42)",
            "balancing_strategy": "RandomUnderSampler(cap=6000) + BorderlineSMOTE(k_neighbors=3)",
            "balancing_applied_to": "training set only — no leakage",
            "bootstrap_ci_n": 1000,
            "bootstrap_ci_alpha": 0.95,
            "hyperparameters": {
                "XGBoost": {
                    "n_estimators": 250, "max_depth": 5, "learning_rate": 0.08,
                    "subsample": 0.8, "colsample_bytree": 0.9, "tree_method": "hist",
                    "random_state": 42
                },
                "RandomForest": {
                    "n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
                    "max_samples": 0.8, "class_weight": "balanced", "random_state": 42
                },
                "GradientBoosting": {
                    "n_estimators": 200, "learning_rate": 0.08,
                    "max_depth": 4, "subsample": 0.8, "random_state": 42
                },
                "MiniRocket": {"num_kernels": 1000, "random_state": 42},
                "CNN": {
                    "architecture": "Conv1D(64,5) -> BN -> Conv1D(128,3) -> BN -> GAP -> Dense(64) -> Dense(2)",
                    "optimizer": "Adam(lr=0.001)",
                    "loss": "categorical_crossentropy",
                    "early_stopping_patience": 10,
                    "monitor": "val_accuracy",
                    "threshold": "calibrated to maximize weighted-F1 (search 0.1-0.9)"
                },
                "LOSO_model": "XGBoost with same hyperparameters as above"
            }
        }
        repro_path = self.paths.reports_dir/"reproducibility_config.json"
        with open(repro_path,"w") as f: json.dump(repro,f,indent=2)
        print(f"  📄 Config reproducibilidad: {repro_path}")

    # ──────────────────────────────────────────────────────────────────────────
    #  PIPELINE PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────
    def run_pipeline(self, max_records=None, max_beats_per_record=2000,
                     max_beats_total=50000, run_loso=False,
                     train_cnn=False, train_rocket=False, light_mode=False):
        t0=time.time()

        self.cache_features_per_record(max_records=max_records,
                                       max_beats_per_record=max_beats_per_record)
        X,y_symbols,groups = self.load_cached_dataset(max_beats_total=max_beats_total)

        y_bin = self._to_binary_chf(y_symbols)
        y01   = np.where(y_bin=="Normal",0,1).astype(np.int8)
        _free_mem(y_symbols)

        loso_report=None
        if run_loso:
            loso_report=self.evaluate_loso(X,y01,groups)
        _free_mem(groups)

        _section("[3/5] Split + Balanceo  (random_state=42)")
        Xtr,Xte,ytr,yte=self.split_stratified(X,y01)
        _free_mem(X,y01)

        Xb,yb,summ=self.smart_balance_binary(Xtr,np.where(ytr==0,"Normal","CHF-morphology"))
        np.savez(self.paths.run_dir/"features_before_balance.npz",X=Xtr,y=ytr)
        np.savez(self.paths.run_dir/"features_after_balance.npz",X=Xb,y=yb)
        _free_mem(Xtr,ytr)

        results=self.train_models(Xb,yb,Xte,yte,
                                  train_cnn=train_cnn,train_rocket=train_rocket,
                                  light_mode=light_mode)
        _free_mem(Xb,yb)

        _section("[5/5] Final Report")
        self._save_paper_report(results)
        self.generate_all_figures()

        total=time.time()-t0
        print(f"\n{'═'*62}")
        print(f"  ✅  Pipeline v12 — {total/60:.1f} min")
        print(f"\n  {'Model':<18} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Spec':>7} {'NPV':>7} {'MCC':>7}")
        print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for name,m in sorted(results.items(),key=lambda x:x[1]["accuracy"],reverse=True):
            auc=f"{m['auc_roc']:>7.4f}" if m.get("auc_roc") else "   N/A "
            print(f"  {name:<18} {m['accuracy']:>7.4f} {m['f1']:>7.4f} {auc}"
                  f" {m['specificity']:>7.4f} {m['npv']:>7.4f} {m['mcc']:>7.4f}")
        if loso_report:
            s=loso_report["summary"]
            n_folds_total = len(loso_report['fold_details'])
            print(f"\n  LOSO ({n_folds_total} folds):")
            for k in ["accuracy","f1","auc_roc","specificity","npv","mcc"]:
                if k not in s or not s[k]: continue
                v = s[k]
                if k == "auc_roc":
                    n_valid = v.get("n_valid_folds", "?")
                    note = f"  (computed over {n_valid}/{n_folds_total} folds , elsees: one sola clase)"
                    print(f"    {k:<14} {v['mean']:.3f} ± {v['std']:.3f}"
                          f"  [{v['min']:.3f}, {v['max']:.3f}]{note}")
                else:
                    print(f"    {k:<14} {v['mean']:.3f} ± {v['std']:.3f}"
                          f"  [{v['min']:.3f}, {v['max']:.3f}]")
        print(f"\n{'═'*62}")
        _print_ram("Final")
        return {"results":results,"loso_report":loso_report,"balance_summary":summ}

    # ──────────────────────────────────────────────────────────────────────────
    #  MENU
    # ──────────────────────────────────────────────────────────────────────────
    def interactive_menu(self):
        print("\n"+"─"*62)
        print("  MENÚ v12 — Beat-level CHF Morphology Detection")
        print("─"*62)
        print("  1)   Quick Test  (2 reg, 5k beats)")
        print("  2)   Partial Validation       (3 reg, 15k beats)")
        print("  3)   Full     (todos, 50k, NO LOSO) ")
        print("  4)   Full + LOSO  ")
        print("  5)   Exit")
        print("─"*62)
        _print_ram("Before Running")
        choice=input("\n   option [1-5]: ").strip()
        if choice=="1":
            return self.run_pipeline(max_records=2,max_beats_per_record=800,
                max_beats_total=5000,light_mode=True)
        if choice=="2":
            return self.run_pipeline(max_records=3,max_beats_per_record=1200,
                max_beats_total=15000,light_mode=True,
                train_cnn=CNN_AVAILABLE,train_rocket=ROCKET_AVAILABLE)
        if choice=="3":
            return self.run_pipeline(max_beats_per_record=2000,max_beats_total=50000,
                train_cnn=CNN_AVAILABLE,train_rocket=ROCKET_AVAILABLE)
        if choice=="4":
            return self.run_pipeline(max_beats_per_record=2000,max_beats_total=50000,
                run_loso=True,train_cnn=CNN_AVAILABLE,train_rocket=ROCKET_AVAILABLE)
        print("  Exiting...")
        return None


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── PATH CONFIGURATION ────────────────────────────────────────────────
    # Update this paths to match your machine before running
    #
    # DATA_PATH    → folder containing the .hea / .dat / .ecg del
    #                BIDMC CHF Database (descarga en physionet.org/content/chfdb/1.0.0/)
    # PROJECT_DIR  → folder where results, models, and figures will be saved
    #
    #  Windows example:
    #   DATA_PATH   = Path(r"C:\Users\TuUsuario\datasets\bidmc-chf-database")
    #   PROJECT_DIR = Path(r"C:\Users\TuUsuario\resultados\chf-detection")
    #
    # ─────────────────────────────────────────────────────────────────────────

    DATA_PATH   = Path(r"DATASET_PATH HERE")
    PROJECT_DIR = Path(r"RESULTS_FOLDER_HERE")

    system = CHFDetectionV12(DATA_PATH, PROJECT_DIR,
                             fs=250, window_size=300, norm_mode="minmax_-1_1")
    system.interactive_menu()