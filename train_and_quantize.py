#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, random, hashlib, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, precision_recall_curve)

SEED=42
random.seed(SEED); np.random.seed(SEED)

# -----------------------------
# Data Load / Cleanup
# -----------------------------
atk_files = [
    './data/decimal/decimal_DoS.csv',
    './data/decimal/decimal_spoofing-GAS.csv',
    './data/decimal/decimal_spoofing-RPM.csv',
    './data/decimal/decimal_spoofing-SPEED.csv',
    './data/decimal/decimal_spoofing-STEERING_WHEEL.csv',
]
benign_file = './data/decimal/decimal_benign.csv'
FEATURES = [f"DATA_{i}" for i in range(8)]

def read_and_clean(path_or_df):
    df = pd.read_csv(path_or_df) if isinstance(path_or_df,str) else path_or_df.copy()
    df = df.drop(columns=['category','specific_class'], errors='ignore')
    need = FEATURES + ['label']
    missing = [c for c in need if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0,'0':0,'attack':1,'1':1})
    if 'ID' not in df.columns:
        # Fixed length group id to prevent leakage of identical rows
        df['ID'] = 'gid_' + (np.arange(len(df)) // 50).astype(str)
    return df

atk_df = pd.concat([pd.read_csv(f).assign(label=1) for f in atk_files], ignore_index=True)
atk_df = read_and_clean(atk_df)
benign_df = read_and_clean(pd.read_csv(benign_file).assign(label=0))

df = pd.concat([atk_df, benign_df], ignore_index=True).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

X = df[FEATURES].values.astype('float32')
y = df['label'].values.astype('int32')

# Prevent leakage using group hash
def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()
groups = np.array([row_hash(r) for r in X])

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"All data shape: X={X.shape}, y={y.shape}, positives={(y==1).sum()} ({100*(y==1).mean():.3f}%)")
print(f"Train/Test sizes: {X_train_raw.shape}/{X_test_raw.shape}")

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# -----------------------------
# Utility: Evaluation + Optimal Threshold
# -----------------------------
def eval_with_thresholds(y_true, scores, thr=0.5, label="Model"):
    # Default threshold
    pred = (scores >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, scores)
    
    # Maximum F1 threshold
    p_curve, r_curve, t_curve = precision_recall_curve(y_true, scores)
    f1_curve = (2*p_curve*r_curve) / (p_curve + r_curve + 1e-12)
    best_idx = int(np.nanargmax(f1_curve))
    bestF1 = float(f1_curve[best_idx])
    
    # precision_recall_curve lacks the last point threshold → correct it
    if best_idx >= len(t_curve): best_idx = len(t_curve)-1
    best_thr = float(t_curve[max(0,best_idx)])
    
    # Recalculate
    pred_best = (scores >= best_thr).astype(int)
    acc_b = accuracy_score(y_true, pred_best)
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(y_true, pred_best, average='binary', zero_division=0)
    
    return {
        "label": label,
        "thr_default": float(thr),
        "acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "auc": float(auc),
        "thr_bestF1": best_thr,
        "acc_b": float(acc_b), "prec_b": float(prec_b), "rec_b": float(rec_b), "f1_b": float(f1_b)
    }

os.makedirs("artifacts", exist_ok=True)

# -----------------------------
# 1) Logistic Regression (balanced)
# -----------------------------
logreg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED)
logreg.fit(X_train, y_train)
scores_lr = logreg.predict_proba(X_test)[:,1]
res_lr = eval_with_thresholds(y_test, scores_lr, thr=0.5, label="LogReg(balanced)")
print(f"[LogReg]  Acc={res_lr['acc']:.4f} Prec={res_lr['prec']:.4f} Rec={res_lr['rec']:.4f} F1={res_lr['f1']:.4f} AUC={res_lr['auc']:.4f}")
print(f"[LogReg-bestF1] Thr={res_lr['thr_bestF1']:.6f} F1={res_lr['f1_b']:.4f}")

# Save parameters (w,b) for the board
# coef_: (1, n_features), intercept_: (1,)
w = logreg.coef_.ravel().astype('float32')
b = np.float32(logreg.intercept_.ravel()[0])
np.savez("artifacts/logreg_params.npz", w=w, b=b)

# -----------------------------
# 2) Gaussian Naive Bayes
# -----------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
scores_gnb = gnb.predict_proba(X_test)[:,1]
res_gnb = eval_with_thresholds(y_test, scores_gnb, thr=0.5, label="GaussianNB")
print(f"[GNB]     Acc={res_gnb['acc']:.4f} Prec={res_gnb['prec']:.4f} Rec={res_gnb['rec']:.4f} F1={res_gnb['f1']:.4f} AUC={res_gnb['auc']:.4f}")
print(f"[GNB-bestF1] Thr={res_gnb['thr_bestF1']:.6f} F1={res_gnb['f1_b']:.4f}")

# Save parameters (mean/variance/prior per class) for the board
# class_count_, class_prior_, theta_(C,F), var_(C,F)
np.savez("artifacts/gnb_params.npz",
         mu=gnb.theta_.astype('float32'),
         var=gnb.var_.astype('float32'),
         prior=gnb.class_prior_.astype('float32'))

# -----------------------------
# 3) ExtraTrees (Recommended setting: shallow and few estimators)
# -----------------------------
et = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=8,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)
et.fit(X_train, y_train)
scores_et = et.predict_proba(X_test)[:,1]
res_et = eval_with_thresholds(y_test, scores_et, thr=0.5, label="ExtraTrees(d=8,100)")
print(f"[ExtraTrees]  Acc={res_et['acc']:.4f} Prec={res_et['prec']:.4f} Rec={res_et['rec']:.4f} F1={res_et['f1']:.4f} AUC={res_et['auc']:.4f}")
print(f"[ExtraTrees-bestF1] Thr={res_et['thr_bestF1']:.6f} F1={res_et['f1_b']:.4f}")

# -----------------------------
# Save Artifacts
# -----------------------------
# 0) Scaler
np.savez("artifacts/scaler.npz",
         mean_=scaler.mean_.astype('float32'),
         scale_=scaler.scale_.astype('float32'))

# 1) Model Objects (for Desktop/Board Python inference)
import joblib
joblib.dump(logreg,      "artifacts/logreg.joblib")
joblib.dump(gnb,         "artifacts/gnb.joblib")
joblib.dump(et,          "artifacts/extratrees.joblib")

# 2) Thresholds/Metadata
thresholds = {
    "features": FEATURES,
    "logreg":   {"thr_default": 0.5, "thr_bestF1": res_lr["thr_bestF1"]},
    "gnb":      {"thr_default": 0.5, "thr_bestF1": res_gnb["thr_bestF1"]},
    "extratrees":{"thr_default": 0.5, "thr_bestF1": res_et["thr_bestF1"]},
    "notes": "All thresholds computed on current test split; reuse as fixed thresholds on board for inference."
}
with open("artifacts/thresholds.json","w") as f:
    json.dump(thresholds, f, indent=2)

# 3) Performance Summary CSV
rows = []
for r in (res_lr, res_gnb, res_et):
    rows.append({
        "model": r["label"],
        "thr_default": r["thr_default"],
        "acc": r["acc"], "prec": r["prec"], "rec": r["rec"], "f1": r["f1"], "auc": r["auc"],
        "thr_bestF1": r["thr_bestF1"],
        "acc_bestF1": r["acc_b"], "prec_bestF1": r["prec_b"], "rec_bestF1": r["rec_b"], "f1_bestF1": r["f1_b"]
    })
pd.DataFrame(rows).to_csv("artifacts/metrics.csv", index=False)

# 4) ExtraTrees feature importances
pd.DataFrame({"feature": FEATURES, "importance": et.feature_importances_})\
  .sort_values("importance", ascending=False)\
  .to_csv("artifacts/extratrees_feature_importances.csv", index=False)

print("✅ Saved artifacts in ./artifacts")
print(" - scaler.npz, thresholds.json, metrics.csv")
print(" - logreg.joblib, gnb.joblib, extratrees.joblib")
print(" - logreg_params.npz (w,b), gnb_params.npz (mu,var,prior)")
print(" - extratrees_feature_importances.csv")