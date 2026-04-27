#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAN-IDS: Binary classification of Decimal dataset (DATA_0..7)
- Leakage Prevention: Group-based split (prevents identical rows from appearing in both train and test)
- Scaling: Fit on Train only, transform on Test
- Imbalance Correction: class_weight='balanced' (+ Keras class_weight)
- Models: MLP(Keras), LogisticRegression, Logistic-L1(saga), LinearSVC, SGD, Ridge, Perceptron, PassiveAggressive,
          DecisionTree(shallow), ExtraTrees(shallow), GaussianNB, ComplementNB(binned), BernoulliNB(binarized)
- Metrics: Acc, Prec, Rec, F1, ROC-AUC + Confusion Matrix / Threshold tuning (max F1)
"""

import os, random, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Fix to CPU (comment out for GPU comparison)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, PassiveAggressiveClassifier,
    Perceptron, RidgeClassifier
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.dummy import DummyClassifier

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 0) Fix Random Seed
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------
# 1) Data Load / Cleanup
# -----------------------------
atk_files = [
    './data/decimal/decimal_DoS.csv',
    './data/decimal/decimal_spoofing-GAS.csv',
    './data/decimal/decimal_spoofing-RPM.csv',
    './data/decimal/decimal_spoofing-SPEED.csv',
    './data/decimal/decimal_spoofing-STEERING_WHEEL.csv',
]
benign_file = './data/decimal/decimal_benign.csv'

def read_and_clean(path_or_df):
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    # Preserve ID, drop other unnecessary columns
    df = df.drop(columns=['category','specific_class'], errors='ignore')

    # Check required columns (ID is optional)
    use_cols = [f'DATA_{i}' for i in range(8)] + ['label']
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type casting
    for c in [f'DATA_{i}' for i in range(8)]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    # Normalize labels (Attack:1, Benign:0)
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0, '0':0, 'attack':1, '1':1})

    # Generate temporary ID if none exists (for sequence grouping)
    if 'ID' not in df.columns:
        df['ID'] = 'gid_' + (np.arange(len(df)) // 50).astype(str)  # Group every 50 records
    return df

# Load Attack/Benign CSV and enforce labels
atk_df = pd.concat([pd.read_csv(f) for f in atk_files], ignore_index=True)
atk_df = atk_df.assign(label=1)
atk_df = read_and_clean(atk_df)

benign_df = read_and_clean(pd.read_csv(benign_file).assign(label=0))

# Merge → Remove NaN → Shuffle
df = pd.concat([atk_df, benign_df], ignore_index=True).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

FEATURES = [f'DATA_{i}' for i in range(8)]
X = df[FEATURES].values.astype('float32')
y = df['label'].values.astype('int32')

print(f"All data shape: X={X.shape}, y={y.shape}, positives={y.sum()} ({y.mean():.3%})")

# -----------------------------
# 2) Group-based Split (Group identical rows)
# -----------------------------
def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()

groups = np.array([row_hash(r) for r in X])

# 80/20 split
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Keep raw features for NB discretization
X_train_raw, X_test_raw = X[train_idx].copy(), X[test_idx].copy()

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
grp_train, grp_test = groups[train_idx], groups[test_idx]

# Check for cross-leakage (should be 0)
inter = set(grp_train).intersection(set(grp_test))
print(f"Group overlap between train/test: {len(inter)} (should be 0)")

# Check class distribution
print("Train dist:", Counter(y_train))
print("Test  dist:", Counter(y_test))

# -----------------------------
# 3) Scaling (Prevent leakage: fit on train → transform on test)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# 4) Evaluation Metrics/Reporting + Threshold Tuning Utility
# -----------------------------
def metrics_report(name, y_true, y_prob=None, y_pred=None, plot_cm=True, thr=None):
    if y_pred is None:
        if y_prob is None:
            raise ValueError("Either y_prob or y_pred is required.")
        # Handle both decision_function scores (negative/positive) or probabilities ([0,1])
        default_thr = 0.0 if (y_prob.min() < 0 or y_prob.max() > 1) else 0.5
        use_thr = default_thr if thr is None else thr
        y_pred = (y_prob >= use_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    try:
        aucv = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    except Exception:
        aucv = np.nan

    print(f"[{name}]  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={aucv:.4f}")

    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Benign","Attack"]).plot(cmap=plt.cm.Blues)
        plt.title(f"{name} - Confusion Matrix")
        plt.grid(False)
        plt.show()

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=aucv)

def best_threshold_by_f1(y_true, y_prob):
    """Find the point of maximum F1 using precision_recall_curve (allows both scores/probs)."""
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2*p*r / (p+r+1e-9)
    idx = np.argmax(f1s[:-1])  # length of thr is 1 less than p/r
    return dict(threshold=float(thr[idx]), precision=float(p[idx]), recall=float(r[idx]), f1=float(f1s[idx]))

def get_scores(clf, X):
    """Consistently obtain score vectors for binary classification from the model."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]           # Probability
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)             # Margin/Score
    return clf.predict(X).astype(float)             # Exception (Not recommended)

def tune_and_report(name, y_true, scores):
    bt = best_threshold_by_f1(y_true, scores)
    print(f"Best threshold by F1 ({name}): {bt}")
    metrics_report(f"{name}-bestF1", y_true, y_prob=scores, thr=bt["threshold"])

# -----------------------------
# 5) Dummy Baseline
# -----------------------------
dum = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_train, y_train)
y_prob_dum = dum.predict_proba(X_test)[:,1] if hasattr(dum, "predict_proba") else None
y_pred_dum = dum.predict(X_test)
print("\n== Dummy (most_frequent) ==")
metrics_report("Dummy", y_test, y_prob=y_prob_dum, y_pred=y_pred_dum)

# -----------------------------
# 6) Keras MLP (Not LSTM!)
# -----------------------------
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {int(c): w for c, w in zip(classes, cw)}

mlp = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
_ = mlp.fit(
    X_train, y_train,
    epochs=50, batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[es], verbose=0
)

y_prob_mlp = mlp.predict(X_test, verbose=0).ravel()
print("\n== MLP (Keras, balanced) ==")
metrics_report("MLP(Keras)", y_test, y_prob=y_prob_mlp)   # Default 0.5
tune_and_report("MLP(Keras)", y_test, y_prob_mlp)         # ★ Optimal F1 threshold
# (Optional) ROC/PR
fpr, tpr, _ = roc_curve(y_test, y_prob_mlp)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("MLP ROC Curve"); plt.grid(True); plt.show()
precv, recv, _ = precision_recall_curve(y_test, y_prob_mlp)
plt.plot(recv, precv); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("MLP PR Curve")
plt.grid(True); plt.show()

# -----------------------------
# 7) Traditional ML Models (Class Weight + Same Scaling)
# -----------------------------
print("\n== Logistic Regression (balanced) ==")
logreg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED)
logreg.fit(X_train, y_train)
scores_lr = get_scores(logreg, X_test)
metrics_report("LogReg(balanced)", y_test, y_prob=scores_lr)   # Default 0.5 or 0.0
tune_and_report("LogReg(balanced)", y_test, scores_lr)         # ★ Optimal F1 threshold

# --- (A) LogReg coefficients: CSV + Figure (Feature Importance)
os.makedirs("figures", exist_ok=True)
FEATURE_NAMES = FEATURES[:]
coefs = np.ravel(logreg.coef_)                 # shape (n_features,)
df_coef = pd.DataFrame({"feature": FEATURE_NAMES, "coef": coefs})
df_coef.to_csv("logreg_coefficients.csv", index=False)
print("✅ Saved: logreg_coefficients.csv")

# Descending order by absolute value, then horizontal bar chart
order = np.argsort(-np.abs(coefs))
feat_sorted = [FEATURE_NAMES[i] for i in order]
coef_sorted = coefs[order]
abs_sorted  = np.abs(coef_sorted)

plt.figure(figsize=(6, 3.8))
y_ax = np.arange(len(abs_sorted))
plt.barh(y_ax, abs_sorted)
plt.yticks(y_ax, feat_sorted)
plt.xlabel("Absolute Coefficient Magnitude")
plt.title("Balanced Logistic Regression — Coefficient Magnitudes")
for i, v in enumerate(abs_sorted):
    plt.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("figures/logreg_coefficients.pdf", dpi=300)
plt.close()
print("✅ Saved: figures/logreg_coefficients.pdf")

print("\n== Logistic Regression L1(saga, balanced) ==")
logreg_l1 = LogisticRegression(
    solver='saga', penalty='l1', C=1.0, class_weight='balanced',
    random_state=SEED, max_iter=2000, n_jobs=-1
)
logreg_l1.fit(X_train, y_train)
scores_l1 = get_scores(logreg_l1, X_test)
metrics_report("LogReg-L1(saga, balanced)", y_test, y_prob=scores_l1)
tune_and_report("LogReg-L1(saga, balanced)", y_test, scores_l1)

print("\n== LinearSVC (balanced) ==")
lsvc = LinearSVC(class_weight='balanced', max_iter=5000, random_state=SEED)
lsvc.fit(X_train, y_train)
scores_svc = get_scores(lsvc, X_test)
metrics_report("LinearSVC(balanced)", y_test, y_prob=scores_svc)   # default thr=0.0
tune_and_report("LinearSVC(balanced)", y_test, scores_svc)

print("\n== SGDClassifier hinge (balanced) ==")
sgd = SGDClassifier(loss='hinge', class_weight='balanced', max_iter=2000, tol=1e-3, random_state=SEED)
sgd.fit(X_train, y_train)
scores_sgd = get_scores(sgd, X_test)
metrics_report("SGD-hinge(balanced)", y_test, y_prob=scores_sgd)
tune_and_report("SGD-hinge(balanced)", y_test, scores_sgd)

print("\n== RidgeClassifier (balanced) ==")
ridge = RidgeClassifier(class_weight='balanced', random_state=SEED)
ridge.fit(X_train, y_train)
scores_rdg = get_scores(ridge, X_test)
metrics_report("RidgeClassifier(balanced)", y_test, y_prob=scores_rdg)
tune_and_report("RidgeClassifier(balanced)", y_test, scores_rdg)

print("\n== Perceptron (balanced) ==")
perc = Perceptron(penalty='l2', class_weight='balanced', random_state=SEED, max_iter=2000, tol=1e-3)
perc.fit(X_train, y_train)
scores_perc = get_scores(perc, X_test)
metrics_report("Perceptron(balanced)", y_test, y_prob=scores_perc)
tune_and_report("Perceptron(balanced)", y_test, scores_perc)

print("\n== PassiveAggressive (balanced) ==")
pa = PassiveAggressiveClassifier(loss='hinge', C=1.0, class_weight='balanced', random_state=SEED, max_iter=2000, tol=1e-3)
pa.fit(X_train, y_train)
scores_pa = get_scores(pa, X_test)
metrics_report("PassiveAggressive(balanced)", y_test, y_prob=scores_pa)
tune_and_report("PassiveAggressive(balanced)", y_test, scores_pa)

print("\n== DecisionTree (shallow, balanced) ==")
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=SEED)
dt.fit(X_train, y_train)
scores_dt = get_scores(dt, X_test)
metrics_report("DecisionTree(shallow)", y_test, y_prob=scores_dt)
tune_and_report("DecisionTree(shallow)", y_test, scores_dt)

print("\n== ExtraTrees (shallow&few, balanced) ==")
et = ExtraTreesClassifier(
    n_estimators=64, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED, n_jobs=-1
)
et.fit(X_train, y_train)
scores_et = get_scores(et, X_test)
metrics_report("ExtraTrees(shallow)", y_test, y_prob=scores_et)
tune_and_report("ExtraTrees(shallow)", y_test, scores_et)

print("\n== GaussianNB ==")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
scores_gnb = get_scores(gnb, X_test)
metrics_report("GaussianNB", y_test, y_prob=scores_gnb)
tune_and_report("GaussianNB", y_test, scores_gnb)

# -----------------------------
# 8) Naive Bayes Variants (ComplementNB, BernoulliNB) — Using binning
# -----------------------------
print("\n== NB variants with KBinsDiscretizer (on raw decimal) ==")
binner = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform')
X_train_bin = binner.fit_transform(X_train_raw)
X_test_bin  = binner.transform(X_test_raw)

print("\n-- ComplementNB --")
cnb = ComplementNB()
cnb.fit(X_train_bin, y_train)
scores_cnb = get_scores(cnb, X_test_bin)
metrics_report("ComplementNB(binned)", y_test, y_prob=scores_cnb)
tune_and_report("ComplementNB(binned)", y_test, scores_cnb)

print("\n-- BernoulliNB --")
binner2 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')  # 2-bin binarization
X_train_bin2 = binner2.fit_transform(X_train_raw)
X_test_bin2  = binner2.transform(X_test_raw)

bnb = BernoulliNB()
bnb.fit(X_train_bin2, y_train)
scores_bnb = get_scores(bnb, X_test_bin2)
metrics_report("BernoulliNB(binarized)", y_test, y_prob=scores_bnb)
tune_and_report("BernoulliNB(binarized)", y_test, scores_bnb)

# ============================
# 10) LSTM (Sequence-based)
# ============================
print("\n== LSTM (sequence) ==")

# A) Sequence Builder: Create windows with continuous timesteps within the same ID
def build_sequences_df_basic(df, features, label_col, group_col='ID',
                             timesteps=10, step=1, timestamp_cols=('timestamp','time','ts')):
    ts_col = next((c for c in timestamp_cols if c in df.columns), None)
    if ts_col is not None:
        df_sorted = df.sort_values([group_col, ts_col], kind='mergesort')
    else:
        df_sorted = df.sort_values([group_col], kind='mergesort')

    X_seq, y_seq, groups = [], [], []
    for gid, g in df_sorted.groupby(group_col, sort=False):
        Xg = g[features].values.astype('float32')
        yg = g[label_col].values.astype('int32')
        if len(g) < timesteps:
            continue
        for i in range(0, len(g) - timesteps + 1, step):
            X_seq.append(Xg[i:i+timesteps])
            y_seq.append(int(yg[i:i+timesteps].max()))  # any-rule
            groups.append(gid)
    return np.array(X_seq), np.array(y_seq), np.array(groups)

TIMESTEPS = 10
SEQ_FEATURES = [f'DATA_{i}' for i in range(8)]
X_seq, y_seq, groups_seq = build_sequences_df_basic(df, SEQ_FEATURES, 'label', group_col='ID',
                                                    timesteps=TIMESTEPS, step=1)

print(f"Seq shapes: X_seq={X_seq.shape}, y_seq={y_seq.shape}, #groups={len(np.unique(groups_seq))}")

# B) Group-based Split
gss_seq = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tr_idx, te_idx = next(gss_seq.split(X_seq, y_seq, groups=groups_seq))
Xtr, Xte = X_seq[tr_idx], X_seq[te_idx]
ytr, yte = y_seq[tr_idx], y_seq[te_idx]
grp_tr, grp_te = groups_seq[tr_idx], groups_seq[te_idx]
assert len(set(grp_tr).intersection(set(grp_te))) == 0, "LSTM group leakage!"
print(f"LSTM split: train={Xtr.shape}, test={Xte.shape}")

# C) Sequence Scaling
scaler_seq = StandardScaler()
Xtr_flat = Xtr.reshape(-1, Xtr.shape[2])
Xte_flat = Xte.reshape(-1, Xte.shape[2])
Xtr_flat = scaler_seq.fit_transform(Xtr_flat)
Xte_flat = scaler_seq.transform(Xte_flat)
Xtr = Xtr_flat.reshape(Xtr.shape)
Xte = Xte_flat.reshape(Xte.shape)

# D) Class Weights
classes = np.unique(ytr)
cw = compute_class_weight('balanced', classes=classes, y=ytr)
class_weight_seq = {int(c): w for c, w in zip(classes, cw)}

# E) LSTM Model (Lightweight)
from tensorflow.keras.layers import LSTM, Bidirectional

tf.keras.utils.set_random_seed(SEED)
lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(Xtr.shape[1], Xtr.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
_ = lstm.fit(Xtr, ytr, epochs=30, batch_size=256, validation_split=0.2,
             class_weight=class_weight_seq, callbacks=[es2], verbose=0)

# F) Evaluation + Threshold Tuning
y_prob_lstm = lstm.predict(Xte, verbose=0).ravel()

def report_seq(name, y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    aucv = roc_auc_score(y_true, y_prob)
    print(f"[{name}] Thr={thr:.3f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={aucv:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign","Attack"]).plot(cmap=plt.cm.Oranges)
    plt.title(f"{name} - Confusion Matrix"); plt.grid(False); plt.show()

print("\n== LSTM(seq) ==")
report_seq("LSTM(seq)", yte, y_prob_lstm, thr=0.5)
bt_lstm = best_threshold_by_f1(yte, y_prob_lstm)
print("Best threshold by F1 (LSTM):", bt_lstm)
report_seq("LSTM(seq)-bestF1", yte, y_prob_lstm, thr=bt_lstm['threshold'])

# (Optional) ROC / PR
fpr, tpr, _ = roc_curve(yte, y_prob_lstm)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("LSTM ROC Curve"); plt.grid(True); plt.show()
precv, recv, _ = precision_recall_curve(yte, y_prob_lstm)
plt.plot(recv, precv); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("LSTM PR Curve")
plt.grid(True); plt.show()

# ============================
# 10) Improved LSTM (sequence + ID embedding + clean labeling)
# ============================
print("\n== LSTM (sequence, improved) ==")

# ── A) Sequence/Label settings
TIMESTEPS   = 20     # Window length (tune between 10~50)
STEP        = 1      # Sliding interval
LABEL_RULE  = "ratio"  # 'last' | 'any' | 'ratio'
POS_RATIO   = 0.3    # Positive ratio threshold when LABEL_RULE='ratio'
ADD_CONTEXT = True   # Whether to add lightweight time-series features like diff/rolling

# ── B) Sequence DataFrame: Preserve "original order" without global shuffle
df_seq = pd.concat([atk_df, benign_df], ignore_index=True)

# (Optional) Add simple time context features
if ADD_CONTEXT:
    ts_col = next((c for c in ['timestamp','time','ts'] if c in df_seq.columns), None)
    sort_cols = ['ID'] + ([ts_col] if ts_col is not None else [])
    df_seq = df_seq.sort_values(sort_cols, kind='mergesort')

    base_feats = [f'DATA_{i}' for i in range(8)]
    for c in base_feats:
        df_seq[f'{c}_diff1']       = df_seq.groupby('ID')[c].diff(1)
        df_seq[f'{c}_roll5_mean'] = (
            df_seq.groupby('ID')[c].rolling(5).mean().reset_index(level=0, drop=True)
        )
        df_seq[f'{c}_roll5_std'] = (
            df_seq.groupby('ID')[c].rolling(5).std().reset_index(level=0, drop=True)
        )
    df_seq = df_seq.fillna(0.0)
    SEQ_FEATURES_IMP = base_feats + [f'{c}_diff1' for c in base_feats] + \
                   [f'{c}_roll5_mean' for c in base_feats] + [f'{c}_roll5_std' for c in base_feats]
else:
    SEQ_FEATURES_IMP = [f'DATA_{i}' for i in range(8)]

# ── C) Create ID Index
id2idx = {idv:i for i, idv in enumerate(df_seq['ID'].astype(str).unique())}
df_seq['id_idx'] = df_seq['ID'].astype(str).map(id2idx).astype('int32')
NUM_IDS = len(id2idx)

# ── D) Sequence Builder
def build_sequences_df(df_in, features, label_col, id_idx_col='id_idx',
                       timesteps=10, step=1):
    ts_col = next((c for c in ['timestamp','time','ts'] if c in df_in.columns), None)
    sort_cols = ['ID'] + ([ts_col] if ts_col is not None else [])
    df_sorted = df_in.sort_values(sort_cols, kind='mergesort')

    X_seq, y_seq, groups, id_seq = [], [], [], []
    for gid, g in df_sorted.groupby('ID', sort=False):
        Xg = g[features].values.astype('float32')
        yg = g[label_col].values.astype('int32')
        ig = g[id_idx_col].values.astype('int32')

        if len(g) < timesteps:
            continue

        for i in range(0, len(g) - timesteps + 1, step):
            Xw = Xg[i:i+timesteps]
            yw = yg[i:i+timesteps]
            if LABEL_RULE == 'last':
                ylab = int(yw[-1])
            elif LABEL_RULE == 'any':
                ylab = int(yw.max())
            else:  # 'ratio'
                ylab = int(yw.mean() >= POS_RATIO)

            X_seq.append(Xw)
            y_seq.append(ylab)
            groups.append(gid)
            id_seq.append(ig[0])   # ID is same within the window

    return np.array(X_seq), np.array(y_seq), np.array(groups), np.array(id_seq)

X_seq_imp, y_seq_imp, grp_seq_imp, id_seq_imp = build_sequences_df(
    df_seq, SEQ_FEATURES_IMP, 'label', timesteps=TIMESTEPS, step=STEP
)

print(f"Seq shapes: X_seq={X_seq_imp.shape}, y_seq={y_seq_imp.shape}, #unique_IDs={NUM_IDS}")

# ── E) Group Split
gss_seq_imp = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tr_idx, te_idx = next(gss_seq_imp.split(X_seq_imp, y_seq_imp, groups=grp_seq_imp))

Xtr_imp, Xte_imp = X_seq_imp[tr_idx], X_seq_imp[te_idx]
ytr_imp, yte_imp = y_seq_imp[tr_idx], y_seq_imp[te_idx]
idtr, idte = id_seq_imp[tr_idx], id_seq_imp[te_idx]
grp_tr_imp, grp_te_imp = grp_seq_imp[tr_idx], grp_seq_imp[te_idx]
assert len(set(grp_tr_imp).intersection(set(grp_te_imp))) == 0, "LSTM group leakage!"
print(f"LSTM split: train={Xtr_imp.shape}, test={Xte_imp.shape}, ids(train/test)={len(np.unique(idtr))}/{len(np.unique(idte))}")

# ── F) Scaling
scaler_seq_imp = StandardScaler(with_mean=True, with_std=True)
Xtr_flat = Xtr_imp.reshape(-1, Xtr_imp.shape[2])
Xte_flat = Xte_imp.reshape(-1, Xte_imp.shape[2])
Xtr_flat = scaler_seq_imp.fit_transform(Xtr_flat)
Xte_flat = scaler_seq_imp.transform(Xte_flat)
Xtr_imp = Xtr_flat.reshape(Xtr_imp.shape)
Xte_imp = Xte_flat.reshape(Xte_imp.shape)

# ── G) Class Weights
classes = np.unique(ytr_imp)
cw = compute_class_weight('balanced', classes=classes, y=ytr_imp)
class_weight_seq_imp = {int(c): w for c, w in zip(classes, cw)}

# ── H) Model (Combine Sequence + ID Embedding)
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Input, Embedding, Flatten, Concatenate

tf.keras.utils.set_random_seed(SEED)

seq_in = Input(shape=(Xtr_imp.shape[1], Xtr_imp.shape[2]), name='seq_in')
id_in  = Input(shape=(), dtype='int32', name='id_in')

x = Bidirectional(LSTM(64, return_sequences=True))(seq_in)
x = Dropout(0.2)(x)
x = LSTM(32)(x)

# ID embedding
id_emb = Embedding(input_dim=NUM_IDS, output_dim=16, name='id_emb')(id_in)
id_emb = Flatten()(id_emb)

h = Concatenate()([x, id_emb])
h = Dense(64, activation='relu')(h)
out = Dense(1, activation='sigmoid')(h)

lstm_model = Model(inputs=[seq_in, id_in], outputs=out)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
_ = lstm_model.fit([Xtr_imp, idtr], ytr_imp, epochs=30, batch_size=256, validation_split=0.2,
                   class_weight=class_weight_seq_imp, callbacks=[es2], verbose=0)

# ── I) Evaluation + Threshold Tuning
y_prob_lstm_imp = lstm_model.predict([Xte_imp, idte], verbose=0).ravel()

def report_seq_imp(name, y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    aucv = roc_auc_score(y_true, y_prob)
    print(f"[{name}] Thr={thr:.3f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={aucv:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign","Attack"]).plot(cmap=plt.cm.Oranges)
    plt.title(f"{name} - Confusion Matrix"); plt.grid(False); plt.show()

print("\n== LSTM(seq+ID) ==")
report_seq_imp("LSTM(seq+ID)", yte_imp, y_prob_lstm_imp, thr=0.5)
bt_lstm_imp = best_threshold_by_f1(yte_imp, y_prob_lstm_imp)
print("Best threshold by F1 (LSTM seq+ID):", bt_lstm_imp)
report_seq_imp("LSTM(seq+ID)-bestF1", yte_imp, y_prob_lstm_imp, thr=bt_lstm_imp['threshold'])

# (Optional) ROC / PR
fpr, tpr, _ = roc_curve(yte_imp, y_prob_lstm_imp)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("LSTM(seq+ID) ROC Curve"); plt.grid(True); plt.show()
precv, recv, _ = precision_recall_curve(yte_imp, y_prob_lstm_imp)
plt.plot(recv, precv); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("LSTM(seq+ID) PR Curve")
plt.grid(True); plt.show()

# -----------------------------
# 9) Recheck Leakage (Exact duplicate row count)
# -----------------------------
def count_exact_dups(X_tr, X_te):
    def hrow(r): return hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest()
    tr = set(hrow(r) for r in X_tr)
    te = [hrow(r) for r in X_te]
    return sum(t in tr for t in te), len(te)

dup_cnt, total_te = count_exact_dups(scaler.inverse_transform(X_train),
                                     scaler.inverse_transform(X_test))
print(f"\nTrain–Test exact duplicate rows: {dup_cnt} / {total_te}  (should be 0)")

print("\nDone.")