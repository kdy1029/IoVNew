#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAN-IDS: Decimal 데이터셋 (DATA_0..7) 이진 분류
- 누수 차단: 그룹 기반 분할(동일 행 중복이 train/test에 동시에 안 들어가게)
- 스케일링: Train에만 fit, Test엔 transform
- 불균형 보정: class_weight='balanced' (+ Keras class_weight)
- 모델: MLP(Keras), LogisticRegression, Logistic-L1(saga), LinearSVC, SGD, Ridge, Perceptron, PassiveAggressive,
        DecisionTree(shallow), ExtraTrees(shallow), GaussianNB, ComplementNB(binned), BernoulliNB(binarized)
- 지표: Acc, Prec, Rec, F1, ROC-AUC + 혼동행렬 / 임계값 튜닝(F1 최대)
"""

import os, random, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU 고정 (GPU 비교시 주석 처리)

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
# 0) 시드 고정
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------
# 1) 데이터 로드/클린업
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
    # 쓰지 않을 수 있는 열 제거
    df = df.drop(columns=['ID','category','specific_class'], errors='ignore')
    # 필수 컬럼 확인
    use_cols = [f'DATA_{i}' for i in range(8)] + ['label']
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    # 타입 캐스팅
    for c in [f'DATA_{i}' for i in range(8)]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    # 라벨 정규화 (공격:1, 정상:0)
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0, '0':0, 'attack':1, '1':1})
    return df

# 공격/정상 CSV 로드 및 강제 라벨
atk_df = pd.concat([pd.read_csv(f) for f in atk_files], ignore_index=True)
atk_df = atk_df.assign(label=1)
atk_df = read_and_clean(atk_df)

benign_df = read_and_clean(pd.read_csv(benign_file).assign(label=0))

# 통합 → NaN 제거 → 셔플
df = pd.concat([atk_df, benign_df], ignore_index=True).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

FEATURES = [f'DATA_{i}' for i in range(8)]
X = df[FEATURES].values.astype('float32')
y = df['label'].values.astype('int32')

print(f"All data shape: X={X.shape}, y={y.shape}, positives={y.sum()} ({y.mean():.3%})")

# -----------------------------
# 2) 그룹 기반 분할(동일 행 중복 그룹화)
#    -> 같은 feature 행은 train/test에 동시에 존재하지 않게
# -----------------------------
def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()

groups = np.array([row_hash(r) for r in X])

# ★ 80/20 split
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# NB 이산화용 원본 보관
X_train_raw, X_test_raw = X[train_idx].copy(), X[test_idx].copy()

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
grp_train, grp_test = groups[train_idx], groups[test_idx]

# 교차 누수 확인 (0이어야 정상)
inter = set(grp_train).intersection(set(grp_test))
print(f"Group overlap between train/test: {len(inter)} (should be 0)")

# 클래스 분포 확인
print("Train dist:", Counter(y_train))
print("Test  dist:", Counter(y_test))

# -----------------------------
# 3) 스케일링 (누수 방지: train fit → test transform)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# 4) 평가지표/리포팅 유틸
# -----------------------------
def metrics_report(name, y_true, y_prob=None, y_pred=None, plot_cm=True):
    if y_pred is None:
        if y_prob is None:
            raise ValueError("y_prob 또는 y_pred 중 하나는 필요합니다.")
        # 선형점수(decision_function)를 그대로 쓸 수도 있으므로 확률이 아닐 수 있음에 유의
        # 분류는 0.5 기준(또는 0 점수 기준)으로 처리
        if y_prob.ndim == 1:
            # decision score 또는 1D proba
            # score일 경우 threshold=0, proba일 경우 threshold=0.5
            thr = 0.0 if (y_prob.min() < 0 or y_prob.max() > 1) else 0.5
            y_pred = (y_prob >= thr).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

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
    # y_prob가 decision score일 수도, 확률일 수도 있음
    # PR curve는 score도 허용
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2*p*r / (p+r+1e-9)
    idx = np.argmax(f1s[:-1])  # thr 길이는 p/r보다 1 작음
    return dict(threshold=thr[idx], precision=float(p[idx]), recall=float(r[idx]), f1=float(f1s[idx]))

# -----------------------------
# 5) 더미 기준선
# -----------------------------
dum = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_train, y_train)
y_prob_dum = dum.predict_proba(X_test)[:,1] if hasattr(dum, "predict_proba") else None
y_pred_dum = dum.predict(X_test)
print("\n== Dummy (most_frequent) ==")
metrics_report("Dummy", y_test, y_prob=y_prob_dum, y_pred=y_pred_dum)

# -----------------------------
# 6) Keras MLP (LSTM 아님!)
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
hist = mlp.fit(
    X_train, y_train,
    epochs=50, batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[es], verbose=0
)

y_prob_mlp = mlp.predict(X_test, verbose=0).ravel()
print("\n== MLP (Keras, balanced) ==")
metrics_report("MLP(Keras)", y_test, y_prob=y_prob_mlp)

bt = best_threshold_by_f1(y_test, y_prob_mlp)
print(f"Best threshold by F1 (MLP): {bt}")

# ROC / PR 곡선 (옵션)
fpr, tpr, _ = roc_curve(y_test, y_prob_mlp)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("MLP ROC Curve"); plt.grid(True); plt.show()

precv, recv, _ = precision_recall_curve(y_test, y_prob_mlp)
plt.plot(recv, precv); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("MLP PR Curve")
plt.grid(True); plt.show()

# -----------------------------
# 7) 전통 ML 모델들 (클래스 가중 + 스케일 동일)
# -----------------------------
print("\n== Logistic Regression (balanced) ==")
logreg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED)
logreg.fit(X_train, y_train)
prob_lr = logreg.predict_proba(X_test)[:,1]
metrics_report("LogReg(balanced)", y_test, y_prob=prob_lr)
print("Best threshold by F1 (LogReg):", best_threshold_by_f1(y_test, prob_lr))

print("\n== Logistic Regression L1(saga, balanced) ==")
logreg_l1 = LogisticRegression(
    solver='saga', penalty='l1', C=1.0, class_weight='balanced',
    random_state=SEED, max_iter=2000, n_jobs=-1
)
logreg_l1.fit(X_train, y_train)
prob_l1 = logreg_l1.predict_proba(X_test)[:,1]
metrics_report("LogReg-L1(saga, balanced)", y_test, y_prob=prob_l1)
print("Best threshold by F1 (LogReg-L1):", best_threshold_by_f1(y_test, prob_l1))

print("\n== LinearSVC (balanced) ==")
lsvc = LinearSVC(class_weight='balanced', max_iter=5000, random_state=SEED)
lsvc.fit(X_train, y_train)
score_svc = lsvc.decision_function(X_test)
pred_svc = (score_svc >= 0).astype(int)
metrics_report("LinearSVC(balanced)", y_test, y_prob=score_svc, y_pred=pred_svc)

print("\n== SGDClassifier hinge (balanced) ==")
sgd = SGDClassifier(loss='hinge', class_weight='balanced', max_iter=2000, tol=1e-3, random_state=SEED)
sgd.fit(X_train, y_train)
score_sgd = sgd.decision_function(X_test)
pred_sgd = (score_sgd >= 0).astype(int)
metrics_report("SGD-hinge(balanced))", y_test, y_prob=score_sgd, y_pred=pred_sgd)

print("\n== RidgeClassifier (balanced) ==")
ridge = RidgeClassifier(class_weight='balanced', random_state=SEED)
ridge.fit(X_train, y_train)
score_rdg = ridge.decision_function(X_test)
pred_rdg = (score_rdg >= 0).astype(int)
metrics_report("RidgeClassifier(balanced)", y_test, y_prob=score_rdg, y_pred=pred_rdg)

print("\n== Perceptron (balanced) ==")
perc = Perceptron(penalty='l2', class_weight='balanced', random_state=SEED, max_iter=2000, tol=1e-3)
perc.fit(X_train, y_train)
score_perc = perc.decision_function(X_test)
pred_perc = (score_perc >= 0).astype(int)
metrics_report("Perceptron(balanced)", y_test, y_prob=score_perc, y_pred=pred_perc)

print("\n== PassiveAggressive (balanced) ==")
pa = PassiveAggressiveClassifier(loss='hinge', C=1.0, class_weight='balanced', random_state=SEED, max_iter=2000, tol=1e-3)
pa.fit(X_train, y_train)
score_pa = pa.decision_function(X_test)
pred_pa = (score_pa >= 0).astype(int)
metrics_report("PassiveAggressive(balanced)", y_test, y_prob=score_pa, y_pred=pred_pa)

print("\n== DecisionTree (shallow, balanced) ==")
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=SEED)
dt.fit(X_train, y_train)
prob_dt = dt.predict_proba(X_test)[:,1]
metrics_report("DecisionTree(shallow)", y_test, y_prob=prob_dt)
print("Best threshold by F1 (DT):", best_threshold_by_f1(y_test, prob_dt))

print("\n== ExtraTrees (shallow&few, balanced) ==")
et = ExtraTreesClassifier(
    n_estimators=64, max_depth=8, min_samples_leaf=10,
    class_weight='balanced', random_state=SEED, n_jobs=-1
)
et.fit(X_train, y_train)
prob_et = et.predict_proba(X_test)[:,1]
metrics_report("ExtraTrees(shallow)", y_test, y_prob=prob_et)
print("Best threshold by F1 (ExtraTrees):", best_threshold_by_f1(y_test, prob_et))

print("\n== GaussianNB ==")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
prob_gnb = gnb.predict_proba(X_test)[:,1]
metrics_report("GaussianNB", y_test, y_prob=prob_gnb)
print("Best threshold by F1 (GNB):", best_threshold_by_f1(y_test, prob_gnb))

# -----------------------------
# 8) Naive Bayes 변종 (ComplementNB, BernoulliNB) — binning 사용
# -----------------------------
print("\n== NB variants with KBinsDiscretizer (on raw decimal) ==")
binner = KBinsDiscretizer(n_bins=16, encode='ordinal', strategy='uniform')
X_train_bin = binner.fit_transform(X_train_raw)
X_test_bin  = binner.transform(X_test_raw)

print("\n-- ComplementNB --")
cnb = ComplementNB()
cnb.fit(X_train_bin, y_train)
prob_cnb = cnb.predict_proba(X_test_bin)[:,1]
metrics_report("ComplementNB(binned)", y_test, y_prob=prob_cnb)
print("Best threshold by F1 (ComplementNB):", best_threshold_by_f1(y_test, prob_cnb))

print("\n-- BernoulliNB --")
binner2 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')  # 2-bin 이진화
X_train_bin2 = binner2.fit_transform(X_train_raw)
X_test_bin2  = binner2.transform(X_test_raw)

bnb = BernoulliNB()
bnb.fit(X_train_bin2, y_train)
prob_bnb = bnb.predict_proba(X_test_bin2)[:,1]
metrics_report("BernoulliNB(binarized)", y_test, y_prob=prob_bnb)
print("Best threshold by F1 (BernoulliNB):", best_threshold_by_f1(y_test, prob_bnb))

# -----------------------------
# 9) 누수 재확인 (정확 행 중복 수)
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
