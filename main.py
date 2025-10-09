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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU 고정 (GPU 비교시 주석 처리)

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
    # ❌ 기존: df.drop(['ID','category','specific_class'], ...)
    # ✅ ID는 보존, 기타 불필요 컬럼만 제거
    df = df.drop(columns=['category','specific_class'], errors='ignore')

    # 필수 컬럼 확인 (ID는 선택)
    use_cols = [f'DATA_{i}' for i in range(8)] + ['label']
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 타입 캐스팅
    for c in [f'DATA_{i}' for i in range(8)]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    # 라벨 정규화 (공격:1, 정상:0)
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0, '0':0, 'attack':1, '1':1})

    # ID가 없으면 임시 ID 생성(시퀀스 묶음용)
    if 'ID' not in df.columns:
        df['ID'] = 'gid_' + (np.arange(len(df)) // 50).astype(str)  # 50개 단위로 같은 그룹

    # (선택) 타임스탬프가 있으면 정규화용으로 보존
    # 예상되는 이름들 중 하나가 있으면 그대로 둡니다.
    # ['timestamp','time','ts'] 중 있는 컬럼만 사용
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
def metrics_report(name, y_true, y_prob=None, y_pred=None, plot_cm=True, thr=None):
    if y_pred is None:
        if y_prob is None:
            raise ValueError("y_prob 또는 y_pred 중 하나는 필요합니다.")
        # 선형점수(decision_function)를 그대로 쓸 수도 있으므로 확률이 아닐 수 있음에 유의
        # 분류는 0.5 기준(또는 0 점수 기준)으로 처리
        if thr is not None:
            y_pred = (y_prob >= thr).astype(int)
        else:
            # 기본: 확률이면 0.5, 점수면 0 기준
            default_thr = 0.0 if (y_prob.min() < 0 or y_prob.max() > 1) else 0.5
            y_pred = (y_prob >= default_thr).astype(int)

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
metrics_report("MLP(Keras)-bestF1", y_test, y_prob=y_prob_mlp, thr=bt["threshold"])
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
metrics_report("LogReg(balanced)", y_test, y_prob=prob_lr, thr=prob_lr["threshold"])
print("Best threshold by F1 (LogReg):", best_threshold_by_f1(y_test, prob_lr))

# --- (A) LogReg coefficients: CSV + Figure (Feature Importance)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)
FEATURE_NAMES = FEATURES[:]
coefs = np.ravel(logreg.coef_)                 # shape (n_features,)
df_coef = pd.DataFrame({"feature": FEATURE_NAMES, "coef": coefs})
df_coef.to_csv("logreg_coefficients.csv", index=False)
print("✅ Saved: logreg_coefficients.csv")

# 절대값 기준 내림차순 정렬 후 가로막대
order = np.argsort(-np.abs(coefs))

feat_sorted = [FEATURE_NAMES[i] for i in order]
coef_sorted = coefs[order]
abs_sorted  = np.abs(coef_sorted)

plt.figure(figsize=(6, 3.8))
y = np.arange(len(abs_sorted))
plt.barh(y, abs_sorted)
plt.yticks(y, feat_sorted)
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


# ============================
# 10) LSTM (시퀀스 기반)
# ============================
print("\n== LSTM (sequence) ==")

# A) 시퀀스 빌더: 같은 ID 안에서 연속 timesteps로 윈도 생성
def build_sequences_df(df, features, label_col, group_col='ID',
                       timesteps=10, step=1, timestamp_cols=('timestamp','time','ts')):
    # 시간 정렬(가능하면)
    ts_col = next((c for c in timestamp_cols if c in df.columns), None)
    if ts_col is not None:
        df_sorted = df.sort_values([group_col, ts_col], kind='mergesort')
    else:
        # 원래 섞은 상태라도 group 내 상대 순서를 유지(mergesort는 안정 정렬)
        df_sorted = df.sort_values([group_col], kind='mergesort')

    X_seq, y_seq, groups = [], [], []
    for gid, g in df_sorted.groupby(group_col, sort=False):
        Xg = g[features].values.astype('float32')
        yg = g[label_col].values.astype('int32')
        if len(g) < timesteps:  # 너무 짧으면 스킵
            continue
        # 슬라이딩 윈도우
        for i in range(0, len(g) - timesteps + 1, step):
            X_seq.append(Xg[i:i+timesteps])
            # 레이블 집계: 윈도 내 하나라도 공격이면 1 (max)
            y_seq.append(int(yg[i:i+timesteps].max()))
            groups.append(gid)
    return np.array(X_seq), np.array(y_seq), np.array(groups)

TIMESTEPS = 10
SEQ_FEATURES = [f'DATA_{i}' for i in range(8)]

# 주의: 시퀀스는 "원본 df"에서 만듭니다(스케일은 나중에 적용)
X_seq, y_seq, groups_seq = build_sequences_df(df, SEQ_FEATURES, 'label', group_col='ID',
                                              timesteps=TIMESTEPS, step=1)

print(f"Seq shapes: X_seq={X_seq.shape}, y_seq={y_seq.shape}, #groups={len(np.unique(groups_seq))}")

# B) 그룹 기반 분할 (동일 ID가 train/test에 동시에 등장 금지)
gss_seq = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)  # 80/20, 필요시 0.30로 변경
tr_idx, te_idx = next(gss_seq.split(X_seq, y_seq, groups=groups_seq))
Xtr, Xte = X_seq[tr_idx], X_seq[te_idx]
ytr, yte = y_seq[tr_idx], y_seq[te_idx]
grp_tr, grp_te = groups_seq[tr_idx], groups_seq[te_idx]
assert len(set(grp_tr).intersection(set(grp_te))) == 0, "LSTM group leakage!"

print(f"LSTM split: train={Xtr.shape}, test={Xte.shape}")

# C) 시퀀스 스케일링 (train에 fit → test에만 transform)
#    각 timestep의 피처 분포를 동일 기준으로 맞추기 위해 flatten→스케일→reshape
scaler_seq = StandardScaler()
Xtr_flat = Xtr.reshape(-1, Xtr.shape[2])
Xte_flat = Xte.reshape(-1, Xte.shape[2])
Xtr_flat = scaler_seq.fit_transform(Xtr_flat)
Xte_flat = scaler_seq.transform(Xte_flat)
Xtr = Xtr_flat.reshape(Xtr.shape)
Xte = Xte_flat.reshape(Xte.shape)

# D) 클래스 가중치
classes = np.unique(ytr)
cw = compute_class_weight('balanced', classes=classes, y=ytr)
class_weight_seq = {int(c): w for c, w in zip(classes, cw)}

# E) LSTM 모델(가볍게)
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
lstm.fit(Xtr, ytr, epochs=30, batch_size=256, validation_split=0.2,
         class_weight=class_weight_seq, callbacks=[es2], verbose=0)

# F) 평가
y_prob_lstm = lstm.predict(Xte, verbose=0).ravel()

def report_seq(name, y_true, y_prob):
    # 임계값 0.5 기준
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    aucv = roc_auc_score(y_true, y_prob)
    print(f"[{name}] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} ROC-AUC={aucv:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign","Attack"]).plot(cmap=plt.cm.Oranges)
    plt.title(f"{name} - Confusion Matrix"); plt.grid(False); plt.show()


# G) F1 최적 임계값
bt_lstm = best_threshold_by_f1(yte, y_prob_lstm)
print("Best threshold by F1 (LSTM):", bt_lstm)

# (선택) ROC/PR 곡선
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

# ── A) 시퀀스/라벨 설정값
TIMESTEPS   = 20     # 윈도 길이 (10~50 사이 튜닝)
STEP        = 1      # 슬라이딩 간격
LABEL_RULE  = "ratio"  # 'last' | 'any' | 'ratio' (권장: 'ratio')
POS_RATIO   = 0.3    # LABEL_RULE='ratio'일 때 양성 비율 임계값
ADD_CONTEXT = True   # diff/rolling 등 경량 시계열 피처 추가 여부

# ── B) 시퀀스용 데이터프레임: 전역 셔플 없는 "원본 순서" 유지
#     -> 위쪽에서 read_and_clean 후 만든 atk_df, benign_df를 사용 (ID 보존 전제)
df_seq = pd.concat([atk_df, benign_df], ignore_index=True)

# (선택) 간단한 시간 맥락 피처 추가: ID 그룹 내 diff/rolling
if ADD_CONTEXT:
    # 시간 정렬 우선 (timestamp 컬럼이 있으면 사용)
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
    SEQ_FEATURES = base_feats + [f'{c}_diff1' for c in base_feats] + \
                   [f'{c}_roll5_mean' for c in base_feats] + [f'{c}_roll5_std' for c in base_feats]
else:
    SEQ_FEATURES = [f'DATA_{i}' for i in range(8)]

# ── C) ID 인덱스 생성 (임베딩 입력용)
id2idx = {idv:i for i, idv in enumerate(df_seq['ID'].astype(str).unique())}
df_seq['id_idx'] = df_seq['ID'].astype(str).map(id2idx).astype('int32')
NUM_IDS = len(id2idx)

# ── D) 시퀀스 빌더
def build_sequences_df(df, features, label_col, id_idx_col='id_idx',
                       timesteps=10, step=1):
    # 시간 정렬(가능하면)
    ts_col = next((c for c in ['timestamp','time','ts'] if c in df.columns), None)
    sort_cols = ['ID'] + ([ts_col] if ts_col is not None else [])
    df_sorted = df.sort_values(sort_cols, kind='mergesort')

    X_seq, y_seq, groups, id_seq = [], [], [], []
    for gid, g in df_sorted.groupby('ID', sort=False):
        Xg = g[features].values.astype('float32')
        yg = g[label_col].values.astype('int32')
        ig = g[id_idx_col].values.astype('int32')

        if len(g) < timesteps:  # 너무 짧은 건 스킵
            continue

        # 슬라이딩 윈도우
        for i in range(0, len(g) - timesteps + 1, step):
            Xw = Xg[i:i+timesteps]
            yw = yg[i:i+timesteps]
            # 라벨링 규칙
            if LABEL_RULE == 'last':
                ylab = int(yw[-1])
            elif LABEL_RULE == 'any':
                ylab = int(yw.max())
            else:  # 'ratio'
                ylab = int(yw.mean() >= POS_RATIO)

            X_seq.append(Xw)
            y_seq.append(ylab)
            groups.append(gid)
            id_seq.append(ig[0])   # 윈도 내 ID는 동일

    return np.array(X_seq), np.array(y_seq), np.array(groups), np.array(id_seq)

X_seq, y_seq, grp_seq, id_seq = build_sequences_df(
    df_seq, SEQ_FEATURES, 'label', timesteps=TIMESTEPS, step=STEP
)

print(f"Seq shapes: X_seq={X_seq.shape}, y_seq={y_seq.shape}, #unique_IDs={NUM_IDS}")

# ── E) 그룹 분할(누수 방지: 동일 ID는 한쪽에만)
gss_seq = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)  # 필요시 0.30로
tr_idx, te_idx = next(gss_seq.split(X_seq, y_seq, groups=grp_seq))

Xtr, Xte = X_seq[tr_idx], X_seq[te_idx]
ytr, yte = y_seq[tr_idx], y_seq[te_idx]
idtr, idte = id_seq[tr_idx], id_seq[te_idx]
grp_tr, grp_te = grp_seq[tr_idx], grp_seq[te_idx]
assert len(set(grp_tr).intersection(set(grp_te))) == 0, "LSTM group leakage!"

print(f"LSTM split: train={Xtr.shape}, test={Xte.shape}, ids(train/test)={len(np.unique(idtr))}/{len(np.unique(idte))}")

# ── F) 스케일링 (train에 fit → test에만 transform)
from sklearn.preprocessing import StandardScaler
scaler_seq = StandardScaler(with_mean=True, with_std=True)
Xtr_flat = Xtr.reshape(-1, Xtr.shape[2])
Xte_flat = Xte.reshape(-1, Xte.shape[2])
Xtr_flat = scaler_seq.fit_transform(Xtr_flat)
Xte_flat = scaler_seq.transform(Xte_flat)
Xtr = Xtr_flat.reshape(Xtr.shape)
Xte = Xte_flat.reshape(Xte.shape)

# ── G) 클래스 가중치
classes = np.unique(ytr)
cw = compute_class_weight('balanced', classes=classes, y=ytr)
class_weight_seq = {int(c): w for c, w in zip(classes, cw)}

# ── H) 모델(시퀀스 + ID 임베딩 결합)
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Input, Embedding, Flatten, Concatenate

tf.keras.utils.set_random_seed(SEED)

seq_in = Input(shape=(Xtr.shape[1], Xtr.shape[2]), name='seq_in')
id_in  = Input(shape=(), dtype='int32', name='id_in')

x = Bidirectional(LSTM(64, return_sequences=True))(seq_in)
x = Dropout(0.2)(x)
x = LSTM(32)(x)

# ID embedding (작게 16차원; 데이터에 맞춰 8~32 튜닝)
id_emb = Embedding(input_dim=NUM_IDS, output_dim=16, name='id_emb')(id_in)
id_emb = Flatten()(id_emb)

h = Concatenate()([x, id_emb])
h = Dense(64, activation='relu')(h)
out = Dense(1, activation='sigmoid')(h)

lstm_model = Model(inputs=[seq_in, id_in], outputs=out)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es2 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit([Xtr, idtr], ytr, epochs=30, batch_size=256, validation_split=0.2,
               class_weight=class_weight_seq, callbacks=[es2], verbose=0)

# ── I) 평가 + 임계값 튜닝
y_prob_lstm = lstm_model.predict([Xte, idte], verbose=0).ravel()

def report_seq(name, y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    aucv = roc_auc_score(y_true, y_prob)
    print(f"[{name}] Thr={thr:.3f}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={aucv:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign","Attack"]).plot(cmap=plt.cm.Oranges)
    plt.title(f"{name} - Confusion Matrix"); plt.grid(False); plt.show()

# 기본 0.5 평가
report_seq("LSTM(seq+ID)", yte, y_prob_lstm, thr=0.5)

# F1 최대 임계값
bt_lstm = best_threshold_by_f1(yte, y_prob_lstm)
print("Best threshold by F1 (LSTM):", bt_lstm)
report_seq("LSTM(seq+ID)-bestF1", yte, y_prob_lstm, thr=bt_lstm['threshold'])

# (선택) ROC / PR
fpr, tpr, _ = roc_curve(yte, y_prob_lstm)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("LSTM ROC Curve"); plt.grid(True); plt.show()

precv, recv, _ = precision_recall_curve(yte, y_prob_lstm)
plt.plot(recv, precv); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("LSTM PR Curve")
plt.grid(True); plt.show()


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
