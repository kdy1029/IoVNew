#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAN-IDS: Decimal(DATA_0..7) 이진 분류 — 학습/검증/테스트 + 아티팩트 저장
- 누수 차단: 그룹 기반 분할(동일 행 중복/ID가 train/val/test 교차 금지)
- 스케일링: Train에만 fit, Val/Test는 transform
- 불균형 보정: class_weight='balanced' (+ Keras class_weight)
- 모델: LogisticRegression(balanced), MLP(Keras)
- 임계값: Validation에서 F1-best 임계값 → meta.json에 저장 → Test/보드 공통 사용
- 산출물: artifacts/{scaler.npz, logreg.joblib, mlp_int8.tflite, meta.json}
"""

import os, random, hashlib, json
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

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
DATA_DIR = "./data/decimal"
atk_files = [
    f'{DATA_DIR}/decimal_DoS.csv',
    f'{DATA_DIR}/decimal_spoofing-GAS.csv',
    f'{DATA_DIR}/decimal_spoofing-RPM.csv',
    f'{DATA_DIR}/decimal_spoofing-SPEED.csv',
    f'{DATA_DIR}/decimal_spoofing-STEERING_WHEEL.csv',
]
benign_file = f'{DATA_DIR}/decimal_benign.csv'
FEATURES = [f"DATA_{i}" for i in range(8)]

def read_and_clean(path_or_df):
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    # 불필요 열 제거(있으면)
    df = df.drop(columns=['category','specific_class'], errors='ignore')
    # 필수 컬럼 확인
    need = FEATURES + ['label']
    missing = [c for c in need if c not in df.columns]
    if missing: raise ValueError(f"필수 컬럼 누락: {missing}")
    # 타입 정리
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0,'0':0,'attack':1,'1':1})
    # ID 없으면 그룹용 임시 ID 생성(50개 단위)
    if 'ID' not in df.columns:
        df['ID'] = 'gid_' + (np.arange(len(df)) // 50).astype(str)
    return df

# 공격/정상 CSV 로드 및 강제 라벨
atk_df = pd.concat([pd.read_csv(f).assign(label=1) for f in atk_files], ignore_index=True)
atk_df = read_and_clean(atk_df)
benign_df = read_and_clean(pd.read_csv(benign_file).assign(label=0))

# 통합 → NaN 제거 → 셔플
df = pd.concat([atk_df, benign_df], ignore_index=True).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print("Loaded:", len(df), "rows")

# -----------------------------
# 2) 그룹 기반 분할(중복/동일행 누수 차단)
# -----------------------------
def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()

X_all = df[FEATURES].values.astype('float32')
y_all = df['label'].values.astype('int32')

# 동일행 기반 그룹(각 행 해시)
row_groups = np.array([row_hash(r) for r in X_all])

# (1) 우선 TrainVal/Test = 80/20
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
trv_idx, te_idx = next(gss.split(X_all, y_all, groups=row_groups))

X_trv, X_te = X_all[trv_idx], X_all[te_idx]
y_trv, y_te = y_all[trv_idx], y_all[te_idx]
grp_trv, grp_te = row_groups[trv_idx], row_groups[te_idx]

# (2) Train/Val = 80/20 (TrainVal 내부에서 그룹 분할)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED+1)
tr_idx, va_idx = next(gss2.split(X_trv, y_trv, groups=grp_trv))

X_tr, X_va = X_trv[tr_idx], X_trv[va_idx]
y_tr, y_va = y_trv[tr_idx], y_trv[va_idx]

print("Split sizes:",
      "train", X_tr.shape, "val", X_va.shape, "test", X_te.shape)
print("Train dist:", Counter(y_tr), "Val dist:", Counter(y_va), "Test dist:", Counter(y_te))

# -----------------------------
# 3) 스케일링 (Train fit → Val/Test transform)
# -----------------------------
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# -----------------------------
# 4) 유틸: 메트릭/임계값
# -----------------------------
def prf1(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return acc, prec, rec, f1

def f1_best_threshold(y_true, y_score):
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2*p*r/(p+r+1e-12)
    idx = np.nanargmax(f1[:-1])  # 마지막점 제외
    return {"threshold": float(thr[idx]),
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f1[idx])}

# -----------------------------
# 5) 모델 1: Logistic Regression (CPU용)
# -----------------------------
logreg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED)
logreg.fit(X_tr_s, y_tr)

# Val에서 F1-best 임계값
prob_va_lr = logreg.predict_proba(X_va_s)[:,1]
bt_lr = f1_best_threshold(y_va, prob_va_lr)
thr_lr = bt_lr["threshold"]
print(f"[LogReg] Val F1-best:", bt_lr)

# Test 평가(동일 임계값)
prob_te_lr = logreg.predict_proba(X_te_s)[:,1]
y_te_lr = (prob_te_lr >= thr_lr).astype(int)
acc, prec, rec, f1 = prf1(y_te, y_te_lr)
auc = roc_auc_score(y_te, prob_te_lr)
print(f"[LogReg] Test  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f} Thr={thr_lr:.4f}")

# -----------------------------
# 6) 모델 2: MLP (보드 NPU용 → TFLite INT8 변환)
# -----------------------------
classes = np.unique(y_tr)
cw = compute_class_weight('balanced', classes=classes, y=y_tr)
class_weight = {int(c): w for c, w in zip(classes, cw)}

mlp = Sequential([
    Input(shape=(X_tr_s.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
mlp.fit(X_tr_s, y_tr, epochs=50, batch_size=256, validation_data=(X_va_s, y_va),
        class_weight=class_weight, callbacks=[es], verbose=0)

# Val F1-best 임계값
prob_va_mlp = mlp.predict(X_va_s, verbose=0).ravel()
bt_mlp = f1_best_threshold(y_va, prob_va_mlp)
thr_mlp = bt_mlp["threshold"]
print(f"[MLP]    Val F1-best:", bt_mlp)

# Test 평가(동일 임계값)
prob_te_mlp = mlp.predict(X_te_s, verbose=0).ravel()
y_te_mlp = (prob_te_mlp >= thr_mlp).astype(int)
acc, prec, rec, f1 = prf1(y_te, y_te_mlp)
auc = roc_auc_score(y_te, prob_te_mlp)
print(f"[MLP]    Test  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f} Thr={thr_mlp:.4f}")

# -----------------------------
# 7) 아티팩트 저장
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

# (a) 스케일러 계수
np.savez("artifacts/scaler.npz",
         mean_=scaler.mean_.astype('float32'),
         scale_=scaler.scale_.astype('float32'))

# (b) sklearn LogReg
import joblib
joblib.dump(logreg, "artifacts/logreg.joblib")

# (c) MLP → TFLite INT8 변환(대표데이터 필요: 스케일된 train 분포)
def rep_gen():
    n = min(500, len(X_tr_s))
    for i in range(n):
        yield [X_tr_s[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tfl = converter.convert()
open("artifacts/mlp_int8.tflite","wb").write(tfl)

# (d) 메타(피처/임계값/셋 크기)
meta = {
    "features": FEATURES,
    "thresholds": {
        "logreg_val_f1_best": float(thr_lr),
        "mlp_val_f1_best": float(thr_mlp)
    },
    "splits": {
        "train": int(len(y_tr)),
        "val": int(len(y_va)),
        "test": int(len(y_te))
    },
    "notes": "Thresholds found on validation; use same for test/board."
}
with open("artifacts/meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved artifacts in ./artifacts")
