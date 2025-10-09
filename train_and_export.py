#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, random, hashlib, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED=42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------
# 데이터 로드
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
    # 필수 컬럼 확인
    need = FEATURES + ['label']
    missing = [c for c in need if c not in df.columns]
    if missing: raise ValueError(f"필수 컬럼 누락: {missing}")
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    df['label'] = df['label'].astype(str).str.lower().map({'benign':0,'0':0,'attack':1,'1':1})
    if 'ID' not in df.columns:
        df['ID'] = 'gid_' + (np.arange(len(df)) // 50).astype(str)
    return df

atk_df = pd.concat([pd.read_csv(f).assign(label=1) for f in atk_files], ignore_index=True)
atk_df = read_and_clean(atk_df)
benign_df = read_and_clean(pd.read_csv(benign_file).assign(label=0))

df = pd.concat([atk_df, benign_df], ignore_index=True).dropna()
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

X = df[FEATURES].values.astype('float32')
y = df['label'].values.astype('int32')

# 그룹 해시로 누수 차단
def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()
groups = np.array([row_hash(r) for r in X])

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 스케일
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# -----------------------------
# Baseline: Logistic Regression (CPU용)
# -----------------------------
logreg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=SEED)
logreg.fit(X_train, y_train)
p = logreg.predict_proba(X_test)[:,1]
pred = (p >= 0.5).astype(int)
acc = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
auc = roc_auc_score(y_test, p)
print(f"[LogReg] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

# -----------------------------
# MLP (보드 NPU용: TFLite INT8로 변환)
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
mlp.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2,
        class_weight=class_weight, callbacks=[es], verbose=0)
p_m = mlp.predict(X_test, verbose=0).ravel()
pred_m = (p_m >= 0.5).astype(int)
acc_m = accuracy_score(y_test, pred_m)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, pred_m, average='binary', zero_division=0)
auc_m = roc_auc_score(y_test, p_m)
print(f"[MLP]    Acc={acc_m:.4f} Prec={prec_m:.4f} Rec={rec_m:.4f} F1={f1_m:.4f} AUC={auc_m:.4f}")

# -----------------------------
# 아티팩트 저장
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

# 1) 스케일러 계수
np.savez("artifacts/scaler.npz", mean_=scaler.mean_.astype('float32'), scale_=scaler.scale_.astype('float32'))

# 2) sklearn LogReg (CPU 추론용)
import joblib
joblib.dump(logreg, "artifacts/logreg.joblib")

# 3) MLP를 TFLite INT8로 변환 (대표 데이터로 양자화)
def rep_gen():
    # 스케일된 train 분포에서 500개 샘플
    n = min(500, len(X_train))
    for i in range(n):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tfl = converter.convert()
open("artifacts/mlp_int8.tflite","wb").write(tfl)

# 4) 메타 정보(컬럼/쓰레시홀드 등)
meta = {
    "features": FEATURES,
    "threshold_default": 0.5,
    "notes": "Scaled with StandardScaler from training set; MLP quantized INT8 for NPU."
}
with open("artifacts/meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved artifacts in ./artifacts")
