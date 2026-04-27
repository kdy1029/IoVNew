import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


SEED = 42
FEATURES = [f"DATA_{i}" for i in range(8)]
ATTACK_FILES = [
    "decimal_DoS.csv",
    "decimal_spoofing-GAS.csv",
    "decimal_spoofing-RPM.csv",
    "decimal_spoofing-SPEED.csv",
    "decimal_spoofing-STEERING_WHEEL.csv",
]
BENIGN_FILE = "decimal_benign.csv"


@dataclass
class PreparedData:
    df: pd.DataFrame
    attack_df: pd.DataFrame
    benign_df: pd.DataFrame
    features: list[str]
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def read_and_clean(path_or_df, features=None):
    features = features or FEATURES
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, (str, Path)) else path_or_df.copy()
    df = df.drop(columns=["category", "specific_class"], errors="ignore")

    required = features + ["label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .map({"benign": 0, "0": 0, "attack": 1, "1": 1})
    )

    if "ID" not in df.columns:
        df["ID"] = "gid_" + (np.arange(len(df)) // 50).astype(str)

    return df


def row_hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(vec).tobytes()).hexdigest()


def load_decimal_data(data_dir="data/decimal", features=None, seed=SEED):
    features = features or FEATURES
    data_dir = Path(data_dir)
    attack_paths = [data_dir / name for name in ATTACK_FILES]
    benign_path = data_dir / BENIGN_FILE

    attack_df = pd.concat([pd.read_csv(path).assign(label=1) for path in attack_paths], ignore_index=True)
    attack_df = read_and_clean(attack_df, features)
    benign_df = read_and_clean(pd.read_csv(benign_path).assign(label=0), features)

    df = pd.concat([attack_df, benign_df], ignore_index=True).dropna()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X = df[features].values.astype("float32")
    y = df["label"].values.astype("int32")
    groups = np.array([row_hash(row) for row in X])
    return df, attack_df, benign_df, X, y, groups


def prepare_decimal_data(data_dir="data/decimal", features=None, seed=SEED, test_size=0.20):
    features = features or FEATURES
    df, attack_df, benign_df, X, y, groups = load_decimal_data(data_dir, features, seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_raw = X[train_idx].copy()
    X_test_raw = X[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    grp_train = groups[train_idx]
    grp_test = groups[test_idx]
    overlap = set(grp_train).intersection(set(grp_test))

    print(f"All data shape: X={X.shape}, y={y.shape}, positives={y.sum()} ({y.mean():.3%})")
    print(f"Group overlap between train/test: {len(overlap)} (should be 0)")
    print("Train dist:", Counter(y_train))
    print("Test  dist:", Counter(y_test))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return PreparedData(
        df=df,
        attack_df=attack_df,
        benign_df=benign_df,
        features=features,
        X=X,
        y=y,
        groups=groups,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )


def build_sequences_df_basic(
    df,
    features,
    label_col,
    group_col="ID",
    timesteps=10,
    step=1,
    timestamp_cols=("timestamp", "time", "ts"),
):
    ts_col = next((col for col in timestamp_cols if col in df.columns), None)
    sort_cols = [group_col] + ([ts_col] if ts_col is not None else [])
    df_sorted = df.sort_values(sort_cols, kind="mergesort")

    X_seq, y_seq, groups = [], [], []
    for gid, group in df_sorted.groupby(group_col, sort=False):
        Xg = group[features].values.astype("float32")
        yg = group[label_col].values.astype("int32")
        if len(group) < timesteps:
            continue
        for i in range(0, len(group) - timesteps + 1, step):
            X_seq.append(Xg[i : i + timesteps])
            y_seq.append(int(yg[i : i + timesteps].max()))
            groups.append(gid)
    return np.array(X_seq), np.array(y_seq), np.array(groups)


def add_sequence_context(df, base_features):
    df_seq = df.copy()
    ts_col = next((col for col in ["timestamp", "time", "ts"] if col in df_seq.columns), None)
    sort_cols = ["ID"] + ([ts_col] if ts_col is not None else [])
    df_seq = df_seq.sort_values(sort_cols, kind="mergesort")

    for col in base_features:
        df_seq[f"{col}_diff1"] = df_seq.groupby("ID")[col].diff(1)
        df_seq[f"{col}_roll5_mean"] = df_seq.groupby("ID")[col].rolling(5).mean().reset_index(level=0, drop=True)
        df_seq[f"{col}_roll5_std"] = df_seq.groupby("ID")[col].rolling(5).std().reset_index(level=0, drop=True)

    df_seq = df_seq.fillna(0.0)
    features = (
        base_features
        + [f"{col}_diff1" for col in base_features]
        + [f"{col}_roll5_mean" for col in base_features]
        + [f"{col}_roll5_std" for col in base_features]
    )
    return df_seq, features


def build_sequences_df(
    df_in,
    features,
    label_col,
    id_idx_col="id_idx",
    timesteps=10,
    step=1,
    label_rule="ratio",
    pos_ratio=0.3,
):
    ts_col = next((col for col in ["timestamp", "time", "ts"] if col in df_in.columns), None)
    sort_cols = ["ID"] + ([ts_col] if ts_col is not None else [])
    df_sorted = df_in.sort_values(sort_cols, kind="mergesort")

    X_seq, y_seq, groups, id_seq = [], [], [], []
    for gid, group in df_sorted.groupby("ID", sort=False):
        Xg = group[features].values.astype("float32")
        yg = group[label_col].values.astype("int32")
        ig = group[id_idx_col].values.astype("int32")

        if len(group) < timesteps:
            continue

        for i in range(0, len(group) - timesteps + 1, step):
            yw = yg[i : i + timesteps]
            if label_rule == "last":
                ylab = int(yw[-1])
            elif label_rule == "any":
                ylab = int(yw.max())
            else:
                ylab = int(yw.mean() >= pos_ratio)

            X_seq.append(Xg[i : i + timesteps])
            y_seq.append(ylab)
            groups.append(gid)
            id_seq.append(ig[0])

    return np.array(X_seq), np.array(y_seq), np.array(groups), np.array(id_seq)


def scale_sequence_split(X_train, X_test):
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    X_test_flat = X_test.reshape(-1, X_test.shape[2])
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    return X_train_flat.reshape(X_train.shape), X_test_flat.reshape(X_test.shape), scaler


def count_exact_dups(X_tr, X_te):
    train_hashes = {row_hash(row) for row in X_tr}
    test_hashes = [row_hash(row) for row in X_te]
    return sum(test_hash in train_hashes for test_hash in test_hashes), len(test_hashes)

