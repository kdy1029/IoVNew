#!/usr/bin/env python
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from src.data import (
    FEATURES,
    SEED,
    add_sequence_context,
    build_sequences_df,
    build_sequences_df_basic,
    count_exact_dups,
    prepare_decimal_data,
    scale_sequence_split,
)
from src.evaluate import best_threshold_by_f1, get_scores, metrics_report, tune_and_report
from src.models import bernoulli_nb, build_lstm, build_lstm_with_id, build_mlp, classical_models, complement_nb, dummy_baseline
from src.plots import save_logreg_coefficients, show_roc_pr


def run(data_dir="data/decimal"):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    prepared = prepare_decimal_data(data_dir=data_dir, seed=SEED)

    dummy = dummy_baseline(seed=SEED).fit(prepared.X_train, prepared.y_train)
    y_prob_dummy = dummy.predict_proba(prepared.X_test)[:, 1] if hasattr(dummy, "predict_proba") else None
    print("\n== Dummy (most_frequent) ==")
    metrics_report("Dummy", prepared.y_test, y_prob=y_prob_dummy, y_pred=dummy.predict(prepared.X_test))

    classes = np.unique(prepared.y_train)
    weights = compute_class_weight("balanced", classes=classes, y=prepared.y_train)
    class_weight = {int(label): weight for label, weight in zip(classes, weights)}

    mlp = build_mlp(prepared.X_train.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mlp.fit(
        prepared.X_train,
        prepared.y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )
    y_prob_mlp = mlp.predict(prepared.X_test, verbose=0).ravel()
    print("\n== MLP (Keras, balanced) ==")
    metrics_report("MLP(Keras)", prepared.y_test, y_prob=y_prob_mlp)
    tune_and_report("MLP(Keras)", prepared.y_test, y_prob_mlp)
    show_roc_pr("MLP", prepared.y_test, y_prob_mlp)

    for name, model in classical_models(seed=SEED):
        print(f"\n== {name} ==")
        model.fit(prepared.X_train, prepared.y_train)
        scores = get_scores(model, prepared.X_test)
        metrics_report(name, prepared.y_test, y_prob=scores)
        tune_and_report(name, prepared.y_test, scores)
        if name == "LogReg(balanced)":
            save_logreg_coefficients(model, prepared.features)

    print("\n== NB variants with KBinsDiscretizer (on raw decimal) ==")
    binner = KBinsDiscretizer(n_bins=16, encode="ordinal", strategy="uniform")
    X_train_bin = binner.fit_transform(prepared.X_train_raw)
    X_test_bin = binner.transform(prepared.X_test_raw)

    print("\n-- ComplementNB --")
    cnb = complement_nb().fit(X_train_bin, prepared.y_train)
    scores_cnb = get_scores(cnb, X_test_bin)
    metrics_report("ComplementNB(binned)", prepared.y_test, y_prob=scores_cnb)
    tune_and_report("ComplementNB(binned)", prepared.y_test, scores_cnb)

    print("\n-- BernoulliNB --")
    binner2 = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform")
    X_train_bin2 = binner2.fit_transform(prepared.X_train_raw)
    X_test_bin2 = binner2.transform(prepared.X_test_raw)
    bnb = bernoulli_nb().fit(X_train_bin2, prepared.y_train)
    scores_bnb = get_scores(bnb, X_test_bin2)
    metrics_report("BernoulliNB(binarized)", prepared.y_test, y_prob=scores_bnb)
    tune_and_report("BernoulliNB(binarized)", prepared.y_test, scores_bnb)

    run_sequence_models(prepared)

    dup_cnt, total_te = count_exact_dups(
        prepared.scaler.inverse_transform(prepared.X_train),
        prepared.scaler.inverse_transform(prepared.X_test),
    )
    print(f"\nTrain-Test exact duplicate rows: {dup_cnt} / {total_te}  (should be 0)")
    print("\nDone.")


def run_sequence_models(prepared):
    print("\n== LSTM (sequence) ==")
    X_seq, y_seq, groups_seq = build_sequences_df_basic(
        prepared.df, FEATURES, "label", group_col="ID", timesteps=10, step=1
    )
    print(f"Seq shapes: X_seq={X_seq.shape}, y_seq={y_seq.shape}, #groups={len(np.unique(groups_seq))}")

    gss_seq = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(gss_seq.split(X_seq, y_seq, groups=groups_seq))
    Xtr, Xte = X_seq[tr_idx], X_seq[te_idx]
    ytr, yte = y_seq[tr_idx], y_seq[te_idx]
    grp_tr, grp_te = groups_seq[tr_idx], groups_seq[te_idx]
    assert len(set(grp_tr).intersection(set(grp_te))) == 0, "LSTM group leakage!"
    print(f"LSTM split: train={Xtr.shape}, test={Xte.shape}")

    Xtr, Xte, _ = scale_sequence_split(Xtr, Xte)
    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight_seq = {int(label): weight for label, weight in zip(classes, weights)}

    tf.keras.utils.set_random_seed(SEED)
    lstm = build_lstm((Xtr.shape[1], Xtr.shape[2]))
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lstm.fit(
        Xtr,
        ytr,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight_seq,
        callbacks=[early_stop],
        verbose=0,
    )
    y_prob_lstm = lstm.predict(Xte, verbose=0).ravel()
    print("\n== LSTM(seq) ==")
    metrics_report("LSTM(seq)", yte, y_prob=y_prob_lstm, thr=0.5)
    bt_lstm = best_threshold_by_f1(yte, y_prob_lstm)
    print("Best threshold by F1 (LSTM):", bt_lstm)
    metrics_report("LSTM(seq)-bestF1", yte, y_prob=y_prob_lstm, thr=bt_lstm["threshold"])
    show_roc_pr("LSTM", yte, y_prob_lstm)

    run_improved_sequence_model(prepared)


def run_improved_sequence_model(prepared):
    print("\n== LSTM (sequence, improved) ==")
    timesteps = 20
    step = 1
    label_rule = "ratio"
    pos_ratio = 0.3

    df_seq = pd.concat([prepared.attack_df, prepared.benign_df], ignore_index=True)
    df_seq, seq_features = add_sequence_context(df_seq, FEATURES)

    id2idx = {idv: idx for idx, idv in enumerate(df_seq["ID"].astype(str).unique())}
    df_seq["id_idx"] = df_seq["ID"].astype(str).map(id2idx).astype("int32")
    num_ids = len(id2idx)

    X_seq, y_seq, grp_seq, id_seq = build_sequences_df(
        df_seq,
        seq_features,
        "label",
        timesteps=timesteps,
        step=step,
        label_rule=label_rule,
        pos_ratio=pos_ratio,
    )
    print(f"Seq shapes: X_seq={X_seq.shape}, y_seq={y_seq.shape}, #unique_IDs={num_ids}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X_seq, y_seq, groups=grp_seq))
    Xtr, Xte = X_seq[tr_idx], X_seq[te_idx]
    ytr, yte = y_seq[tr_idx], y_seq[te_idx]
    idtr, idte = id_seq[tr_idx], id_seq[te_idx]
    grp_tr, grp_te = grp_seq[tr_idx], grp_seq[te_idx]
    assert len(set(grp_tr).intersection(set(grp_te))) == 0, "LSTM group leakage!"
    print(f"LSTM split: train={Xtr.shape}, test={Xte.shape}, ids(train/test)={len(np.unique(idtr))}/{len(np.unique(idte))}")

    Xtr, Xte, _ = scale_sequence_split(Xtr, Xte)
    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(label): weight for label, weight in zip(classes, weights)}

    tf.keras.utils.set_random_seed(SEED)
    model = build_lstm_with_id((Xtr.shape[1], Xtr.shape[2]), num_ids)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        [Xtr, idtr],
        ytr,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )

    y_prob = model.predict([Xte, idte], verbose=0).ravel()
    print("\n== LSTM(seq+ID) ==")
    metrics_report("LSTM(seq+ID)", yte, y_prob=y_prob, thr=0.5)
    best = best_threshold_by_f1(yte, y_prob)
    print("Best threshold by F1 (LSTM seq+ID):", best)
    metrics_report("LSTM(seq+ID)-bestF1", yte, y_prob=y_prob, thr=best["threshold"])
    show_roc_pr("LSTM(seq+ID)", yte, y_prob)


if __name__ == "__main__":
    run()
