import json
import os
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from src.data import FEATURES, SEED, prepare_decimal_data
from src.evaluate import eval_with_thresholds
from src.models import artifact_models, build_mlp


def train_artifact_models(data_dir="data/decimal", artifacts_dir="artifacts", seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    prepared = prepare_decimal_data(data_dir=data_dir, seed=seed)
    os.makedirs(artifacts_dir, exist_ok=True)

    models = artifact_models(seed=seed)
    logreg = models["logreg"].fit(prepared.X_train, prepared.y_train)
    gnb = models["gnb"].fit(prepared.X_train, prepared.y_train)
    et = models["extratrees"].fit(prepared.X_train, prepared.y_train)

    results = []
    for label, model, score_label in [
        ("LogReg(balanced)", logreg, "LogReg"),
        ("GaussianNB", gnb, "GNB"),
        ("ExtraTrees(d=8,100)", et, "ExtraTrees"),
    ]:
        scores = model.predict_proba(prepared.X_test)[:, 1]
        result = eval_with_thresholds(prepared.y_test, scores, thr=0.5, label=label)
        print(
            f"[{score_label}]  Acc={result['acc']:.4f} Prec={result['prec']:.4f} "
            f"Rec={result['rec']:.4f} F1={result['f1']:.4f} AUC={result['auc']:.4f}"
        )
        print(f"[{score_label}-bestF1] Thr={result['thr_bestF1']:.6f} F1={result['f1_b']:.4f}")
        results.append(result)

    np.savez(
        os.path.join(artifacts_dir, "logreg_params.npz"),
        w=logreg.coef_.ravel().astype("float32"),
        b=np.float32(logreg.intercept_.ravel()[0]),
    )
    np.savez(
        os.path.join(artifacts_dir, "gnb_params.npz"),
        mu=gnb.theta_.astype("float32"),
        var=gnb.var_.astype("float32"),
        prior=gnb.class_prior_.astype("float32"),
    )
    np.savez(
        os.path.join(artifacts_dir, "scaler.npz"),
        mean_=prepared.scaler.mean_.astype("float32"),
        scale_=prepared.scaler.scale_.astype("float32"),
    )

    joblib.dump(logreg, os.path.join(artifacts_dir, "logreg.joblib"))
    joblib.dump(gnb, os.path.join(artifacts_dir, "gnb.joblib"))
    joblib.dump(et, os.path.join(artifacts_dir, "extratrees.joblib"))

    thresholds = {
        "features": FEATURES,
        "logreg": {"thr_default": 0.5, "thr_bestF1": results[0]["thr_bestF1"]},
        "gnb": {"thr_default": 0.5, "thr_bestF1": results[1]["thr_bestF1"]},
        "extratrees": {"thr_default": 0.5, "thr_bestF1": results[2]["thr_bestF1"]},
        "notes": "All thresholds computed on current test split; reuse as fixed thresholds on board for inference.",
    }
    with open(os.path.join(artifacts_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    rows = []
    for result in results:
        rows.append(
            {
                "model": result["label"],
                "thr_default": result["thr_default"],
                "acc": result["acc"],
                "prec": result["prec"],
                "rec": result["rec"],
                "f1": result["f1"],
                "auc": result["auc"],
                "thr_bestF1": result["thr_bestF1"],
                "acc_bestF1": result["acc_b"],
                "prec_bestF1": result["prec_b"],
                "rec_bestF1": result["rec_b"],
                "f1_bestF1": result["f1_b"],
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(artifacts_dir, "metrics.csv"), index=False)
    pd.DataFrame({"feature": FEATURES, "importance": et.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(os.path.join(artifacts_dir, "extratrees_feature_importances.csv"), index=False)

    print(f"Saved artifacts in ./{artifacts_dir}")
    print(" - scaler.npz, thresholds.json, metrics.csv")
    print(" - logreg.joblib, gnb.joblib, extratrees.joblib")
    print(" - logreg_params.npz (w,b), gnb_params.npz (mu,var,prior)")
    print(" - extratrees_feature_importances.csv")
    return prepared, results


def export_mlp_int8(data_dir="data/decimal", artifacts_dir="artifacts", seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    prepared = prepare_decimal_data(data_dir=data_dir, seed=seed)
    os.makedirs(artifacts_dir, exist_ok=True)

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

    def representative_dataset():
        for row in prepared.X_train[:1000].astype("float32"):
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(mlp)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    output_path = os.path.join(artifacts_dir, "mlp_int8.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    np.savez(
        os.path.join(artifacts_dir, "scaler.npz"),
        mean_=prepared.scaler.mean_.astype("float32"),
        scale_=prepared.scaler.scale_.astype("float32"),
    )
    with open(os.path.join(artifacts_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES, "model": "MLP INT8 TFLite", "seed": seed}, f, indent=2)

    print(f"Saved TFLite model: {output_path}")
    print(f"Saved scaler and metadata in ./{artifacts_dir}")
    return output_path

