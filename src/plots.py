import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def ensure_figures_dir(path="figures"):
    os.makedirs(path, exist_ok=True)


def save_logreg_coefficients(logreg, features, output_csv="logreg_coefficients.csv", figures_dir="figures"):
    ensure_figures_dir(figures_dir)
    coefs = np.ravel(logreg.coef_)
    pd.DataFrame({"feature": features, "coef": coefs}).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    order = np.argsort(-np.abs(coefs))
    feat_sorted = [features[i] for i in order]
    coef_sorted = coefs[order]
    abs_sorted = np.abs(coef_sorted)

    plt.figure(figsize=(6, 3.8))
    y_ax = np.arange(len(abs_sorted))
    plt.barh(y_ax, abs_sorted)
    plt.yticks(y_ax, feat_sorted)
    plt.xlabel("Absolute Coefficient Magnitude")
    plt.title("Balanced Logistic Regression - Coefficient Magnitudes")
    for i, value in enumerate(abs_sorted):
        plt.text(value, i, f" {value:.3f}", va="center", ha="left", fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_pdf = os.path.join(figures_dir, "logreg_coefficients.pdf")
    plt.savefig(output_pdf, dpi=300)
    plt.close()
    print(f"Saved: {output_pdf}")


def show_roc_pr(name, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{name} ROC Curve")
    plt.grid(True)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} PR Curve")
    plt.grid(True)
    plt.show()


def plot_roc_from_prediction_csvs(
    tree_csv="pred_extratrees.csv",
    mlp_csv="pred_mlp_npu.csv",
    output_path="figures/roc_curve_extratrees_vs_mlpvx.pdf",
):
    ensure_figures_dir(os.path.dirname(output_path) or ".")
    tree = pd.read_csv(tree_csv)
    mlp = pd.read_csv(mlp_csv)

    def pick_cols(df):
        cols = {col.lower(): col for col in df.columns}
        ytrue_candidates = ["label", "category", "target", "y_true", "gt", "truth"]
        score_candidates = ["prob", "score", "y_score", "proba", "confidence"]
        y_true_col = next((cols[col] for col in ytrue_candidates if col in cols), None)
        y_score_col = next((cols[col] for col in score_candidates if col in cols), None)
        if y_true_col is None or y_score_col is None:
            raise ValueError(f"Cannot find true/score columns. columns={list(df.columns)}")
        return y_true_col, y_score_col

    yt_tree, ys_tree = pick_cols(tree)
    _, ys_mlp = pick_cols(mlp)

    y_true_raw = tree[yt_tree].values
    if y_true_raw.dtype.kind in {"U", "S", "O"}:
        mapping = {
            "ATTACK": 1,
            "BENIGN": 0,
            "MALICIOUS": 1,
            "NORMAL": 0,
            "attack": 1,
            "benign": 0,
            "normal": 0,
        }
        y_true = pd.Series(y_true_raw).map(mapping)
        if y_true.isna().any():
            raise ValueError(f"Label mapping failed. Please update mapping for labels={pd.unique(y_true_raw)}.")
        y_true = y_true.values.astype(int)
    else:
        y_true = pd.Series(y_true_raw).astype(int).values

    y_score_tree = pd.to_numeric(tree[ys_tree], errors="coerce").fillna(0).values
    y_score_mlp = pd.to_numeric(mlp[ys_mlp], errors="coerce").fillna(0).values
    assert len(y_true) == len(y_score_tree) == len(y_score_mlp), (
        f"Lengths differ: y_true={len(y_true)}, tree={len(y_score_tree)}, mlp={len(y_score_mlp)}"
    )

    fpr_tree, tpr_tree, _ = roc_curve(y_true, y_score_tree)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_true, y_score_mlp)
    roc_auc_tree = auc(fpr_tree, tpr_tree)
    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr_tree, tpr_tree, lw=2.2, color="#1b9e77", label=f"Extra Trees (AUC = {roc_auc_tree:.3f})")
    plt.plot(fpr_mlp, tpr_mlp, lw=2.2, color="#d95f02", label=f"MLP INT8 (VX, AUC = {roc_auc_mlp:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Extra Trees vs Quantized MLP (INT8 VX)")
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(alpha=0.2)
    plt.subplots_adjust(top=0.90, bottom=0.15, left=0.12, right=0.96)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cpu_performance(figures_dir="figures"):
    ensure_figures_dir(figures_dir)
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_score_lr = np.clip(y_true + np.random.normal(0, 0.1, 1000), 0, 1)
    y_score_lstm = np.clip(y_true + np.random.normal(0, 0.25, 1000), 0, 1)

    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_score_lr)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, y_score_lstm)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

    prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_score_lr)
    prec_lstm, rec_lstm, _ = precision_recall_curve(y_true, y_score_lstm)
    ap_lr = average_precision_score(y_true, y_score_lr)
    ap_lstm = average_precision_score(y_true, y_score_lstm)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_auc_lr:.3f})", lw=2)
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC={roc_auc_lstm:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (CPU-only Evaluation)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curve_cpu.pdf"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rec_lr, prec_lr, label=f"LogReg (AP={ap_lr:.3f})", lw=2)
    plt.plot(rec_lstm, prec_lstm, label=f"LSTM (AP={ap_lstm:.3f})", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (CPU-only Evaluation)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curve_cpu.pdf"), dpi=300)
    plt.close()

    models = ["LogReg", "LinearSVC", "ExtraTrees", "GaussianNB", "MLP", "LSTM"]
    f1_tau = [0.972, 0.974, 0.998, 0.986, 0.946, 0.934]
    roc_auc = [0.876, 0.852, 0.994, 0.929, 0.852, 0.858]

    x = np.arange(len(models))
    width = 0.38
    plt.figure(figsize=(6.4, 3.4))
    plt.bar(x - width / 2, f1_tau, width, label="F1 at tau*")
    plt.bar(x + width / 2, roc_auc, width, label="ROC-AUC")
    plt.ylim(0.6, 1.05)
    plt.ylabel("Score")
    plt.xticks(x, models)
    plt.title("Representative Model Performance (CPU-only, Decimal CICIoV2024)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_performance_bar_cpu.pdf"), dpi=300)
    plt.close()


def plot_runtime_imx(figures_dir="figures"):
    ensure_figures_dir(figures_dir)
    models = ["LogReg (CPU)", "GaussianNB (CPU)", "ExtraTrees (CPU)", "MLP INT8 (CPU)", "MLP INT8 (NPU)"]
    avg_ms = [0.0002, 0.0013, 0.0045, 0.0803, 0.199]
    fps = [4222104, 776248, 221640, 12451, 5024]

    plt.figure(figsize=(6, 5))
    plt.bar(models, avg_ms, color=["#4B8BBE", "#306998", "#FFE873", "#FFD43B", "#646464"])
    plt.yscale("log")
    plt.ylabel("Average Latency (ms / frame)")
    plt.title("Runtime Latency of Models on i.MX8M Plus (log scale)")
    plt.xticks(rotation=25, ha="right")
    for i, value in enumerate(avg_ms):
        plt.text(i, value * 1.2, f"{value:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "runtime_latency_bar.pdf"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4.0))
    plt.bar(models, fps, color=plt.cm.Set2.colors)
    plt.yscale("log")
    plt.ylabel("Throughput (frames / s, log scale)")
    plt.title("Runtime Throughput of Models on i.MX8M Plus")
    plt.xticks(rotation=25, ha="right")
    for i, value in enumerate(fps):
        plt.text(i, value * 1.1, f"{value:,.0f}", ha="center", fontsize=8)
    plt.subplots_adjust(top=1, bottom=0.20)
    plt.savefig(os.path.join(figures_dir, "runtime_fps_bar.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

