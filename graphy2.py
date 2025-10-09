# ================================
# IoV IDS Figures (No seaborn)
# ================================
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 0) Utility: ensure output dir
# -------------------------------
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# 1) Feature Importance (LogReg)
# -------------------------------
def plot_logreg_coefficients(model=None,
                             coefs=None,
                             feature_names=None,
                             title="Logistic Regression Feature Importance",
                             outfile=os.path.join(OUT_DIR, "logreg_coefficients.pdf")):
    """
    Plot absolute coefficient magnitudes for a trained LogisticRegression (binary).
    - Pass either `model` (sklearn LogisticRegression with coef_) or `coefs` (1D array-like).
    - `feature_names` must match the number of coefficients.
    """
    # Pull coefficients
    if coefs is None:
        if model is None or not hasattr(model, "coef_"):
            raise ValueError("Provide either `model` (with coef_) or `coefs`.")
        # sklearn: coef_.shape = (1, n_features) for binary
        coefs = np.ravel(model.coef_)
    else:
        coefs = np.asarray(coefs).ravel()

    n = len(coefs)
    if feature_names is None:
        # Fallback names if not provided
        feature_names = [f"feat_{i}" for i in range(n)]
    if len(feature_names) != n:
        raise ValueError("`feature_names` length must match number of coefficients.")

    # Sort by absolute magnitude (descending)
    abs_coefs = np.abs(coefs)
    order = np.argsort(-abs_coefs)
    coefs_sorted = coefs[order]
    abs_sorted = abs_coefs[order]
    names_sorted = [feature_names[i] for i in order]

    # Plot (horizontal bar)
    plt.figure(figsize=(8.5, 3.8))
    y = np.arange(n)
    plt.barh(y, abs_sorted)
    plt.yticks(y, names_sorted)
    plt.xlabel("Absolute Coefficient Magnitude")
    plt.title(title)

    # Annotate values on bars
    for i, v in enumerate(abs_sorted):
        plt.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=8)

    plt.gca().invert_yaxis()  # largest on top
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved: {outfile}")

# -------------------------------
# 2) Model Performance Bar Chart
# -------------------------------
def plot_model_performance_bar(models,
                               f1_scores,
                               roc_auc_scores,
                               title="Representative Model Performance (Decimal CICIoV2024, GPU)",
                               outfile=os.path.join(OUT_DIR, "model_performance_bar.pdf")):
    """
    Grouped bars for F1 and ROC-AUC across models.
    """
    models = list(models)
    f1 = np.asarray(f1_scores, dtype=float)
    roc = np.asarray(roc_auc_scores, dtype=float)

    if not (len(models) == len(f1) == len(roc)):
        raise ValueError("`models`, `f1_scores`, `roc_auc_scores` must have same length.")

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, f1, width, label="F1-score")
    plt.bar(x + width/2, roc, width, label="ROC-AUC")
    plt.ylim(0.6, 1.05)
    plt.ylabel("Score")
    plt.xticks(x, models, rotation=0)
    plt.title(title)
    plt.legend()

    # Annotate bars
    for i, v in enumerate(f1):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(roc):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅ Saved: {outfile}")

# -------------------------------
# 3) Example usage (fill in real values)
# -------------------------------

# A) FEATURE IMPORTANCE
# If you already have a trained sklearn LogisticRegression model:
# from joblib import load
# logreg = load("logreg_balanced.pkl")
# feature_names = ["DATA_0","DATA_1","DATA_2","DATA_3","DATA_4","DATA_5","DATA_6","DATA_7"]
# plot_logreg_coefficients(model=logreg, feature_names=feature_names)

# Or, if you only have coefficients (example length 8: DATA_0..DATA_7):
example_coefs = np.array([ 0.45, -0.12, 0.31, -0.52, 0.18, 0.07, -0.26, 0.09 ])  # <-- replace with your coef_
coefs = np.array([
    -0.290361258713655,   # DATA_0
    -1.92613197905109,    # DATA_1
    -0.635614962639666,   # DATA_2
     2.26984241755507,    # DATA_3
    -0.222772820313925,   # DATA_4
    -2.26574691236934,    # DATA_5
     0.0113146179606015,  # DATA_6
    -0.666378206801643    # DATA_7
])
feature_names = ["DATA_0","DATA_1","DATA_2","DATA_3","DATA_4","DATA_5","DATA_6","DATA_7"]
plot_logreg_coefficients(coefs=coefs, feature_names=feature_names,
                         title="Balanced Logistic Regression — Coefficient Magnitudes")

# B) MODEL PERFORMANCE BAR (대표 5모델)
models = ["LogReg","SVC","ExtraTrees","MLP","LSTM"]
f1_scores = [0.9315, 0.9315, 0.8330, 0.8331, 0.6472]
roc_auc_scores = [0.9911, 0.9911, 0.9977, 0.9299, 0.9407]
plot_model_performance_bar(models, f1_scores, roc_auc_scores)
