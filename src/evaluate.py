import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def metrics_report(name, y_true, y_prob=None, y_pred=None, plot_cm=True, thr=None, cmap=None):
    if y_pred is None:
        if y_prob is None:
            raise ValueError("Either y_prob or y_pred is required.")
        default_thr = 0.0 if (y_prob.min() < 0 or y_prob.max() > 1) else 0.5
        use_thr = default_thr if thr is None else thr
        y_pred = (y_prob >= use_thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        aucv = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    except Exception:
        aucv = np.nan

    print(f"[{name}]  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={aucv:.4f}")

    if plot_cm:
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(
            cmap=cmap or plt.cm.Blues
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.grid(False)
        plt.show()

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": aucv}


def best_threshold_by_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1s[:-1])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1s[idx]),
    }


def get_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict(X).astype(float)


def tune_and_report(name, y_true, scores):
    best = best_threshold_by_f1(y_true, scores)
    print(f"Best threshold by F1 ({name}): {best}")
    metrics_report(f"{name}-bestF1", y_true, y_prob=scores, thr=best["threshold"])


def eval_with_thresholds(y_true, scores, thr=0.5, label="Model"):
    pred = (scores >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, scores)

    p_curve, r_curve, t_curve = precision_recall_curve(y_true, scores)
    f1_curve = (2 * p_curve * r_curve) / (p_curve + r_curve + 1e-12)
    best_idx = int(np.nanargmax(f1_curve))
    best_f1 = float(f1_curve[best_idx])
    if best_idx >= len(t_curve):
        best_idx = len(t_curve) - 1
    best_thr = float(t_curve[max(0, best_idx)])

    pred_best = (scores >= best_thr).astype(int)
    acc_b = accuracy_score(y_true, pred_best)
    prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
        y_true, pred_best, average="binary", zero_division=0
    )

    return {
        "label": label,
        "thr_default": float(thr),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "thr_bestF1": best_thr,
        "acc_b": float(acc_b),
        "prec_b": float(prec_b),
        "rec_b": float(rec_b),
        "f1_b": float(f1_b),
        "bestF1": best_f1,
    }

