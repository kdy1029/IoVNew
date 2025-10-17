# ==========================================
# IoV IDS Visualization (CPU-only Results)
# ==========================================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

os.makedirs("figures", exist_ok=True)

# -----------------------------
# 1) ROC & PR Curves (LogReg vs LSTM)  -- CPU-only
# -----------------------------
# ⬇️ 실제 결과 배열로 교체하세요.
# y_true = np.load('figures/y_true_test.npy')
# y_score_lr = np.load('figures/y_score_logreg_test.npy')   # predict_proba[:,1] or decision_function normalized
# y_score_lstm = np.load('figures/y_score_lstm_test.npy')   # sigmoid outputs

# --- Fallback mock (삭제 가능): 형상 확인용 ---
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_score_lr = np.clip(y_true + np.random.normal(0, 0.1, 1000), 0, 1)
y_score_lstm = np.clip(y_true + np.random.normal(0, 0.25, 1000), 0, 1)

# ROC
fpr_lr, tpr_lr, _ = roc_curve(y_true, y_score_lr)
fpr_lstm, tpr_lstm, _ = roc_curve(y_true, y_score_lstm)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# PR
prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_score_lr)
prec_lstm, rec_lstm, _ = precision_recall_curve(y_true, y_score_lstm)
ap_lr = average_precision_score(y_true, y_score_lr)
ap_lstm = average_precision_score(y_true, y_score_lstm)

# --- Plot ROC ---
plt.figure(figsize=(6, 4))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={roc_auc_lr:.3f})', lw=2)
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC={roc_auc_lstm:.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (CPU-only Evaluation)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/roc_curve_cpu.pdf', dpi=300)
plt.close()

# --- Plot PR ---
plt.figure(figsize=(6, 4))
plt.plot(rec_lr, prec_lr, label=f'LogReg (AP={ap_lr:.3f})', lw=2)
plt.plot(rec_lstm, prec_lstm, label=f'LSTM (AP={ap_lstm:.3f})', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (CPU-only Evaluation)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('figures/pr_curve_cpu.pdf', dpi=300)
plt.close()

print("✅ Saved: figures/roc_curve_cpu.pdf, figures/pr_curve_cpu.pdf")

# -----------------------------
# 2) Model Performance Bar Chart (CPU-only, F1@τ* from validation)
# -----------------------------
# 논문 표(τ* from validation)와 수치를 일치시킴
models = ['LogReg', 'LinearSVC', 'ExtraTrees', 'GaussianNB', 'MLP', 'LSTM']
f1_tau =  [0.972,   0.974,       0.998,        0.986,        0.946,  0.934]
roc_auc  = [0.876,   0.852,       0.994,        0.929,        0.852,  0.858]

x = np.arange(len(models))
width = 0.38

plt.figure(figsize=(6.4, 3.4))
plt.bar(x - width/2, f1_tau, width, label='F1@τ*')
plt.bar(x + width/2, roc_auc, width, label='ROC–AUC')
plt.ylim(0.6, 1.05)
plt.ylabel('Score')
plt.xticks(x, models)
plt.title('Representative Model Performance (CPU-only, Decimal CICIoV2024)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/model_performance_bar_cpu.pdf', dpi=300)
plt.close()

print("✅ Saved: figures/model_performance_bar_cpu.pdf")
