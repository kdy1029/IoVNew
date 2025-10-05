# ==========================================
# IoV IDS Visualization (GPU Results)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# -----------------------------
# 1️⃣ ROC & PR Curves (LogReg vs LSTM)
# -----------------------------

# === Replace with your actual y_true, y_score ===
# Example: if you have them from model.predict_proba()
# y_true = np.load('y_true.npy')
# y_score_lr = np.load('y_score_logreg.npy')
# y_score_lstm = np.load('y_score_lstm.npy')

# For demonstration: mock data (keep same shape as real)
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_score_lr = np.clip(y_true + np.random.normal(0, 0.1, 1000), 0, 1)
y_score_lstm = np.clip(y_true + np.random.normal(0, 0.25, 1000), 0, 1)

# --- ROC curves ---
fpr_lr, tpr_lr, _ = roc_curve(y_true, y_score_lr)
fpr_lstm, tpr_lstm, _ = roc_curve(y_true, y_score_lstm)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

# --- PR curves ---
prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_score_lr)
prec_lstm, rec_lstm, _ = precision_recall_curve(y_true, y_score_lstm)

# --- Plot ROC ---
plt.figure(figsize=(6,4))
plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={roc_auc_lr:.3f})', lw=2)
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC={roc_auc_lstm:.3f})', lw=2)
plt.plot([0,1],[0,1],'k--',lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (GPU Evaluation)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/roc_curve.pdf', dpi=300)
plt.close()

# --- Plot PR ---
plt.figure(figsize=(6,4))
plt.plot(rec_lr, prec_lr, label='LogReg', lw=2)
plt.plot(rec_lstm, prec_lstm, label='LSTM', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (GPU Evaluation)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('figures/pr_curve.pdf', dpi=300)
plt.close()

print("✅ Saved: figures/roc_curve.pdf, figures/pr_curve.pdf")

# -----------------------------
# 2️⃣ Model Performance Bar Chart
# -----------------------------
models = ['LogReg', 'SVC', 'ExtraTrees', 'MLP', 'LSTM']
f1_scores = [0.9315, 0.9315, 0.8330, 0.8331, 0.6472]
roc_auc = [0.9911, 0.9911, 0.9977, 0.9299, 0.9407]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(6,3))
plt.bar(x - width/2, f1_scores, width, label='F1-score')
plt.bar(x + width/2, roc_auc, width, label='ROC-AUC')
plt.ylim(0.6, 1.05)
plt.ylabel('Score')
plt.xticks(x, models)
plt.legend()
plt.title('Representative Model Performance (Decimal CICIoV2024)')
plt.tight_layout()
plt.savefig('figures/model_performance_bar.pdf', dpi=300)
plt.close()

print("✅ Saved: figures/model_performance_bar.pdf")
