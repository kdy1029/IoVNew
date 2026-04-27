# ==========================================
# ROC Curve: Extra Trees vs MLP INT8 (VX) - robust loader
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# File paths
TREE_CSV = "pred_extratrees.csv"
MLP_CSV  = "pred_mlp_npu.csv"

# 1) Load CSV
tree = pd.read_csv(TREE_CSV)
mlp  = pd.read_csv(MLP_CSV)

# 2) Automatic column detection for labels and scores
def pick_cols(df):
    cols = {c.lower(): c for c in df.columns}
    # Candidates for y_true
    ytrue_candidates = ['label', 'category', 'target', 'y_true', 'gt', 'truth']
    y_true_col = next((cols[c] for c in ytrue_candidates if c in cols), None)
    # Candidates for score
    score_candidates = ['prob', 'score', 'y_score', 'proba', 'confidence']
    y_score_col = next((cols[c] for c in score_candidates if c in cols), None)
    if y_true_col is None or y_score_col is None:
        raise ValueError(f"Cannot find true/score columns. columns={list(df.columns)}")
    return y_true_col, y_score_col

yt_tree, ys_tree = pick_cols(tree)
yt_mlp,  ys_mlp  = pick_cols(mlp)

# 3) True label vector (must be identical for both files) — use from one side only
y_true_raw = tree[yt_tree].values

# Map string labels to 0/1
if y_true_raw.dtype.kind in {'U','S','O'}:
    # Common label mappings
    mapping = {
        'ATTACK': 1, 'BENIGN': 0,
        'MALICIOUS': 1, 'NORMAL': 0,
        'attack': 1, 'benign': 0, 'normal': 0
    }
    y_true = pd.Series(y_true_raw).map(mapping)
    if y_true.isna().any():
        uniq = pd.unique(y_true_raw)
        raise ValueError(f"Label mapping failed. Please update mapping for labels={uniq}.")
    y_true = y_true.values.astype(int)
else:
    # Cast to int if already numeric
    y_true = pd.Series(y_true_raw).astype(int).values

# 4) Score vector
y_score_tree = pd.to_numeric(tree[ys_tree], errors='coerce').fillna(0).values
y_score_mlp  = pd.to_numeric(mlp[ys_mlp],  errors='coerce').fillna(0).values

# Verify lengths
assert len(y_true) == len(y_score_tree) == len(y_score_mlp), \
    f"Lengths differ: y_true={len(y_true)}, tree={len(y_score_tree)}, mlp={len(y_score_mlp)}"

# 5) Calculate ROC
fpr_tree, tpr_tree, _ = roc_curve(y_true, y_score_tree)
fpr_mlp,  tpr_mlp,  _ = roc_curve(y_true, y_score_mlp)
roc_auc_tree = auc(fpr_tree, tpr_tree)
roc_auc_mlp  = auc(fpr_mlp,  tpr_mlp)

# 6) Plot
plt.figure(figsize=(6, 4))
plt.plot(fpr_tree, tpr_tree, lw=2.2, color='#1b9e77',
         label=f'Extra Trees (AUC = {roc_auc_tree:.3f})')
plt.plot(fpr_mlp,  tpr_mlp,  lw=2.2, color='#d95f02',
         label=f'MLP INT8 (VX, AUC = {roc_auc_mlp:.3f})')
plt.plot([0,1],[0,1],'k--',lw=1,label='Chance (AUC = 0.5)')
plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Extra Trees vs Quantized MLP (INT8 VX)')
plt.legend(loc='lower right', fontsize=9, frameon=True)
plt.grid(alpha=0.2)
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.12, right=0.96)
plt.savefig('figures/roc_curve_extratrees_vs_mlpvx.pdf',
            dpi=300, bbox_inches='tight', pad_inches=0.3)
plt.close()
print("✅ Saved: figures/roc_curve_extratrees_vs_mlpvx.pdf")
