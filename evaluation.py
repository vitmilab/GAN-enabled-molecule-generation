import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve
import seaborn as sns
import numpy as np

# --- Predicted probabilities and labels from your test set ---
y_true = y_test
y_prob = y_prob_test
y_pred = y_pred_test

# -----------------------
# 1) ROC Curve
# -----------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'ROC_curve.png'), dpi=300)
plt.show()

# -----------------------
# 2) Precision-Recall Curve
# -----------------------
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
avg_prec = average_precision_score(y_true, y_prob)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_prec:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'PR_curve.png'), dpi=300)
plt.show()

# -----------------------
# 3) Confusion Matrix Heatmap
# -----------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'Confusion_Matrix.png'), dpi=300)
plt.show()

# -----------------------
# 4) Calibration Plot
# -----------------------
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', lw=2, label='XGBoost')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Fraction')
plt.title('Calibration Plot (Test Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'Calibration_plot.png'), dpi=300)
plt.show()


from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred_test)
print(f"MCC: {mcc:.4f}")

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_prob_test)
mcc = matthews_corrcoef(y_test, y_pred_test)

print("\nTEST PERFORMANCE (COMPREHENSIVE)")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"MCC: {mcc:.4f}")
