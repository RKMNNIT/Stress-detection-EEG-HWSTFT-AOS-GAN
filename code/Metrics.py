import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def PerformanceMetrics():
    mat = scipy.io.loadmat('Selected_Features.mat')
    y_true = mat['labels'].ravel()
    pred_mat = scipy.io.loadmat('Predictions.mat')
    y_pred = pred_mat['y_pred'].ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== PERFORMANCE METRICS =====\n")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(8, 5))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [acc*100, precision*100, recall*100, f1*100]
    plt.bar(metrics, values, color=['purple','red','blue','green'])
    plt.ylim(0, 100)
    plt.ylabel('Score (%)')
    plt.title('Model Performance Metrics')
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    print("\n===== METRICS COMPUTED SUCCESSFULLY =====\n")
