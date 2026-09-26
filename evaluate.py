"""
Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Training Time
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def evaluate_model(name, model, X_test, y_test, fit_time, threshold=0.35):
    preds = model.predict(X_test, threshold=threshold)
    probs = model.predict_proba(X_test)

    if isinstance(probs, np.ndarray) and probs.ndim > 1:
        probs = probs[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else float("nan"),
        "training_time_sec": round(fit_time, 2),
    }