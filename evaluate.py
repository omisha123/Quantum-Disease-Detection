"""
Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Training Time
Compares Classical Ensemble (Swin Features) vs Quantum Hybrid Ensemble (Swin PCA Features)
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def evaluate_model(name, model, X_test, y_test, fit_time):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Ensure 1D probability array for positive class (Malignant = 1)
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


def run_full_comparison(X_train, X_test, y_train, y_test,
                         X_train_q, X_test_q,
                         n_qubits=4, quantum_subset=80, vqc_epochs=20):
    from classical_model import ClassicalEnsemble
    from image_pipeline import QuantumSwinHybridEnsemble

    results = []

    # --- Classical path (Full 768 Swin Features) ---
    print("=" * 50)
    print("Training Classical Ensemble (Swin Transformer Features)...")
    t0 = time.time()
    classical_model = ClassicalEnsemble().fit(X_train, y_train)
    classical_time = time.time() - t0
    results.append(evaluate_model("Classical Swin Ensemble (SVM+RF)",
                                   classical_model, X_test, y_test, classical_time))

    # --- Quantum path (PCA Reduced Swin Features) ---
    print("=" * 50)
    print("Training Quantum Hybrid Ensemble (Swin PCA -> QSVM+VQC)...")
    t0 = time.time()
    quantum_model = QuantumSwinHybridEnsemble(
        n_qubits=n_qubits, vqc_epochs=vqc_epochs
    ).fit(X_train_q[:quantum_subset], y_train[:quantum_subset])
    quantum_time = time.time() - t0
    results.append(evaluate_model("Quantum Swin Hybrid Ensemble (QSVM+VQC)",
                                   quantum_model, X_test_q, y_test, quantum_time))

    return results, classical_model, quantum_model


if __name__ == "__main__":
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from image_pipeline import load_mias_dataset

    print("1. Extracting Swin Transformer features for MIAS mammograms...")
    X_swin, y = load_mias_dataset(data_dir="./data/MIAS")

    # PCA reduction for quantum path
    N_QUBITS = 4
    pca = PCA(n_components=N_QUBITS)
    X_quantum = pca.fit_transform(X_swin)

    # Train / Test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_swin, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_q, X_test_q, _, _ = train_test_split(
        X_quantum, y, test_size=0.2, random_state=42, stratify=y
    )

    results, classical_model, quantum_model = run_full_comparison(
        X_train, X_test, y_train, y_test, X_train_q, X_test_q,
        n_qubits=N_QUBITS, quantum_subset=80, vqc_epochs=20
    )

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("FINAL COMPARISON RESULTS (MIAS SWIN-QUANTUM)")
    print("=" * 50)
    print(df.to_string(index=False))
    df.to_csv("results_comparison.csv", index=False)
    print("\nSaved evaluation benchmark to results_comparison.csv")