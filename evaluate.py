"""
Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Training Time
Compares Classical Ensemble (Prediction B) vs Quantum Hybrid Ensemble (Prediction A)
"""

import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def evaluate_model(name, model, X_test, y_test, fit_time):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc_roc": roc_auc_score(y_test, probs),
        "training_time_sec": round(fit_time, 2),
    }


def run_full_comparison(X_train, X_test, y_train, y_test,
                         X_train_q, X_test_q,
                         n_qubits=4, quantum_subset=80, vqc_epochs=20):
    from classical_model import ClassicalEnsemble
    from quantum_model import QuantumHybridEnsemble

    results = []

    # --- Classical path (full features) ---
    print("=" * 50)
    print("Training Classical Ensemble (full feature set)...")
    t0 = time.time()
    classical_model = ClassicalEnsemble().fit(X_train, y_train)
    classical_time = time.time() - t0
    results.append(evaluate_model("Classical Ensemble (SVM+RF)",
                                   classical_model, X_test, y_test, classical_time))

    # --- Quantum path (reduced features) ---
    print("=" * 50)
    print("Training Quantum Hybrid Ensemble (reduced feature set)...")
    t0 = time.time()
    quantum_model = QuantumHybridEnsemble(
        n_qubits=n_qubits, vqc_epochs=vqc_epochs
    ).fit(X_train_q[:quantum_subset], y_train[:quantum_subset])
    quantum_time = time.time() - t0
    results.append(evaluate_model("Quantum Hybrid Ensemble (QSVM+VQC)",
                                   quantum_model, X_test_q, y_test, quantum_time))

    return results, classical_model, quantum_model


if __name__ == "__main__":
    import pandas as pd
    from data_pipeline import load_breast_cancer_data, clean_and_scale, quantum_only_reduction

    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=4)

    results, classical_model, quantum_model = run_full_comparison(
        X_train, X_test, y_train, y_test, X_train_q, X_test_q,
        n_qubits=4, quantum_subset=80, vqc_epochs=20
    )

    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    print(df.to_string(index=False))
    df.to_csv("results_comparison.csv", index=False)
    print("\nSaved to results_comparison.csv")
