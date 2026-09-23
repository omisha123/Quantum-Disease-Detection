import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from image_pipeline import load_mias_dataset, QuantumSwinHybridEnsemble
from classical_model import ClassicalEnsemble
from evaluate import evaluate_model

def run_experiment(n_qubits=4, quantum_subset=80, vqc_epochs=20):
    print("⏳ Loading MIAS dataset & extracting Swin features...")
    X_raw, y = load_mias_dataset(data_dir="./data/MIAS")

    # PCA Reduction
    pca = PCA(n_components=n_qubits)
    X_quantum = pca.fit_transform(X_raw)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_q, X_test_q, _, _ = train_test_split(
        X_quantum, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Train Classical Ensemble
    print("🔵 Training Classical Ensemble...")
    t0 = time.time()
    classical_model = ClassicalEnsemble().fit(X_train, y_train)
    classical_time = time.time() - t0

    # 2. Train Quantum Hybrid Ensemble
    print("🟣 Training Quantum Hybrid Ensemble...")
    t0 = time.time()
    quantum_model = QuantumSwinHybridEnsemble(n_qubits=n_qubits, vqc_epochs=vqc_epochs)
    quantum_model.fit(X_train_q[:quantum_subset], y_train[:quantum_subset])
    quantum_time = time.time() - t0

    # 3. Evaluate Models
    print("📊 Evaluating metrics...")
    classical_res = evaluate_model("Classical Swin Ensemble", classical_model, X_test, y_test, classical_time)
    quantum_res = evaluate_model("Quantum Swin Hybrid Ensemble", quantum_model, X_test_q, y_test, quantum_time)

    # Clean numeric types for JSON
    def clean_metrics(res):
        cleaned = {}
        for k, v in res.items():
            if isinstance(v, (np.floating, float)):
                cleaned[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                cleaned[k] = int(v)
            else:
                cleaned[k] = str(v)
        return cleaned

    c_clean = clean_metrics(classical_res)
    q_clean = clean_metrics(quantum_res)

    # Generate sample test cases for visual preview
    sample_cases = []
    for idx in range(min(4, len(X_test))):
        c_pred = classical_model.predict(X_test[idx:idx+1])[0]
        q_pred = quantum_model.predict(X_test_q[idx:idx+1])[0]
        actual = "Malignant" if y_test[idx] == 1 else "Benign"
        
        sample_cases.append({
            "case_id": idx,
            "actual_label": actual,
            "thumbnail": f"https://via.placeholder.com/130/171c2a/8c94a8?text=Case+{idx+1}",
            "classical": {"prediction": "Malignant" if c_pred == 1 else "Benign"},
            "quantum": {"prediction": "Malignant" if q_pred == 1 else "Benign"}
        })

    # Combined payload for both HTML views
    results_payload = {
        "imaging": {
            "evaluation": {
                "classical": c_clean,
                "quantum": q_clean
            },
            "cases": sample_cases
        },
        "datasets": {
            "mias_benchmarks": {
                "evaluation": {
                    "classical": c_clean,
                    "quantum": q_clean
                }
            }
        }
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("✅ Successfully exported to results.json")

if __name__ == "__main__":
    run_experiment()