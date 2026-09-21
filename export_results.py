"""
Pre-computes results for ALL tabular datasets and saves to results.json for the dashboard.
Run this ONCE before your demo: python3 export_results.py
Then just open dashboard.html - no live training needed during the demo.

Resumable per-dataset: if interrupted, re-running skips datasets/patients already done.
"""

import json
import os
import time
import numpy as np

from data_pipeline import (
    load_breast_cancer_data, load_diabetes_risk_data, load_genomics_data,
    clean_and_scale, quantum_only_reduction
)
from classical_model import ClassicalEnsemble
from quantum_model import QuantumHybridEnsemble
from evaluate import evaluate_model
from explainability import explain_model, top_features_for_patient

N_QUBITS = 4
QUANTUM_SUBSET = 80
VQC_EPOCHS = 20
N_PATIENTS_TO_EXPLAIN = 5
RESULTS_PATH = "results.json"


def get_dataset_configs():
    configs = {}

    configs["breast_cancer"] = {
        "display_name": "Breast Cancer Wisconsin (Diagnostic)",
        "framing": "Diagnostic confirmation (post-biopsy) - benchmark dataset for direct comparison with published QML research",
        "loader": lambda: (*load_breast_cancer_data(), True)
    }

    configs["diabetes"] = {
        "display_name": "Pima Diabetes (Early Risk Screening)",
        "framing": "Genuinely early/pre-diagnosis - glucose, BMI, age, family history risk factors",
        "loader": lambda: (*load_diabetes_risk_data(), True)
    }

    configs["genomics"] = {
        "display_name": "Leukemia Gene Expression (Genomics)",
        "framing": "High-dimensional genomics data - illustrates classical ML's curse-of-dimensionality challenge",
        "loader": lambda: load_genomics_data()  # already returns (X, y, is_real)
    }

    return configs


def process_dataset(key, cfg, output):
    print(f"\n{'='*60}\nDataset: {cfg['display_name']}\n{'='*60}", flush=True)

    X, y, is_real = cfg["loader"]()
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)
    n_qubits = min(N_QUBITS, X_train.shape[1])
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=n_qubits)

    if key not in output["datasets"]:
        output["datasets"][key] = {
            "display_name": cfg["display_name"],
            "framing": cfg["framing"],
            "is_real_data": is_real,
            "evaluation": None,
            "dataset_info": None,
            "patients": []
        }
    ds_out = output["datasets"][key]
    ds_out["is_real_data"] = is_real

    if ds_out["evaluation"] is None:
        print("Training classical ensemble...", flush=True)
        t0 = time.time()
        classical_model = ClassicalEnsemble().fit(X_train, y_train)
        classical_time = time.time() - t0

        print("Training quantum hybrid ensemble...", flush=True)
        t0 = time.time()
        quantum_model = QuantumHybridEnsemble(n_qubits=n_qubits, vqc_epochs=VQC_EPOCHS)
        subset = min(QUANTUM_SUBSET, len(X_train_q))
        quantum_model.fit(X_train_q[:subset], y_train[:subset])
        quantum_time = time.time() - t0

        print("Evaluating...", flush=True)
        classical_result = evaluate_model("Classical Ensemble (SVM+RF)", classical_model, X_test, y_test, classical_time)
        quantum_result = evaluate_model("Quantum Hybrid Ensemble (QSVM+VQC)", quantum_model, X_test_q, y_test, quantum_time)

        ds_out["evaluation"] = {"classical": classical_result, "quantum": quantum_result}
        ds_out["dataset_info"] = {
            "name": cfg["display_name"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features_classical": X_train.shape[1],
            "n_features_quantum": n_qubits,
            "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 3)
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print("Trained and evaluated. Checkpoint saved.", flush=True)
    else:
        print("Already evaluated - re-training to explain patients...", flush=True)
        classical_model = ClassicalEnsemble().fit(X_train, y_train)
        quantum_model = QuantumHybridEnsemble(n_qubits=n_qubits, vqc_epochs=VQC_EPOCHS)
        subset = min(QUANTUM_SUBSET, len(X_train_q))
        quantum_model.fit(X_train_q[:subset], y_train[:subset])

    already_done = {p["patient_id"] for p in ds_out["patients"]}
    n_to_explain = min(N_PATIENTS_TO_EXPLAIN, len(X_test))

    for i in range(n_to_explain):
        if i in already_done:
            print(f"  Patient {i+1}/{n_to_explain} - already done, skipping", flush=True)
            continue

        print(f"  Patient {i+1}/{n_to_explain} - computing...", flush=True)

        c_risk = float(classical_model.predict_proba(X_test[i:i+1])[0])
        shap_vals_c, _ = explain_model(
            classical_model, X_train, X_test[i:i+1], feature_names, n_background=15, nsamples=60
        )
        top_c = top_features_for_patient(shap_vals_c[0], feature_names, top_n=6)

        q_risk = float(quantum_model.predict_proba(X_test_q[i:i+1])[0])
        pca_feature_names = [f"PC{j+1}" for j in range(n_qubits)]
        shap_vals_q, _ = explain_model(
            quantum_model, X_train_q[:subset], X_test_q[i:i+1],
            pca_feature_names, n_background=8, nsamples=30
        )
        top_q = top_features_for_patient(shap_vals_q[0], pca_feature_names, top_n=4)

        ds_out["patients"].append({
            "patient_id": i,
            "actual_label": "Positive/High-risk" if y_test[i] == 0 else "Negative/Low-risk",
            "classical": {
                "risk_score_pct": round((1 - c_risk) * 100, 1),
                "prediction": "Negative/Low-risk" if c_risk >= 0.5 else "Positive/High-risk",
                "top_features": [{"feature": f, "impact": v} for f, v in top_c]
            },
            "quantum": {
                "risk_score_pct": round((1 - q_risk) * 100, 1),
                "prediction": "Negative/Low-risk" if q_risk >= 0.5 else "Positive/High-risk",
                "top_features": [{"feature": f, "impact": v} for f, v in top_q]
            }
        })

        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Patient {i+1} done and saved.", flush=True)


if __name__ == "__main__":
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            output = json.load(f)
        if "datasets" not in output:
            output = {"datasets": {}}
    else:
        output = {"datasets": {}}

    configs = get_dataset_configs()
    for key, cfg in configs.items():
        process_dataset(key, cfg, output)

    print(f"\n{'='*60}\nDone! Results for {len(output['datasets'])} datasets saved in {RESULTS_PATH}")
    print("Now open dashboard.html in your browser - no more live training needed for the demo.")
