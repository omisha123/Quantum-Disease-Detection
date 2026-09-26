"""
Pre-computes results for tabular datasets and saves to results.json for the dashboard.
Run: python export_results.py
Resumable per-dataset: if interrupted, re-running skips datasets/patients already done.
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from classical_model import ClassicalEnsemble
from evaluate import evaluate_model
from explainability import explain_model, top_features_for_patient
from tabular_quantum_model import (
    qga_feature_selection, qpso_search_vqc_hyperparams, TabularQuantumEnsemble
)

N_QUBITS = 6
QUANTUM_SUBSET = 80
VQC_EPOCHS = 20
THRESHOLD = 0.35
N_PATIENTS_TO_EXPLAIN = 5
RESULTS_PATH = "results.json"


# ==========================================
# Data Loaders
# ==========================================

def load_diabetes_risk_data():
    np.random.seed(42)
    n = 300
    glucose = np.random.normal(120, 30, n)
    bmi = np.random.normal(30, 6, n)
    age = np.random.normal(45, 12, n)
    bp = np.random.normal(70, 10, n)
    insulin = np.random.normal(80, 25, n)
    pedigree = np.random.normal(0.5, 0.2, n)

    logits = 0.03 * glucose + 0.05 * bmi + 0.02 * age - 5.0
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > 0.5).astype(int)

    X = np.column_stack([glucose, bmi, age, bp, insulin, pedigree])
    feature_names = ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "DiabetesPedigree"]
    return X, y, feature_names, False


def load_genomics_data():
    np.random.seed(101)
    n_samples, n_genes = 120, 200
    X = np.random.randn(n_samples, n_genes)
    logits = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5
    y = (logits > 0).astype(int)
    feature_names = [f"Gene_{i+1}" for i in range(n_genes)]
    return X, y, feature_names, False


def load_heart_disease_data(csv_path="data/heart.csv"):
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        url = ("https://raw.githubusercontent.com/shubham20315/"
               "Prediction-of-likeliness-of-cardiovascular-disease-using-machine-learning-algorithms/"
               "main/heart.csv")
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Heart disease: deduplicated {before} rows -> {len(df)} unique rows")

    y = df["target"].to_numpy()
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].to_numpy()
    return X, y, feature_names, True


def clean_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def quantum_only_reduction(X_train_scaled, X_test_scaled, n_components=4):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    def rescale_to_pi(arr, ref_min, ref_max):
        denom = np.where((ref_max - ref_min) == 0, 1e-8, ref_max - ref_min)
        return -np.pi + (arr - ref_min) * (2 * np.pi) / denom

    train_min, train_max = X_train_pca.min(axis=0), X_train_pca.max(axis=0)
    X_train_q = rescale_to_pi(X_train_pca, train_min, train_max)
    X_test_q = rescale_to_pi(X_test_pca, train_min, train_max)
    X_test_q = np.clip(X_test_q, -np.pi, np.pi)
    return X_train_q, X_test_q, pca


def get_dataset_configs():
    configs = {}

    configs["heart_disease"] = {
        "display_name": "Heart Disease (UCI Cleveland)",
        "framing": "Coronary heart disease risk from standard clinical features "
                   "(chest pain type, resting BP, cholesterol, ST depression, etc.) "
                   "— primary target condition, using QGA-style feature selection + "
                   "QPSO-style hyperparameter search + real QSVM+VQC",
        "loader": load_heart_disease_data
    }

    configs["diabetes"] = {
        "display_name": "Pima Diabetes (Early Risk Screening)",
        "framing": "Genuinely early/pre-diagnosis - glucose, BMI, age, family history risk factors",
        "loader": load_diabetes_risk_data
    }

    configs["genomics"] = {
        "display_name": "Leukemia Gene Expression (Genomics)",
        "framing": "High-dimensional genomics data - illustrates classical ML's curse-of-dimensionality challenge",
        "loader": load_genomics_data
    }

    return configs


def extract_risk_score(model, X_sample):
    try:
        probs = model.predict_proba(X_sample)
        if isinstance(probs, np.ndarray):
            if probs.ndim == 2 and probs.shape[1] > 1:
                return float(probs[0, 1])
            elif probs.ndim == 1:
                return float(probs[0])
        return float(probs)
    except Exception:
        pred = model.predict(X_sample)
        return float(pred[0])


def process_dataset(key, cfg, output):
    print(f"\n{'='*60}\nDataset: {cfg['display_name']}\n{'='*60}", flush=True)

    X, y, feature_names, is_real = cfg["loader"]()
    X_train, X_test, y_train, y_test, scaler = clean_and_scale(X, y)

    n_qubits = min(N_QUBITS, X_train.shape[1])
    X_train_q, X_test_q, selected_names = qga_feature_selection(
        X_train, X_test, y_train, feature_names, n_components=n_qubits
    )

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

    # Always (re)train both models fresh — needed for SHAP regardless of resume state
    print("Training classical ensemble...", flush=True)
    t0 = time.time()
    classical_model = ClassicalEnsemble().fit(X_train, y_train)
    classical_time = time.time() - t0

    print("Running QPSO-style hyperparameter search for VQC...", flush=True)
    best_params = qpso_search_vqc_hyperparams(X_train_q, y_train, n_qubits)

    print("Training quantum model with best-found hyperparameters...", flush=True)
    t0 = time.time()
    quantum_model = TabularQuantumEnsemble(
        n_qubits=n_qubits,
        vqc_layers=best_params["vqc_layers"],
        vqc_epochs=best_params["vqc_epochs"]
    )
    quantum_model.fit(X_train_q, y_train)
    quantum_time = time.time() - t0
    subset = len(X_train_q)

    if ds_out["evaluation"] is None:
        print("Evaluating...", flush=True)
        classical_result = evaluate_model(
            "Classical Ensemble (SVM+RF)", classical_model, X_test, y_test, classical_time, threshold=THRESHOLD
        )
        quantum_result = evaluate_model(
            "Quantum Ensemble (QGA+QPSO+QSVM+VQC)", quantum_model, X_test_q, y_test, quantum_time, threshold=THRESHOLD
        )

        print(f"\n>>> {cfg['display_name']}: classical acc={classical_result['accuracy']:.3f}, "
              f"quantum acc={quantum_result['accuracy']:.3f}\n")

        ds_out["evaluation"] = {"classical": classical_result, "quantum": quantum_result}
        ds_out["dataset_info"] = {
            "name": cfg["display_name"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features_classical": X_train.shape[1],
            "n_features_quantum": n_qubits,
            "qga_selected_features": selected_names,
            "qpso_best_params": best_params,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print("Trained and evaluated. Checkpoint saved.", flush=True)
    else:
        print("Already evaluated - models retrained above for patient explainability.", flush=True)

    already_done = {p["patient_id"] for p in ds_out["patients"]}
    n_to_explain = min(N_PATIENTS_TO_EXPLAIN, len(X_test))

    for i in range(n_to_explain):
        if i in already_done:
            print(f"  Patient {i+1}/{n_to_explain} - already done, skipping", flush=True)
            continue

        print(f"  Patient {i+1}/{n_to_explain} - computing SHAP explanations...", flush=True)

        c_prob_pos = extract_risk_score(classical_model, X_test[i:i+1])
        shap_vals_c, _ = explain_model(
            classical_model, X_train, X_test[i:i+1], feature_names, n_background=15, nsamples=60
        )
        top_c = top_features_for_patient(shap_vals_c[0], feature_names, top_n=6)

        q_prob_pos = extract_risk_score(quantum_model, X_test_q[i:i+1])
        shap_vals_q, _ = explain_model(
            quantum_model,
            X_train_q[:subset],
            X_test_q[i:i+1],
            selected_names,
            n_background=3,
            nsamples=10
        )
        top_q = top_features_for_patient(shap_vals_q[0], selected_names, top_n=4)

        ds_out["patients"].append({
            "patient_id": i,
            "actual_label": "Positive/High-risk" if y_test[i] == 1 else "Negative/Low-risk",
            "classical": {
                "risk_score_pct": round(c_prob_pos * 100, 1),
                "prediction": "Positive/High-risk" if c_prob_pos >= THRESHOLD else "Negative/Low-risk",
                "top_features": [{"feature": str(f), "impact": float(v)} for f, v in top_c]
            },
            "quantum": {
                "risk_score_pct": round(q_prob_pos * 100, 1),
                "prediction": "Positive/High-risk" if q_prob_pos >= THRESHOLD else "Negative/Low-risk",
                "top_features": [{"feature": str(f), "impact": float(v)} for f, v in top_q]
            }
        })

        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Patient {i+1} done and saved.", flush=True)

if __name__ == "__main__":
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                output = json.load(f)
            if "datasets" not in output:
                output["datasets"] = {}
        except Exception:
            output = {"datasets": {}}
    else:
        output = {"datasets": {}}

    # Remove any leftover breast_cancer tabular entry (now MIAS-only, under "imaging")
    output["datasets"].pop("breast_cancer", None)

    configs = get_dataset_configs()
    for key, cfg in configs.items():
        process_dataset(key, cfg, output)

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}\nDone! Pre-computed results for {len(output['datasets'])} datasets saved in '{RESULTS_PATH}'.")
    print("Now open dashboard.html in your browser.")