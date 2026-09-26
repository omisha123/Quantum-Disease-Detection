"""
Pre-computes results for MIAS Mammography and merges into results.json under
"imaging". Uses the SAME classical model (ClassicalEnsemble) and SAME quantum
model (TabularQuantumEnsemble, with QGA feature selection + QPSO hyperparameter
search) as export_results.py's heart disease path — Swin Transformer output is
a 768-dim numeric vector, i.e. tabular data, so the identical pipeline applies.

Run before demo: python export_image_results.py
"""

import base64
import io
import json
import os
import time
import glob
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

from image_pipeline import load_mias_dataset
from classical_model import ClassicalEnsemble
from tabular_quantum_model import (
    qga_feature_selection, qpso_search_vqc_hyperparams, TabularQuantumEnsemble
)

RESULTS_PATH = "results.json"
N_CASES_TO_SHOW = 8
N_QUBITS = 6
THRESHOLD = 0.35
DATA_DIR = "./data/MIAS"


def eval_metrics(name, y_true, preds, probs, fit_time):
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan"),
        "training_time_sec": round(fit_time, 2),
    }


def image_file_to_data_uri(img_path, display_size=112):
    try:
        im = Image.open(img_path).convert("L").resize((display_size, display_size), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"Could not load thumbnail for {img_path}: {e}")
        return ""


def find_metadata_csv():
    possible_paths = [
        os.path.join("data", "mias_info.csv"),
        os.path.join(DATA_DIR, "mias_info.csv"),
        "mias_info.csv",
        "./data/mias_info.csv"
    ] + glob.glob(os.path.join(DATA_DIR, "*info*.csv")) + glob.glob("data/*info*.csv")

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def get_all_dataset_images():
    valid_exts = ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.PGM", "*.PNG", "*.JPG")
    found_images = []
    search_dirs = [DATA_DIR, "./data", "data/MIAS", "MIAS"]
    for d in search_dirs:
        if os.path.exists(d):
            for ext in valid_exts:
                found_images.extend(glob.glob(os.path.join(d, ext)))
    return sorted(list(set(found_images)))


def main():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            output = json.load(f)
    else:
        output = {}
    output.setdefault("datasets", {})

    print("1. Loading MIAS dataset and extracting Swin Transformer embeddings...")
    X_swin, y = load_mias_dataset(data_dir=DATA_DIR)
    swin_feature_names = [f"swin_dim_{i}" for i in range(X_swin.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X_swin, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Classical Path (same ClassicalEnsemble used for heart disease) ---
    print("\n2. Training Classical Ensemble (Swin Features -> SVM + RF)...")
    t0 = time.time()
    classical_model = ClassicalEnsemble()
    classical_model.fit(X_train_scaled, y_train)
    classical_time = time.time() - t0

    classical_probs = classical_model.predict_proba(X_test_scaled)
    classical_probs_pos = classical_probs[:, 1] if classical_probs.ndim == 2 else classical_probs
    classical_preds = classical_model.predict(X_test_scaled, threshold=THRESHOLD)

    # --- Quantum Path: QGA feature selection + QPSO hyperparameter search + QSVM+VQC ---
    print("\n3. QGA-style feature selection (768 Swin dims -> 6 quantum features)...")
    X_train_q, X_test_q, selected_names = qga_feature_selection(
        X_train_scaled, X_test_scaled, y_train, swin_feature_names, n_components=N_QUBITS
    )

    print("4. QPSO-style hyperparameter search for VQC...")
    best_params = qpso_search_vqc_hyperparams(X_train_q, y_train, N_QUBITS)

    print("5. Training real QSVM + VQC with best-found hyperparameters...")
    t0 = time.time()
    quantum_model = TabularQuantumEnsemble(
        n_qubits=N_QUBITS,
        vqc_layers=best_params["vqc_layers"],
        vqc_epochs=best_params["vqc_epochs"]
    )
    quantum_model.fit(X_train_q, y_train)
    quantum_time = time.time() - t0

    quantum_probs = quantum_model.predict_proba(X_test_q)
    quantum_probs_pos = quantum_probs[:, 1] if quantum_probs.ndim == 2 else quantum_probs
    quantum_preds = quantum_model.predict(X_test_q, threshold=THRESHOLD)

    # --- Evaluate ---
    classical_result = eval_metrics("Classical Ensemble (SVM+RF)", y_test, classical_preds, classical_probs_pos, classical_time)
    quantum_result = eval_metrics("Quantum Ensemble (QGA+QPSO+QSVM+VQC)", y_test, quantum_preds, quantum_probs_pos, quantum_time)

    print(f"\nClassical Results: {classical_result}")
    print(f"Quantum Results:   {quantum_result}")
    print(f"\n>>> MIAS Mammography: classical acc={classical_result['accuracy']:.3f}, "
          f"quantum acc={quantum_result['accuracy']:.3f}\n")

    # --- Thumbnails ---
    csv_path = find_metadata_csv()
    df = None
    id_col = None
    if csv_path:
        print(f"\nUsing metadata CSV for thumbnails: {csv_path}")
        try:
            raw_rows = []
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].upper() not in ['REFNUM', 'MIAS']:
                        raw_rows.append(parts[0])
            df = pd.DataFrame({'REFNUM': raw_rows})
            id_col = 'REFNUM'
        except Exception as e:
            print(f"Could not parse CSV for thumbnails: {e}")

    fallback_images = get_all_dataset_images()
    print(f"Found {len(fallback_images)} image files in local directories.")

    n_cases = min(N_CASES_TO_SHOW, len(X_test))
    cases = []

    for i in range(n_cases):
        actual = "Malignant" if y_test[i] == 1 else "Benign/Normal"
        c_pred = "Malignant" if classical_preds[i] == 1 else "Benign/Normal"
        q_pred = "Malignant" if quantum_preds[i] == 1 else "Benign/Normal"

        thumb_uri = ""
        if df is not None and id_col in df.columns and i < len(df):
            file_id = str(df.iloc[i][id_col]).strip()
            matches = glob.glob(os.path.join(DATA_DIR, f"{file_id}*"))
            if not matches:
                matches = glob.glob(os.path.join("./data", f"{file_id}*"))
            if matches:
                thumb_uri = image_file_to_data_uri(matches[0])

        if not thumb_uri and i < len(fallback_images):
            thumb_uri = image_file_to_data_uri(fallback_images[i])

        cases.append({
            "case_id": i,
            "thumbnail": thumb_uri,
            "actual_label": actual,
            "classical": {
                "prediction": c_pred,
                "risk_score_pct": round(float(classical_probs_pos[i]) * 100, 1),
            },
            "quantum": {
                "prediction": q_pred,
                "risk_score_pct": round(float(quantum_probs_pos[i]) * 100, 1),
            },
        })

    # --- Save under "datasets" (same shape/location as heart disease etc.) ---
    output["datasets"]["mias_mammography"] = {
        "display_name": "MIAS Mammography (Swin Transformer)",
        "framing": "Mammography image screening, Swin embeddings treated as tabular "
                   "features — same QGA+QPSO+QSVM+VQC pipeline as heart disease",
        "is_real_data": True,
        "evaluation": {"classical": classical_result, "quantum": quantum_result},
        "dataset_info": {
            "name": "MIAS Mammography",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features_classical": X_train_scaled.shape[1],
            "n_features_quantum": N_QUBITS,
            "qga_selected_features": selected_names,
            "qpso_best_params": best_params,
        },
        "cases": cases,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved MIAS results (unified QGA+QPSO pipeline, {n_cases} cases) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()