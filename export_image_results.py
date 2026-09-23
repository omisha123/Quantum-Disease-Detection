"""
Pre-computes results for the MIAS Mammography Swin-Quantum path and
merges them into results.json under the "imaging" key for the dashboard.

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
from sklearn.decomposition import PCA

# Suppress sklearn future warnings regarding SVC probability
warnings.filterwarnings("ignore", category=FutureWarning)

# Imports from your actual Swin/Quantum project files
from image_pipeline import load_mias_dataset, QuantumSwinHybridEnsemble
from classical_model import ClassicalEnsemble

RESULTS_PATH = "results.json"
N_CASES_TO_SHOW = 8
N_QUBITS = 4
DATA_DIR = "./data/MIAS"


def eval_metrics(name, y_true, preds, probs, fit_time):
    # Ensure 1D probability array for positive class
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
    """Converts a local MIAS mammogram file (.pgm/.png/.jpg) into a base64 URI for web dashboard rendering."""
    try:
        im = Image.open(img_path).convert("L").resize((display_size, display_size), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"Could not load thumbnail for {img_path}: {e}")
        return ""


def find_metadata_csv():
    """Robust lookup for the MIAS metadata CSV across multiple directory structures."""
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
    """Scans dataset directories directly for image files as a fallback."""
    valid_exts = ("*.pgm", "*.png", "*.jpg", "*.jpeg", "*.PGM", "*.PNG", "*.JPG")
    found_images = []
    search_dirs = [DATA_DIR, "./data", "data/MIAS", "MIAS"]
    
    for d in search_dirs:
        if os.path.exists(d):
            for ext in valid_exts:
                found_images.extend(glob.glob(os.path.join(d, ext)))
                
    # Deduplicate and sort
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

    # Dimensionality Reduction for Quantum Circuit
    pca = PCA(n_components=N_QUBITS)
    X_quantum = pca.fit_transform(X_swin)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_swin, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_q, X_test_q, _, _ = train_test_split(
        X_quantum, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Train Classical Path ---
    print("\n2. Training Classical Ensemble (Swin Features -> SVM + RF)...")
    t0 = time.time()
    classical_model = ClassicalEnsemble()
    classical_model.fit(X_train, y_train)
    classical_time = time.time() - t0

    classical_probs = classical_model.predict_proba(X_test)
    if classical_probs.ndim == 2:
        classical_probs_pos = classical_probs[:, 1]
    else:
        classical_probs_pos = classical_probs
    
    # Lower threshold to 0.35 to catch positive malignant cases
    classical_preds = (classical_probs_pos >= 0.35).astype(int)

    # --- 3. Train Quantum Path ---
    print("\n3. Training Quantum Hybrid Ensemble (Swin PCA -> QSVM + VQC)...")
    t0 = time.time()
    quantum_model = QuantumSwinHybridEnsemble(n_qubits=N_QUBITS, vqc_epochs=20)
    quantum_model.fit(X_train_q, y_train)
    quantum_time = time.time() - t0

    quantum_probs = quantum_model.predict_proba(X_test_q)
    if quantum_probs.ndim == 2:
        quantum_probs_pos = quantum_probs[:, 1]
    else:
        quantum_probs_pos = quantum_probs
    
    # Predict using threshold = 0.35 to match class imbalance fix
    quantum_preds = quantum_model.predict(X_test_q, threshold=0.35)

    # --- 4. Evaluate Metrics ---
    classical_result = eval_metrics("Classical Swin Ensemble", y_test, classical_preds, classical_probs_pos, classical_time)
    quantum_result = eval_metrics("Quantum Swin Hybrid Ensemble", y_test, quantum_preds, quantum_probs_pos, quantum_time)

    print(f"\nClassical Results: {classical_result}")
    print(f"Quantum Results:   {quantum_result}")

    # --- 5. Generate Test Case Thumbnails for Dashboard ---
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

    # Pre-fetch direct images as fallback
    fallback_images = get_all_dataset_images()
    print(f"Found {len(fallback_images)} image files in local directories.")

    n_cases = min(N_CASES_TO_SHOW, len(X_test))
    cases = []

    for i in range(n_cases):
        actual = "Malignant" if y_test[i] == 1 else "Benign/Normal"
        c_pred = "Malignant" if classical_preds[i] == 1 else "Benign/Normal"
        q_pred = "Malignant" if quantum_preds[i] == 1 else "Benign/Normal"

        thumb_uri = ""

        # Strategy A: Try metadata CSV ID lookup
        if df is not None and id_col in df.columns and i < len(df):
            file_id = str(df.iloc[i][id_col]).strip()
            matches = glob.glob(os.path.join(DATA_DIR, f"{file_id}*"))
            if not matches:
                matches = glob.glob(os.path.join("./data", f"{file_id}*"))
            if matches:
                thumb_uri = image_file_to_data_uri(matches[0])

        # Strategy B: Fall back to direct image list if Strategy A failed
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

    # --- 6. Save to JSON ---
    output["imaging"] = {
        "display_name": "MIAS Mammography",
        "framing": "Mammography image screening — direct comparison between Classical CNN/Swin vs Quantum-Hybrid (Swin + VQC)",
        "is_real_data": True,
        "engine": "Swin Transformer + PennyLane Quantum Hybrid",
        "evaluation": {
            "classical": {
                "model": "Classical Ensemble (SVM+RF)",
                "accuracy": classical_result["accuracy"],
                "precision": classical_result["precision"],
                "recall": classical_result["recall"],
                "f1": classical_result["f1"],
                "auc_roc": classical_result["auc_roc"],
                "training_time_sec": classical_result["training_time_sec"]
            },
            "quantum": {
                "model": "Quantum Hybrid Ensemble (QSVM+VQC)",
                "accuracy": quantum_result["accuracy"],
                "precision": quantum_result["precision"],
                "recall": quantum_result["recall"],
                "f1": quantum_result["f1"],
                "auc_roc": quantum_result["auc_roc"],
                "training_time_sec": quantum_result["training_time_sec"]
            }
        },
        "dataset_info": {
            "name": "MIAS Mammography",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_qubits": N_QUBITS,
        },
        "cases": cases,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved MIAS Swin-Quantum imaging results ({n_cases} cases) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()