"""
Pre-computes results for the IMAGING path (breast ultrasound screening) and
merges them into results.json under the "imaging" key, alongside the tabular
"datasets" already produced by export_results.py.

Run this once before your demo: python3 export_image_results.py
Then open dashboard.html - the new "Imaging Screening" tab reads this data.

Mirrors export_results.py's pattern: real classical CNN / quantum hybrid
CNN+VQC training (image_model.py) is used when torch + pennylane are
available. If they aren't installed in this environment, this script falls
back to a lightweight scikit-learn approximation of the same two paths
(PCA+RandomForest for "classical", PCA + closed-form angle-embedding
feature map + LogisticRegression for "quantum-hybrid") so the dashboard can
still be demoed end-to-end. This mirrors the same real-data/synthetic-data
fallback pattern image_model.py and image_pipeline.py already use.
"""

import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from image_pipeline import load_breast_ultrasound_images, make_synthetic_ultrasound_data

RESULTS_PATH = "results.json"
N_CASES_TO_SHOW = 8
N_QUBITS = 4


def get_data():
    try:
        X_train, y_train, X_test, y_test = load_breast_ultrasound_images()
        is_real = True
    except Exception as e:
        print(f"Real BreastMNIST unavailable here ({type(e).__name__}), using SYNTHETIC "
              f"placeholder images shaped like the real dataset (see README).")
        X_train, y_train, X_test, y_test = make_synthetic_ultrasound_data(n_train=160, n_test=40)
        is_real = False
    return X_train, y_train, X_test, y_test, is_real


def eval_metrics(name, y_true, preds, probs, fit_time):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "training_time_sec": round(fit_time, 2),
    }


def try_real_pipeline(X_train, y_train, X_test, y_test):
    """Real path from image_model.py: Classical CNN vs CNN-feature-extractor + VQC."""
    import torch
    from image_model import (
        train_classical_cnn, cnn_predict_proba,
        train_hybrid_cnn_quantum, hybrid_cnn_predict_proba
    )

    subset = min(120, len(X_train))

    t0 = time.time()
    cnn_model = train_classical_cnn(X_train[:subset], y_train[:subset], epochs=15)
    classical_time = time.time() - t0
    classical_probs = cnn_predict_proba(cnn_model, X_test)
    classical_preds = (classical_probs >= 0.5).astype(int)

    t0 = time.time()
    hybrid_model = train_hybrid_cnn_quantum(X_train[:subset], y_train[:subset], n_qubits=N_QUBITS, epochs=10)
    quantum_time = time.time() - t0
    quantum_probs = hybrid_cnn_predict_proba(hybrid_model, X_test)
    quantum_preds = (quantum_probs >= 0.5).astype(int)

    return classical_probs, classical_preds, classical_time, quantum_probs, quantum_preds, quantum_time


def fallback_pipeline(X_train, y_train, X_test, y_test):
    """
    Lightweight scikit-learn stand-in for when torch/pennylane aren't installed.
    Same architecture idea, shape-for-shape:
      Classical: flattened pixels -> PCA(50) -> RandomForest  (proxy for the CNN)
      Quantum:   flattened pixels -> PCA(N_QUBITS) -> closed-form RY angle-embedding
                 expectation values (cos of the scaled pixel-PCA angle per qubit,
                 the same quantity VQC's AngleEmbedding + PauliZ readout would give
                 for a single-layer, no-entanglement circuit) -> LogisticRegression
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr_flat = X_train.reshape(len(X_train), -1)
    Xte_flat = X_test.reshape(len(X_test), -1)

    # ---- Classical path ----
    t0 = time.time()
    n_comp_c = min(50, Xtr_flat.shape[1], Xtr_flat.shape[0] - 1)
    pca_c = PCA(n_components=n_comp_c).fit(Xtr_flat)
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(pca_c.transform(Xtr_flat), y_train)
    classical_time = time.time() - t0
    classical_probs = clf.predict_proba(pca_c.transform(Xte_flat))[:, 1]
    classical_preds = (classical_probs >= 0.5).astype(int)

    # ---- Quantum-hybrid path ----
    t0 = time.time()
    n_qubits = min(N_QUBITS, Xtr_flat.shape[1], Xtr_flat.shape[0] - 1)
    pca_q = PCA(n_components=n_qubits).fit(Xtr_flat)
    scaler_q = StandardScaler().fit(pca_q.transform(Xtr_flat))

    def angle_features(X_raw):
        angles = np.clip(scaler_q.transform(pca_q.transform(X_raw)), -np.pi, np.pi)
        return np.cos(angles)  # <Z> expectation of an RY(angle) rotation from |0>

    meta = LogisticRegression().fit(angle_features(Xtr_flat), y_train)
    quantum_time = time.time() - t0
    quantum_probs = meta.predict_proba(angle_features(Xte_flat))[:, 1]
    quantum_preds = (quantum_probs >= 0.5).astype(int)

    return classical_probs, classical_preds, classical_time, quantum_probs, quantum_preds, quantum_time


def image_to_data_uri(img_2d, display_size=112):
    arr = (np.clip(img_2d, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L").resize((display_size, display_size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            output = json.load(f)
    else:
        output = {}
    output.setdefault("datasets", {})

    X_train, y_train, X_test, y_test, is_real = get_data()
    print(f"Imaging data: train={X_train.shape}, test={X_test.shape}, real_data={is_real}")

    try:
        print("Attempting real CNN / quantum-hybrid CNN+VQC training (torch + pennylane)...")
        (classical_probs, classical_preds, classical_time,
         quantum_probs, quantum_preds, quantum_time) = try_real_pipeline(X_train, y_train, X_test, y_test)
        engine = "real (ClassicalCNN vs CNN-feature-extractor+VQC, from image_model.py)"
    except Exception as e:
        print(f"torch/pennylane not available/working here ({type(e).__name__}: {e}); "
              f"using scikit-learn fallback approximation "
              f"of the same classical-vs-quantum-hybrid architecture.")
        (classical_probs, classical_preds, classical_time,
         quantum_probs, quantum_preds, quantum_time) = fallback_pipeline(X_train, y_train, X_test, y_test)
        engine = "fallback (PCA+RandomForest vs PCA+angle-embedding+LogisticRegression - " \
                 "install torch+pennylane and rerun for the real CNN/VQC models)"

    classical_result = eval_metrics("Classical CNN", y_test, classical_preds, classical_probs, classical_time)
    quantum_result = eval_metrics("Quantum Hybrid CNN+VQC", y_test, quantum_preds, quantum_probs, quantum_time)

    print(f"Classical: {classical_result}")
    print(f"Quantum:   {quantum_result}")

    n_cases = min(N_CASES_TO_SHOW, len(X_test))
    cases = []
    for i in range(n_cases):
        # BreastMNIST convention: 0 = malignant, 1 = normal/benign
        actual = "Benign" if y_test[i] == 1 else "Malignant"
        c_pred = "Benign" if classical_preds[i] == 1 else "Malignant"
        q_pred = "Benign" if quantum_preds[i] == 1 else "Malignant"
        cases.append({
            "case_id": i,
            "thumbnail": image_to_data_uri(X_test[i]),
            "actual_label": actual,
            "classical": {
                "prediction": c_pred,
                "risk_score_pct": round((1 - float(classical_probs[i])) * 100, 1),
            },
            "quantum": {
                "prediction": q_pred,
                "risk_score_pct": round((1 - float(quantum_probs[i])) * 100, 1),
            },
        })

    output["imaging"] = {
        "display_name": "Breast Ultrasound Screening (BreastMNIST)",
        "framing": "Screening modality (pre-biopsy imaging) - earlier in the care pathway than the "
                   "Wisconsin biopsy-derived tabular dataset.",
        "is_real_data": is_real,
        "engine": engine,
        "evaluation": {"classical": classical_result, "quantum": quantum_result},
        "dataset_info": {
            "name": "BreastMNIST (28x28 grayscale ultrasound)",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_qubits": N_QUBITS,
        },
        "cases": cases,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved imaging results ({n_cases} cases) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
