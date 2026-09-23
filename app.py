"""
Results Dashboard: Streamlit side-by-side comparison for MIAS Image Dataset
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

# Import updated Swin-Quantum image pipeline
from image_pipeline import load_mias_dataset, QuantumSwinHybridEnsemble
from classical_model import ClassicalEnsemble
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

st.set_page_config(page_title="Hybrid QML Disease Detection (MIAS)", layout="wide")

st.title("🧬 Hybrid Quantum-Classical ML Platform for Mammogram Classification")
st.caption("Swin Transformer Feature Extraction -> Classical Ensemble vs Quantum Hybrid Ensemble (QSVM + VQC)")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Configuration")
view_mode = st.sidebar.radio(
    "View mode",
    ["🩺 Clinician View", "🔬 Researcher / Comparison View"],
    help="Clinician View: focuses on one patient's mammogram risk. "
         "Researcher View: classical vs quantum model benchmarking."
)
dataset_choice = st.sidebar.selectbox("Dataset", ["MIAS Mammography Dataset"])
n_qubits = st.sidebar.slider("Qubits (PCA feature count)", 2, 8, 4)
quantum_subset = st.sidebar.slider(
    "Quantum training subset size", 20, 200, 80,
    help="Smaller = faster demo, since quantum kernel calculation scales with sample size"
)
vqc_epochs = st.sidebar.slider("VQC training epochs", 5, 50, 20)
run_button = st.sidebar.button("🚀 Run / Load Models", type="primary")


# ---------------- Cache heavy Swin extraction & model training ----------------
@st.cache_resource(show_spinner=False)
def get_mias_swin_features():
    # Extract 768-dim Swin Transformer embeddings from MIAS images
    X_raw, y = load_mias_dataset(data_dir="./data/MIAS")
    return X_raw, y

@st.cache_resource(show_spinner=False)
def train_classical_model(_X_train, _y_train):
    return ClassicalEnsemble().fit(_X_train, _y_train)

@st.cache_resource(show_spinner=False)
def train_quantum_model(_X_train_q, _y_train, n_qubits, subset, epochs):
    model = QuantumSwinHybridEnsemble(n_qubits=n_qubits, vqc_epochs=epochs)
    model.fit(_X_train_q[:subset], _y_train[:subset])
    return model


if run_button:
    with st.spinner("Extracting Swin Transformer features from MIAS mammograms..."):
        X_raw, y = get_mias_swin_features()

    # PCA Reduction: Swin Features (768) -> Qubits (n_qubits)
    pca = PCA(n_components=n_qubits)
    X_quantum = pca.fit_transform(X_raw)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_q, X_test_q, _, _ = train_test_split(
        X_quantum, y, test_size=0.2, random_state=42, stratify=y
    )

    with st.spinner("Training Classical Ensemble on Swin Embeddings..."):
        t0 = time.time()
        classical_model = train_classical_model(X_train, y_train)
        classical_time = time.time() - t0

    with st.spinner("Training Quantum Hybrid Ensemble (simulating circuits)..."):
        t0 = time.time()
        quantum_model = train_quantum_model(X_train_q, y_train, n_qubits, quantum_subset, vqc_epochs)
        quantum_time = time.time() - t0

    classical_result = evaluate_model("Classical Ensemble", classical_model, X_test, y_test, classical_time)
    quantum_result = evaluate_model("Quantum Hybrid Ensemble", quantum_model, X_test_q, y_test, quantum_time)

    # =========================================================
    # CLINICIAN VIEW — patient-first, plain language
    # =========================================================
    if view_mode == "🩺 Clinician View":
        st.success("Models ready.")
        st.markdown("---")
        st.subheader("Patient Risk Assessment (Mammogram Screening)")

        patient_idx = st.slider("Select patient mammogram case", 0, len(X_test) - 1, 0)

        risk_prob = classical_model.predict_proba(X_test[patient_idx:patient_idx + 1])[0]
        prediction = "MALIGNANT (High Risk)" if risk_prob >= 0.5 else "BENIGN / NORMAL (Low Risk)"
        risk_score_display = risk_prob * 100

        risk_col1, risk_col2 = st.columns([1, 2])
        with risk_col1:
            if risk_prob >= 0.5:
                st.error(f"### ⚠️ {prediction}")
            else:
                st.success(f"### ✅ {prediction}")
            st.metric("Malignancy Risk Score", f"{risk_score_display:.1f}%")

        with risk_col2:
            actual_label = "Malignant" if y_test[patient_idx] == 1 else "Benign/Normal"
            st.write(f"**Actual diagnosis on record:** {actual_label}")
            st.write("This score reflects the Swin Transformer representation of the mammogram image, "
                     "compared against learned patterns from the MIAS clinical dataset. "
                     "**This is a decision-support estimate, not a standalone diagnosis.**")

        st.markdown("---")
        with st.expander("🔬 See quantum model's assessment (research/comparison)"):
            q_risk_prob = quantum_model.predict_proba(X_test_q[patient_idx:patient_idx + 1])[0]
            q_prediction = "Malignant (High Risk)" if q_risk_prob >= 0.5 else "Benign/Normal (Low Risk)"
            st.write(f"Quantum model assessment: **{q_prediction}** "
                     f"(malignancy risk score: {q_risk_prob * 100:.1f}%)")
            agree = "✅ Agrees with classical model" if (risk_prob >= 0.5) == (q_risk_prob >= 0.5) \
                else "⚠️ Disagrees with classical model — worth a closer look"
            st.write(agree)

    # =========================================================
    # RESEARCHER / COMPARISON VIEW — benchmarking view
    # =========================================================
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔵 Classical Ensemble Path")
            st.markdown("Swin Features (768-dim) → SVM + Random Forest → Meta Combiner")
            st.success(f"Done in {classical_time:.2f}s")
        with col2:
            st.subheader("🟣 Quantum Hybrid Path")
            st.markdown(f"Swin Features → PCA ({n_qubits} qubits) → QSVM + VQC → Ensemble")
            st.success(f"Done in {quantum_time:.2f}s")

        st.markdown("---")
        st.subheader("📊 Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Training Time")
        results_df = pd.DataFrame([classical_result, quantum_result]).set_index("model")
        st.dataframe(results_df.style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "auc_roc": "{:.3f}", "training_time_sec": "{:.2f}s"
        }), use_container_width=True)

        metric_cols = st.columns(4)
        for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
            with metric_cols[i]:
                delta = quantum_result[metric] - classical_result[metric]
                st.metric(f"Classical {metric}", f"{classical_result[metric]:.3f}")
                st.metric(f"Quantum {metric}", f"{quantum_result[metric]:.3f}", delta=f"{delta:+.3f}")

        actual_label = "Malignant" if y_test[patient_idx] == 1 else "Benign/Normal"
        st.info(f"Ground truth for patient index {patient_idx}: **{actual_label}**")

else:
    if view_mode == "🩺 Clinician View":
        st.info("👈 Click **Run / Load Models** to run Swin Feature Extraction and show the patient risk assessment.")
    else:
        st.info("👈 Configure settings in the sidebar and click **Run / Load Models** to benchmark Swin-Classical vs Swin-Quantum.")