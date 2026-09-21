"""
Results Dashboard: Streamlit side-by-side comparison
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

from data_pipeline import load_breast_cancer_data, clean_and_scale, quantum_only_reduction
from classical_model import ClassicalEnsemble
from quantum_model import QuantumHybridEnsemble
from evaluate import evaluate_model
from explainability import explain_model, top_features_for_patient

st.set_page_config(page_title="Hybrid QML Disease Detection", layout="wide")

st.title("🧬 Hybrid Quantum-Classical ML Platform for Early Disease Detection")
st.caption("Classical Ensemble (SVM + Random Forest) vs Quantum Hybrid Ensemble (QSVM + VQC)")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Configuration")
view_mode = st.sidebar.radio(
    "View mode",
    ["🩺 Clinician View", "🔬 Researcher / Comparison View"],
    help="Clinician View: focuses on one patient, plain-language risk factors. "
         "Researcher View: classical vs quantum model comparison for benchmarking."
)
dataset_choice = st.sidebar.selectbox("Dataset", ["Breast Cancer Wisconsin"])  # Heart Disease: add CSV to enable
n_qubits = st.sidebar.slider("Qubits (quantum feature count)", 2, 8, 4)
quantum_subset = st.sidebar.slider("Quantum training subset size", 20, 200, 80,
                                    help="Smaller = faster demo, since quantum kernel cost grows with samples")
vqc_epochs = st.sidebar.slider("VQC training epochs", 5, 50, 20)
run_button = st.sidebar.button("🚀 Run / Load Models", type="primary")

# ---------------- Cache heavy training ----------------
@st.cache_resource(show_spinner=False)
def load_data():
    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)
    return X_train, X_test, y_train, y_test, scaler, feature_names, X

@st.cache_resource(show_spinner=False)
def train_classical(_X_train, _y_train):
    return ClassicalEnsemble().fit(_X_train, _y_train)

@st.cache_resource(show_spinner=False)
def train_quantum(_X_train_q, _y_train, n_qubits, subset, epochs):
    model = QuantumHybridEnsemble(n_qubits=n_qubits, vqc_epochs=epochs)
    model.fit(_X_train_q[:subset], _y_train[:subset])
    return model

if run_button:
    X_train, X_test, y_train, y_test, scaler, feature_names, X_raw = load_data()
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=n_qubits)

    with st.spinner("Training classical ensemble..."):
        t0 = time.time()
        classical_model = train_classical(X_train, y_train)
        classical_time = time.time() - t0

    with st.spinner("Training quantum hybrid ensemble (simulating circuits, may take a minute)..."):
        t0 = time.time()
        quantum_model = train_quantum(X_train_q, y_train, n_qubits, quantum_subset, vqc_epochs)
        quantum_time = time.time() - t0

    classical_result = evaluate_model("Classical Ensemble", classical_model,
                                       X_test, y_test, classical_time)
    quantum_result = evaluate_model("Quantum Hybrid Ensemble", quantum_model,
                                     X_test_q, y_test, quantum_time)

    # =========================================================
    # CLINICIAN VIEW — patient-first, plain language
    # =========================================================
    if view_mode == "🩺 Clinician View":
        st.success("Models ready.")
        st.markdown("---")
        st.subheader("Patient Risk Assessment")

        patient_idx = st.slider("Select patient (test case)", 0, len(X_test) - 1, 0)

        # Use the classical ensemble as the primary clinical-facing model —
        # it uses the full feature set (real clinical measurements, not PCA-compressed values),
        # so its explanation is directly interpretable by a doctor.
        risk_prob = classical_model.predict_proba(X_test[patient_idx:patient_idx + 1])[0]
        prediction = "MALIGNANT (High Risk)" if risk_prob < 0.5 else "BENIGN (Low Risk)"
        risk_score_display = (1 - risk_prob) * 100  # malignant-risk %, more intuitive for clinicians

        risk_col1, risk_col2 = st.columns([1, 2])
        with risk_col1:
            if risk_prob < 0.5:
                st.error(f"### ⚠️ {prediction}")
            else:
                st.success(f"### ✅ {prediction}")
            st.metric("Malignancy Risk Score", f"{risk_score_display:.1f}%")

        with risk_col2:
            actual_label = "Benign" if y_test[patient_idx] == 1 else "Malignant"
            st.write(f"**Actual diagnosis on record:** {actual_label}")
            st.write("This score reflects the model's confidence based on this patient's "
                     "biopsy measurements, compared against patterns learned from prior patient data. "
                     "**This is a decision-support estimate, not a standalone diagnosis.**")

        st.markdown("---")
        st.subheader("What's driving this assessment")
        st.caption("Top clinical measurements pushing this patient's result toward malignant or benign, "
                   "compared against this patient's own value vs. the typical range seen in the training data.")

        with st.spinner("Analyzing contributing factors..."):
            shap_vals_c, _ = explain_model(
                classical_model, X_train, X_test[patient_idx:patient_idx + 1],
                feature_names, n_background=20, nsamples=80
            )

        top_feats = top_features_for_patient(shap_vals_c[0], feature_names, top_n=6)

        for feat, val in top_feats:
            direction = "🔴 pushes toward MALIGNANT" if val < 0 else "🟢 pushes toward BENIGN"
            strength = min(abs(val) * 20, 1.0)  # rough visual scaling for the bar
            colc1, colc2, colc3 = st.columns([2, 3, 2])
            with colc1:
                st.write(f"**{feat}**")
            with colc2:
                st.progress(strength)
            with colc3:
                st.write(direction)

        st.markdown("---")
        st.caption(
            "⚕️ **Clinical note:** This tool is a research prototype for early-detection screening support. "
            "It is not a certified diagnostic device. All flagged results should be confirmed through "
            "standard clinical workflow (biopsy, radiology, specialist review) before any treatment decision."
        )

        with st.expander("🔬 See quantum model's assessment (research/comparison, optional)"):
            st.caption("The quantum hybrid model uses a compressed feature set due to current qubit "
                       "hardware limits, so its per-feature explanation is less directly clinically "
                       "readable — shown here for research transparency, not primary clinical use.")
            q_risk_prob = quantum_model.predict_proba(X_test_q[patient_idx:patient_idx + 1])[0]
            q_prediction = "Malignant (High Risk)" if q_risk_prob < 0.5 else "Benign (Low Risk)"
            st.write(f"Quantum model assessment: **{q_prediction}** "
                     f"(malignancy risk score: {(1-q_risk_prob)*100:.1f}%)")
            agree = "✅ Agrees with classical model" if (risk_prob < 0.5) == (q_risk_prob < 0.5) \
                else "⚠️ Disagrees with classical model — worth a closer look"
            st.write(agree)

    # =========================================================
    # RESEARCHER / COMPARISON VIEW — original benchmarking view
    # =========================================================
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔵 Classical Ensemble Path")
            st.markdown("Full feature set (30 features) → SVM + Random Forest → Meta Combiner")
            st.success(f"Done in {classical_time:.2f}s")
        with col2:
            st.subheader("🟣 Quantum Hybrid Path")
            st.markdown(f"Reduced features ({n_qubits} qubits) → QSVM + VQC → Meta Combiner")
            st.success(f"Done in {quantum_time:.2f}s")

        st.markdown("---")
        st.subheader("📊 Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Training Time")
        results_df = pd.DataFrame([classical_result, quantum_result]).set_index("model")
        st.dataframe(results_df.style.format({
            "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "auc_roc": "{:.3f}", "training_time_sec": "{:.2f}s"
        }), width="stretch")

        metric_cols = st.columns(4)
        for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
            with metric_cols[i]:
                delta = classical_result[metric] - quantum_result[metric]
                st.metric(f"Classical {metric}", f"{classical_result[metric]:.3f}")
                st.metric(f"Quantum {metric}", f"{quantum_result[metric]:.3f}", delta=f"{-delta:+.3f}")

        st.markdown("---")
        st.subheader("🔍 Explainability (KernelSHAP)")
        st.caption("Same model-agnostic method applied to both paths — works regardless of the underlying model type.")

        patient_idx = st.slider("Select a test patient to explain", 0, len(X_test) - 1, 0)
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("**Classical model — top features**")
            with st.spinner("Computing SHAP values..."):
                shap_vals_c, _ = explain_model(
                    classical_model, X_train, X_test[patient_idx:patient_idx + 1],
                    feature_names, n_background=20, nsamples=80
                )
            for feat, val in top_features_for_patient(shap_vals_c[0], feature_names):
                direction = "→ malignant risk" if val < 0 else "→ benign"
                st.write(f"- **{feat}**: {val:+.4f} ({direction})")

        with exp_col2:
            st.markdown("**Quantum model — top features (PCA components)**")
            pca_feature_names = [f"PC{i+1}" for i in range(n_qubits)]
            with st.spinner("Computing SHAP values for quantum model (this is the slow part)..."):
                shap_vals_q, _ = explain_model(
                    quantum_model, X_train_q[:quantum_subset], X_test_q[patient_idx:patient_idx + 1],
                    pca_feature_names, n_background=8, nsamples=30
                )
            for feat, val in top_features_for_patient(shap_vals_q[0], pca_feature_names):
                direction = "→ malignant risk" if val < 0 else "→ benign"
                st.write(f"- **{feat}**: {val:+.4f} ({direction})")

        actual_label = "benign" if y_test[patient_idx] == 1 else "malignant"
        st.info(f"Ground truth for this patient: **{actual_label}**")

else:
    if view_mode == "🩺 Clinician View":
        st.info("👈 Click **Run / Load Models** to load a patient's risk assessment.")
        st.markdown("""
        ### Clinician View shows, per patient:
        - Overall risk score (malignant vs benign) in plain terms
        - The **specific clinical measurements** driving that assessment, ranked by influence
        - Whether the quantum model agrees, as a secondary cross-check
        """)
    else:
        st.info("👈 Configure settings in the sidebar and click **Run / Load Models** to see the full comparison.")
        st.markdown("""
        ### Researcher View shows:
        1. Shared preprocessing across both paths
        2. Classical Ensemble (SVM + Random Forest) vs Quantum Hybrid Ensemble (QSVM + VQC)
        3. Full benchmark: accuracy, precision, recall, F1, AUC-ROC, training time
        4. KernelSHAP explainability for both models
        """)
