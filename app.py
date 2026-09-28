"""
Live Streamlit dashboard: classical vs quantum hybrid ensemble on the tabular datasets.
Run with: streamlit run app.py
(For a reliable live demo, prefer dashboard.html + export_results.py - no training on stage.)
"""

import time
import pandas as pd
import streamlit as st

from classical_model import ClassicalEnsemble
from evaluate import evaluate_model
from export_results import get_dataset_configs, clean_and_scale
from tabular_quantum_model import qga_feature_selection, TabularQuantumEnsemble

st.set_page_config(page_title="Hybrid QML Disease Detection", layout="wide")
st.title("🧬 Hybrid Quantum-Classical ML Platform")
st.caption("Classical Ensemble (SVM + RF) vs Quantum Hybrid Ensemble (QSVM + VQC)")

configs = get_dataset_configs()

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")
view_mode = st.sidebar.radio("View mode", ["🩺 Clinician View", "🔬 Researcher View"])
dataset_key = st.sidebar.selectbox(
    "Dataset", list(configs.keys()), format_func=lambda k: configs[k]["display_name"]
)
n_qubits = st.sidebar.slider("Qubits (selected features)", 2, 8, 4)
vqc_layers = st.sidebar.slider("VQC layers", 1, 4, 3)
vqc_epochs = st.sidebar.slider("VQC training epochs", 10, 80, 40)
qsvm_ref = st.sidebar.slider(
    "QSVM training samples", 20, 200, 100,
    help="Kernel cost grows with this; keep small for a live demo."
)
run_button = st.sidebar.button("🚀 Run / Load Models", type="primary")


# ---------------- Cached loading / training ----------------
@st.cache_resource(show_spinner=False)
def load_data(key):
    X, y, feature_names, is_real = configs[key]["loader"]()
    X_train, X_test, y_train, y_test, _ = clean_and_scale(X, y)
    return X_train, X_test, y_train, y_test, feature_names, is_real


@st.cache_resource(show_spinner=False)
def train_classical(key):
    X_train, _, y_train, _, _, _ = load_data(key)
    t0 = time.time()
    model = ClassicalEnsemble().fit(X_train, y_train)
    return model, time.time() - t0


@st.cache_resource(show_spinner=False)
def train_quantum(key, n_qubits, layers, epochs, ref_size):
    X_train, X_test, y_train, _, feature_names, _ = load_data(key)
    k = min(n_qubits, X_train.shape[1])
    X_train_q, X_test_q, selected = qga_feature_selection(
        X_train, X_test, y_train, feature_names, n_components=k
    )
    t0 = time.time()
    model = TabularQuantumEnsemble(
        n_qubits=k, vqc_layers=layers, vqc_epochs=epochs, qsvm_ref_size=ref_size
    ).fit(X_train_q, y_train)
    return model, X_test_q, selected, time.time() - t0


if not run_button:
    st.info("👈 Pick a dataset and click **Run / Load Models**.")
    st.stop()

cfg = configs[dataset_key]
X_train, X_test, y_train, y_test, feature_names, is_real = load_data(dataset_key)
if not is_real:
    st.warning("This dataset is SYNTHETIC (real file not found) - results are illustrative only.")
st.caption(cfg["framing"])

with st.spinner("Training classical ensemble..."):
    classical_model, classical_time = train_classical(dataset_key)
with st.spinner("Training quantum hybrid ensemble (simulating circuits)..."):
    quantum_model, X_test_q, selected, quantum_time = train_quantum(
        dataset_key, n_qubits, vqc_layers, vqc_epochs, qsvm_ref
    )

classical_result = evaluate_model("Classical Ensemble", classical_model, X_test, y_test, classical_time)
quantum_result = evaluate_model("Quantum Hybrid Ensemble", quantum_model, X_test_q, y_test, quantum_time)

# ---------------- Clinician view ----------------
if view_mode == "🩺 Clinician View":
    st.subheader("Patient Risk Assessment")
    idx = st.slider("Select patient case", 0, len(X_test) - 1, 0)

    risk = float(classical_model.predict_proba(X_test[idx:idx + 1])[0])
    high = risk >= 0.5
    c1, c2 = st.columns([1, 2])
    with c1:
        (st.error if high else st.success)(f"### {'⚠️ HIGH RISK' if high else '✅ LOW RISK'}")
        st.metric("Risk Score", f"{risk * 100:.1f}%")
    with c2:
        st.write(f"**Actual outcome on record:** {'Positive' if y_test[idx] == 1 else 'Negative'}")
        st.write("This is a decision-support estimate, not a standalone diagnosis.")

    with st.expander("🔬 Quantum model's assessment"):
        q_risk = float(quantum_model.predict_proba(X_test_q[idx:idx + 1])[0])
        st.write(f"Quantum risk score: **{q_risk * 100:.1f}%**")
        st.write("✅ Agrees with classical model" if (q_risk >= 0.5) == high
                 else "⚠️ Disagrees with classical model - worth a closer look")

# ---------------- Researcher view ----------------
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔵 Classical Ensemble")
        st.markdown(f"{X_train.shape[1]} features -> SVM + Random Forest -> Meta Combiner")
        st.success(f"Trained in {classical_time:.2f}s")
    with col2:
        st.subheader("🟣 Quantum Hybrid")
        st.markdown(f"Selected features: {', '.join(selected)} -> QSVM + VQC (averaged)")
        st.success(f"Trained in {quantum_time:.2f}s")

    st.subheader("📊 Evaluation")
    df = pd.DataFrame([classical_result, quantum_result]).set_index("model")
    st.dataframe(df.style.format({
        "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
        "f1": "{:.3f}", "auc_roc": "{:.3f}", "training_time_sec": "{:.2f}s"
    }), use_container_width=True)

    if quantum_result["recall"] == 1.0 and quantum_result["precision"] == quantum_result["accuracy"]:
        st.warning("Quantum model appears to predict every case as positive - "
                   "try more epochs/qubits before trusting these numbers.")