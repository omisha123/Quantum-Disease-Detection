# Hybrid Quantum ML Platform for Disease Detection

Tabular pipeline: dataset loader -> scaling -> **Classical Ensemble** (all features) and
**Quantum Hybrid Ensemble** (selected features encoded on qubits) -> shared evaluation ->
KernelSHAP explainability -> dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `classical_model.py` | SVM (RBF) + Random Forest -> Logistic Regression meta combiner |
| `quantum_model.py` | QSVM (ZZ-style quantum kernel, statevectors computed once per sample) and VQC (RY encoding, StronglyEntanglingLayers, Pauli-Z readout, dense head) |
| `tabular_quantum_model.py` | Feature selection (ANOVA F-score, "QGA-style"), swarm hyperparameter search ("QPSO-style"), and `TabularQuantumEnsemble` (QSVM + VQC, probabilities averaged) |
| `evaluate.py` | Accuracy, precision, recall, F1, AUC-ROC, training time |
| `explainability.py` | Model-agnostic KernelSHAP for both models |
| `dataset_leukemia.py` | Converts the Kaggle Golub leukemia files into `data/leukemia_gene_expression.csv` |
| `export_results.py` | Trains everything, writes `results.json` (resumable) |
| `dashboard.html` | Static dashboard that reads `results.json` (recommended for live demos) |
| `app.py` | Live Streamlit version (trains on demand; slower and more fragile) |

## Run order

```bash
# Optional: real leukemia data (download from Kaggle into data/ first, see below)
python3 dataset_leukemia.py

python3 classical_model.py      # smoke test on Wisconsin data
python3 quantum_model.py        # smoke test of the quantum ensemble (slower)
python3 explainability.py       # SHAP demo

python3 export_results.py       # builds results.json for all four datasets
python3 -m http.server 8000     # open http://localhost:8000/dashboard.html

streamlit run app.py            # optional live version
```

## Datasets

| Dataset | Source | Framing |
|---|---|---|
| Pima Diabetes | auto-downloaded to `data/pima.csv` | Early risk screening (pre-diagnosis) |
| UCI Heart Disease | auto-downloaded to `data/heart.csv` (deduplicated) | Diagnostic risk from clinical features |
| Wisconsin Breast Cancer | scikit-learn built-in | **Diagnostic confirmation** (features come from an already-taken biopsy) - not early detection |
| Leukemia Gene Expression | Kaggle: `crawford/gene-expression`, place files in `data/` and run `dataset_leukemia.py` | High-dimensional genomics; few samples, many features |

If `data/leukemia_gene_expression.csv` is missing, `export_results.py` falls back to
**synthetic** genomics data and the dashboard labels it "Synthetic". Do not present
synthetic results as real leukemia results.

## Tuning notes

- QSVM kernel cost grows quadratically with training-set size; `TabularQuantumEnsemble` caps
  the QSVM reference set (`qsvm_ref_size`, default 150).
- 4-6 qubits is the practical range on a simulator; more qubits are slower, not necessarily better.
- Features are encoded in [0, pi/2]. A wider range made angles wrap and hurt the kernel.
- `app.py` trains live; `export_results.py` + `dashboard.html` avoids waiting during a demo.
- Always check the quantum row for a collapsed model (recall 1.0 with precision equal to
  accuracy means it predicts everything positive). `app.py` warns when this happens.

## Known limitations

- Results depend on a small held-out test split; report them with that caveat.
- Quantum runs are simulated. Kernel simulation does not scale beyond a few hundred samples.
- No quantum advantage is claimed or demonstrated; the value is a working, benchmarked hybrid
  pipeline that is ready for better hardware.
- "QGA" and "QPSO" here are classical proxies (F-score selection, swarm-style search), not
  true quantum-genetic or qubit-encoded algorithms.
- Explanations for the quantum model use very small SHAP budgets (speed), so they are noisy.
- This is a research demo, not a clinical decision tool.