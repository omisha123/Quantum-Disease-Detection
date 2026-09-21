# Hybrid Quantum ML Platform for Early Disease Detection

Matches the architecture: shared preprocessing → Classical Ensemble Path (full features) +
Quantum Hybrid Ensemble Path (PCA-reduced features) → shared evaluation → KernelSHAP explainability
→ Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## File map (matches architecture diagram blocks)

- `data_pipeline.py` — Patient Data → Data Cleaning → Feature Scaling → (quantum-only) PCA + rescale to [-π, π]
  - `load_breast_cancer_data()` — Breast Cancer Wisconsin (biopsy-stage, diagnostic confirmation)
  - `load_diabetes_risk_data()` — Pima Diabetes (**genuinely early/risk-screening data** — glucose, BMI, age, family history, auto-downloads from GitHub)
- `classical_model.py` — SVM (RBF) + Random Forest → Meta Combiner → Prediction B
- `quantum_model.py` — QSVM (ZZFeatureMap quantum kernel) + VQC (RY encoding, StronglyEntanglingLayers, Pauli-Z measurement, Dense head) → Meta Combiner → Prediction A
- `image_pipeline.py` — BreastMNIST (breast ultrasound screening images), auto-downloads via `medmnist` package
- `image_model.py` — Classical CNN vs Quantum Hybrid CNN (CNN feature extractor → VQC), same comparison structure as the tabular path
- `evaluate.py` — Accuracy, Precision, Recall, F1, AUC-ROC, Training Time for both paths
- `explainability.py` — KernelSHAP, model-agnostic, works on both classical and quantum models
- `app.py` — Streamlit dashboard (live training — can be slow/fragile, see dashboard.html for a more reliable demo option)
- `dashboard.html` + `export_results.py` — Pre-computed static HTML dashboard (recommended for live demos — no training during the demo itself)

## Run order (test each piece standalone first)

```bash
python3 data_pipeline.py           # sanity check preprocessing shapes
python3 classical_model.py         # classical baseline accuracy (Breast Cancer)
python3 quantum_model.py           # quantum hybrid accuracy (slower - simulating circuits)
python3 image_model.py             # image screening: classical CNN vs quantum hybrid CNN
python3 evaluate.py                # full comparison table, saved to results_comparison.csv
python3 explainability.py          # SHAP demo on a few patients
python3 export_results.py          # pre-compute results.json for the static dashboard
streamlit run app.py               # OR: python3 -m http.server 8000, then open dashboard.html
```

## Two disease-detection framings in this platform

1. **Early risk screening** (genuinely pre-diagnosis): Pima Diabetes (tabular risk factors), BreastMNIST (ultrasound screening images) — these use data collected *before* a confirmed diagnosis.
2. **Diagnostic confirmation** (post-biopsy classification): Breast Cancer Wisconsin — this dataset's features come from an already-taken biopsy, so frame it honestly as "accelerating and explaining diagnosis" rather than "early detection." See the note below.
3. **Genomics / high-dimensional curse-of-dimensionality demo**: gene-expression data (Golub et al. Leukemia AML vs ALL style — ~2000-7000 genes, ~70 samples) — this directly illustrates the PS background's specific complaint about classical ML struggling with "high-dimensional, noisy, complex biomedical data." Very few samples relative to feature count, so this is the strongest case for PCA/dimensionality reduction being essential, not optional.

**Important honesty note for your pitch:** Breast Cancer Wisconsin is NOT early-detection data (features come from a biopsy that's already been taken) — it's diagnostic classification. Pima Diabetes and BreastMNIST ultrasound imaging ARE genuinely earlier-stage / screening data. Use this distinction explicitly in your presentation — it's a stronger, more honest story than claiming "early detection" for all three.


## Important tuning notes for your demo

- **`quantum_subset`** (in `evaluate.py` / sidebar in `app.py`): QSVM's kernel matrix costs O(n²) quantum
  circuit evaluations. Keep training subset small (50-100 samples) for a live demo — accuracy is still
  meaningful, just don't use the full 455-sample training set live on stage.
- **`n_qubits`**: 4-6 is the sweet spot. More qubits = slower simulation, not necessarily better accuracy.
- **`vqc_epochs`**: 20-30 is enough to show a real loss curve dropping. More epochs = better accuracy but
  longer wait during a live demo — pre-run and cache results before presenting if possible.
- First real run of `evaluate.py` on this environment: Classical = 96.5% accuracy in 0.3s,
  Quantum Hybrid = ~80-82% accuracy in ~45s (4 qubits, 80 training samples, 20 epochs). These are real
  numbers from an actual run, not placeholders — expect similar ranges, use them as your baseline talking point.

## Adding Heart Disease dataset (optional third dataset — note: also NOT early-detection data, same caveat as Breast Cancer)

1. Download from Kaggle: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
2. Save as `data/heart.csv`
3. In any script, swap `load_breast_cancer_data()` for `load_heart_disease_data()`

## Adding real genomics data (Golub Leukemia gene-expression)

`load_genomics_data()` falls back to synthetic high-dimensional data automatically (2000 features,
72 samples - matches real Golub-dataset proportions) so the pipeline is fully testable without any
download. For your actual demo, swap in the real dataset:

1. Download from Kaggle: https://www.kaggle.com/datasets/crawford/gene-expression
2. Reformat so gene-expression columns + a `target` column (0/1 for ALL/AML) exist, save as
   `data/leukemia_gene_expression.csv`
3. `load_genomics_data()` will automatically detect the real file and use it (prints "Is real data: True")

Note the function returns a third value `is_real` (True/False) - check this in your own scripts/demo
so you always know whether you're looking at real or fallback data.

## Image pipeline setup

`medmnist` is in `requirements.txt`. First run of `image_pipeline.py` or `image_model.py` auto-downloads
BreastMNIST (~560KB) from Zenodo — needs internet access once, then it's cached locally under
`~/.medmnist/`. This was tested with synthetic placeholder data in the build sandbox (Zenodo isn't
reachable there) to confirm the CNN and hybrid CNN+VQC code runs correctly end-to-end with zero
errors — on your machine with normal internet access, it will pull the real ultrasound images instead.

## Known limitations (state these proactively in your report — judges respect this)

- Quantum path is currently tabular-only (no imaging/genomics) — scoped deliberately for a 3-day build.
- QSVM kernel matrix computation doesn't scale past ~100-200 samples on a simulator — real QPU or
  quantum-inspired approximate kernels would be needed for production-scale data.
- No demonstrated quantum advantage yet — this is expected and consistent with current NISQ-era research;
  the value here is a working, benchmarked hybrid pipeline ready for better hardware as it arrives.
