"""
Explainability: KernelSHAP - model agnostic
Applied to both the classical full-feature model and the quantum reduced-feature model.
"""

import numpy as np
import shap


def explain_model(model, X_train, X_explain, feature_names, n_background=30, nsamples=100):
    """
    Generic KernelSHAP wrapper - works for ANY model with a predict_proba(X) method,
    whether it's the classical ensemble or the quantum hybrid ensemble.

    n_background: size of background dataset (smaller = faster, less precise)
    nsamples: number of coalition samples KernelSHAP draws per prediction (smaller = faster)
    """
    # Background dataset: summarize with k-means to keep it fast
    k = min(n_background, len(X_train))
    kmeans_obj = shap.kmeans(X_train, k)
    background = kmeans_obj.data if hasattr(kmeans_obj, 'data') else kmeans_obj

    explainer = shap.KernelExplainer(model.predict_proba, background)
    raw_shap_values = explainer.shap_values(X_explain, nsamples=nsamples)

    # Handle multi-class / binary output from predict_proba
    # shap_values can be a list [array_class0, array_class1] or a 3D array
    if isinstance(raw_shap_values, list):
        # Select SHAP values for positive class (Class 1 / Malignant)
        shap_values = raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    return shap_values, explainer


def top_features_for_patient(shap_values_row, feature_names, top_n=5):
    """Return the top N features driving one patient's prediction, ranked by |impact|."""
    idx_sorted = np.argsort(-np.abs(shap_values_row))[:top_n]
    return [(feature_names[i], round(float(shap_values_row[i]), 4)) for i in idx_sorted]


if __name__ == "__main__":
    from data_pipeline import load_breast_cancer_data, clean_and_scale, quantum_only_reduction
    from classical_model import ClassicalEnsemble

    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)

    print("Training classical model for explainability demo...")
    classical_model = ClassicalEnsemble().fit(X_train, y_train)

    print("Running KernelSHAP on classical model (a few test patients)...")
    shap_values, explainer = explain_model(
        classical_model, X_train, X_test[:5], feature_names,
        n_background=20, nsamples=80
    )

    for i in range(5):
        print(f"\nPatient {i} - top features driving prediction:")
        for feat, val in top_features_for_patient(shap_values[i], feature_names):
            direction = "toward malignant risk" if val > 0 else "toward benign"
            print(f"  {feat}: {val:+.4f} ({direction})")