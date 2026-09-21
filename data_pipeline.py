"""
Data Pipeline: Patient Data -> Cleaning -> Scaling
Shared preprocessing for both Classical and Quantum paths.
Matches architecture blocks: "Patient Data", "Data Cleaning", "Feature Scaling"
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def load_breast_cancer_data():
    """Load Breast Cancer Wisconsin dataset (built into sklearn)."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign
    return X, y


def load_heart_disease_data(csv_path="data/heart.csv"):
    """
    Load Heart Disease dataset (Kaggle / UCI Cleveland format).
    Expects a CSV with a 'target' column (0 = no disease, 1 = disease).
    Download from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    """
    df = pd.read_csv(csv_path)
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def load_diabetes_risk_data(csv_path="data/pima_diabetes.csv"):
    """
    Load Pima Indians Diabetes dataset - genuinely an EARLY / RISK-SCREENING dataset
    (uses pre-diagnosis risk factors: glucose, BMI, age, family history, etc.)
    rather than post-biopsy/post-diagnosis features like Breast Cancer or Heart Disease.

    Auto-downloads from a public GitHub mirror if the file isn't present locally.
    Source: https://github.com/jbrownlee/Datasets (pima-indians-diabetes.data.csv)
    """
    import os
    columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "target"
    ]
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, header=None, names=columns)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    # Note: 0-values in glucose/blood_pressure/skin_thickness/insulin/bmi are missing-data
    # placeholders in this dataset, not real zero readings - treat as NaN so imputation handles them.
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df[col] = df[col].replace(0, np.nan)

    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def load_genomics_data(csv_path="data/leukemia_gene_expression.csv"):
    """
    Load a gene-expression genomics dataset (e.g. Golub et al. Leukemia AML vs ALL).
    Genuinely high-dimensional (thousands of gene-expression features, few dozen samples) -
    the classic "curse of dimensionality" case the PS's background section refers to.

    Expects a CSV with a 'target' column (0/1) and gene expression columns.
    Recommended source (requires free Kaggle account - not auto-downloadable without
    Kaggle API credentials, so this is a manual download like Heart Disease):
      https://www.kaggle.com/datasets/crawford/gene-expression
    (Golub et al. 1999 - Acute Myeloid vs Acute Lymphoblastic Leukemia, ~7000 genes, 72 patients)

    If the file isn't found, falls back to a SYNTHETIC high-dimensional dataset shaped like
    real gene-expression data (many features, few samples) so the pipeline can still be
    demonstrated/tested. Clearly flagged - swap in the real CSV before your actual demo.
    """
    import os
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        y = df["target"]
        X = df.drop(columns=["target"])
        return X, y, True  # True = real data

    print(f"WARNING: {csv_path} not found - using SYNTHETIC gene-expression-shaped data. "
          f"Download the real dataset from Kaggle (see docstring) before your demo.")
    from sklearn.datasets import make_classification
    X_arr, y_arr = make_classification(
        n_samples=72, n_features=2000, n_informative=30, n_redundant=100,
        n_clusters_per_class=2, weights=[0.65, 0.35], random_state=42
    )
    gene_names = [f"gene_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=gene_names)
    y = pd.Series(y_arr, name="target")
    return X, y, False  # False = synthetic placeholder


def clean_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Block: 'Data Cleaning: Missing value imputation'
    Block: 'Feature Scaling: StandardScaler'
    Returns train/test splits, already cleaned + scaled.
    """
    # --- Data Cleaning ---
    imputer = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # --- Train/test split (before scaling to avoid leakage) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, X.columns.tolist()


def quantum_only_reduction(X_train_scaled, X_test_scaled, n_components=6):
    """
    Block: 'PCA: reduce to K=4-8 features' (Quantum-Only Reduction subgraph)
    Block: 'Rescale to -pi to pi for angle encoding'

    IMPORTANT: This is applied ONLY to the quantum path.
    The classical path uses the full feature set (X_train_scaled / X_test_scaled directly).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Rescale each PCA component independently to [-pi, pi]
    # so it's valid input for RY/RZ angle-encoding gates.
    def rescale_to_pi(arr, ref_min, ref_max):
        return -np.pi + (arr - ref_min) * (2 * np.pi) / (ref_max - ref_min)

    train_min, train_max = X_train_pca.min(axis=0), X_train_pca.max(axis=0)
    X_train_q = rescale_to_pi(X_train_pca, train_min, train_max)
    X_test_q = rescale_to_pi(X_test_pca, train_min, train_max)
    # clip test set in case it falls slightly outside train range
    X_test_q = np.clip(X_test_q, -np.pi, np.pi)

    return X_train_q, X_test_q, pca


if __name__ == "__main__":
    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=6)

    print(f"Classical path shapes -> train: {X_train.shape}, test: {X_test.shape}")
    print(f"Quantum path shapes   -> train: {X_train_q.shape}, test: {X_test_q.shape}")
    print(f"Quantum data range    -> min: {X_train_q.min():.3f}, max: {X_train_q.max():.3f}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
