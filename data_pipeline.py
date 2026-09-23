"""
Data Pipeline: MIAS Mammogram Data -> Swin Transformer Embeddings -> Preprocessing & Scaling
Shared preprocessing for both Classical and Quantum paths.
Matches architecture blocks: "Patient Data", "Feature Extraction", "Feature Scaling", "PCA Reduction"
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Import image feature extractor from image_pipeline
from image_pipeline import load_mias_dataset


def load_mias_swin_data(data_dir="./data/MIAS"):
    """
    Loads MIAS mammogram images and extracts 768-dimensional 
    Swin Transformer embeddings alongside binary target labels.
    """
    X_features, y_labels = load_mias_dataset(data_dir=data_dir)
    return X_features, y_labels


def clean_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Block: 'Feature Scaling: StandardScaler'
    Returns train/test splits for full Swin feature embeddings.
    """
    # Train/test split before scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale Swin Transformer feature embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = [f"swin_dim_{i}" for i in range(X.shape[1])]

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def quantum_only_reduction(X_train_scaled, X_test_scaled, n_components=4):
    """
    Block: 'PCA: reduce Swin embeddings to K=4-8 features'
    Block: 'Rescale to -pi to pi for quantum angle encoding'

    IMPORTANT: Applied ONLY to the quantum path.
    The classical path uses the full 768 Swin Transformer embeddings.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Rescale each PCA component independently to [-pi, pi] for quantum gates
    def rescale_to_pi(arr, ref_min, ref_max):
        # Prevent division by zero if min equals max
        denom = np.where((ref_max - ref_min) == 0, 1e-8, ref_max - ref_min)
        return -np.pi + (arr - ref_min) * (2 * np.pi) / denom

    train_min, train_max = X_train_pca.min(axis=0), X_train_pca.max(axis=0)
    X_train_q = rescale_to_pi(X_train_pca, train_min, train_max)
    X_test_q = rescale_to_pi(X_test_pca, train_min, train_max)

    # Clip test set in case values extend beyond training distribution bounds
    X_test_q = np.clip(X_test_q, -np.pi, np.pi)

    return X_train_q, X_test_q, pca


if __name__ == "__main__":
    print("Loading MIAS images and extracting Swin Transformer embeddings...")
    X, y = load_mias_swin_data(data_dir="./data/MIAS")

    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=4)

    print(f"\nClassical path shapes (Full Swin Embeddings) -> train: {X_train.shape}, test: {X_test.shape}")
    print(f"Quantum path shapes (PCA Reduced Angles)     -> train: {X_train_q.shape}, test: {X_test_q.shape}")
    print(f"Quantum data range                           -> min: {X_train_q.min():.3f}, max: {X_train_q.max():.3f}")
    print(f"PCA explained variance ratio sum             -> {pca.explained_variance_ratio_.sum():.3f}")