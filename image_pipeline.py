import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoImageProcessor, SwinModel

# Quantum components imported from quantum_model.py
from quantum_model import QSVM, train_vqc, vqc_predict_proba


# =========================================================
# 1. Swin Transformer Feature Extractor (Image Model)
# =========================================================

class SwinImageFeatureExtractor:
    """Uses official Hugging Face Swin Transformer to extract high-level feature vectors from images."""
    def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224"):
        print(f"Loading Swin Transformer: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image_path):
        try:
            # Open grayscale MIAS image and convert to RGB format expected by Swin
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Swin outputs pooler_output or spatial average pooling
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    features = outputs.pooler_output.squeeze().numpy()
                else:
                    # Global average pool over sequence/spatial tokens
                    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return features
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            return None


# =========================================================
# 2. MIAS Image Dataset Loader
# =========================================================

def load_mias_dataset(data_dir="./data/MIAS"):
    """Reads mias_info.csv and extracts Swin Transformer features for all found mammogram images."""
    csv_candidates = [
        os.path.join(data_dir, "mias_info.csv"),
        os.path.join(data_dir, "mias_derived_info.csv"),
        os.path.join("./data", "mias_info.csv")
    ]
    
    csv_path = None
    for path in csv_candidates:
        if os.path.exists(path):
            csv_path = path
            break

    if not csv_path:
        raise FileNotFoundError("Could not find mias_info.csv in data/ or data/MIAS/")

    # Parse metadata handling flexible space/comma formatting
    df = pd.read_csv(csv_path, sep=r'\s+|,', engine='python')
    df.columns = df.columns.str.strip().str.upper()

    # Identify metadata columns
    id_col = 'REFNUM' if 'REFNUM' in df.columns else df.columns[0]
    severity_col = 'SEVERITY' if 'SEVERITY' in df.columns else df.columns[2]

    # Map target severity: Normal/Benign -> 0, Malignant -> 1
    df['SEVERITY_CLEAN'] = df[severity_col].fillna('N').astype(str).str.strip()
    label_map = {'N': 0, 'B': 0, 'M': 1}
    df['target'] = df['SEVERITY_CLEAN'].map(label_map).fillna(0).astype(int)

    extractor = SwinImageFeatureExtractor()
    features_list = []
    labels_list = []

    print("\nReading MIAS directory and extracting Swin Transformer features...")
    
    # Process images matched with CSV IDs
    for _, row in df.iterrows():
        file_id = str(row[id_col]).strip()
        matches = glob.glob(os.path.join(data_dir, f"{file_id}*"))
        
        if matches:
            img_path = matches[0]
            feat = extractor.extract_features(img_path)
            if feat is not None:
                features_list.append(feat)
                labels_list.append(row['target'])

    X_features = np.array(features_list)
    y_labels = np.array(labels_list)

    print(f"Extracted features from {len(X_features)} images. Swin Embedding Shape: {X_features.shape}")
    return X_features, y_labels


# =========================================================
# 3. Quantum Hybrid Pipeline
# =========================================================

class QuantumSwinHybridEnsemble:
    def __init__(self, n_qubits=4, vqc_layers=3, vqc_epochs=20):
        self.n_qubits = n_qubits
        self.vqc_layers = vqc_layers
        self.vqc_epochs = vqc_epochs
        self.pca = PCA(n_components=n_qubits)  # Ensures input is reduced to n_qubits
        self.qsvm = QSVM(n_qubits)
        self.vqc_model = None

    def fit(self, X_train, y_train):
        # 1. Automatically reduce 768 Swin features down to n_qubits (4)
        if X_train.shape[1] > self.n_qubits:
            print(f"Reducing {X_train.shape[1]} Swin features to {self.n_qubits} quantum features via PCA...")
            X_train_q = self.pca.fit_transform(X_train)
        else:
            X_train_q = X_train

        print("\n1. Training Quantum SVM (QSVM) on 4 Quantum Features...")
        self.qsvm.fit(X_train_q, y_train)

        print("2. Training Variational Quantum Circuit (VQC)...")
        self.vqc_model = train_vqc(
            X_train_q, y_train, self.n_qubits, self.vqc_layers, self.vqc_epochs
        )
        return self

    def predict_proba(self, X):
        # Apply fitted PCA reduction to test data
        if X.shape[1] > self.n_qubits:
            X_q = self.pca.transform(X)
        else:
            X_q = X

        qsvm_probs = self.qsvm.predict_proba(X_q)
        vqc_probs = vqc_predict_proba(self.vqc_model, X_q)
        
        # Average ensemble probabilities
        return (qsvm_probs + vqc_probs) / 2.0

    def predict(self, X, threshold=0.35):
        probs = self.predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        return (probs >= threshold).astype(int)


# =========================================================
# Execution Entry Point
# =========================================================

if __name__ == "__main__":
    # 1. Load MIAS dataset images & extract Swin Transformer embeddings
    X_swin_features, y = load_mias_dataset(data_dir="./data/MIAS")

    # 2. PCA Reduction: Swin Features (768 dims) -> Quantum Qubits (4 dims)
    N_QUBITS = 4
    pca = PCA(n_components=N_QUBITS)
    X_quantum = pca.fit_transform(X_swin_features)

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Quantum Ensemble
    ensemble = QuantumSwinHybridEnsemble(n_qubits=N_QUBITS, vqc_epochs=20)
    ensemble.fit(X_train, y_train)

    # 5. Evaluate
    preds = ensemble.predict(X_test)
    print(f"\nMIAS Swin-Quantum Hybrid Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Benign/Normal", "Malignant"]))