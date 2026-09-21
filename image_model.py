"""
Image Screening Models:
- Classical: small CNN (Conv -> Pool -> Conv -> Pool -> Dense)
- Quantum Hybrid: same CNN as a feature extractor -> reduce to few features -> VQC (reused from quantum_model.py)

This mirrors the tabular architecture exactly: same-shape comparison,
only the model core changes between classical and quantum paths.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_model import VQC  # reuse the same VQC block used for tabular data


# =========================================================
# Classical CNN
# =========================================================

class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))   # -> (batch, 8, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (batch, 16, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train_classical_cnn(X_train, y_train, epochs=15, lr=0.001, batch_size=32):
    model = ClassicalCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # add channel dim
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            preds = model(X_t[idx])
            loss = loss_fn(preds, y_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Classical CNN epoch {epoch+1}/{epochs} - avg loss: {total_loss/(n/batch_size):.4f}")

    return model


def cnn_predict_proba(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(X_t).squeeze(-1).numpy()


# =========================================================
# Quantum Hybrid CNN: CNN feature extractor -> reduce -> VQC
# =========================================================

class HybridCNNFeatureExtractor(nn.Module):
    """Same conv backbone as ClassicalCNN, but outputs a small feature vector
    (n_qubits-sized) instead of a final classification - feeds into the VQC."""
    def __init__(self, n_qubits):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc_reduce = nn.Linear(16 * 7 * 7, n_qubits)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_reduce(x)) * np.pi  # squash to [-pi, pi] for angle encoding
        return x


class HybridCNNQuantum(nn.Module):
    """Full pipeline: CNN feature extractor -> VQC -> prediction. Trained end-to-end."""
    def __init__(self, n_qubits, vqc_layers=3):
        super().__init__()
        self.feature_extractor = HybridCNNFeatureExtractor(n_qubits)
        self.vqc = VQC(n_qubits, vqc_layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.vqc(features)


def train_hybrid_cnn_quantum(X_train, y_train, n_qubits=4, vqc_layers=2, epochs=10, lr=0.01, batch_size=16):
    model = HybridCNNQuantum(n_qubits, vqc_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            preds = model(X_t[idx])
            loss = loss_fn(preds, y_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  Hybrid CNN+VQC epoch {epoch+1}/{epochs} - avg loss: {total_loss/n_batches:.4f}")

    return model


def hybrid_cnn_predict_proba(model, X):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return model(X_t).squeeze(-1).numpy()


if __name__ == "__main__":
    from image_pipeline import load_breast_ultrasound_images, make_synthetic_ultrasound_data
    from sklearn.metrics import accuracy_score

    try:
        X_train, y_train, X_test, y_test = load_breast_ultrasound_images()
        print("Using REAL BreastMNIST data.")
    except Exception as e:
        print(f"Real data unavailable here ({type(e).__name__}), using SYNTHETIC data "
              f"to validate the model code runs correctly end-to-end.")
        print("On your machine with internet access, this will use real BreastMNIST data.")
        X_train, y_train, X_test, y_test = make_synthetic_ultrasound_data(n_train=120, n_test=30)

    # Keep it small/fast for a quick validation run
    subset = min(80, len(X_train))

    print("\nTraining Classical CNN...")
    cnn_model = train_classical_cnn(X_train[:subset], y_train[:subset], epochs=10)
    cnn_preds = (cnn_predict_proba(cnn_model, X_test) >= 0.5).astype(int)
    print(f"Classical CNN accuracy: {accuracy_score(y_test, cnn_preds):.4f}")

    print("\nTraining Quantum Hybrid CNN...")
    hybrid_model = train_hybrid_cnn_quantum(X_train[:subset], y_train[:subset], n_qubits=4, epochs=8)
    hybrid_preds = (hybrid_cnn_predict_proba(hybrid_model, X_test) >= 0.5).astype(int)
    print(f"Quantum Hybrid CNN accuracy: {accuracy_score(y_test, hybrid_preds):.4f}")
