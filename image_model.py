"""
Image Screening Models:
- Classical CNN: Flexible conv network (Conv -> Pool -> Conv -> Pool -> FC)
- Hybrid Swin/CNN + VQC: Translates image representations/Swin features into 
  quantum state angles for VQC processing (reused from quantum_model.py).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum_model import VQC  # reuse the VQC block from quantum_model.py


# =========================================================
# Classical CNN
# =========================================================

class ClassicalCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # LazyLinear dynamically computes flatten dimension on first forward pass
        self.fc1 = nn.LazyLinear(32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train_classical_cnn(X_train, y_train, epochs=15, lr=0.001, batch_size=32):
    model = ClassicalCNN(in_channels=1 if X_train.ndim == 3 else X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # Format input tensor
    if X_train.ndim == 3:
        X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    else:
        X_t = torch.tensor(X_train, dtype=torch.float32)

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
        if (epoch + 1) % 5 == 0:
            print(f"  Classical CNN epoch {epoch+1}/{epochs} - avg loss: {total_loss/n_batches:.4f}")

    return model


def cnn_predict_proba(model, X):
    if X.ndim == 3:
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    else:
        X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_t).squeeze()
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        return preds.numpy()


# =========================================================
# Quantum Hybrid Pathways (Swin & CNN backbones)
# =========================================================

class HybridCNNFeatureExtractor(nn.Module):
    """Conv backbone reducing images to n_qubits features squashed to [-pi, pi]."""
    def __init__(self, n_qubits=4, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc_reduce = nn.LazyLinear(n_qubits)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc_reduce(x)) * np.pi  # Squash to [-pi, pi] for angle encoding
        return x


class HybridCNNQuantum(nn.Module):
    """Pipeline: CNN Extractor -> VQC -> Dense Output."""
    def __init__(self, n_qubits=4, vqc_layers=2, in_channels=1):
        super().__init__()
        self.feature_extractor = HybridCNNFeatureExtractor(n_qubits, in_channels)
        self.vqc = VQC(n_qubits, vqc_layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.vqc(features)


class HybridSwinQuantum(nn.Module):
    """Pipeline for pre-extracted Swin Transformer feature vectors -> VQC -> Dense Output."""
    def __init__(self, in_features, n_qubits=4, vqc_layers=2):
        super().__init__()
        self.fc_reduce = nn.Linear(in_features, n_qubits)
        self.vqc = VQC(n_qubits, vqc_layers)

    def forward(self, x):
        features = torch.tanh(self.fc_reduce(x)) * np.pi
        return self.vqc(features)


def train_hybrid_cnn_quantum(X_train, y_train, n_qubits=4, vqc_layers=2, epochs=10, lr=0.01, batch_size=16):
    is_1d_features = (X_train.ndim == 2)
    
    if is_1d_features:
        model = HybridSwinQuantum(in_features=X_train.shape[1], n_qubits=n_qubits, vqc_layers=vqc_layers)
        X_t = torch.tensor(X_train, dtype=torch.float32)
    else:
        in_ch = 1 if X_train.ndim == 3 else X_train.shape[1]
        model = HybridCNNQuantum(n_qubits=n_qubits, vqc_layers=vqc_layers, in_channels=in_ch)
        X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) if X_train.ndim == 3 else torch.tensor(X_train, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
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
        print(f"  Hybrid Quantum epoch {epoch+1}/{epochs} - avg loss: {total_loss/n_batches:.4f}")

    return model


def hybrid_cnn_predict_proba(model, X):
    if X.ndim == 3:
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    else:
        X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_t).squeeze()
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        return preds.numpy()


if __name__ == "__main__":
    from image_pipeline import load_mias_swin_data, make_synthetic_ultrasound_data
    from sklearn.metrics import accuracy_score

    try:
        X_train, y_train, X_test, y_test = load_mias_swin_data(data_dir="./data/MIAS")
        print("Successfully loaded MIAS Swin Transformer features.")
    except Exception as e:
        print(f"MIAS data unreadable ({type(e).__name__}), running test with SYNTHETIC image vectors...")
        X_train, y_train, X_test, y_test = make_synthetic_ultrasound_data(n_train=120, n_test=30)

    subset = min(80, len(X_train))

    if X_train.ndim == 2:
        # Pre-extracted 1D embeddings (e.g., Swin Transformer)
        print("\nTraining Classical Dense baseline on Swin embeddings...")
        classical_fc = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Run a quick train loop for validation
        opt = torch.optim.Adam(classical_fc.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()
        X_t = torch.tensor(X_train[:subset], dtype=torch.float32)
        y_t = torch.tensor(y_train[:subset], dtype=torch.float32).unsqueeze(1)
        
        for ep in range(10):
            opt.zero_grad()
            loss = loss_fn(classical_fc(X_t), y_t)
            loss.backward()
            opt.step()
            
        with torch.no_grad():
            preds = (classical_fc(torch.tensor(X_test, dtype=torch.float32)).squeeze() >= 0.5).int().numpy()
        print(f"Classical Classifier accuracy: {accuracy_score(y_test, preds):.4f}")

    else:
        # 2D Image inputs
        print("\nTraining Classical CNN...")
        cnn_model = train_classical_cnn(X_train[:subset], y_train[:subset], epochs=10)
        cnn_preds = (cnn_predict_proba(cnn_model, X_test) >= 0.5).astype(int)
        print(f"Classical CNN accuracy: {accuracy_score(y_test, cnn_preds):.4f}")

    print("\nTraining Hybrid Quantum Model...")
    hybrid_model = train_hybrid_cnn_quantum(X_train[:subset], y_train[:subset], n_qubits=4, epochs=8)
    hybrid_preds = (hybrid_cnn_predict_proba(hybrid_model, X_test) >= 0.5).astype(int)
    print(f"Hybrid Quantum accuracy: {accuracy_score(y_test, hybrid_preds):.4f}")