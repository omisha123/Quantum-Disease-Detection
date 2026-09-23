"""
Quantum Hybrid Ensemble Path - Reduced Features
Blocks: QSVM (ZZFeatureMap quantum kernel) -> \
                                             Meta Combiner -> Prediction A
        VQC (RY encoding + StronglyEntanglingLayers) -> /
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# =========================================================
# QSVM: Quantum kernel via a ZZ-style feature map
# =========================================================

def make_quantum_kernel(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    def feature_map(x):
        # ZZFeatureMap-style encoding: Hadamard + RZ + entangling ZZ interactions
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ((np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

    def get_state(x):
        @qml.qnode(dev)
        def circuit(x):
            feature_map(x)
            return qml.state()
        return circuit(x)

    def kernel(x1, x2):
        # |<phi(x1)|phi(x2)>|^2 computed directly from statevectors
        state1 = get_state(x1)
        state2 = get_state(x2)
        overlap = np.vdot(state1, state2)
        return np.abs(overlap) ** 2

    return kernel


class QSVM:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.kernel_fn = make_quantum_kernel(n_qubits)
        # Added class_weight="balanced" to handle class imbalance
        self.svm = SVC(kernel="precomputed", probability=True, class_weight="balanced")
        self.X_train_ref = None

    def _kernel_matrix(self, A, B):
        return np.array([[self.kernel_fn(a, b) for b in B] for a in A])

    def fit(self, X_train, y_train):
        self.X_train_ref = X_train
        K_train = self._kernel_matrix(X_train, X_train)
        self.svm.fit(K_train, y_train)
        return self

    def predict_proba(self, X):
        K_test = self._kernel_matrix(X, self.X_train_ref)
        return self.svm.predict_proba(K_test)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# =========================================================
# VQC: RY angle encoding + StronglyEntanglingLayers + Dense head
# =========================================================

class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.dense_head = nn.Linear(n_qubits, 1)

    def forward(self, x):
        q_out = self.qlayer(x)          # Measurement: Pauli-Z expectation values
        out = self.dense_head(q_out)    # Classical Dense head output logits
        return out


def train_vqc(X_train, y_train, n_qubits, n_layers=3, epochs=30, lr=0.05):
    model = VQC(n_qubits, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Calculate pos_weight to handle class imbalance dynamically
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"   VQC epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    return model


def vqc_predict_proba(model, X):
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t).squeeze()
        preds = torch.sigmoid(logits)
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        return preds.numpy()


# =========================================================
# Quantum Hybrid Ensemble: QSVM + VQC -> Meta Combiner -> Prediction A
# =========================================================

class QuantumHybridEnsemble:
    def __init__(self, n_qubits=4, vqc_layers=3, vqc_epochs=30):
        self.n_qubits = n_qubits
        self.qsvm = QSVM(n_qubits)
        self.vqc_layers = vqc_layers
        self.vqc_epochs = vqc_epochs
        self.vqc_model = None
        # Added class_weight="balanced" to Meta Combiner
        self.meta_combiner = LogisticRegression(class_weight="balanced")

    def fit(self, X_train, y_train):
        print("Training QSVM...")
        self.qsvm.fit(X_train, y_train)

        print("Training VQC...")
        self.vqc_model = train_vqc(
            X_train, y_train, self.n_qubits, self.vqc_layers, self.vqc_epochs
        )

        qsvm_probs = self.qsvm.predict_proba(X_train)
        vqc_probs = vqc_predict_proba(self.vqc_model, X_train)
        meta_X = np.column_stack([qsvm_probs, vqc_probs])
        self.meta_combiner.fit(meta_X, y_train)
        return self

    def predict_proba(self, X):
        qsvm_probs = self.qsvm.predict_proba(X)
        vqc_probs = vqc_predict_proba(self.vqc_model, X)
        meta_X = np.column_stack([qsvm_probs, vqc_probs])
        return self.meta_combiner.predict_proba(meta_X)[:, 1]

    def predict(self, X):
        """Prediction A - final binary output."""
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    from data_pipeline import load_mias_swin_data, clean_and_scale, quantum_only_reduction
    from sklearn.metrics import accuracy_score, classification_report

    print("Loading MIAS dataset with Swin embeddings...")
    X, y = load_mias_swin_data(data_dir="./data/MIAS")
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)

    N_QUBITS = 4
    X_train_q, X_test_q, pca = quantum_only_reduction(X_train, X_test, n_components=N_QUBITS)

    subset = 80
    print("\nTraining Quantum Hybrid Ensemble...")
    model = QuantumHybridEnsemble(n_qubits=N_QUBITS, vqc_epochs=20)
    model.fit(X_train_q[:subset], y_train[:subset])

    preds = model.predict(X_test_q)
    print(f"\nQuantum Hybrid Ensemble Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Benign/Normal", "Malignant"]))