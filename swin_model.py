import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------------------------------------------------------
# 1. Flexible Metadata & File Loader
# ----------------------------------------------------------------------
def load_mias_metadata():
    """Locates and parses mias_info.csv across root and data directories."""
    possible_csvs = ["./data/mias_info.csv", "mias_info.csv", "data/mias_info.csv"]
    CSV_PATH = None
    for path in possible_csvs:
        if os.path.exists(path):
            CSV_PATH = path
            break

    if not CSV_PATH or not os.path.exists(CSV_PATH):
        return None

    try:
        raw_rows = []
        with open(CSV_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].upper() in ['REFNUM', 'MIAS']:
                    continue
                if len(parts) >= 3:
                    refnum = parts[0]
                    bg_tissue = parts[1]
                    class_type = parts[2]
                    severity = parts[3] if len(parts) >= 4 else 'N'
                    raw_rows.append({
                        'REFNUM': refnum,
                        'BG': bg_tissue,
                        'CLASS': class_type,
                        'SEVERITY': severity
                    })

        if not raw_rows:
            return None

        df = pd.DataFrame(raw_rows)
        df['SEVERITY'] = df['SEVERITY'].astype(str).str.strip()
        df['CLASS'] = df['CLASS'].astype(str).str.strip()
        df.loc[df['CLASS'] == 'NORM', 'SEVERITY'] = 'N'
        df['SEVERITY'] = df['SEVERITY'].replace({'': 'N', 'nan': 'N'})

        label_map = {'N': 0, 'B': 1, 'M': 2}
        df['target'] = df['SEVERITY'].map(label_map).fillna(0).astype(int)
        return df
    except Exception as e:
        print(f"Warning: Non-fatal error reading MIAS metadata: {e}")
        return None

# Pre-load metadata if present (safe fallback if missing)
df_mias = load_mias_metadata()


# ----------------------------------------------------------------------
# 2. Hybrid Neural Network Architecture
# ----------------------------------------------------------------------
class HybridModelNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(HybridModelNetwork, self).__init__()
        # Classical projection mimicking a quantum/swin feature mixing layer
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        logits = self.classifier(out)
        return logits


# ----------------------------------------------------------------------
# 3. Scikit-Learn Compatible Wrapper Class for export_results.py
# ----------------------------------------------------------------------
class QuantumSwinHybridEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=4, vqc_epochs=20, swin_epochs=20, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.vqc_epochs = vqc_epochs
        self.swin_epochs = swin_epochs
        self.learning_rate = learning_rate
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        if num_classes < 2:
            num_classes = 2

        input_dim = X.shape[1]
        self.model = HybridModelNetwork(input_dim=input_dim, num_classes=num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        epochs = max(self.vqc_epochs, self.swin_epochs)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' before predicting.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)