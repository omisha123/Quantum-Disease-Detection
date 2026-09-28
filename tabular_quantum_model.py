"""
Real quantum model for heart disease:
- QGA-style feature selection (ANOVA F-score proxy for fitness-guided selection)
- QPSO-style hyperparameter search (swarm-based, informed by particle best/global best,
  proxy for true qubit-encoded QPSO — same search principle: exploit + explore)
- Genuine QSVM + VQC from quantum_model.py
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from quantum_model import QSVM, train_vqc, vqc_predict_proba


def qga_feature_selection(X_train_scaled, X_test_scaled, y_train, feature_names, n_components=6):
    """Target-aware feature selection — picks features that actually separate
    the classes, instead of PCA's blind variance-maximizing components."""
    selector = SelectKBest(score_func=f_classif, k=min(n_components, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    selected_names = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"QGA-style selection picked: {selected_names}")

    # Encoding range [0, pi/2]. The old [-pi, pi] range made angles wrap (-pi == +pi
    # under RY) and pushed the ZZ kernel toward near-identity; a narrower range
    # improved held-out AUC on heart and Pima in our tests.
    ENC_SPAN = np.pi / 2

    def rescale(arr, ref_min, ref_max):
        denom = np.where((ref_max - ref_min) == 0, 1e-8, ref_max - ref_min)
        return np.clip((arr - ref_min) / denom, 0, 1) * ENC_SPAN

    train_min, train_max = X_train_sel.min(axis=0), X_train_sel.max(axis=0)
    X_train_q = rescale(X_train_sel, train_min, train_max)
    X_test_q = rescale(X_test_sel, train_min, train_max)
    return X_train_q, X_test_q, selected_names


def qpso_search_vqc_hyperparams(X_train_q, y_train, n_qubits, n_particles=6, n_iterations=4):
    """
    QPSO-style search over VQC hyperparameters (vqc_layers, epochs).
    Particles = candidate hyperparameter sets. Each iteration, particles move
    toward the best-found combination so far (global best), with some random
    exploration (mimicking QPSO's quantum-tunnelling-inspired jumps).
    This is a lightweight, honest proxy for true qubit-encoded QPSO —
    same search principle (swarm converging on a global best), run classically.
    """
    rng = np.random.default_rng(42)
    layer_choices = [2, 3, 4]
    epoch_choices = [15, 20, 30]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_q, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    best_score = -1
    best_params = {"vqc_layers": 3, "vqc_epochs": 20}

    particles = [
        {"vqc_layers": rng.choice(layer_choices), "vqc_epochs": rng.choice(epoch_choices)}
        for _ in range(n_particles)
    ]

    for it in range(n_iterations):
        print(f"  QPSO iteration {it+1}/{n_iterations}...")
        for p in particles:
            model = train_vqc(X_tr, y_tr, n_qubits, p["vqc_layers"], p["vqc_epochs"])
            probs = vqc_predict_proba(model, X_val)
            preds = (probs >= 0.5).astype(int)
            score = accuracy_score(y_val, preds)

            if score > best_score:
                best_score = score
                best_params = dict(p)

        # Move particles toward global best (with light random exploration)
        new_particles = []
        for _ in range(n_particles):
            if rng.random() < 0.6:
                new_particles.append(dict(best_params))
            else:
                new_particles.append({
                    "vqc_layers": rng.choice(layer_choices),
                    "vqc_epochs": rng.choice(epoch_choices)
                })
        particles = new_particles

        best_params = {k: int(v) for k, v in best_params.items()}
    print(f"  QPSO best params: {best_params} (val accuracy {best_score:.3f})")
    return best_params


class TabularQuantumEnsemble:
    def __init__(self, n_qubits=6, vqc_layers=3, vqc_epochs=20, qsvm_ref_size=150):
        self.n_qubits = n_qubits
        self.qsvm = QSVM(n_qubits)
        self.vqc_layers = vqc_layers
        self.vqc_epochs = vqc_epochs
        self.vqc_model = None
        self.qsvm_ref_size = qsvm_ref_size

    def fit(self, X_train, y_train):
        print("Training real QSVM...")
        # Cap QSVM's reference set — prediction cost scales with this size,
        # and SHAP calls predict_proba hundreds of times per explanation
        if len(X_train) > self.qsvm_ref_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), self.qsvm_ref_size, replace=False)
            X_qsvm, y_qsvm = X_train[idx], y_train[idx]
        else:
            X_qsvm, y_qsvm = X_train, y_train
        self.qsvm.fit(X_qsvm, y_qsvm)

        print(f"Training real VQC (layers={self.vqc_layers}, epochs={self.vqc_epochs})...")
        self.vqc_model = train_vqc(X_train, y_train, self.n_qubits, self.vqc_layers, self.vqc_epochs)
        return self
    
    def predict_proba(self, X):
        qsvm_probs = self.qsvm.predict_proba(X)
        vqc_probs = vqc_predict_proba(self.vqc_model, X)
        return (qsvm_probs + vqc_probs) / 2.0

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        return (probs >= threshold).astype(int)