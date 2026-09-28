"""
Classical Ensemble Path - Full Feature Set
Blocks: SVM RBF kernel -> \
                        Meta Combiner (Logistic Regression) -> Prediction B
        Random Forest  -> /
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ClassicalEnsemble:
    def __init__(self, random_state=42):
        # class_weight="balanced" handles imbalanced medical datasets (most are)
        self.svm = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        )
        self.rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state
        )
        self.meta_combiner = LogisticRegression(class_weight="balanced")
        self.is_fitted = False

    def fit(self, X_train, y_train):
        # --- Base models (Tier 1) ---
        self.svm.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        # --- Build meta-features: base model probability outputs ---
        svm_probs = self.svm.predict_proba(X_train)[:, 1]
        rf_probs = self.rf.predict_proba(X_train)[:, 1]
        meta_X = np.column_stack([svm_probs, rf_probs])

        # --- Meta Combiner (Tier 2) ---
        self.meta_combiner.fit(meta_X, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Returns probability of class 1 (used by SHAP + evaluation)."""
        svm_probs = self.svm.predict_proba(X)[:, 1]
        rf_probs = self.rf.predict_proba(X)[:, 1]
        meta_X = np.column_stack([svm_probs, rf_probs])
        return self.meta_combiner.predict_proba(meta_X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Prediction B - final binary output with customizable probability decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    # Self-contained smoke test - sklearn's built-in Wisconsin Breast Cancer
    # dataset, no other project files required. Just checks the ensemble
    # trains and predicts correctly; the real benchmark run is export_results.py.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    data = load_breast_cancer()
    X, y = data.data, 1 - data.target  # flip so 1 = malignant

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training Classical Ensemble model on Wisconsin Breast Cancer (smoke test)...")
    model = ClassicalEnsemble().fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    print(f"\nClassical Ensemble Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Benign", "Malignant"], zero_division=0))