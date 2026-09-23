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
        # Added class_weight="balanced" to handle MIAS severe class imbalance
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

    def predict(self, X, threshold=0.35):
        """Prediction B - final binary output with customizable probability decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    from data_pipeline import load_mias_swin_data, clean_and_scale
    from sklearn.metrics import accuracy_score, classification_report

    print("Loading MIAS mammography dataset with Swin Transformer embeddings...")
    X, y = load_mias_swin_data(data_dir="./data/MIAS")
    X_train, X_test, y_train, y_test, scaler, feature_names = clean_and_scale(X, y)

    print("\nTraining Classical Ensemble model...")
    model = ClassicalEnsemble().fit(X_train, y_train)
    
    # Predict using the tuned decision threshold
    preds = model.predict(X_test, threshold=0.35)

    print(f"\nClassical Ensemble Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Benign/Normal", "Malignant"], zero_division=0))