"""
Image Screening Path: BreastMNIST (breast ultrasound images, benign/malignant)
Genuinely a SCREENING dataset - ultrasound is typically used before/instead of biopsy,
so this is a earlier-stage screening modality than the Wisconsin biopsy-derived dataset.

Auto-downloads via the `medmnist` package (~560KB, one-time download).
Requires internet access on first run: pip install medmnist
"""

import numpy as np


def load_breast_ultrasound_images(size=28):
    """
    Returns (X_train, y_train, X_test, y_test) as numpy arrays.
    X shape: (n_samples, size, size) grayscale, values in [0, 1]
    y: 0 = malignant, 1 = benign (matches BreastMNIST's own convention... see note below)

    Note: BreastMNIST source labels are 0=malignant, 1=normal+benign (binary merged).
    """
    from medmnist import BreastMNIST

    train_ds = BreastMNIST(split="train", download=True, size=size)
    test_ds = BreastMNIST(split="test", download=True, size=size)

    def to_arrays(ds):
        X = np.array([np.array(img) for img, _ in ds], dtype=np.float32) / 255.0
        y = np.array([label[0] for _, label in ds], dtype=np.int64)
        return X, y

    X_train, y_train = to_arrays(train_ds)
    X_test, y_test = to_arrays(test_ds)
    return X_train, y_train, X_test, y_test


def make_synthetic_ultrasound_data(n_train=200, n_test=50, size=28, seed=42):
    """
    FALLBACK ONLY - for testing pipeline code without network access to Zenodo.
    Generates random data shaped identically to real BreastMNIST so the model
    architecture can be validated end-to-end. DO NOT use this for real results/demo -
    it's for code verification only. Real demo must use load_breast_ultrasound_images().
    """
    rng = np.random.default_rng(seed)
    X_train = rng.random((n_train, size, size), dtype=np.float32)
    y_train = rng.integers(0, 2, n_train)
    X_test = rng.random((n_test, size, size), dtype=np.float32)
    y_test = rng.integers(0, 2, n_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = load_breast_ultrasound_images()
        print("Loaded REAL BreastMNIST data.")
    except Exception as e:
        print(f"Could not download real data ({e}), using synthetic data for code testing only.")
        X_train, y_train, X_test, y_test = make_synthetic_ultrasound_data()

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")
