import os
import time
import numpy as np

from src.dataset import build_synthetic_dataset
from src.real_split import load_real_train_val
from src.model import SignClassifier
from src.sign_config import SIGN_CATALOGUE
from src.model import SignClassifier, extract_features


def main():

    print("=" * 60)
    print("HYBRID TRAFFIC SIGN CLASSIFIER TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Generate synthetic training data
    # --------------------------------------------------
    print("\nGenerating synthetic training dataset...")

    t0 = time.time()

    X_synthetic, y_synthetic = build_synthetic_dataset(
        samples_per_class=300
    )

    print(
        f"Synthetic training images: {len(X_synthetic)}"
    )
    print(f"Time: {time.time() - t0:.1f}s")

    # --------------------------------------------------
    # 2. Load real training + validation data
    # --------------------------------------------------
    print("\nLoading real GTSRB dataset...")

    X_real_train, y_real_train, X_real_val, y_real_val = (
        load_real_train_val()
    )

    # --------------------------------------------------
    # 3. Combine synthetic + real training data
    # --------------------------------------------------
    X_train = list(X_synthetic) + list(X_real_train)

    y_train = np.concatenate([
        y_synthetic,
        y_real_train
    ])

    print("\n============================================")
    print("TRAINING DATA")
    print("============================================")
    print(f"Synthetic: {len(X_synthetic)}")
    print(f"Real:      {len(X_real_train)}")
    print(f"TOTAL:     {len(X_train)}")

    print("\n============================================")
    print("VALIDATION DATA")
    print("============================================")
    print(f"Real validation: {len(X_real_val)}")

    # --------------------------------------------------
    # 4. Train classifier
    # --------------------------------------------------
    print("\n============================================")
    print("TRAINING RANDOM FOREST")
    print("============================================")

    clf = SignClassifier()

    t0 = time.time()

    clf.fit(
    X_train,
    y_train,
    test_size=None,
    verbose=False
    )

    print(
        f"\nTraining completed in "
        f"{time.time() - t0:.1f}s"
    )

    # --------------------------------------------------
    # 5. Evaluate on REAL validation data
    # --------------------------------------------------
    print("\n============================================")
    print("REAL-WORLD VALIDATION")
    print("============================================")

    X_val_features = np.array([
    extract_features(img)
    for img in X_real_val
    ])

    val_accuracy = clf.clf.score(
        X_val_features,
        y_real_val
    )

    print(
        f"Real validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------
    # 6. Save model
    # --------------------------------------------------
    os.makedirs("models", exist_ok=True)

    clf.save(
        "models/sign_classifier_hybrid.joblib"
    )

    print("\n============================================")
    print("HYBRID MODEL SAVED")
    print("============================================")
    print(
        "models/sign_classifier_hybrid.joblib"
    )


if __name__ == "__main__":
    main()