import numpy as np

from src.dataset import build_synthetic_dataset
from src.real_dataset import load_real_dataset


def build_hybrid_dataset(
    synthetic_per_class=300,
    real_per_class=300,
    random_state=42
):
    print("=" * 60)
    print("BUILDING HYBRID DATASET")
    print("=" * 60)

    # -----------------------------
    # 1. Synthetic dataset
    # -----------------------------
    print("\nGenerating synthetic dataset...")

    X_synthetic, y_synthetic = build_synthetic_dataset(
        samples_per_class=synthetic_per_class
    )

    print(f"Synthetic images: {len(X_synthetic)}")

    # -----------------------------
    # 2. Real dataset
    # -----------------------------
    print("\nLoading real dataset...")

    X_real, y_real = load_real_dataset()

    # -----------------------------
    # 3. Balance real dataset
    # -----------------------------
    rng = np.random.default_rng(random_state)

    selected_indices = []

    for class_id in np.unique(y_real):

        class_indices = np.where(y_real == class_id)[0]

        n_select = min(real_per_class, len(class_indices))

        chosen = rng.choice(
            class_indices,
            size=n_select,
            replace=False
        )

        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)

    X_real_balanced = [
        X_real[i] for i in selected_indices
    ]

    y_real_balanced = y_real[selected_indices]

    print(
        f"\nReal images after balancing: "
        f"{len(X_real_balanced)}"
    )

    # -----------------------------
    # 4. Combine datasets
    # -----------------------------
    X = list(X_synthetic) + X_real_balanced

    y = np.concatenate([
        y_synthetic,
        y_real_balanced
    ])

    # Shuffle
    indices = rng.permutation(len(X))

    X = [X[i] for i in indices]
    y = y[indices]

    print("\n" + "=" * 60)
    print("HYBRID DATASET READY")
    print("=" * 60)

    print(f"Total images: {len(X)}")
    print(f"Labels shape: {y.shape}")

    print("\nClass distribution:")

    for class_id in sorted(np.unique(y)):
        count = np.sum(y == class_id)
        print(f"Class {class_id}: {count}")

    return X, y