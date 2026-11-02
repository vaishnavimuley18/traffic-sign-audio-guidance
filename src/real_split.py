import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.real_dataset import REAL_CLASS_TO_ID


def load_real_train_val(
    data_dir="data/real",
    val_size=0.2,
    random_state=42
):
    X_train = []
    y_train = []

    X_val = []
    y_val = []

    for class_name, class_id in REAL_CLASS_TO_ID.items():

        class_dir = os.path.join(data_dir, class_name)

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        images = []
        labels = []

        for filename in files:

            path = os.path.join(class_dir, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            images.append(img)
            labels.append(class_id)

        if len(images) < 2:
            continue

        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=val_size,
            random_state=random_state,
            stratify=labels
        )

        X_train.extend(train_imgs)
        y_train.extend(train_labels)

        X_val.extend(val_imgs)
        y_val.extend(val_labels)

    print("\n============================================")
    print("REAL DATASET SPLIT")
    print("============================================")
    print(f"Real training images:   {len(X_train)}")
    print(f"Real validation images: {len(X_val)}")

    return (
        X_train,
        np.array(y_train),
        X_val,
        np.array(y_val)
    )