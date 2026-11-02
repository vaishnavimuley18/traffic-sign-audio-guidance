"""
model.py
========
Feature extraction + classifier for traffic sign recognition.

DESIGN NOTE
This project uses classical hand-crafted features (HOG for shape/edge
structure + HSV color histograms for sign color, which is a very
strong signal in traffic signs: red=prohibition/danger, blue=mandatory
/informational, yellow=warning) fed into a RandomForest classifier
from scikit-learn.

This choice is deliberate for this environment: it trains in seconds
on CPU with no GPU or large framework (torch/tensorflow) download,
while still demonstrating the full "computer vision -> classification"
stage the project requires.

UPGRADING TO A CNN (recommended for production / a real dataset)
The interface is intentionally identical to what a CNN wrapper would
expose (`extract_features`/`SignClassifier.predict`), so you can swap
`SignClassifier` for a `torch.nn.Module`-based classifier later
without touching `pipeline.py`, `nlp_instructions.py` or `tts_engine.py`.
A ready-made upgrade path: replace this file with a small CNN
(e.g. 4 conv blocks + FC head) trained on real GTSRB images resized to
32x32 or 64x64 - the rest of the pipeline stays the same.
"""

import numpy as np
import cv2
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.sign_config import NUM_CLASSES


def extract_features(img_bgr):
    """Convert a BGR sign image into a fixed-length feature vector."""
    img = cv2.resize(img_bgr, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Shape / edge structure
    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True,
    )

    # Color signal (sign color is highly discriminative: red/blue/yellow)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-6)
    hist_s = hist_s / (hist_s.sum() + 1e-6)

    return np.concatenate([hog_feat, hist_h, hist_s]).astype(np.float32)


def build_feature_matrix(images):
    return np.array([extract_features(im) for im in images], dtype=np.float32)


class SignClassifier:
    """Thin wrapper around the trained model + label metadata."""

    def __init__(self, clf=None):
        self.clf = clf or RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        )

    def fit(self, X_imgs, y, test_size=0.15, verbose=True):
        """
        Train the classifier.

        If test_size is None:
            Train on all supplied data without an internal split.

        Otherwise:
                Perform the original train/test split.
        """

        feats = build_feature_matrix(X_imgs)

        if test_size is None:
            self.clf.fit(feats, y)

            if verbose:
                print(f"Training on all {len(y)} images.")

            return None

        Xtr, Xte, ytr, yte = train_test_split(
            feats,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        self.clf.fit(Xtr, ytr)

        preds = self.clf.predict(Xte)

        acc = accuracy_score(yte, preds)

        if verbose:
            print(f"Held-out test accuracy: {acc*100:.2f}%")
            print(
                classification_report(
                        yte,
                        preds,
                        zero_division=0
                    )
                )

        return acc

    def predict(self, img_bgr):
        """Return (class_id, confidence) for a single cropped sign image."""
        feat = extract_features(img_bgr).reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return class_id, confidence

    def predict_topk(self, img_bgr, k=3):
        feat = extract_features(img_bgr).reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top_idx]

    def save(self, path="models/sign_classifier.joblib"):
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path="models/sign_classifier.joblib"):
        clf = joblib.load(path)
        return cls(clf)
