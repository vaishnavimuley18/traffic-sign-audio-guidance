"""
dataset.py
==========
Generates a synthetic-but-realistic traffic sign image dataset.

WHY SYNTHETIC DATA?
This environment has no access to image datasets like GTSRB (German
Traffic Sign Recognition Benchmark) or Mapillary Traffic Sign Dataset,
since they live behind hosts (Kaggle, GTSRB servers, etc.) outside the
sandbox's network allowlist. To still deliver a fully working,
trainable, end-to-end pipeline, this module procedurally draws each
sign class (correct shape, color and icon/text per real-world
conventions) and applies photometric + geometric augmentation
(rotation, scale, blur, brightness, noise, occlusion) to mimic
real-world variation.

SWAPPING IN REAL DATA
For production use, replace `build_synthetic_dataset()` with a loader
that reads a real labeled folder structure, e.g.:

    data/train/<class_name>/*.png
    data/test/<class_name>/*.png

...and keep `CLASS_NAMES` in `sign_config.py` aligned with your
folder names. `train.py` and `model.py` do not need to change.
"""

import os
import random
import numpy as np
import cv2

from src.sign_config import SIGN_CATALOGUE

IMG_SIZE = 64  # width & height of each generated sign image


def _draw_base_shape(canvas, shape, color_bgr):
    """Draw the sign's outer shape (border) filled with a light interior."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    r = int(w * 0.42)

    if shape == "circle":
        cv2.circle(canvas, (cx, cy), r, color_bgr, thickness=-1)
        cv2.circle(canvas, (cx, cy), int(r * 0.82), (255, 255, 255), thickness=-1)
    elif shape == "octagon":
        pts = []
        for i in range(8):
            ang = np.pi / 8 + i * (np.pi / 4)
            pts.append((int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))))
        pts = np.array(pts, np.int32)
        cv2.fillPoly(canvas, [pts], color_bgr)
    elif shape == "triangle":
        pts = np.array([
            (cx, cy - r),
            (cx - int(r * 0.95), cy + int(r * 0.8)),
            (cx + int(r * 0.95), cy + int(r * 0.8)),
        ], np.int32)
        cv2.fillPoly(canvas, [pts], color_bgr)
        inner = np.array([
            (cx, cy - int(r * 0.72)),
            (cx - int(r * 0.68), cy + int(r * 0.58)),
            (cx + int(r * 0.68), cy + int(r * 0.58)),
        ], np.int32)
        cv2.fillPoly(canvas, [inner], (255, 255, 255))
    elif shape == "inverted_triangle":
        pts = np.array([
            (cx - int(r * 0.95), cy - int(r * 0.8)),
            (cx + int(r * 0.95), cy - int(r * 0.8)),
            (cx, cy + r),
        ], np.int32)
        cv2.fillPoly(canvas, [pts], color_bgr)
        inner = np.array([
            (cx - int(r * 0.68), cy - int(r * 0.58)),
            (cx + int(r * 0.68), cy - int(r * 0.58)),
            (cx, cy + int(r * 0.72)),
        ], np.int32)
        cv2.fillPoly(canvas, [inner], (255, 255, 255))
    elif shape == "square":
        half = int(r * 0.85)
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), color_bgr, -1)
    return canvas


def _draw_icon(canvas, sign):
    """Draw a simple representative icon or text for the sign's meaning."""
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2
    black = (30, 30, 30)

    if sign.name == "stop":
        cv2.putText(canvas, "STOP", (cx - 24, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    elif sign.name.startswith("speed_limit"):
        cv2.putText(canvas, sign.value, (cx - 15, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, 2)
    elif sign.name == "pedestrian_crossing":
        cv2.line(canvas, (cx - 6, cy - 12), (cx - 6, cy - 4), black, 2)
        cv2.circle(canvas, (cx - 6, cy - 16), 3, black, -1)
        cv2.line(canvas, (cx - 6, cy - 4), (cx - 12, cy + 12), black, 2)
        cv2.line(canvas, (cx - 6, cy - 4), (cx, cy + 12), black, 2)
    elif sign.name == "no_entry":
        cv2.rectangle(canvas, (cx - 16, cy - 5), (cx + 16, cy + 5), (255, 255, 255), -1)
    elif sign.name == "no_overtaking":
        cv2.circle(canvas, (cx - 8, cy), 6, (0, 0, 0), -1)
        cv2.circle(canvas, (cx + 8, cy), 6, (0, 0, 255), -1)
        cv2.line(canvas, (cx - 18, cy + 15), (cx + 18, cy - 15), (0, 0, 255), 3)
    elif sign.name == "school_zone":
        cv2.circle(canvas, (cx, cy - 10), 5, black, -1)
        cv2.line(canvas, (cx, cy - 5), (cx, cy + 8), black, 2)
    elif sign.name == "traffic_signal_ahead":
        for i, col in enumerate([(0, 0, 255), (0, 255, 255), (0, 255, 0)]):
            cv2.circle(canvas, (cx, cy - 12 + i * 10), 3, col, -1)
    elif sign.name == "roundabout_ahead":
        cv2.circle(canvas, (cx, cy), 10, black, 2)
        cv2.arrowedLine(canvas, (cx + 10, cy - 10), (cx + 10, cy + 2), black, 2, tipLength=0.5)
    elif sign.name == "hospital":
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 4)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 4)
    elif sign.name == "parking":
        cv2.putText(canvas, "P", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif sign.name == "one_way":
        cv2.arrowedLine(canvas, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 4, tipLength=0.4)
    elif sign.name == "yield":
        pass  # inverted triangle alone conveys yield
    return canvas


def _color_bgr(color_name):
    return {"red": (40, 40, 200), "blue": (180, 90, 20), "yellow": (30, 200, 220)}[color_name]


def generate_sign_image(sign, augment=True, size=IMG_SIZE):
    """Render a single sign image, optionally with augmentation noise."""
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)  # light background
    canvas = _draw_base_shape(canvas, sign.shape, _color_bgr(sign.color))
    canvas = _draw_icon(canvas, sign)

    if augment:
        # Rotation
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, random.uniform(0.85, 1.1))
        canvas = cv2.warpAffine(canvas, M, (size, size), borderValue=(250, 250, 250))
        # Brightness / contrast jitter
        alpha = random.uniform(0.75, 1.25)   # contrast
        beta = random.uniform(-25, 25)       # brightness
        canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        # Gaussian noise
        noise = np.random.normal(0, 8, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Occasional blur (simulates motion / low resolution camera)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    return canvas


def build_synthetic_dataset(samples_per_class=300, seed=42):
    """Return (X, y) numpy arrays: images and integer class ids."""
    random.seed(seed)
    np.random.seed(seed)
    X, y = [], []
    for sign in SIGN_CATALOGUE:
        for i in range(samples_per_class):
            img = generate_sign_image(sign, augment=(i > 0))  # keep sample 0 "clean"
            X.append(img)
            y.append(sign.class_id)
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    # shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def save_sample_gallery(out_path="samples/class_gallery.png"):
    """Save a grid image showing one clean example of every sign class (for README/demo)."""
    imgs = [generate_sign_image(s, augment=False) for s in SIGN_CATALOGUE]
    cols = 5
    rows = int(np.ceil(len(imgs) / cols))
    grid = np.full((rows * IMG_SIZE, cols * IMG_SIZE, 3), 255, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * IMG_SIZE:(r + 1) * IMG_SIZE, c * IMG_SIZE:(c + 1) * IMG_SIZE] = im
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path


if __name__ == "__main__":
    path = save_sample_gallery()
    print(f"Saved class gallery to {path}")
