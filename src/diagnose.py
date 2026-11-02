import os
import cv2

from src.model import SignClassifier


MODEL_PATH = "models/sign_classifier_hybrid.joblib"

TEST_IMAGES = [
    "samples/real_sign_1.jpg",
    "samples/real_sign_2.jpg",
    "samples/real_sign_3.jpg",
    "samples/real_stop.jpg",
]


CLASS_NAMES = {
    0: "Stop",
    1: "Yield",
    2: "Pedestrian Crossing",
    3: "Speed Limit 20",
    4: "Speed Limit 60",
    5: "Speed Limit 80",
    6: "Speed Limit 40",
    7: "No Entry",
    8: "No Overtaking",
    9: "School Zone",
    10: "Traffic Signal Ahead",
    11: "Roundabout Ahead",
    12: "Hospital",
    13: "Parking",
    14: "One Way",
}


clf = SignClassifier.load(MODEL_PATH)

print("=" * 70)
print("EXTERNAL REAL-WORLD TEST")
print("=" * 70)

for image_path in TEST_IMAGES:

    img = cv2.imread(image_path)

    if img is None:
        print(f"\nCould not read: {image_path}")
        continue

    predictions = clf.predict_topk(img, k=5)

    print("\n" + "-" * 70)
    print(f"Image: {image_path}")
    print("-" * 70)

    for rank, (class_id, confidence) in enumerate(predictions, start=1):
        name = CLASS_NAMES.get(class_id, f"Class {class_id}")

        print(
            f"{rank}. {name:<25} "
            f"{confidence * 100:.2f}%"
        )