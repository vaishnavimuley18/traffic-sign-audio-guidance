import os
import cv2
import numpy as np


# Maps real GTSRB-derived folders to YOUR project's class IDs
REAL_CLASS_TO_ID = {
    "speed_limit_20": 3,
    "speed_limit_60": 4,
    "speed_limit_80": 5,
    "no_overtaking": 8,
    "yield": 1,
    "stop": 0,
    "no_entry": 7,
    "traffic_signal": 10,
    "pedestrian_crossing": 2,
    "school_zone": 9,
    "roundabout": 11,
}


def load_real_dataset(data_dir="data/real"):
    images = []
    labels = []

    for class_name, class_id in REAL_CLASS_TO_ID.items():

        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"{class_name}: {len(files)} images")

        for filename in files:

            path = os.path.join(class_dir, filename)

            img = cv2.imread(path)

            if img is None:
                continue

            images.append(img)
            labels.append(class_id)

    print("\n============================================")
    print(f"Total real images loaded: {len(images)}")
    print("============================================")

    return images, np.array(labels)