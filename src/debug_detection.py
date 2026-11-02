import cv2
import os
from src.detect import detect_candidate_regions, crop_regions

IMAGE_PATH = "samples/real_stop.jpg"
OUTPUT_DIR = "outputs/detected_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError(f"Could not read image: {IMAGE_PATH}")

boxes = detect_candidate_regions(img)
crops = crop_regions(img, boxes)

print(f"Detected candidate regions: {len(boxes)}")

for i, ((x, y, w, h, color), crop) in enumerate(zip(boxes, crops)):

    crop_path = os.path.join(
        OUTPUT_DIR,
        f"candidate_{i}_{color}.jpg"
    )

    cv2.imwrite(crop_path, crop)

    print(
        f"Candidate {i}: "
        f"x={x}, y={y}, w={w}, h={h}, color={color}"
    )

    print(f"Saved crop: {crop_path}")

print("\nOpen the saved crop image and check whether:")
print("1. The Stop sign is clearly visible")
print("2. There is too much background around it")
print("3. The sign is very small inside the crop")