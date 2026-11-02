"""
detect.py
=========
Locates candidate traffic-sign regions inside a full scene photo
(as opposed to an already-cropped sign image).

APPROACH
Traffic signs are designed to be visually salient: strong, saturated
red / blue / yellow against a natural background. This module:
  1. Converts to HSV and thresholds for the three sign color families.
  2. Finds contours in each color mask.
  3. Filters by area, aspect ratio, and solidity to reject noise.
  4. Returns bounding boxes of candidate regions, which are then cropped
     and passed to SignClassifier for recognition.

This is a classical-CV pre-filter, not a learned detector. It is fast,
needs no training data, and is a reasonable stand-in for this project's
scope. For production-grade detection in cluttered scenes (multiple
signs, partial occlusion, night-time), swap this module for a trained
object detector such as YOLOv8/YOLO-NAS fine-tuned on traffic-sign
bounding-box datasets — the rest of the pipeline (classify -> NLP ->
TTS) does not need to change, since it just consumes cropped regions.
"""

import cv2
import numpy as np

# HSV ranges tuned for typical sign colors (OpenCV H: 0-180)
COLOR_RANGES = {
    "red": [((0, 90, 80), (10, 255, 255)), ((170, 90, 80), (180, 255, 255))],
    "blue": [((95, 90, 60), (130, 255, 255))],
    "yellow": [((18, 90, 100), (35, 255, 255))],
}


def _color_mask(hsv, color):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def detect_candidate_regions(img_bgr, min_area_frac=0.001, max_area_frac=0.5):
    """Return a list of (x, y, w, h, color) candidate sign bounding boxes."""
    h, w = img_bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    boxes = []
    for color in COLOR_RANGES:
        mask = _color_mask(hsv, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_frac * img_area or area > max_area_frac * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / float(bh)
            if not (0.65 <= aspect <= 1.5):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.55:
                continue
            boxes.append((x, y, bw, bh, color))

    return boxes


def crop_regions(img_bgr, boxes, pad_frac=0.15):
    """Crop each candidate box (with a little padding) out of the image."""
    h, w = img_bgr.shape[:2]
    crops = []
    for (x, y, bw, bh, color) in boxes:
        pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        crops.append(img_bgr[y0:y1, x0:x1])
    return crops
