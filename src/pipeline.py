"""
pipeline.py
===========
Wires the four stages of the system together:

    image --> [detect] --> [classify] --> [NLP instruction] --> [TTS audio]

Two entry points are provided:
  - `process_cropped_sign(path)`   : input is already a cropped sign image
  - `process_scene(path)`          : input is a full driving-scene photo;
                                      candidate sign regions are located
                                      first via src.detect

Both return a list of result dicts and write one audio file per
detected sign into outputs/.
"""

import os
import cv2

from src.model import SignClassifier
from src.nlp_instructions import generate_instruction
from src.tts_engine import speak_to_file
from src.detect import detect_candidate_regions, crop_regions
from src.sign_config import ID_TO_SIGN

MODEL_PATH = "models/sign_classifier_hybrid.joblib"


def _load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return SignClassifier.load(MODEL_PATH)


def _classify_and_narrate(img_bgr, clf, out_prefix, index=0):
    class_id, confidence = clf.predict(img_bgr)
    sign = ID_TO_SIGN[class_id]
    instruction = generate_instruction(class_id, confidence)
    audio_path = f"outputs/{out_prefix}_{index}.wav"
    speak_to_file(instruction, audio_path)
    return {
        "sign_name": sign.display_name,
        "class_id": class_id,
        "confidence": round(confidence, 3),
        "instruction": instruction,
        "audio_path": audio_path,
    }


def process_cropped_sign(image_path: str, clf=None):
    clf = clf or _load_classifier()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    base = os.path.splitext(os.path.basename(image_path))[0]
    return [_classify_and_narrate(img, clf, base)]


def process_scene(image_path: str, clf=None):
    clf = clf or _load_classifier()
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    boxes = detect_candidate_regions(img)
    crops = crop_regions(img, boxes)

    if not crops:
        return []

    base = os.path.splitext(os.path.basename(image_path))[0]

    # --------------------------------------------------
    # Classify all candidate regions first
    # --------------------------------------------------
    candidates = []

    for i, crop in enumerate(crops):

        if crop.size == 0:
            continue

        class_id, confidence = clf.predict(crop)

        candidates.append({
            "crop": crop,
            "class_id": class_id,
            "confidence": confidence,
            "index": i
        })

    if not candidates:
        return []

    # --------------------------------------------------
    # Select the highest-confidence detection
    # --------------------------------------------------
    best = max(
        candidates,
        key=lambda x: x["confidence"]
    )

    # --------------------------------------------------
    # Generate NLP instruction + ONE audio file
    # --------------------------------------------------
    result = _classify_and_narrate(
        best["crop"],
        clf,
        base,
        index=best["index"]
    )

    return [result]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Sign Recognition + Audio Guidance")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument(
        "--mode", choices=["cropped", "scene"], default="cropped",
        help="'cropped' if the image is already a single cropped sign, "
             "'scene' if it's a full driving photo that may contain a sign somewhere in it",
    )
    args = parser.parse_args()

    fn = process_cropped_sign if args.mode == "cropped" else process_scene
    results = fn(args.image)

    if not results:
        print("No traffic sign recognized.")
    for r in results:
        print(f"\nDetected: {r['sign_name']}  (confidence: {r['confidence']*100:.1f}%)")
        print(f"Instruction: {r['instruction']}")
        print(f"Audio saved to: {r['audio_path']}")
