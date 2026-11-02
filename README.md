# Intelligent Traffic Sign Recognition & Natural-Language Audio Guidance

An end-to-end pipeline that detects a traffic sign in an image, classifies it,
converts the classification into a clear natural-language instruction, and
speaks it aloud — e.g. detecting a **Speed Limit 40** sign produces the audio
*"The speed limit ahead is 40 kilometres per hour. Please reduce your speed."*
rather than just announcing the sign's name.

```
   image/video frame
         │
         ▼
 ┌───────────────┐   classical CV: HSV color threshold + contour/shape
 │  1. DETECT    │   filtering → candidate sign bounding boxes
 └───────┬───────┘
         ▼
 ┌───────────────┐   HOG (shape) + HSV color-histogram features
 │  2. CLASSIFY  │   → RandomForest classifier → class + confidence
 └───────┬───────┘
         ▼
 ┌───────────────┐   template selection + slot filling (e.g. speed value)
 │  3. NLP / NLG │   + severity-based framing + confidence hedging
 └───────┬───────┘
         ▼
 ┌───────────────┐   offline TTS (espeak-ng) → .wav
 │  4. SPEECH    │
 └───────────────┘
```

## Why this design

| Stage | Technique used | Why |
|---|---|---|
| Detection | HSV color threshold + contour/shape filtering (`src/detect.py`) | Signs are designed to be color-salient (red/blue/yellow); this needs no training data and runs instantly. Swappable for a trained detector (YOLOv8 etc.) for cluttered/real-world scenes. |
| Classification | HOG + color-histogram features → RandomForest (`src/model.py`) | Trains in seconds on CPU, no GPU/deep-learning framework download needed in this environment, and the interface is identical to what a CNN wrapper would expose — see "Upgrading to a CNN" below. |
| NLP / instruction generation | Template + slot-filling + severity/confidence conditioning (`src/nlp_instructions.py`) | Deterministic, offline, and safety-appropriate for a bounded sign vocabulary; a generative-LLM fallback for open-vocabulary/unfamiliar signs is stubbed for future extension. |
| Speech | `espeak-ng` via subprocess (`src/tts_engine.py`) | Fully offline — important for an in-vehicle assistive feature that shouldn't depend on connectivity. Swappable for Amazon Polly / Google TTS for more natural voices. |

## A note on the training data

This sandbox has no network access to real traffic-sign datasets (GTSRB,
Mapillary, etc. — they live on hosts outside the environment's allowlist).
`src/dataset.py` therefore **procedurally generates** a labeled dataset of
15 common sign classes (correct shape/color/icon per real-world convention)
with rotation, scale, brightness, noise and blur augmentation, so the whole
pipeline is genuinely trainable and testable end-to-end.

Swapping in a real dataset later is a one-file change:
1. Point `src/dataset.py` (or a new loader) at real labeled image folders.
2. Keep class names aligned with `src/sign_config.py`.
3. Re-run `python -m src.train` — nothing else changes.

## Project layout

```
traffic-sign-audio-guidance/
├── src/
│   ├── sign_config.py       # sign catalogue + instruction templates (the "knowledge base")
│   ├── dataset.py           # synthetic dataset generator
│   ├── model.py             # feature extraction + classifier
│   ├── train.py             # trains and saves the classifier
│   ├── detect.py            # locates candidate sign regions in a full scene
│   ├── nlp_instructions.py  # class -> natural language instruction
│   ├── tts_engine.py        # text -> speech (.wav)
│   └── pipeline.py          # wires everything together + CLI
├── models/                  # trained classifier (.joblib)
├── samples/                 # example sign images + class gallery
├── outputs/                 # generated audio files land here
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
apt-get install espeak-ng      # offline TTS engine (Debian/Ubuntu)
```

## Usage

**1. Train the classifier** (generates the synthetic dataset and trains in ~30s):
```bash
python -m src.train
```

**2. Run on a single cropped sign image:**
```bash
python -m src.pipeline path/to/sign.png --mode cropped
```

**3. Run on a full driving-scene photo** (detects sign location first):
```bash
python -m src.pipeline path/to/scene.jpg --mode scene
```

Each run prints the detected sign, the generated instruction, and the path
to the generated `.wav` audio file in `outputs/`.

### Example output
```
Detected: Speed Limit 40  (confidence: 91.0%)
Instruction: The speed limit ahead is 40 kilometres per hour. Please reduce your speed.
Audio saved to: outputs/speed_limit_40_0.wav
```

## Supported sign classes (15)

Stop · Yield · Pedestrian Crossing · Speed Limit 20/40/60/80 · No Entry ·
No Overtaking · School Zone · Traffic Signal Ahead · Roundabout Ahead ·
Hospital · Parking · One Way

Add a new sign by adding one `SignDefinition` entry in `src/sign_config.py`
and re-running `python -m src.train`.

## Handling uncertainty

If the classifier's confidence for a detection falls below a threshold
(`LOW_CONFIDENCE_THRESHOLD` in `nlp_instructions.py`), the generated
instruction is automatically hedged ("A sign that may be '...' was detected,
but visibility was limited...") instead of stating an uncertain guess as
fact — important for a driver-facing safety feature.

## Extension paths for production

- **Detection**: replace `src/detect.py`'s color/contour heuristic with a
  fine-tuned YOLOv8/YOLO-NAS object detector for robust detection in
  cluttered, low-light, or multi-sign scenes.
- **Classification**: replace `SignClassifier` in `src/model.py` with a
  small CNN trained on a real dataset (e.g. GTSRB) — `predict()`'s
  signature can stay identical so `pipeline.py` needs no changes.
- **NLP**: for open-vocabulary or text-based signs, route through
  `generate_instruction_via_llm()` in `nlp_instructions.py`, which is
  stubbed to call an LLM (e.g. the Claude API) to paraphrase OCR'd text
  into a short, actionable instruction.
- **Speech**: swap `tts_engine.py`'s `espeak-ng` call for a cloud TTS API
  (Amazon Polly, Google Cloud TTS) for more natural-sounding voices, or for
  on-device neural TTS (e.g. Piper) for higher quality while staying offline.
- **Video**: run `process_scene()` per frame with simple temporal smoothing
  (e.g. only speak once per N consecutive frames agreeing) to avoid
  chattering announcements from a live dashcam feed.

## Limitations of this build

- The classifier is trained on procedurally generated sign renders, not
  real photographs — real-world accuracy will depend on retraining with an
  authentic dataset and, likely, a CNN rather than hand-crafted features.
- The detector is a classical color/shape heuristic, not a learned model —
  it will miss faded, occluded, or unusually lit signs that a trained
  detector would catch.
- TTS uses `espeak-ng`'s robotic default voice; intelligibility is fine but
  not naturalistic.
