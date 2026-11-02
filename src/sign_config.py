"""
sign_config.py
================
Central knowledge base for the project.

Each traffic sign class is defined with:
  - shape / color (used by the synthetic data generator AND by the
    classical-CV pre-detector)
  - a human label
  - an NLP instruction *template*. Templates can contain a {value}
    placeholder that gets filled in dynamically (e.g. the number on a
    speed-limit sign), which is what makes this a light NLG step rather
    than a static lookup table.

To extend the system with a real dataset (e.g. GTSRB - German Traffic
Sign Recognition Benchmark, or a country-specific sign set), you only
need to:
  1. Add/edit entries here with the correct class name.
  2. Point `src/dataset.py` at the real image folders instead of the
     synthetic generator.
  3. Retrain with `src/train.py` (the model architecture and pipeline
     do not change).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SignDefinition:
    class_id: int
    name: str                 # internal class name, e.g. "speed_limit_40"
    display_name: str         # human readable, e.g. "Speed Limit 40"
    shape: str                # circle | triangle | octagon | inverted_triangle | square
    color: str                # red | blue | yellow
    instruction_template: str # may contain {value}
    value: Optional[str] = None  # dynamic value substituted into template, e.g. "40"
    severity: str = "info"    # info | caution | warning  (affects TTS pacing/prefix)


# ---------------------------------------------------------------------------
# The sign catalogue. This is intentionally a small, representative set of
# common regulatory / warning / informational signs so the whole pipeline
# (detect -> classify -> generate instruction -> speak) can be demonstrated
# end-to-end. Adding a new sign is a one-line addition here plus retraining.
# ---------------------------------------------------------------------------
SIGN_CATALOGUE = [
    SignDefinition(0, "stop", "Stop", "octagon", "red",
                   "Stop sign ahead. Come to a complete stop and check for traffic before proceeding.",
                   severity="warning"),

    SignDefinition(1, "yield", "Yield", "inverted_triangle", "red",
                   "Yield ahead. Slow down and give way to crossing or oncoming traffic.",
                   severity="caution"),

    SignDefinition(2, "pedestrian_crossing", "Pedestrian Crossing", "triangle", "yellow",
                   "Pedestrian crossing ahead. Please slow down and watch for pedestrians.",
                   severity="caution"),

    SignDefinition(3, "speed_limit_20", "Speed Limit 20", "circle", "red",
                   "The speed limit ahead is {value} kilometres per hour. Please reduce your speed.",
                   value="20", severity="info"),

    SignDefinition(4, "speed_limit_40", "Speed Limit 40", "circle", "red",
                   "The speed limit ahead is {value} kilometres per hour. Please reduce your speed.",
                   value="40", severity="info"),

    SignDefinition(5, "speed_limit_60", "Speed Limit 60", "circle", "red",
                   "The speed limit ahead is {value} kilometres per hour. Please adjust your speed accordingly.",
                   value="60", severity="info"),

    SignDefinition(6, "speed_limit_80", "Speed Limit 80", "circle", "red",
                   "The speed limit ahead is {value} kilometres per hour. Please adjust your speed accordingly.",
                   value="80", severity="info"),

    SignDefinition(7, "no_entry", "No Entry", "circle", "red",
                   "No entry ahead. This road is closed to vehicles from this direction. Please turn back or choose another route.",
                   severity="warning"),

    SignDefinition(8, "no_overtaking", "No Overtaking", "circle", "red",
                   "Overtaking is not allowed ahead. Please stay in your lane.",
                   severity="caution"),

    SignDefinition(9, "school_zone", "School Zone", "triangle", "yellow",
                   "School zone ahead. Please slow down and watch for children.",
                   severity="caution"),

    SignDefinition(10, "traffic_signal_ahead", "Traffic Signal Ahead", "triangle", "yellow",
                   "Traffic signal ahead. Be prepared to stop.",
                   severity="caution"),

    SignDefinition(11, "roundabout_ahead", "Roundabout Ahead", "triangle", "yellow",
                   "Roundabout ahead. Slow down and give way to traffic already on the roundabout.",
                   severity="caution"),

    SignDefinition(12, "hospital", "Hospital", "square", "blue",
                   "Hospital ahead. Please avoid honking and drive quietly.",
                   severity="info"),

    SignDefinition(13, "parking", "Parking", "square", "blue",
                   "Parking area ahead.",
                   severity="info"),

    SignDefinition(14, "one_way", "One Way", "square", "blue",
                   "This is a one-way road. Traffic moves in a single direction only.",
                   severity="info"),
]

CLASS_NAMES = [s.name for s in SIGN_CATALOGUE]
ID_TO_SIGN = {s.class_id: s for s in SIGN_CATALOGUE}
NAME_TO_SIGN = {s.name: s for s in SIGN_CATALOGUE}
NUM_CLASSES = len(SIGN_CATALOGUE)

SEVERITY_PREFIX = {
    "warning": "Attention. ",
    "caution": "",
    "info": "",
}
