"""
tts_engine.py
=============
Converts a natural-language instruction string into spoken audio.

Uses `pyttsx3` (backed by `espeak-ng` on Linux), which works fully
offline - important for an in-vehicle safety feature that should not
depend on network availability. Saves output as a .wav file.

For higher-quality, more natural voices in production, swap this
module for a cloud TTS API (Amazon Polly, Google Cloud TTS, ElevenLabs)
behind the same `speak_to_file()` function signature - nothing else in
the pipeline needs to change.
"""

import os
import subprocess


def speak_to_file(text: str, out_path: str = "outputs/instruction.wav", rate: int = 165) -> str:
    """
    Render `text` to a wav file at `out_path` using espeak-ng directly
    (more reliable in headless/sandboxed environments than pyttsx3's
    event loop). Returns the output path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        "espeak-ng" if _has("espeak-ng") else "espeak",
        "-s", str(rate),
        "-v", "en-us",
        "-w", out_path,
        text,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def _has(binary: str) -> bool:
    from shutil import which
    return which(binary) is not None


if __name__ == "__main__":
    path = speak_to_file(
        "Pedestrian crossing ahead. Please slow down and watch for pedestrians.",
        "outputs/demo.wav",
    )
    print(f"Saved audio to {path}")
