"""
nlp_instructions.py
====================
The "NLP module" of the pipeline: turns a recognized sign class into a
clear, spoken-style natural-language instruction, instead of just
reading out the sign's name.

DESIGN
Rather than a single flat lookup, generation happens in two stages so
it is genuinely a *generation* step, not just a dictionary read:

  1. Template selection - each SignDefinition carries an
     `instruction_template` (see sign_config.py) with an optional
     {value} slot (e.g. a speed limit number).
  2. Contextual composition - a severity-based prefix ("Attention.")
     is prepended for signs that demand urgent action (stop, no entry),
     and confidence-based hedging is added when the vision model is
     not fully sure, so the system never states an uncertain detection
     as a hard fact.

This keeps the module lightweight and fully deterministic/offline
(no external LLM call needed for a bounded, safety-relevant sign
vocabulary) while still demonstrating template-based NLG with slot
filling and context-conditioning.

EXTENDING WITH A GENERATIVE LLM
For open-vocabulary signs (e.g. novel text-based signs an OCR module
reads), `generate_instruction_via_llm()` shows how you would instead
call an LLM (e.g. the Claude API) to paraphrase/compose an instruction
on the fly - useful as a fallback for the "unfamiliar sign" scenario
named in the problem statement.
"""

from src.sign_config import ID_TO_SIGN, SEVERITY_PREFIX

LOW_CONFIDENCE_THRESHOLD = 0.55


def generate_instruction(class_id: int, confidence: float = 1.0) -> str:
    """Compose the final spoken instruction for a recognized sign."""
    sign = ID_TO_SIGN[class_id]
    base = sign.instruction_template.format(value=sign.value) if sign.value else sign.instruction_template
    prefix = SEVERITY_PREFIX.get(sign.severity, "")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        # Hedge rather than assert an uncertain detection as fact.
        hedge = f"A sign that may be '{sign.display_name}' was detected ahead, but visibility was limited. "
        return hedge + base

    return f"{prefix}{base}"


def generate_instruction_via_llm(ocr_text: str) -> str:
    """
    Fallback path for unfamiliar / text-based signs not in the fixed
    catalogue (e.g. a construction notice board read via OCR).

    In a deployed system this would call the Claude API with a system
    prompt constraining it to produce a single short, actionable,
    driver-facing sentence from the raw OCR text. Left as a stub here
    to keep this project fully offline and dependency-light; wire in
    `anthropic.Anthropic().messages.create(...)` with your API key to
    activate it.
    """
    return (
        f"A road sign reading \"{ocr_text.strip()}\" was detected ahead. "
        "Please read it carefully and adjust your driving accordingly."
    )


if __name__ == "__main__":
    # quick manual check of every catalogue entry
    for sid, sign in ID_TO_SIGN.items():
        print(f"[{sign.display_name}] -> {generate_instruction(sid)}")
