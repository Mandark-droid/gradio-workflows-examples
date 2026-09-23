"""Build a FLUX prompt from the meaning.

Deterministic and LLM-free, so it costs nothing. Deity names and ritual
specifics are stripped because image models render them poorly and often
disrespectfully; scenes stay abstract and landscape-led.
"""
from __future__ import annotations

import re

MAX_PROMPT_CHARS = 400

_STRIP_TERMS = [
    "krishna", "kṛṣṇa", "arjuna", "vishnu", "viṣṇu", "shiva", "śiva",
    "brahma", "brahmā", "rama", "rāma", "hanuman", "hanumān", "ganesha",
    "gaṇeśa", "devi", "devī", "lord", "god", "goddess", "deity", "idol",
    "temple", "worship", "ritual", "sacrifice", "yajña", "yajna",
]

_STYLE = (
    "serene Indian landscape, wide vista, soft dawn light, "
    "muted ochre and indigo palette, no people, no text, painterly"
)


def illustration_prompt(meaning: str) -> str:
    text = str(meaning or "").strip()

    for term in _STRIP_TERMS:
        text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")

    if not text:
        return _STYLE

    budget = MAX_PROMPT_CHARS - len(_STYLE) - 2
    if len(text) > budget:
        text = text[:budget].rsplit(" ", 1)[0]

    return f"{text}, {_STYLE}"[:MAX_PROMPT_CHARS]
