"""Detect the input script, convert to both Devanagari and IAST, split pādas.

Pure Python and deterministic — free to run and easy to test.
"""
from __future__ import annotations

import json
import re

from indic_transliteration import detect as _detect
from indic_transliteration.sanscript import DEVANAGARI, IAST, transliterate

DANDA_RE = re.compile(r"[।॥|]+")

_VOWEL_RE = re.compile(r"ai|au|[āīūṝḹeoaiuṛḷ]")

_SCHEME_BY_NAME = {
    "devanagari": DEVANAGARI,
    "iast": IAST,
}


def _detect_scheme(text: str) -> str:
    try:
        return str(_detect.detect(text)).lower()
    except Exception:
        return "iast"


def _syllable_count(text: str) -> int:
    """In Sanskrit a syllable is one vowel nucleus, so counting vowels counts
    syllables."""
    return len(_VOWEL_RE.findall(text.lower()))


def split_padas(text: str) -> list[str]:
    """Split on daṇḍas; otherwise fall back to the most syllable-balanced
    word boundary.

    ASR output carries no daṇḍas, so this fallback is the live path. Splitting
    at the word midpoint misreports metre whenever the words are unevenly
    sized, so the boundary that best balances syllable counts is chosen
    instead.
    """
    parts = [p.strip() for p in DANDA_RE.split(text) if p.strip()]
    if len(parts) > 1:
        return parts
    single = parts[0] if parts else ""
    if not single:
        return []
    words = single.split()
    if len(words) < 2:
        return [single]

    best_index, best_delta = 1, None
    for index in range(1, len(words)):
        left = _syllable_count(" ".join(words[:index]))
        right = _syllable_count(" ".join(words[index:]))
        delta = abs(left - right)
        if best_delta is None or delta < best_delta:
            best_index, best_delta = index, delta
    return [" ".join(words[:best_index]), " ".join(words[best_index:])]


def normalize(source_json: str) -> str:
    try:
        payload = json.loads(source_json)
        if not isinstance(payload, dict):
            raise ValueError
        text = str(payload.get("text", ""))
        source = str(payload.get("source", "unknown"))
    except (json.JSONDecodeError, ValueError):
        text, source = str(source_json), "unknown"

    text = text.strip()
    if not text:
        return json.dumps(
            {"devanagari": "", "iast": "", "padas": [], "scheme": "", "source": source},
            ensure_ascii=False,
        )

    scheme_name = _detect_scheme(text)
    scheme = _SCHEME_BY_NAME.get(scheme_name, IAST)

    devanagari = text if scheme == DEVANAGARI else transliterate(text, scheme, DEVANAGARI)
    iast = text if scheme == IAST else transliterate(text, scheme, IAST)

    return json.dumps(
        {
            "devanagari": devanagari,
            "iast": iast,
            "padas": split_padas(iast),
            "scheme": scheme_name,
            "source": source,
        },
        ensure_ascii=False,
    )


def verse_text(normalized_json: str) -> str:
    """Extract the bare Devanagari verse from normalize's JSON payload.

    The Ślōkārtha Space expects a verse string. Handing it the whole
    normalized object would ask the model to interpret JSON.
    """
    try:
        payload = json.loads(normalized_json)
        if isinstance(payload, dict):
            return str(payload.get("devanagari") or payload.get("iast") or "")
    except (json.JSONDecodeError, TypeError):
        pass
    return str(normalized_json or "")
