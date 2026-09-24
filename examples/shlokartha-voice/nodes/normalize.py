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
    """Detect the input script.

    Falls back to IAST because it is the safest default: transliterating
    IAST-as-IAST is a no-op, so a misdetection degrades to leaving the text
    alone rather than mangling it.
    """
    try:
        return str(_detect.detect(text)).lower()
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        return "iast"


def _syllable_count(text: str) -> int:
    """In Sanskrit a syllable is one vowel nucleus, so counting vowels counts
    syllables."""
    return len(_VOWEL_RE.findall(text.lower()))


# Syllable counts of the pādas this pipeline can recognise. A segment whose
# count is not one of these is probably several pādas run together.
PADA_LENGTHS = frozenset({8, 11, 12, 14, 15, 17, 19})
_MAX_SPLIT_DEPTH = 2


def _balanced_halves(text: str) -> tuple[str, str] | None:
    """Split at the word boundary that most evenly balances syllables."""
    words = text.split()
    if len(words) < 2:
        return None
    best_index, best_delta = None, None
    for index in range(1, len(words)):
        left = _syllable_count(" ".join(words[:index]))
        right = _syllable_count(" ".join(words[index:]))
        delta = abs(left - right)
        if best_delta is None or delta < best_delta:
            best_index, best_delta = index, delta
    if best_index is None:
        return None
    return " ".join(words[:best_index]), " ".join(words[best_index:])


def _split_to_padas(segment: str, depth: int = 0) -> list[str]:
    """Halve a segment until each part is a plausible pāda.

    A written verse puts a daṇḍa between half-lines, not between pādas, so one
    daṇḍa segment of an Anuṣṭubh is 16 syllables — two pādas. Without this the
    metre matcher never sees an 8-syllable pāda and reports Unknown for every
    complete verse.
    """
    count = _syllable_count(segment)
    if count in PADA_LENGTHS or depth >= _MAX_SPLIT_DEPTH:
        return [segment]
    if count >= 4 and count % 2 == 0:
        halves = _balanced_halves(segment)
        if halves and all(_syllable_count(h) for h in halves):
            return (
                _split_to_padas(halves[0], depth + 1)
                + _split_to_padas(halves[1], depth + 1)
            )
    return [segment]


def split_padas(text: str) -> list[str]:
    """Split a verse into pādas.

    Daṇḍas mark half-lines, so each segment is split further until its parts
    are plausible pāda lengths. ASR output carries no daṇḍas at all, which is
    why the fallback must do the same work.
    """
    segments = [p.strip() for p in DANDA_RE.split(text) if p.strip()]
    if not segments:
        return []
    padas: list[str] = []
    for segment in segments:
        padas.extend(_split_to_padas(segment))
    return [p for p in padas if p.strip()]


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
