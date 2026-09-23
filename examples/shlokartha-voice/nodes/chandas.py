"""Sanskrit metre (chandas) detection from IAST.

A syllable is guru (heavy) if its vowel is long, or it carries an anusvāra or
visarga, or it is followed by a conjunct. Otherwise laghu (light). Per-pāda
weight strings are matched against a metre table.

Metre detection doubles as an ASR sanity check: a known Anuṣṭubh verse that
fails scansion almost always means transcription drift.
"""
from __future__ import annotations

import json

# Longest-first so digraphs win over their first character.
VOWELS = ["ai", "au", "ā", "ī", "ū", "ṝ", "ḹ", "e", "o", "a", "i", "u", "ṛ", "ḷ"]
LONG_VOWELS = {"ā", "ī", "ū", "ṝ", "ḹ", "e", "ai", "o", "au"}
CONSONANTS = [
    "kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh",
    "k", "g", "ṅ", "c", "j", "ñ", "ṭ", "ḍ", "ṇ", "t", "d", "n",
    "p", "b", "m", "y", "r", "l", "v", "ś", "ṣ", "s", "h",
]
MODIFIERS = {"ṃ", "ṁ", "ḥ"}

GANA_PATTERNS = {
    "ma": "GGG", "na": "LLL", "bha": "GLL", "ya": "LGG",
    "ja": "LGL", "ra": "GLG", "sa": "LLG", "ta": "GGL",
}
_PATTERN_TO_GANA = {v: k for k, v in GANA_PATTERNS.items()}

# Fixed-pattern metres, expressed as gana sequences plus trailing singles.
GANA_METRES: dict[str, list[str]] = {
    "Indravajrā": ["ta", "ta", "ja", "ga", "ga"],
    "Upendravajrā": ["ja", "ta", "ja", "ga", "ga"],
    "Vasantatilakā": ["ta", "bha", "ja", "ja", "ga", "ga"],
    "Mālinī": ["na", "na", "ma", "ya", "ya"],
    "Śikhariṇī": ["ya", "ma", "na", "sa", "bha", "la", "ga"],
    "Śārdūlavikrīḍita": ["ma", "sa", "ja", "sa", "ta", "ta", "ga"],
}


def _expand(sequence: list[str]) -> str:
    out = []
    for token in sequence:
        if token == "ga":
            out.append("G")
        elif token == "la":
            out.append("L")
        else:
            out.append(GANA_PATTERNS[token])
    return "".join(out)


METRE_WEIGHTS = {name: _expand(seq) for name, seq in GANA_METRES.items()}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    lowered = text.lower()
    while i < len(lowered):
        ch = lowered[i]
        if ch.isspace() or ch in "'’.,-":
            i += 1
            continue
        if ch in MODIFIERS:
            tokens.append(("M", ch))
            i += 1
            continue
        for cons in CONSONANTS:
            if lowered.startswith(cons, i):
                tokens.append(("C", cons))
                i += len(cons)
                break
        else:
            for vowel in VOWELS:
                if lowered.startswith(vowel, i):
                    tokens.append(("V", vowel))
                    i += len(vowel)
                    break
            else:
                i += 1
    return tokens


def syllabify(iast: str) -> list[str]:
    """Group tokens into syllables: onset consonants + vowel + modifiers."""
    syllables: list[str] = []
    current = ""
    for kind, value in _tokenize(iast):
        if kind == "C":
            if current and _has_vowel(current):
                syllables.append(current)
                current = ""
            current += value
        elif kind == "V":
            current += value
        else:
            current += value
    if current:
        syllables.append(current)
    return [s for s in syllables if _has_vowel(s)]


def _has_vowel(chunk: str) -> bool:
    return any(v in chunk for v in VOWELS)


def _vowel_of(syllable: str) -> str:
    i = 0
    found = ""
    while i < len(syllable):
        for vowel in VOWELS:
            if syllable.startswith(vowel, i):
                found = vowel
                i += len(vowel)
                break
        else:
            i += 1
    return found


def _onset_length(syllable: str) -> int:
    """How many consonants precede this syllable's vowel."""
    count, i = 0, 0
    while i < len(syllable):
        for cons in CONSONANTS:
            if syllable.startswith(cons, i):
                count += 1
                i += len(cons)
                break
        else:
            break
    return count


def weights(iast: str) -> str:
    syllables = syllabify(iast)
    out: list[str] = []
    for index, syllable in enumerate(syllables):
        vowel = _vowel_of(syllable)
        heavy = vowel in LONG_VOWELS
        if not heavy and any(m in syllable for m in MODIFIERS):
            heavy = True
        if not heavy and index + 1 < len(syllables):
            if _onset_length(syllables[index + 1]) >= 2:
                heavy = True
        out.append("G" if heavy else "L")
    return "".join(out)


def ganas(weight_string: str) -> list[str]:
    out: list[str] = []
    for i in range(0, len(weight_string) - len(weight_string) % 3, 3):
        out.append(_PATTERN_TO_GANA[weight_string[i : i + 3]])
    for remainder in weight_string[len(weight_string) - len(weight_string) % 3 :]:
        out.append("ga" if remainder == "G" else "la")
    return out


def _matches(actual: str, expected: str) -> bool:
    """Final syllable of a pāda is metrically anceps — either weight matches."""
    if len(actual) != len(expected):
        return False
    return all(a == e for a, e in zip(actual[:-1], expected[:-1]))


def _is_anustubh_pada(weight_string: str, index: int) -> bool:
    if len(weight_string) != 8:
        return False
    if weight_string[4] != "L" or weight_string[5] != "G":
        return False
    expected_seventh = "G" if index % 2 == 0 else "L"
    return weight_string[6] == expected_seventh


def chandas_detect(normalized_json: str) -> str:
    try:
        payload = json.loads(normalized_json)
        padas = payload.get("padas") or []
        if not padas and payload.get("iast"):
            padas = [payload["iast"]]
    except (json.JSONDecodeError, AttributeError, TypeError):
        padas = []

    per_pada = [weights(p) for p in padas]
    per_pada = [w for w in per_pada if w]

    if not per_pada:
        return json.dumps(
            {
                "metre": "Unknown", "ganas_per_pada": [], "syllables_per_pada": [],
                "weights_per_pada": [], "confidence": 0.0,
            },
            ensure_ascii=False,
        )

    best_name, best_fraction = "Unknown", 0.0

    anustubh_hits = sum(
        1 for i, w in enumerate(per_pada) if _is_anustubh_pada(w, i)
    )
    if anustubh_hits:
        best_name = "Anuṣṭubh"
        best_fraction = anustubh_hits / len(per_pada)

    for name, expected in METRE_WEIGHTS.items():
        hits = sum(1 for w in per_pada if _matches(w, expected))
        fraction = hits / len(per_pada)
        if fraction > best_fraction:
            best_name, best_fraction = name, fraction

    # Upajāti: pādas that are individually Indravajrā or Upendravajrā.
    if best_name in ("Indravajrā", "Upendravajrā") and best_fraction < 1.0:
        mixed = sum(
            1 for w in per_pada
            if _matches(w, METRE_WEIGHTS["Indravajrā"])
            or _matches(w, METRE_WEIGHTS["Upendravajrā"])
        )
        if mixed / len(per_pada) > best_fraction:
            best_name = "Upajāti"
            best_fraction = mixed / len(per_pada)

    return json.dumps(
        {
            "metre": best_name if best_fraction > 0 else "Unknown",
            "ganas_per_pada": [ganas(w) for w in per_pada],
            "syllables_per_pada": [len(w) for w in per_pada],
            "weights_per_pada": per_pada,
            "confidence": round(best_fraction, 3),
        },
        ensure_ascii=False,
    )
