import json
import pytest
from nodes.chandas import syllabify, weights, ganas, chandas_detect, GANA_PATTERNS


def test_syllabify_counts_gita_1_1_first_pada():
    # dharmakṣetre kurukṣetre = 8 syllables (dhar-ma-kṣe-tre-ku-ru-kṣe-tre)
    assert len(syllabify("dharmakṣetre kurukṣetre")) == 8


def test_syllabify_handles_aspirated_digraphs_as_one_consonant():
    assert syllabify("bha") == ["bha"]


def test_syllabify_treats_diphthong_as_one_vowel():
    assert syllabify("kau") == ["kau"]


def test_long_vowel_is_guru():
    assert weights("rā") == "G"


def test_short_vowel_alone_is_laghu():
    assert weights("ra") == "L"


def test_e_and_o_are_always_guru_in_sanskrit():
    assert weights("re") == "G"
    assert weights("ro") == "G"


def test_anusvara_makes_preceding_syllable_guru():
    assert weights("raṃ") == "G"


def test_visarga_makes_preceding_syllable_guru():
    assert weights("raḥ") == "G"


def test_conjunct_makes_preceding_syllable_guru():
    # "dhar-ma": 'a' is short but followed by 'rm' -> conjunct -> guru
    assert weights("dharma")[0] == "G"


def test_all_eight_ganas_are_three_syllables():
    for name, pattern in GANA_PATTERNS.items():
        assert len(pattern) == 3, name


def test_ganas_splits_into_triples_with_remainder():
    assert ganas("GGLGGLLGLGG") == ["ta", "ta", "ja", "ga", "ga"]


def test_detects_anustubh_on_gita_1_1():
    normalized = json.dumps({
        "iast": "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ",
        "padas": ["dharmakṣetre kurukṣetre", "samavetā yuyutsavaḥ"],
    })
    out = json.loads(chandas_detect(normalized))
    assert out["metre"] == "Anuṣṭubh"
    assert out["syllables_per_pada"] == [8, 8]
    assert out["confidence"] == 1.0


def test_unknown_metre_reports_low_confidence_not_an_exception():
    normalized = json.dumps({"iast": "ka", "padas": ["ka"]})
    out = json.loads(chandas_detect(normalized))
    assert out["metre"] == "Unknown"
    assert out["confidence"] == 0.0


def test_empty_input_is_safe():
    out = json.loads(chandas_detect(json.dumps({"iast": "", "padas": []})))
    assert out["metre"] == "Unknown" and out["confidence"] == 0.0


def test_normalize_then_chandas_detects_a_full_verse():
    """The seam that matters: split_padas feeding chandas_detect. Testing them
    separately is what let a complete verse report Unknown in production."""
    import json as _json
    from nodes.normalize import normalize

    full = (
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। "
        "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"
    )
    normalized = normalize(_json.dumps({"text": full, "source": "typed"}))
    result = _json.loads(chandas_detect(normalized))
    assert result["metre"] == "Anuṣṭubh", result
    assert result["syllables_per_pada"] == [8, 8, 8, 8]
    assert result["confidence"] >= 0.75
