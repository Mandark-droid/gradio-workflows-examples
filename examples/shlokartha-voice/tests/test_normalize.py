import json
from nodes.normalize import normalize, split_padas


GITA_1_1_DEVA = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"


def test_devanagari_input_is_detected_and_kept():
    out = json.loads(normalize(json.dumps({"text": GITA_1_1_DEVA, "source": "typed"})))
    assert out["scheme"] == "devanagari"
    assert out["devanagari"].startswith("धर्मक्षेत्रे")


def test_devanagari_is_transliterated_to_iast():
    out = json.loads(normalize(json.dumps({"text": GITA_1_1_DEVA, "source": "typed"})))
    assert out["iast"] == "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ"


def test_iast_input_is_transliterated_to_devanagari():
    out = json.loads(normalize(json.dumps({"text": "dharmakṣetre", "source": "typed"})))
    assert out["devanagari"] == "धर्मक्षेत्रे"


def test_bare_string_input_is_accepted():
    out = json.loads(normalize(GITA_1_1_DEVA))
    assert out["iast"].startswith("dharmakṣetre")
    assert out["source"] == "unknown"


def test_source_is_carried_through():
    out = json.loads(normalize(json.dumps({"text": "rāma", "source": "asr"})))
    assert out["source"] == "asr"


def test_empty_input_yields_empty_fields_not_an_exception():
    out = json.loads(normalize(json.dumps({"text": "", "source": "asr"})))
    assert out["iast"] == "" and out["padas"] == []


def test_danda_splits_padas():
    assert split_padas("alpha । beta ॥") == ["alpha", "beta"]


def test_padas_fall_back_to_balanced_syllable_halves_without_dandas():
    padas = split_padas("dharmakṣetre kurukṣetre samavetā yuyutsavaḥ")
    assert padas == ["dharmakṣetre kurukṣetre", "samavetā yuyutsavaḥ"]


def test_fallback_balances_syllables_not_word_count():
    # "a" is one syllable, "bhārata" is three. A word-midpoint split would
    # give 1 vs 6 syllables; balancing must not.
    from nodes.normalize import _syllable_count

    left, right = split_padas("a bhārata bhārata")
    assert abs(_syllable_count(left) - _syllable_count(right)) <= 1


def test_double_danda_is_not_treated_as_two_empty_padas():
    assert "" not in split_padas("alpha ॥ beta")
