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
    # 4 words, 6 syllables: "a", "a", "a" (1 each) then "bhārata" (3). The
    # word-count midpoint (2 words each side) would give 2 vs 4 syllables;
    # balancing must instead cut after the third word for an exact 3 vs 3.
    from nodes.normalize import _syllable_count

    left, right = split_padas("a a a bhārata")
    assert abs(_syllable_count(left) - _syllable_count(right)) <= 1


def test_double_danda_is_not_treated_as_two_empty_padas():
    assert "" not in split_padas("alpha ॥ beta")


def test_verse_text_extracts_devanagari():
    from nodes.normalize import verse_text

    payload = json.dumps({"devanagari": "धर्मक्षेत्रे", "iast": "dharmakṣetre"})
    assert verse_text(payload) == "धर्मक्षेत्रे"


def test_verse_text_falls_back_to_iast_then_raw():
    from nodes.normalize import verse_text

    assert verse_text(json.dumps({"iast": "rāma"})) == "rāma"
    assert verse_text("not json") == "not json"


FULL_VERSE_IAST = (
    "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ । "
    "māmakāḥ pāṇḍavāścaiva kimakurvata sañjaya ॥"
)


def test_full_verse_with_dandas_yields_four_padas():
    padas = split_padas(FULL_VERSE_IAST)
    from nodes.normalize import _syllable_count

    assert len(padas) == 4, padas
    assert all(_syllable_count(p) == 8 for p in padas), [
        _syllable_count(p) for p in padas
    ]


def test_full_verse_without_dandas_yields_four_padas():
    from nodes.normalize import _syllable_count

    padas = split_padas(FULL_VERSE_IAST.replace("।", " ").replace("॥", " "))
    assert len(padas) == 4, padas
    assert all(_syllable_count(p) == 8 for p in padas)


def test_half_verse_still_yields_two_padas():
    padas = split_padas("dharmakṣetre kurukṣetre samavetā yuyutsavaḥ")
    assert len(padas) == 2
