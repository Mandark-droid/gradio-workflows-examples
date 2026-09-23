import pytest
from eval.metrics import character_error_rate, word_f1, load_testset


def test_cer_is_zero_for_identical_strings():
    assert character_error_rate("rāmaḥ", "rāmaḥ") == 0.0


def test_cer_counts_a_single_substitution():
    assert character_error_rate("rama", "rana") == pytest.approx(0.25)


def test_cer_counts_deletion():
    assert character_error_rate("rama", "ram") == pytest.approx(0.25)


def test_cer_of_empty_reference_is_zero_when_hypothesis_also_empty():
    assert character_error_rate("", "") == 0.0


def test_cer_of_empty_reference_with_output_is_one():
    assert character_error_rate("", "x") == 1.0


def test_word_f1_is_one_for_exact_match():
    assert word_f1(["a", "b"], ["a", "b"]) == 1.0


def test_word_f1_is_zero_for_disjoint():
    assert word_f1(["a"], ["b"]) == 0.0


def test_word_f1_handles_partial_overlap():
    assert word_f1(["a", "b"], ["a", "c"]) == pytest.approx(0.5)


def test_word_f1_of_two_empties_is_one():
    assert word_f1([], []) == 1.0


def test_testset_has_five_verses_with_required_fields():
    rows = load_testset()
    assert len(rows) == 6
    for row in rows:
        assert row["id"] and row["devanagari"] and row["iast"]
        assert row["metre"] and isinstance(row["gold_padaccheda"], list)
        assert isinstance(row["padas"], list) and len(row["padas"]) in (2, 4)
