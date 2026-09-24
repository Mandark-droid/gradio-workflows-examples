import pytest
from driver.analyse import bootstrap_ci, bradley_terry, percentiles, small_sample_note


def test_bootstrap_ci_of_constant_values_is_that_constant():
    low, high = bootstrap_ci([1.0, 1.0, 1.0], iterations=200, seed=0)
    assert low == 1.0 and high == 1.0


def test_bootstrap_ci_brackets_the_mean():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    low, high = bootstrap_ci(values, iterations=500, seed=0)
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_ci_is_deterministic_for_a_seed():
    values = [0.0, 1.0, 0.5]
    assert bootstrap_ci(values, 300, 7) == bootstrap_ci(values, 300, 7)


def test_bootstrap_ci_of_empty_input_is_zero():
    assert bootstrap_ci([], 100, 0) == (0.0, 0.0)


def test_bradley_terry_ranks_a_clear_winner_highest():
    pairs = [
        {"a": "x", "b": "y", "winner": "x", "agreed": True},
        {"a": "x", "b": "y", "winner": "x", "agreed": True},
        {"a": "x", "b": "z", "winner": "x", "agreed": True},
    ]
    ratings = bradley_terry(pairs)
    assert ratings["x"] == max(ratings.values())


def test_bradley_terry_ignores_unagreed_pairs():
    pairs = [{"a": "x", "b": "y", "winner": "x", "agreed": False}]
    ratings = bradley_terry(pairs)
    assert ratings["x"] == pytest.approx(ratings["y"])


def test_bradley_terry_of_no_pairs_is_empty():
    assert bradley_terry([]) == {}


def test_percentiles_reports_p50_and_p95():
    stats = percentiles([float(i) for i in range(1, 101)])
    assert stats["p50"] == pytest.approx(50.0, abs=1.5)
    assert stats["p95"] == pytest.approx(95.0, abs=1.5)


def test_percentiles_of_empty_input_is_zero():
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0}


def test_small_sample_note_is_explicit():
    assert "not statistically meaningful" in small_sample_note(5).lower()


def test_small_sample_note_reports_the_actual_row_count():
    assert "n=5" in small_sample_note(5)
    assert "n=42" in small_sample_note(42)
