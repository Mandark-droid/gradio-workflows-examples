import json
import pytest
from nodes import sandhi


def test_splits_gita_1_1_into_words():
    normalized = json.dumps({"iast": "dharmakṣetre kurukṣetre"})
    out = json.loads(sandhi.sandhi_split(normalized))
    assert out["splits"]
    assert "dharmakṣetre" in out["splits"][0]["words"]


def test_returns_at_most_top_k():
    normalized = json.dumps({"iast": "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ"})
    out = json.loads(sandhi.sandhi_split(normalized))
    assert len(out["splits"]) <= 3


def test_scores_descend_by_rank_and_are_labelled():
    normalized = json.dumps({"iast": "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ"})
    out = json.loads(sandhi.sandhi_split(normalized))
    scores = [s["score"] for s in out["splits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(s["score_kind"] == "rank" for s in out["splits"])


def test_words_are_iast_not_slp1():
    normalized = json.dumps({"iast": "dharmakṣetre"})
    out = json.loads(sandhi.sandhi_split(normalized))
    joined = " ".join(out["splits"][0]["words"])
    assert "z" not in joined and "D" not in joined  # SLP1 markers


def test_empty_input_returns_empty_splits():
    out = json.loads(sandhi.sandhi_split(json.dumps({"iast": ""})))
    assert out["splits"] == [] and out["truncated"] is False


def test_timeout_falls_back_to_unsplit_verse(monkeypatch):
    def _slow(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(sandhi, "_run_parser", _slow)
    out = json.loads(sandhi.sandhi_split(json.dumps({"iast": "dharmakṣetre kurukṣetre"})))
    assert out["truncated"] is True
    assert out["splits"][0]["words"] == ["dharmakṣetre", "kurukṣetre"]


def test_parser_failure_falls_back_rather_than_raising(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(sandhi, "_run_parser", _boom)
    out = json.loads(sandhi.sandhi_split(json.dumps({"iast": "rāma"})))
    assert out["truncated"] is True
