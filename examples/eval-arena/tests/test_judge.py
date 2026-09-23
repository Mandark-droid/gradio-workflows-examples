import json
import pytest
from nodes import judge, arena_io


def _env(model_id, output="out"):
    return json.dumps({
        "model_id": model_id, "output": output, "latency_ms": 1,
        "tokens_out": 5, "reasoning_chars": 0, "error": None,
    })


def test_three_candidates_yield_three_pairs(monkeypatch):
    monkeypatch.setattr(arena_io, "chat",
                        lambda *a, **k: ('{"winner":"A","confidence":0.9,"rationale":"x"}', 1, 1, 0))
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    assert len(result["pairs"]) == 3


def test_six_judge_calls_are_made(monkeypatch):
    calls = []

    def _chat(model_id, prompt, *a, **k):
        calls.append(prompt)
        return '{"winner":"A","confidence":0.9,"rationale":"x"}', 1, 1, 0

    monkeypatch.setattr(arena_io, "chat", _chat)
    judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p")
    assert len(calls) == 6


def test_judge_calls_run_concurrently_not_serially(monkeypatch):
    import time

    def _slow(*args, **kwargs):
        time.sleep(0.3)
        return '{"winner":"A","confidence":0.9,"rationale":"x"}', 1, 1, 0

    monkeypatch.setattr(arena_io, "chat", _slow)
    started = time.perf_counter()
    judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p")
    elapsed = time.perf_counter() - started

    # 6 calls x 0.3s = 1.8s if serial; concurrent should be well under half that.
    assert elapsed < 0.9, f"judge calls appear to run serially: {elapsed:.2f}s"


def test_agreeing_orderings_produce_a_decisive_winner(monkeypatch):
    # Position A wins in the first ordering; position B wins in the swapped
    # ordering. Both name the same underlying model, so it is a real win.
    responses = iter([
        '{"winner":"A","confidence":0.9,"rationale":"x"}',
        '{"winner":"B","confidence":0.9,"rationale":"x"}',
    ] * 3)
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: (next(responses), 1, 1, 0))
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    first = result["pairs"][0]
    assert first["agreed"] is True
    assert first["winner"] == first["a"]


def test_disagreeing_orderings_produce_a_tie(monkeypatch):
    # Position A wins both times, which means the judge just prefers position A.
    monkeypatch.setattr(arena_io, "chat",
                        lambda *a, **k: ('{"winner":"A","confidence":0.9,"rationale":"x"}', 1, 1, 0))
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    assert all(pair["agreed"] is False for pair in result["pairs"])
    assert all(pair["winner"] is None for pair in result["pairs"])


def test_rationale_is_capped_to_the_configured_word_count(monkeypatch):
    long_rationale = " ".join(["word"] * 500)
    monkeypatch.setattr(
        arena_io, "chat",
        lambda *a, **k: (json.dumps({"winner": "A", "confidence": 0.5,
                                     "rationale": long_rationale}), 1, 1, 0),
    )
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    cap = arena_io.load_candidates()["judge"]["max_rationale_words"]
    assert len(result["pairs"][0]["rationale"].split()) <= cap


def test_unparseable_judge_reply_becomes_a_tie_not_a_crash(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("garbage", 1, 1, 0))
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    assert all(pair["winner"] is None for pair in result["pairs"])


def test_judge_failure_does_not_crash_the_node(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("judge down")

    monkeypatch.setattr(arena_io, "chat", _boom)
    result = json.loads(judge.pairwise_judge(_env("m1"), _env("m2"), _env("m3"), "p"))
    assert len(result["pairs"]) == 3
    assert all(pair["winner"] is None for pair in result["pairs"])


def test_candidate_with_an_error_envelope_loses(monkeypatch):
    calls = []
    monkeypatch.setattr(
        arena_io, "chat",
        lambda *a, **k: (calls.append(1),
                         ('{"winner":"A","confidence":1,"rationale":"x"}', 1, 1, 0))[1],
    )
    broken = json.dumps({"model_id": "m1", "output": "", "latency_ms": 0,
                         "tokens_out": 0, "reasoning_chars": 0, "error": "boom"})
    result = json.loads(judge.pairwise_judge(broken, _env("m2"), _env("m3"), "p"))
    pair = next(p for p in result["pairs"] if "m1" in (p["a"], p["b"]))
    assert pair["winner"] != "m1"
