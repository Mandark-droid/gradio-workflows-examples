import json
import pytest
from nodes import candidate, arena_io


def test_envelope_has_exactly_the_agreed_keys(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("out", 10, 4, 120))
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert set(envelope) == set(candidate.ENVELOPE_KEYS)


def test_envelope_carries_the_slot_model_id(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("out", 10, 4, 120))
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert envelope["model_id"] == arena_io.slot("a")["model_id"]


def test_envelope_reports_latency_and_tokens(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("out", 123, 45, 7))
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert envelope["latency_ms"] == 123 and envelope["tokens_out"] == 45


def test_envelope_reports_reasoning_volume(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("161", 900, 345, 1003))
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert envelope["tokens_out"] == 345
    assert envelope["reasoning_chars"] == 1003


def test_error_is_captured_not_raised(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(arena_io, "chat", _boom)
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert envelope["error"] and "provider is down" in envelope["error"]
    assert envelope["output"] == ""


def test_error_envelope_still_has_every_key(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(arena_io, "chat", _boom)
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert set(envelope) == set(candidate.ENVELOPE_KEYS)


def test_error_envelope_zeroes_the_numeric_fields(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(arena_io, "chat", _boom)
    envelope = json.loads(candidate.run_candidate("a", "hi"))
    assert envelope["latency_ms"] == 0
    assert envelope["tokens_out"] == 0
    assert envelope["reasoning_chars"] == 0


def test_all_three_slots_produce_the_same_shape(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("out", 1, 1, 0))
    shapes = [
        set(json.loads(candidate.run_candidate(s, "hi")))
        for s in ("a", "b", "c")
    ]
    assert shapes[0] == shapes[1] == shapes[2]


def test_wrappers_delegate_to_their_slots(monkeypatch):
    monkeypatch.setattr(arena_io, "chat", lambda *a, **k: ("out", 1, 1, 0))
    for fn, slot_name in (
        (candidate.candidate_a, "a"),
        (candidate.candidate_b, "b"),
        (candidate.candidate_c, "c"),
    ):
        assert json.loads(fn("hi"))["model_id"] == arena_io.slot(slot_name)["model_id"]


def test_unknown_slot_yields_an_error_envelope_not_a_crash():
    envelope = json.loads(candidate.run_candidate("z", "hi"))
    assert envelope["error"]


def test_caller_token_is_passed_through_to_chat(monkeypatch):
    seen = {}

    def _capture(model_id, prompt, max_new_tokens, greedy, hf_token=""):
        seen["token"] = hf_token
        return ("ok", 1, 1, 0)

    monkeypatch.setattr(arena_io, "chat", _capture)
    candidate.candidate_a("hi", "hf_visitor_token")
    assert seen["token"] == "hf_visitor_token"


def test_error_envelope_never_contains_the_token(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(arena_io, "chat", _boom)
    envelope = candidate.run_candidate("a", "hi", "hf_secret_abc")
    assert "hf_secret_abc" not in envelope
