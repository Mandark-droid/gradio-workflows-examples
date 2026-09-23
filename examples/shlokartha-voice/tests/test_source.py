import json
import pytest
from nodes import source, hf_io


def test_asr_returns_empty_string_for_none():
    assert source.asr(None) == ""


def test_asr_returns_empty_string_for_blank():
    assert source.asr("   ") == ""


def test_asr_never_raises_when_the_space_fails(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("space down")

    monkeypatch.setattr(hf_io, "call_space", _boom)
    assert source.asr("some.wav") == ""


def test_asr_returns_the_transcribed_verse(monkeypatch):
    monkeypatch.setattr(hf_io, "call_space", lambda *a, **k: "rāmaḥ")
    assert source.asr("some.wav") == "rāmaḥ"


def test_select_source_prefers_typed_text():
    out = json.loads(source.select_source("from asr", "typed verse"))
    assert out == {"text": "typed verse", "source": "typed"}


def test_select_source_falls_back_to_asr():
    out = json.loads(source.select_source("from asr", ""))
    assert out == {"text": "from asr", "source": "asr"}


def test_select_source_with_both_empty_is_safe():
    out = json.loads(source.select_source("", "   "))
    assert out == {"text": "", "source": "none"}


def test_select_source_strips_whitespace():
    out = json.loads(source.select_source("", "  verse  "))
    assert out["text"] == "verse"
