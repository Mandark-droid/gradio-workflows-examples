import json
import pytest
from nodes import source, hf_io


@pytest.fixture(autouse=True)
def _some_wav_exists(tmp_path, monkeypatch):
    """asr() now wraps its path with gradio_client.handle_file, which — same
    as it would for a real recorded audio file — requires the path to exist
    on disk or be an http(s) URL. The tests below call asr("some.wav")
    literally; give that name a real file in a throwaway cwd so handle_file's
    existence check passes exactly as it would for genuine mic input.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "some.wav").write_bytes(b"RIFF")


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


def test_meaning_returns_a_readable_message_for_an_empty_verse():
    from nodes.source import shlokartha_meaning, NO_VERSE_MESSAGE

    assert shlokartha_meaning("") == NO_VERSE_MESSAGE
    assert shlokartha_meaning("   ") == NO_VERSE_MESSAGE
    assert shlokartha_meaning(None) == NO_VERSE_MESSAGE


def test_meaning_never_raises_when_the_space_fails(monkeypatch):
    from nodes.source import shlokartha_meaning

    def _boom(*a, **k):
        raise RuntimeError("space down")

    monkeypatch.setattr(hf_io, "call_space", _boom)
    out = shlokartha_meaning("धर्मक्षेत्रे")
    assert out.startswith("Meaning unavailable:")


def test_asr_sends_a_file_handle_not_a_bare_path(monkeypatch):
    seen = {}

    def _capture(space_id, api_name, *args, **kwargs):
        seen["arg"] = args[0] if args else None
        return "rāmaḥ"

    monkeypatch.setattr(hf_io, "call_space", _capture)
    source.asr("some.wav")
    # handle_file returns a dict describing the file, never a bare string.
    assert isinstance(seen["arg"], dict), f"sent {type(seen['arg']).__name__}"
