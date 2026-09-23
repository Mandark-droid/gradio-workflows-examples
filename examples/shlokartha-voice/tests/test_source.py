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
    # A dead backend must not be silently indistinguishable from silent
    # audio (both used to return ""), so a failure now reports the sentinel
    # instead of an empty string. It still never raises.
    assert source.asr("some.wav") == source.ASR_ERROR_SENTINEL


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


def test_select_source_reports_asr_error_on_backend_failure():
    out = json.loads(source.select_source(source.ASR_ERROR_SENTINEL, ""))
    assert out == {"text": "", "source": "asr_error"}


def test_select_source_reports_none_for_empty_audio():
    out = json.loads(source.select_source("", ""))
    assert out == {"text": "", "source": "none"}


def test_select_source_sentinel_never_leaks_into_text():
    out = json.loads(source.select_source(source.ASR_ERROR_SENTINEL, ""))
    assert "\x00" not in out["text"]
    assert out["text"] == ""


def test_select_source_typed_text_wins_over_an_asr_error():
    out = json.loads(source.select_source(source.ASR_ERROR_SENTINEL, "typed verse"))
    assert out == {"text": "typed verse", "source": "typed"}


def test_meaning_reports_asr_error_instead_of_blaming_the_user():
    from nodes.source import shlokartha_meaning, ASR_ERROR_MESSAGE, NO_VERSE_MESSAGE

    normalized = json.dumps({"source": "asr_error"})
    out = shlokartha_meaning("", normalized_json=normalized)
    assert out == ASR_ERROR_MESSAGE
    assert out != NO_VERSE_MESSAGE


def test_meaning_low_trust_prefix_below_threshold(monkeypatch):
    from nodes.source import shlokartha_meaning, LOW_TRUST_PREFIX

    monkeypatch.setattr(hf_io, "call_space", lambda *a, **k: "a fixed commentary")
    metre_json = json.dumps({"confidence": 0.5})
    out = shlokartha_meaning("dharmakṣetre", metre_json=metre_json)
    assert out.startswith(LOW_TRUST_PREFIX)
    assert "a fixed commentary" in out


def test_meaning_no_low_trust_prefix_above_threshold(monkeypatch):
    from nodes.source import shlokartha_meaning, LOW_TRUST_PREFIX

    monkeypatch.setattr(hf_io, "call_space", lambda *a, **k: "a fixed commentary")
    metre_json = json.dumps({"confidence": 1.0})
    out = shlokartha_meaning("dharmakṣetre", metre_json=metre_json)
    assert not out.startswith(LOW_TRUST_PREFIX)
    assert out == "a fixed commentary"
