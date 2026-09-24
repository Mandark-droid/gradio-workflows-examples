import json
import os
from pathlib import Path

import pytest
from nodes import hf_io


@pytest.fixture(autouse=True)
def _isolated_fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_io, "FIXTURE_DIR", tmp_path)
    monkeypatch.setenv("WORKFLOW_IO_MODE", "replay")
    hf_io.load_config.cache_clear()


def test_replay_mode_is_the_default(monkeypatch):
    monkeypatch.delenv("WORKFLOW_IO_MODE", raising=False)
    assert hf_io.mode() == "replay"


def test_replay_without_fixture_raises_rather_than_calling_network():
    with pytest.raises(hf_io.FixtureMissing):
        hf_io.call_space("some/space", "/endpoint", "arg")


def test_replay_returns_a_written_fixture(tmp_path):
    key = hf_io.fixture_key("space", "some/space", "/endpoint", ("arg",))
    hf_io.write_fixture(key, {"value": "recorded"})
    assert hf_io.call_space("some/space", "/endpoint", "arg") == "recorded"


def test_fixture_key_is_stable_across_calls():
    a = hf_io.fixture_key("space", "s", "/e", ("x", 1))
    b = hf_io.fixture_key("space", "s", "/e", ("x", 1))
    assert a == b


def test_fixture_key_varies_with_arguments():
    a = hf_io.fixture_key("space", "s", "/e", ("x",))
    b = hf_io.fixture_key("space", "s", "/e", ("y",))
    assert a != b


def test_result_index_selects_from_a_tuple_result(tmp_path):
    key = hf_io.fixture_key("space", "s", "/e", ("a",))
    hf_io.write_fixture(key, {"value": ["first", "second"]})
    assert hf_io.call_space("s", "/e", "a", result_index=1) == "second"


def test_binary_fixture_round_trips(tmp_path):
    key = hf_io.fixture_key("model", "m", "text_to_image", ("prompt", 8, 8))
    hf_io.write_fixture(key, {"value": None}, binary=b"\x89PNG-bytes")
    assert hf_io.read_fixture(key)[1] == b"\x89PNG-bytes"


def test_config_loads_and_has_required_keys():
    cfg = hf_io.load_config()
    assert cfg["shlokartha"]["space_id"]
    assert cfg["tts"]["space_id"]
    assert cfg["flux"]["model_id"]


class _FakeClient:
    """Stands in for gradio_client.Client: predict() returns whatever the
    test wants, simulating what a live Space call would hand back."""

    def __init__(self, result):
        self._result = result

    def predict(self, *args, api_name=None):
        return self._result


def _patch_client(monkeypatch, result):
    monkeypatch.setattr("gradio_client.Client", lambda *a, **k: _FakeClient(result))


def test_recording_a_file_result_does_not_leak_the_local_path(
    tmp_path, monkeypatch
):
    # Simulate what gradio_client does for a Space that returns a file: it
    # downloads to a local temp path on the recording machine and hands that
    # path back as a plain string.
    downloaded = tmp_path / "a-local-download-dir" / "clip.mp3"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"FAKE-MP3-BYTES")
    _patch_client(monkeypatch, str(downloaded))
    monkeypatch.setenv("WORKFLOW_IO_MODE", "record")

    hf_io.call_space("some/space", "/tts", "hello")

    key = hf_io.fixture_key("space", "some/space", "/tts", ("hello",))
    fixture_text = (hf_io.FIXTURE_DIR / f"{key}.json").read_text(encoding="utf-8")
    assert str(downloaded) not in fixture_text
    assert os.sep.join(["a-local-download-dir", "clip.mp3"]) not in fixture_text
    assert (hf_io.FIXTURE_DIR / f"{key}.bin").exists()

    record = json.loads(fixture_text)
    assert record["value"] is None
    assert record["file_suffix"] == ".mp3"


def test_replaying_a_file_result_returns_a_readable_path_with_matching_bytes(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "clip.mp3"
    downloaded.write_bytes(b"FAKE-MP3-BYTES")
    _patch_client(monkeypatch, str(downloaded))
    monkeypatch.setenv("WORKFLOW_IO_MODE", "record")
    hf_io.call_space("some/space", "/tts", "hello")

    monkeypatch.setenv("WORKFLOW_IO_MODE", "replay")
    replayed_path = hf_io.call_space("some/space", "/tts", "hello")

    assert replayed_path != str(downloaded)
    assert replayed_path.endswith(".mp3")
    assert Path(replayed_path).read_bytes() == b"FAKE-MP3-BYTES"


def test_recording_and_replaying_a_file_within_a_list_result(tmp_path, monkeypatch):
    downloaded = tmp_path / "audio.wav"
    downloaded.write_bytes(b"FAKE-WAV-BYTES")
    _patch_client(monkeypatch, [str(downloaded), "some text"])
    monkeypatch.setenv("WORKFLOW_IO_MODE", "record")
    hf_io.call_space("some/space", "/multi", "hello", result_index=0)

    key = hf_io.fixture_key("space", "some/space", "/multi", ("hello",))
    fixture_text = (hf_io.FIXTURE_DIR / f"{key}.json").read_text(encoding="utf-8")
    assert str(downloaded) not in fixture_text
    assert (hf_io.FIXTURE_DIR / f"{key}.bin").exists()

    monkeypatch.setenv("WORKFLOW_IO_MODE", "replay")
    replayed_path = hf_io.call_space("some/space", "/multi", "hello", result_index=0)
    assert replayed_path != str(downloaded)
    assert Path(replayed_path).read_bytes() == b"FAKE-WAV-BYTES"


def test_plain_text_space_result_still_round_trips_unchanged(monkeypatch):
    _patch_client(monkeypatch, "just some ordinary text output")
    monkeypatch.setenv("WORKFLOW_IO_MODE", "record")
    recorded = hf_io.call_space("some/space", "/text-endpoint", "hello")
    assert recorded == "just some ordinary text output"

    key = hf_io.fixture_key("space", "some/space", "/text-endpoint", ("hello",))
    assert not (hf_io.FIXTURE_DIR / f"{key}.bin").exists()
    record = json.loads((hf_io.FIXTURE_DIR / f"{key}.json").read_text(encoding="utf-8"))
    assert record == {"value": "just some ordinary text output"}

    monkeypatch.setenv("WORKFLOW_IO_MODE", "replay")
    replayed = hf_io.call_space("some/space", "/text-endpoint", "hello")
    assert replayed == "just some ordinary text output"
