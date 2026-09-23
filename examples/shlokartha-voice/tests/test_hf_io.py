import json
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
