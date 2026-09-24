import pytest
from nodes import arena_io


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(arena_io, "FIXTURE_DIR", tmp_path)
    monkeypatch.setenv("ARENA_IO_MODE", "replay")
    arena_io.load_candidates.cache_clear()
    # config_hash is also lru_cache'd and derives from load_candidates(), so
    # it must be cleared alongside it — otherwise a test that reads
    # config_hash() before any mutation can get a value cached by an earlier
    # test that already mutated the config, making a before/after comparison
    # vacuous depending on test order.
    arena_io.config_hash.cache_clear()


def test_replay_is_the_default(monkeypatch):
    monkeypatch.delenv("ARENA_IO_MODE", raising=False)
    assert arena_io.mode() == "replay"


def test_three_candidate_slots_are_configured():
    cfg = arena_io.load_candidates()
    assert sorted(c["slot"] for c in cfg["candidates"]) == ["a", "b", "c"]


def test_candidates_are_three_distinct_models():
    ids = [c["model_id"] for c in arena_io.load_candidates()["candidates"]]
    assert len(set(ids)) == 3


def test_judge_is_out_of_family_from_every_candidate():
    cfg = arena_io.load_candidates()
    judge_org = cfg["judge"]["model_id"].split("/")[0].lower()
    for candidate in cfg["candidates"]:
        assert candidate["model_id"].split("/")[0].lower() != judge_org


def test_slot_lookup_returns_the_right_candidate():
    assert arena_io.slot("b")["slot"] == "b"


def test_unknown_slot_raises():
    with pytest.raises(KeyError):
        arena_io.slot("z")


def test_chat_replays_a_fixture():
    key = arena_io.fixture_key("chat", "some/model", ("hello", 8, True))
    arena_io.write_fixture(
        key, {"text": "hi", "latency_ms": 12, "tokens_out": 3, "reasoning_chars": 40}
    )
    assert arena_io.chat("some/model", "hello", 8, True) == ("hi", 12, 3, 40)


def test_chat_fixture_without_reasoning_field_still_loads():
    key = arena_io.fixture_key("chat", "some/model", ("legacy", 8, True))
    arena_io.write_fixture(key, {"text": "hi", "latency_ms": 1, "tokens_out": 1})
    assert arena_io.chat("some/model", "legacy", 8, True) == ("hi", 1, 1, 0)


def test_chat_without_fixture_raises_rather_than_calling_network():
    with pytest.raises(arena_io.FixtureMissing):
        arena_io.chat("some/model", "unseen", 8, True)


def test_config_hash_is_stable():
    assert arena_io.config_hash() == arena_io.config_hash()


def test_config_hash_changes_when_a_candidate_changes(monkeypatch):
    before = arena_io.config_hash()
    cfg = arena_io.load_candidates()
    monkeypatch.setitem(cfg["candidates"][0], "model_id", "other/model")
    arena_io.config_hash.cache_clear()
    assert arena_io.config_hash() != before


def test_caller_token_is_excluded_from_the_fixture_key():
    a = arena_io.fixture_key("chat", "m", ("hello", 8, True))
    arena_io.write_fixture(a, {"text": "hi", "latency_ms": 1, "tokens_out": 1,
                               "reasoning_chars": 0})
    # Two different callers, two different tokens, one shared recording.
    assert arena_io.chat("m", "hello", 8, True, "hf_tokenAAA") == ("hi", 1, 1, 0)
    assert arena_io.chat("m", "hello", 8, True, "hf_tokenBBB") == ("hi", 1, 1, 0)


def test_caller_token_is_never_written_to_a_fixture(tmp_path):
    key = arena_io.fixture_key("chat", "m", ("p", 8, True))
    arena_io.write_fixture(key, {"text": "t", "latency_ms": 1, "tokens_out": 1,
                                 "reasoning_chars": 0})
    arena_io.chat("m", "p", 8, True, "hf_secretvalue123")
    blob = (arena_io.FIXTURE_DIR / f"{key}.json").read_text(encoding="utf-8")
    assert "hf_secretvalue123" not in blob


def test_no_candidate_or_judge_entry_carries_a_revision_key():
    # Inference Providers expose no way to request a specific model revision
    # (InferenceClient takes no revision argument, on init or on
    # chat_completion), so a `revision` field here would imply a weight pin
    # that cannot exist. Guards against it being re-added.
    cfg = arena_io.load_candidates()
    entries = list(cfg["candidates"]) + [cfg["judge"]]
    for entry in entries:
        assert "revision" not in entry
