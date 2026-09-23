import json
import pytest
from driver.checkpoint import load_done, append
from driver.run_batch import project_cost, CALLS_PER_ROW


def test_missing_checkpoint_is_an_empty_set(tmp_path):
    assert load_done(tmp_path / "none.jsonl") == set()


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "run.jsonl"
    append(path, 0, {"row_id": "a"})
    append(path, 3, {"row_id": "d"})
    assert load_done(path) == {0, 3}


def test_resume_does_not_lose_earlier_rows(tmp_path):
    path = tmp_path / "run.jsonl"
    append(path, 0, {"row_id": "a"})
    assert load_done(path) == {0}
    append(path, 1, {"row_id": "b"})
    assert load_done(path) == {0, 1}


def test_corrupt_checkpoint_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "run.jsonl"
    append(path, 0, {"row_id": "a"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert load_done(path) == {0}


def test_checkpoint_records_the_payload(tmp_path):
    path = tmp_path / "run.jsonl"
    append(path, 2, {"row_id": "c", "scores": {}})
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["row_index"] == 2 and record["payload"]["row_id"] == "c"


def test_calls_per_row_is_three_candidates_plus_six_judge():
    assert CALLS_PER_ROW == 9


def test_cost_projection_scales_with_rows():
    calls_5, usd_5 = project_cost(5)
    calls_10, usd_10 = project_cost(10)
    assert calls_5 == 45 and calls_10 == 90
    assert usd_10 > usd_5


def test_cost_projection_of_zero_rows_is_zero():
    assert project_cost(0) == (0, 0.0)


def test_resume_does_not_call_the_endpoint_for_completed_rows(tmp_path, monkeypatch):
    from driver import run_batch

    path = tmp_path / "run.jsonl"
    append(path, 0, {"row_id": "a"})
    append(path, 1, {"row_id": "b"})

    calls = []

    def _fake_run_row(client, row_index):
        calls.append(row_index)
        return {"row_id": f"row{row_index}"}

    monkeypatch.setattr(run_batch, "_run_row", _fake_run_row)
    monkeypatch.setattr(run_batch, "_resolve_token", lambda: "test-token")

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(run_batch, "Client", _FakeClient, raising=False)

    run_batch.run("owner/name", 4, path, max_calls=100, workers=1, assume_yes=True)

    assert 0 not in calls and 1 not in calls, f"re-ran completed rows: {calls}"
    assert sorted(calls) == [2, 3]


def test_ceiling_refuses_before_making_any_call(tmp_path, monkeypatch):
    from driver import run_batch

    calls = []
    monkeypatch.setattr(run_batch, "_run_row", lambda c, i: calls.append(i))
    rc = run_batch.run("owner/name", 500, tmp_path / "x.jsonl",
                       max_calls=100, workers=1, assume_yes=True)
    assert rc == 1
    assert calls == [], "made calls despite exceeding the ceiling"
