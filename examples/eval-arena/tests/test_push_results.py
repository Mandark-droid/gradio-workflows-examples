import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import push_results


class _FakeApi:
    def __init__(self, token=None):
        self.token = token
        self.uploaded_text = None

    def create_repo(self, *a, **k):
        pass

    def upload_file(self, path_or_fileobj, **k):
        self.uploaded_text = Path(path_or_fileobj).read_text(encoding="utf-8")


def _write_checkpoint(path: Path) -> None:
    payload = {
        "row_id": "r1",
        "per_model": {
            "m1": {"score": 1.0, "failure_tag": None},
            "m2": {"score": 0.0, "failure_tag": None},
        },
        "latency": {},
        "wins": {"m1": 1, "m2": 0},
        # judge.pairwise_judge's pairs, as aggregate() and run_batch's merge
        # pass them through unchanged.
        "pairs": [
            {
                "a": "m1", "b": "m2", "winner": "m1", "agreed": True,
                "confidence": 0.9, "rationale": "m1 is more precise",
                "judge_error": None,
            },
        ],
        "workflow_json_hash": "abc",
        "candidate_config_hash": "def",
    }
    path.write_text(
        json.dumps({"row_index": 0, "payload": payload}) + "\n", encoding="utf-8"
    )


def test_judge_outcomes_attached_per_model(tmp_path, monkeypatch):
    """`pairs` survives the driver's merge but used to die at publish time —
    the spec's schema names judge_outcomes. This asserts each model's own
    published record carries the pairs it was involved in, with rationale
    omitted to keep records small."""
    ckpt = tmp_path / "run.ckpt.jsonl"
    _write_checkpoint(ckpt)

    fake = _FakeApi()
    monkeypatch.setattr(push_results, "_resolve_token", lambda: "fake-token")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: fake)
    monkeypatch.setattr(
        sys, "argv",
        ["push_results.py", "--checkpoint", str(ckpt), "--repo", "owner/name",
         "--run-id", "testrun"],
    )

    assert push_results.main() == 0
    records = [json.loads(line) for line in fake.uploaded_text.splitlines()]
    by_model = {r["model_id"]: r for r in records}

    assert by_model["m1"]["judge_outcomes"] == [
        {"opponent": "m2", "winner": "m1", "agreed": True, "confidence": 0.9}
    ]
    assert by_model["m2"]["judge_outcomes"] == [
        {"opponent": "m1", "winner": "m1", "agreed": True, "confidence": 0.9}
    ]
    # rationale is dropped to keep records small.
    for outcomes in (by_model["m1"]["judge_outcomes"], by_model["m2"]["judge_outcomes"]):
        for outcome in outcomes:
            assert "rationale" not in outcome


def test_judge_outcomes_empty_list_when_no_pairs(tmp_path, monkeypatch):
    payload = {
        "row_id": "r1",
        "per_model": {"m1": {"score": 1.0, "failure_tag": None}},
        "latency": {}, "wins": {"m1": 0}, "pairs": [],
        "workflow_json_hash": "abc", "candidate_config_hash": "def",
    }
    ckpt = tmp_path / "run.ckpt.jsonl"
    ckpt.write_text(json.dumps({"row_index": 0, "payload": payload}) + "\n", encoding="utf-8")

    fake = _FakeApi()
    monkeypatch.setattr(push_results, "_resolve_token", lambda: "fake-token")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: fake)
    monkeypatch.setattr(
        sys, "argv",
        ["push_results.py", "--checkpoint", str(ckpt), "--repo", "owner/name",
         "--run-id", "testrun"],
    )

    assert push_results.main() == 0
    records = [json.loads(line) for line in fake.uploaded_text.splitlines()]
    assert records[0]["judge_outcomes"] == []
