import json
from nodes.aggregate import aggregate


def _env(model_id, latency_ms=100, tokens_out=20, reasoning_chars=150):
    return json.dumps({
        "model_id": model_id, "output": "o", "latency_ms": latency_ms,
        "tokens_out": tokens_out, "reasoning_chars": reasoning_chars, "error": None,
    })


SCORER = json.dumps({"per_model": {
    "m1": {"score": 1.0, "failure_tag": None},
    "m2": {"score": 0.0, "failure_tag": "wrong"},
    "m3": {"score": 1.0, "failure_tag": None},
}})

JUDGE = json.dumps({"pairs": [
    {"a": "m1", "b": "m2", "winner": "m1", "agreed": True, "confidence": 0.9, "rationale": "r"},
    {"a": "m1", "b": "m3", "winner": None, "agreed": False, "confidence": 0.4, "rationale": "r"},
    {"a": "m2", "b": "m3", "winner": "m3", "agreed": True, "confidence": 0.8, "rationale": "r"},
]})


def test_returns_exactly_three_payloads():
    result = aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "row-1")
    assert len(result) == 3


def test_scores_payload_carries_row_id_and_per_model():
    scores = json.loads(aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "row-1")[0])
    assert scores["row_id"] == "row-1"
    assert set(scores["per_model"]) == {"m1", "m2", "m3"}


def test_wins_are_counted_from_agreed_pairs_only():
    scores = json.loads(aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "row-1")[0])
    assert scores["wins"] == {"m1": 1, "m2": 0, "m3": 1}


def test_latency_payload_has_one_entry_per_model():
    latency = json.loads(aggregate(SCORER, JUDGE, _env("m1", 100), _env("m2", 200), _env("m3", 300), "r")[1])
    assert latency["m1"]["latency_ms"] == 100
    assert latency["m3"]["tokens_out"] == 20
    assert latency["m1"]["reasoning_chars"] == 150


def test_verdict_payload_passes_pairs_through():
    verdict = json.loads(aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "r")[2])
    assert len(verdict["pairs"]) == 3


def test_verdict_embeds_scores_and_latency_for_single_endpoint_use():
    verdict = json.loads(aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "r")[2])
    assert "per_model" in verdict and "latency" in verdict


def test_malformed_inputs_do_not_crash_the_node():
    result = aggregate("bad", "bad", "bad", "bad", "bad", "r")
    assert len(result) == 3
    assert json.loads(result[0])["row_id"] == "r"


def test_config_hash_is_embedded_for_traceability():
    scores = json.loads(aggregate(SCORER, JUDGE, _env("m1"), _env("m2"), _env("m3"), "r")[0])
    assert scores["candidate_config_hash"]
