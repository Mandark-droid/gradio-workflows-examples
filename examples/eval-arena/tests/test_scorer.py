import json
import pytest
from nodes.scorer import deterministic_scorer, score_one, FAILURE_TAGS


def _env(model_id="m", output="", error=None, tokens_out=10):
    # Six keys, matching nodes.candidate.ENVELOPE_KEYS exactly: three of the
    # four candidates are reasoning models, so reasoning_chars is part of
    # every real envelope even though the scorer itself never reads it.
    return json.dumps({
        "model_id": model_id, "output": output, "latency_ms": 1,
        "tokens_out": tokens_out, "reasoning_chars": 0, "error": error,
    })


def _gold(gold, answer_type):
    return json.dumps({"gold": gold, "answer_type": answer_type})


def test_exact_match_scores_one():
    assert score_one("Tokyo", "tokyo", "exact", 10, 512)[0] == 1.0


def test_exact_match_ignores_case_and_punctuation():
    assert score_one("  Tokyo.  ", "tokyo", "exact", 10, 512)[0] == 1.0


def test_exact_mismatch_scores_zero_and_tags_wrong():
    score, tag = score_one("Osaka", "tokyo", "exact", 10, 512)
    assert score == 0.0 and tag == "wrong"


def test_numeric_within_tolerance_scores_one():
    assert score_one("161.0000001", "161", "numeric", 10, 512)[0] == 1.0


def test_numeric_outside_tolerance_is_wrong():
    score, tag = score_one("162", "161", "numeric", 10, 512)
    assert score == 0.0 and tag == "wrong"


def test_numeric_extracts_a_number_from_prose():
    assert score_one("The answer is 161.", "161", "numeric", 10, 512)[0] == 1.0


def test_non_numeric_output_for_numeric_gold_is_unparseable():
    score, tag = score_one("no idea", "161", "numeric", 10, 512)
    assert score == 0.0 and tag == "unparseable"


def test_valid_matching_json_scores_one():
    assert score_one('{"age": 31, "name": "Asha"}',
                     '{"age":31,"name":"Asha"}', "json", 10, 512)[0] == 1.0


def test_json_in_a_code_fence_is_still_parsed():
    out = '```json\n{"name": "Asha", "age": 31}\n```'
    assert score_one(out, '{"age":31,"name":"Asha"}', "json", 10, 512)[0] == 1.0


def test_invalid_json_is_unparseable():
    score, tag = score_one("not json", '{"a":1}', "json", 10, 512)
    assert score == 0.0 and tag == "unparseable"


def test_freeform_is_left_to_the_judge_with_a_neutral_score():
    score, tag = score_one("anything", "gold text", "free-form", 10, 512)
    assert score is None and tag is None


def test_output_at_the_token_cap_is_tagged_truncated():
    score, tag = score_one("1 2 3", "END", "truncation", 512, 512)
    assert tag == "truncated"


def test_error_envelope_is_tagged_error():
    result = json.loads(deterministic_scorer(
        _env("m1", "", error="boom"), _env("m2", "Tokyo"), _env("m3", "Tokyo"),
        _gold("tokyo", "exact"),
    ))
    assert result["per_model"]["m1"]["failure_tag"] == "error"
    assert result["per_model"]["m1"]["score"] == 0.0


def test_all_three_models_appear_in_the_result():
    result = json.loads(deterministic_scorer(
        _env("m1", "Tokyo"), _env("m2", "Osaka"), _env("m3", "Tokyo"),
        _gold("tokyo", "exact"),
    ))
    assert set(result["per_model"]) == {"m1", "m2", "m3"}


def test_every_tag_used_is_a_declared_tag():
    result = json.loads(deterministic_scorer(
        _env("m1", "", error="x"), _env("m2", "Osaka"), _env("m3", "Tokyo"),
        _gold("tokyo", "exact"),
    ))
    for entry in result["per_model"].values():
        assert entry["failure_tag"] in FAILURE_TAGS or entry["failure_tag"] is None


def test_malformed_envelope_does_not_crash_the_node():
    result = json.loads(deterministic_scorer(
        "not json", _env("m2", "Tokyo"), _env("m3", "Tokyo"), _gold("tokyo", "exact")
    ))
    assert "per_model" in result
