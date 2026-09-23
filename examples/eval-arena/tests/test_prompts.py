import json
from nodes.prompts import prompt_builder, gold_extract, TEMPLATES


def test_every_answer_type_has_a_template():
    for task_type in ("exact", "numeric", "json", "free-form", "truncation"):
        assert task_type in TEMPLATES


def test_prompt_is_wrapped_with_its_template():
    built = prompt_builder("numeric", "What is 2+2?")
    assert "What is 2+2?" in built
    assert built != "What is 2+2?"


def test_numeric_template_asks_for_a_bare_number():
    assert "number" in prompt_builder("numeric", "x").lower()


def test_json_template_asks_for_json_only():
    assert "json" in prompt_builder("json", "x").lower()


def test_unknown_task_type_falls_back_without_raising():
    assert "x" in prompt_builder("nonsense", "x")


def test_gold_extract_normalizes_whitespace_and_case_for_exact():
    out = json.loads(gold_extract("  Tokyo  ", "exact"))
    assert out == {"gold": "tokyo", "answer_type": "exact"}


def test_gold_extract_keeps_numeric_gold_parseable():
    out = json.loads(gold_extract("161", "numeric"))
    assert float(out["gold"]) == 161.0


def test_gold_extract_compacts_json_gold():
    out = json.loads(gold_extract('{"name": "Asha", "age": 31}', "json"))
    assert json.loads(out["gold"]) == {"name": "Asha", "age": 31}


def test_gold_extract_preserves_freeform_case():
    out = json.loads(gold_extract("Tracing Links A Request", "free-form"))
    assert out["gold"] == "Tracing Links A Request"


def test_gold_extract_of_empty_gold_is_safe():
    out = json.loads(gold_extract("", "exact"))
    assert out["gold"] == ""
