"""Deterministic scoring — the primary signal.

Dispatches on the gold answer type. Free-form rows return a None score and
are left to the judge, which is the only place a model's wording is graded
by another model.
"""
from __future__ import annotations

import json
import re

FAILURE_TAGS = ("wrong", "unparseable", "truncated", "error")

NUMERIC_TOLERANCE = 1e-3
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _normalize(text: str) -> str:
    text = " ".join(str(text).split()).strip().lower()
    return text.strip(" .,;:!?\"'")


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def score_one(
    output: str, gold: str, answer_type: str, tokens_out: int, token_cap: int
) -> tuple[float | None, str | None]:
    """Return (score, failure_tag). A None score means 'judge decides'."""
    text = str(output or "")

    if answer_type == "free-form":
        return None, None

    if tokens_out >= token_cap and answer_type == "truncation":
        return 0.0, "truncated"

    if answer_type == "exact":
        if _normalize(text) == _normalize(gold):
            return 1.0, None
        return 0.0, "wrong"

    if answer_type == "numeric":
        found = _NUMBER_RE.findall(text.replace(",", ""))
        if not found:
            return 0.0, "unparseable"
        try:
            expected = float(gold)
        except (TypeError, ValueError):
            return 0.0, "unparseable"
        for candidate in found:
            actual = float(candidate)
            denominator = abs(expected) or 1.0
            if abs(actual - expected) / denominator <= NUMERIC_TOLERANCE:
                return 1.0, None
        return 0.0, "wrong"

    if answer_type == "json":
        try:
            parsed = json.loads(_strip_fence(text))
        except json.JSONDecodeError:
            return 0.0, "unparseable"
        try:
            expected = json.loads(gold)
        except json.JSONDecodeError:
            return 0.0, "unparseable"
        return (1.0, None) if parsed == expected else (0.0, "wrong")

    if answer_type == "truncation":
        if _normalize(gold) in _normalize(text):
            return 1.0, None
        return 0.0, "wrong"

    return (1.0, None) if _normalize(text) == _normalize(gold) else (0.0, "wrong")


def _load(envelope_json: str) -> dict:
    try:
        payload = json.loads(envelope_json)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass
    return {"model_id": "", "output": "", "tokens_out": 0, "error": "malformed envelope"}


def deterministic_scorer(
    env_a: str, env_b: str, env_c: str, gold_json: str
) -> str:
    from nodes import arena_io

    try:
        gold_payload = json.loads(gold_json)
    except (json.JSONDecodeError, TypeError):
        gold_payload = {"gold": "", "answer_type": ""}

    gold = str(gold_payload.get("gold", ""))
    answer_type = str(gold_payload.get("answer_type", ""))

    caps = {c["slot"]: int(c["max_new_tokens"]) for c in
            arena_io.load_candidates()["candidates"]}

    per_model: dict[str, dict] = {}
    for slot_name, raw in zip(("a", "b", "c"), (env_a, env_b, env_c)):
        envelope = _load(raw)
        model_id = envelope.get("model_id") or f"slot_{slot_name}"

        if envelope.get("error"):
            per_model[model_id] = {"score": 0.0, "failure_tag": "error"}
            continue

        score, tag = score_one(
            envelope.get("output", ""),
            gold,
            answer_type,
            int(envelope.get("tokens_out") or 0),
            caps.get(slot_name, 512),
        )
        per_model[model_id] = {"score": score, "failure_tag": tag}

    return json.dumps({"per_model": per_model}, ensure_ascii=False)
