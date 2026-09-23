"""The only full join in the graph.

gr.Workflow gives one endpoint per connected component, so /scores returns
all three subjects. The verdict payload therefore embeds scores and latency
too: a caller who only reads one subject still has everything, and the driver
never needs a second call.
"""
from __future__ import annotations

import json

from nodes import arena_io


def _load(text: str, default: dict) -> dict:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else default
    except (json.JSONDecodeError, TypeError):
        return default


def aggregate(
    scorer_json: str, judge_json: str,
    env_a: str, env_b: str, env_c: str, row_id: str,
) -> tuple[str, str, str]:
    scorer = _load(scorer_json, {"per_model": {}})
    judge = _load(judge_json, {"pairs": []})

    per_model = scorer.get("per_model") or {}
    pairs = judge.get("pairs") or []

    latency: dict[str, dict] = {}
    for raw in (env_a, env_b, env_c):
        envelope = _load(raw, {})
        model_id = envelope.get("model_id")
        if model_id:
            latency[model_id] = {
                "latency_ms": int(envelope.get("latency_ms") or 0),
                "tokens_out": int(envelope.get("tokens_out") or 0),
                "reasoning_chars": int(envelope.get("reasoning_chars") or 0),
            }

    wins = {model_id: 0 for model_id in per_model}
    for pair in pairs:
        for model_id in (pair.get("a"), pair.get("b")):
            wins.setdefault(model_id, 0)
        winner = pair.get("winner")
        if pair.get("agreed") and winner:
            wins[winner] = wins.get(winner, 0) + 1
    wins.pop(None, None)

    try:
        config_hash = arena_io.config_hash()
    except Exception:
        config_hash = ""

    scores_payload = {
        "row_id": row_id,
        "per_model": per_model,
        "wins": wins,
        "candidate_config_hash": config_hash,
    }
    latency_payload = latency
    verdict_payload = {
        "row_id": row_id,
        "pairs": pairs,
        "per_model": per_model,
        "latency": latency,
        "candidate_config_hash": config_hash,
    }

    dump = lambda payload: json.dumps(payload, ensure_ascii=False)
    return dump(scores_payload), dump(latency_payload), dump(verdict_payload)
