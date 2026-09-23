"""Pairwise LLM judging with position-swap control.

Three candidates give three pairs. Each pair is judged twice with the
positions swapped, so six calls per row, issued concurrently. A pair counts
as a win only when both orderings name the same model; if the judge names the
same *position* both times it is expressing position bias, and the pair is
recorded as a tie.

The judge is out-of-family from every candidate, pinned in candidates.yaml.
"""
from __future__ import annotations

import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor

from nodes import arena_io

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

JUDGE_TEMPLATE = """You are grading two answers to the same task.

Task:
{prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Reply with a single JSON object and nothing else:
{{"winner": "A" or "B" or "tie", "confidence": 0.0-1.0, "rationale": "at most {cap} words"}}
"""


def _load(envelope_json: str) -> dict:
    try:
        payload = json.loads(envelope_json)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass
    return {"model_id": "", "output": "", "error": "malformed envelope"}


def _parse_verdict(text: str) -> dict:
    match = _FENCE_RE.search(text)
    raw = match.group(1) if match else text
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        return {"winner": "tie", "confidence": 0.0, "rationale": ""}

    winner = str(payload.get("winner", "tie")).strip().upper()
    if winner not in ("A", "B"):
        winner = "TIE"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "winner": winner,
        "confidence": confidence,
        "rationale": str(payload.get("rationale", "")),
    }


def _ask(prompt_text: str, answer_a: str, answer_b: str, hf_token: str = "") -> dict:
    cfg = arena_io.load_candidates()["judge"]
    prompt = JUDGE_TEMPLATE.format(
        prompt=prompt_text, answer_a=answer_a, answer_b=answer_b,
        cap=cfg["max_rationale_words"],
    )
    try:
        text, _, _, _ = arena_io.chat(
            cfg["model_id"], prompt, int(cfg["max_new_tokens"]), True, hf_token
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if hf_token:
            # A visitor-supplied credential must never reach a result record,
            # even indirectly through an HTTP client's error message.
            message = message.replace(hf_token, "[redacted]")
        return {"winner": "TIE", "confidence": 0.0, "rationale": "", "error": message}
    verdict = _parse_verdict(text)
    verdict["error"] = None
    return verdict


def _cap_words(text: str, cap: int) -> str:
    words = str(text).split()
    return " ".join(words[:cap])


def pairwise_judge(env_a: str, env_b: str, env_c: str, prompt: str, hf_token: str = "") -> str:
    cfg = arena_io.load_candidates()["judge"]
    cap = int(cfg["max_rationale_words"])

    envelopes = [_load(env_a), _load(env_b), _load(env_c)]
    for index, envelope in enumerate(envelopes):
        envelope.setdefault("model_id", f"slot_{index}")

    combos = list(itertools.combinations(range(3), 2))

    jobs = []
    for i, j in combos:
        jobs.append((i, j, envelopes[i].get("output", ""), envelopes[j].get("output", "")))
        jobs.append((j, i, envelopes[j].get("output", ""), envelopes[i].get("output", "")))

    # All six calls are submitted up front so they run concurrently; the
    # `with` block's shutdown(wait=True) only blocks until *these* futures
    # are done, which is fine since we need every verdict before scoring
    # the pairs. What must never happen is dispatching one job at a time.
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(_ask, prompt, job[2], job[3], hf_token) for job in jobs
        ]
        verdicts = [future.result() for future in futures]

    pairs = []
    for index, (i, j) in enumerate(combos):
        forward = verdicts[index * 2]
        swapped = verdicts[index * 2 + 1]

        # Translate each verdict from a position into a candidate index.
        forward_pick = {"A": i, "B": j}.get(forward["winner"])
        swapped_pick = {"A": j, "B": i}.get(swapped["winner"])

        broken_i = bool(envelopes[i].get("error"))
        broken_j = bool(envelopes[j].get("error"))

        if broken_i and not broken_j:
            winner_index, agreed = j, True
        elif broken_j and not broken_i:
            winner_index, agreed = i, True
        elif forward_pick is not None and forward_pick == swapped_pick:
            winner_index, agreed = forward_pick, True
        else:
            winner_index, agreed = None, False

        # A None judge_error means both the forward and swapped calls
        # succeeded; otherwise it is the first error seen, so a tie caused by
        # an unreachable judge is never silently indistinguishable from a
        # tie the judge actually reached.
        judge_error = forward.get("error") or swapped.get("error")

        pairs.append({
            "a": envelopes[i]["model_id"],
            "b": envelopes[j]["model_id"],
            "winner": envelopes[winner_index]["model_id"] if winner_index is not None else None,
            "agreed": agreed,
            "confidence": round(
                (forward["confidence"] + swapped["confidence"]) / 2, 3
            ),
            "rationale": _cap_words(forward["rationale"], cap),
            "judge_error": judge_error,
        })

    return json.dumps({"pairs": pairs}, ensure_ascii=False)
