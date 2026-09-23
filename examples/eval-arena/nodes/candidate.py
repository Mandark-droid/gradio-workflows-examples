"""One runner for every candidate slot.

Candidates are `fn` nodes rather than `model` nodes because model-kind nodes
expose no latency or token counts — verified against the gr.Workflow schema.
Wrapping InferenceClient is what makes the uniform envelope possible.

The node reads a slot letter, never a model name, so swapping a candidate is
a candidates.yaml edit with no change to this file or anything downstream.

Three of the four models behind these slots are reasoning models: they spend
completion tokens on chain-of-thought (reasoning_content) before the final
answer lands in content. reasoning_chars carries that spend through the
envelope so a reader can tell "thinking" from "stalling" instead of staring
at a gap between tokens_out and the length of output.
"""
from __future__ import annotations

import json

from nodes import arena_io

ENVELOPE_KEYS = (
    "model_id",
    "output",
    "latency_ms",
    "tokens_out",
    "reasoning_chars",
    "error",
)


def _envelope(
    model_id: str,
    output: str,
    latency_ms: int,
    tokens_out: int,
    reasoning_chars: int,
    error: str | None,
) -> str:
    return json.dumps(
        {
            "model_id": model_id,
            "output": output,
            "latency_ms": latency_ms,
            "tokens_out": tokens_out,
            "reasoning_chars": reasoning_chars,
            "error": error,
        },
        ensure_ascii=False,
    )


def run_candidate(slot_name: str, prompt: str, hf_token: str = "") -> str:
    try:
        config = arena_io.slot(slot_name)
    except KeyError as exc:
        return _envelope("", "", 0, 0, 0, f"unknown slot: {exc}")

    model_id = config["model_id"]
    try:
        text, latency_ms, tokens_out, reasoning_chars = arena_io.chat(
            model_id,
            str(prompt or ""),
            int(config["max_new_tokens"]),
            bool(config["greedy"]),
            hf_token,
        )
    except Exception as exc:
        # A dead provider must not fail the row; the scorer sees the error tag.
        return _envelope(model_id, "", 0, 0, 0, f"{type(exc).__name__}: {exc}")

    return _envelope(model_id, text, latency_ms, tokens_out, reasoning_chars, None)


def candidate_a(prompt: str, hf_token: str = "") -> str:
    return run_candidate("a", prompt, hf_token)


def candidate_b(prompt: str, hf_token: str = "") -> str:
    return run_candidate("b", prompt, hf_token)


def candidate_c(prompt: str, hf_token: str = "") -> str:
    return run_candidate("c", prompt, hf_token)
