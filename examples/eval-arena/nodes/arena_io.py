"""Every model call in the arena goes through here.

Three modes via ARENA_IO_MODE: replay (default), record, live. Tests run in
replay, so they never touch the network and development never re-spends.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"


class FixtureMissing(RuntimeError):
    """Replay mode was asked for a call that has never been recorded."""


def mode() -> str:
    return os.environ.get("ARENA_IO_MODE", "replay").strip().lower()


@lru_cache(maxsize=1)
def load_candidates() -> dict:
    return yaml.safe_load((ROOT / "candidates.yaml").read_text(encoding="utf-8"))


def slot(name: str) -> dict:
    for candidate in load_candidates()["candidates"]:
        if candidate["slot"] == name:
            return candidate
    raise KeyError(f"No candidate slot {name!r}")


@lru_cache(maxsize=1)
def config_hash() -> str:
    """Ties every result record to the exact candidate configuration."""
    payload = json.dumps(load_candidates(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fixture_key(kind: str, target: str, args: tuple) -> str:
    payload = json.dumps([kind, target, list(args)], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def write_fixture(key: str, record: dict) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / f"{key}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_fixture(key: str) -> dict:
    path = FIXTURE_DIR / f"{key}.json"
    if not path.exists():
        raise FixtureMissing(
            f"No fixture {key}. Re-run with ARENA_IO_MODE=record to create it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_token() -> str | None:
    """Never defaulted, never logged."""
    return os.environ.get("HF_TOKEN") or None


def chat(
    model_id: str, prompt: str, max_new_tokens: int, greedy: bool
) -> tuple[str, int, int, int]:
    """One chat completion. Returns (text, latency_ms, tokens_out, reasoning_chars).

    Several candidates are reasoning models: they emit chain-of-thought into
    reasoning_content (field name varies by provider) and put only the final
    answer in content. reasoning_chars makes that spend visible instead of
    letting it vanish into an unexplained gap between tokens_out and the
    length of the returned text.
    """
    key = fixture_key("chat", model_id, (prompt, max_new_tokens, greedy))
    current = mode()

    if current == "replay":
        record = read_fixture(key)
        return (
            record["text"],
            record["latency_ms"],
            record["tokens_out"],
            record.get("reasoning_chars", 0),
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=_hf_token())
    started = time.perf_counter()
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model_id,
        max_tokens=max_new_tokens,
        temperature=0.0 if greedy else 0.7,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    msg = response.choices[0].message
    text = (getattr(msg, "content", None) or "").strip()
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )
    reasoning_chars = len(reasoning)
    usage = getattr(response, "usage", None)
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)

    if current == "record":
        write_fixture(
            key,
            {
                "text": text,
                "latency_ms": latency_ms,
                "tokens_out": tokens_out,
                "reasoning_chars": reasoning_chars,
            },
        )
    return text, latency_ms, tokens_out, reasoning_chars
