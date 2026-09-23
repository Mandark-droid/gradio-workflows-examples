"""Every network call in this example goes through here.

Three modes, set by WORKFLOW_IO_MODE:

  replay (default) — read a recorded fixture; raise FixtureMissing if absent.
                     Tests run in this mode, so they never touch the network.
  record           — call live, then write the fixture.
  live             — call live, write nothing.

This is what keeps the project inside its budget: development replays, and
only an explicit record/live run spends anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"


class FixtureMissing(RuntimeError):
    """Replay mode was asked for a call that has never been recorded."""


def _utf8_stdout() -> None:
    """Windows consoles default to cp1252 and crash on Devanagari or IAST."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def mode() -> str:
    return os.environ.get("WORKFLOW_IO_MODE", "replay").strip().lower()


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def fixture_key(kind: str, target: str, endpoint: str, args: tuple) -> str:
    payload = json.dumps(
        [kind, target, endpoint, list(args)], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def write_fixture(key: str, record: dict, binary: bytes | None = None) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    if binary is not None:
        blob = FIXTURE_DIR / f"{key}.bin"
        blob.write_bytes(binary)
        record = {**record, "binary": blob.name}
    (FIXTURE_DIR / f"{key}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_fixture(key: str) -> tuple[dict, bytes | None]:
    path = FIXTURE_DIR / f"{key}.json"
    if not path.exists():
        raise FixtureMissing(
            f"No fixture {key}. Re-run with WORKFLOW_IO_MODE=record to create it."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    binary = None
    if record.get("binary"):
        binary = (FIXTURE_DIR / record["binary"]).read_bytes()
    return record, binary


def _hf_token() -> str | None:
    """Never defaulted, never logged."""
    return os.environ.get("HF_TOKEN") or None


def call_space(
    space_id: str, api_name: str, *args: Any, result_index: int | None = None
) -> Any:
    key = fixture_key("space", space_id, api_name, args)
    current = mode()

    if current == "replay":
        record, _ = read_fixture(key)
        value = record["value"]
    else:
        from gradio_client import Client

        client = Client(space_id, token=_hf_token())
        value = client.predict(*args, api_name=api_name)
        if isinstance(value, tuple):
            value = list(value)
        if current == "record":
            write_fixture(key, {"value": value})

    if result_index is not None and isinstance(value, list):
        return value[result_index]
    return value


def call_model_text_to_image(
    model_id: str, prompt: str, width: int, height: int
) -> bytes:
    key = fixture_key("model", model_id, "text_to_image", (prompt, width, height))
    current = mode()

    if current == "replay":
        _, binary = read_fixture(key)
        if binary is None:
            raise FixtureMissing(f"Fixture {key} has no binary payload.")
        return binary

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=_hf_token())
    image = client.text_to_image(
        prompt, model=model_id, width=width, height=height
    )
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    binary = buffer.getvalue()
    if current == "record":
        write_fixture(key, {"value": None}, binary=binary)
    return binary
