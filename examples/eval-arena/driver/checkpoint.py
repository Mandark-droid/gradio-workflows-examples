"""Append-only JSONL checkpoint so an interrupted run resumes without
re-spending on rows that already completed."""
from __future__ import annotations

import json
from pathlib import Path


def load_done(path: Path) -> set[int]:
    if not Path(path).exists():
        return set()
    done: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(int(json.loads(line)["row_index"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # a torn final line must not poison the resume
    return done


def append(path: Path, row_index: int, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"row_index": int(row_index), "payload": payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
