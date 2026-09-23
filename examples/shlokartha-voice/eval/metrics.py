"""Metrics shared by the evaluation harnesses."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "gita_5.json"
RESULTS = Path(__file__).resolve().parent / "results"


def load_testset() -> list[dict]:
    return json.loads(DATA.read_text(encoding="utf-8"))["verses"]


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference, hypothesis = reference.strip(), hypothesis.strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def word_f1(gold: list[str], predicted: list[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    gold_set, pred_set = set(gold), set(predicted)
    overlap = len(gold_set & pred_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def write_results(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


SMALL_SAMPLE_NOTE = (
    "n=6. Not statistically meaningful. Restore the full test set in "
    "eval/data/ and re-run to obtain a reportable number."
)
