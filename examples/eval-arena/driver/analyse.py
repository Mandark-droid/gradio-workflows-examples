"""Run-level analysis: mean score with bootstrap CI, Bradley-Terry ratings
from pairwise outcomes, and latency percentiles.

At five rows none of these is statistically meaningful. The implementations
are the deliverable; every reported number carries SMALL_SAMPLE_NOTE, and the
command to re-run at a real sample size is printed alongside it.

    python driver/analyse.py --checkpoint runs/run_x.ckpt.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SMALL_SAMPLE_NOTE = (
    "n=5. These figures are NOT statistically meaningful. A bootstrap CI, a "
    "Bradley-Terry rating and a judge-agreement rate all need far more rows. "
    "Re-run with a larger dataset to obtain reportable numbers."
)


def bootstrap_ci(
    values: list[float], iterations: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[min(int(0.975 * len(means)), len(means) - 1)]
    return round(low, 4), round(high, 4)


def bradley_terry(
    pairs: list[dict], iterations: int = 200, tolerance: float = 1e-9
) -> dict[str, float]:
    """Minorization-maximization fit of Bradley-Terry strengths."""
    models: set[str] = set()
    for pair in pairs:
        models.update({pair.get("a"), pair.get("b")} - {None})
    if not models:
        return {}

    wins = {m: 0.0 for m in models}
    meetings: dict[tuple[str, str], float] = {}
    for pair in pairs:
        a, b = pair.get("a"), pair.get("b")
        if not a or not b:
            continue
        key = tuple(sorted((a, b)))
        meetings[key] = meetings.get(key, 0.0) + 1
        if pair.get("agreed") and pair.get("winner"):
            wins[pair["winner"]] = wins.get(pair["winner"], 0.0) + 1
        else:
            wins[a] = wins.get(a, 0.0) + 0.5
            wins[b] = wins.get(b, 0.0) + 0.5

    strength = {m: 1.0 for m in models}
    for _ in range(iterations):
        updated = {}
        for m in models:
            denominator = 0.0
            for (x, y), count in meetings.items():
                if m not in (x, y):
                    continue
                other = y if x == m else x
                denominator += count / (strength[m] + strength[other])
            updated[m] = wins[m] / denominator if denominator else strength[m]
        total = sum(updated.values()) or 1.0
        updated = {m: v * len(models) / total for m, v in updated.items()}
        shift = max(abs(updated[m] - strength[m]) for m in models)
        strength = updated
        if shift < tolerance:
            break
    return {m: round(v, 4) for m, v in strength.items()}


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return float(ordered[index])

    return {"p50": round(pick(0.5), 2), "p95": round(pick(0.95), 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"No checkpoint at {path}")
        return 2

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line)["payload"])
            except (json.JSONDecodeError, KeyError):
                continue

    scores: dict[str, list[float]] = {}
    latencies: dict[str, list[float]] = {}
    reasoning: dict[str, list[float]] = {}
    pairs: list[dict] = []

    for payload in records:
        for model_id, entry in (payload.get("per_model") or {}).items():
            if entry.get("score") is not None:
                scores.setdefault(model_id, []).append(float(entry["score"]))
        for model_id, entry in (payload.get("latency") or {}).items():
            latencies.setdefault(model_id, []).append(float(entry.get("latency_ms", 0)))
            reasoning.setdefault(model_id, []).append(float(entry.get("reasoning_chars", 0)))
        pairs.extend(payload.get("pairs") or [])

    print(f"Rows analysed: {len(records)}\n")
    print("| model | mean score | 95% CI | p50 ms | p95 ms | mean reasoning chars | BT rating |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    ratings = bradley_terry(pairs)
    for model_id in sorted(set(scores) | set(latencies) | set(ratings)):
        values = scores.get(model_id, [])
        mean = sum(values) / len(values) if values else 0.0
        low, high = bootstrap_ci(values)
        stats = percentiles(latencies.get(model_id, []))
        reasoning_values = reasoning.get(model_id, [])
        mean_reasoning = sum(reasoning_values) / len(reasoning_values) if reasoning_values else 0.0
        print(
            f"| {model_id} | {mean:.3f} | [{low}, {high}] | {stats['p50']} "
            f"| {stats['p95']} | {int(round(mean_reasoning))} | {ratings.get(model_id, 0.0)} |"
        )

    print(f"\n{SMALL_SAMPLE_NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
