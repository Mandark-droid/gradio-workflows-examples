"""Metre accuracy on the typed test set. Pure CPU — costs nothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import SMALL_SAMPLE_NOTE, load_testset, write_results
from nodes.chandas import chandas_detect
from nodes.hf_io import _utf8_stdout
from nodes.normalize import normalize


def main() -> int:
    _utf8_stdout()
    rows, gold_hits, split_hits = load_testset(), 0, 0
    details = []

    print("| id | gold metre | gold-pada detected | split detected | confidence (gold/split) | ok (gold/split) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        gold_payload = json.dumps({"iast": row["iast"], "padas": row["padas"]})
        gold_result = json.loads(chandas_detect(gold_payload))

        split_payload = normalize(json.dumps({"text": row["iast"], "source": "typed"}))
        split_result = json.loads(chandas_detect(split_payload))

        gold_ok = gold_result["metre"] == row["metre"]
        split_ok = split_result["metre"] == row["metre"]
        gold_hits += gold_ok
        split_hits += split_ok
        details.append({
            "id": row["id"],
            "gold": row["metre"],
            "gold_result": gold_result,
            "split_result": split_result,
            "gold_ok": gold_ok,
            "split_ok": split_ok,
        })
        print(
            f"| {row['id']} | {row['metre']} | {gold_result['metre']} "
            f"| {split_result['metre']} | {gold_result['confidence']}/{split_result['confidence']} "
            f"| {'yes' if gold_ok else 'no'}/{'yes' if split_ok else 'no'} |"
        )

    gold_accuracy = gold_hits / len(rows)
    split_accuracy = split_hits / len(rows)
    print(f"\nMetre accuracy (gold padas):  {gold_hits}/{len(rows)} = {gold_accuracy:.0%}")
    print(f"Metre accuracy (split_padas): {split_hits}/{len(rows)} = {split_accuracy:.0%}")
    print(
        "The split-path accuracy is the number that reflects what a user "
        "actually gets, since production never has gold pada boundaries."
    )
    print(SMALL_SAMPLE_NOTE)
    write_results("chandas", {
        "gold_accuracy": gold_accuracy,
        "split_accuracy": split_accuracy,
        "n": len(rows),
        "details": details,
        "note": SMALL_SAMPLE_NOTE,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
