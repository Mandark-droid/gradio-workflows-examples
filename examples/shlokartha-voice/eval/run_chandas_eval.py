"""Metre accuracy on the typed test set. Pure CPU — costs nothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import SMALL_SAMPLE_NOTE, load_testset, write_results
from nodes.chandas import chandas_detect
from nodes.hf_io import _utf8_stdout


def main() -> int:
    _utf8_stdout()
    rows, hits = load_testset(), 0
    details = []

    print("| id | gold metre | detected | confidence | ok |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        normalized = json.dumps({"iast": row["iast"], "padas": row["padas"]})
        result = json.loads(chandas_detect(normalized))
        ok = result["metre"] == row["metre"]
        hits += ok
        details.append({"id": row["id"], "gold": row["metre"], **result, "ok": ok})
        print(
            f"| {row['id']} | {row['metre']} | {result['metre']} "
            f"| {result['confidence']} | {'yes' if ok else 'no'} |"
        )

    accuracy = hits / len(rows)
    print(f"\nMetre accuracy: {hits}/{len(rows)} = {accuracy:.0%}")
    print(SMALL_SAMPLE_NOTE)
    write_results("chandas", {"accuracy": accuracy, "n": len(rows), "details": details,
                              "note": SMALL_SAMPLE_NOTE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
