"""Top-1 and top-3 word F1 against gold padaccheda. Pure CPU — costs nothing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import SMALL_SAMPLE_NOTE, load_testset, word_f1, write_results
from nodes.hf_io import _utf8_stdout
from nodes.normalize import normalize
from nodes.sandhi import sandhi_split


def main() -> int:
    _utf8_stdout()
    rows = load_testset()
    top1_scores, top3_scores, details = [], [], []

    print("| id | top-1 F1 | top-3 F1 |")
    print("| --- | --- | --- |")
    for row in rows:
        normalized = normalize(json.dumps({"text": row["iast"], "source": "typed"}))
        splits = json.loads(sandhi_split(normalized))["splits"]
        gold = row["gold_padaccheda"]

        top1 = word_f1(gold, splits[0]["words"]) if splits else 0.0
        top3 = max((word_f1(gold, s["words"]) for s in splits[:3]), default=0.0)
        top1_scores.append(top1)
        top3_scores.append(top3)
        details.append({"id": row["id"], "top1": top1, "top3": top3})
        print(f"| {row['id']} | {top1:.2f} | {top3:.2f} |")

    mean1 = sum(top1_scores) / len(top1_scores)
    mean3 = sum(top3_scores) / len(top3_scores)
    print(f"\nMean top-1 F1: {mean1:.3f}   Mean top-3 F1: {mean3:.3f}")
    print(SMALL_SAMPLE_NOTE)
    write_results("sandhi", {"top1": mean1, "top3": mean3, "n": len(rows),
                             "details": details, "note": SMALL_SAMPLE_NOTE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
