"""Answer the spec's open question: does feeding padaccheda improve meaning?

Arm A: /sanskrit_commentator(verse)             -- what the graph uses.
Arm B: /interpret_sanskrit_verse(system_prompt) -- word split injected into
       the system prompt, which is the only way this Space accepts it.

If arm B wins, rewire op_meaning to an fn wrapping /interpret_sanskrit_verse.

    WORKFLOW_IO_MODE=record python eval/run_meaning_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import SMALL_SAMPLE_NOTE, load_testset, write_results
from nodes import hf_io
from nodes.hf_io import _utf8_stdout
from nodes.normalize import normalize
from nodes.sandhi import sandhi_split

SYSTEM_PROMPT = (
    "You are a Sanskrit commentator. The verse has been split into these "
    "words (padaccheda): {words}. Use the split to resolve compounds, then "
    "give the meaning in English."
)


def main() -> int:
    _utf8_stdout()
    cfg = hf_io.load_config()["shlokartha"]
    rows, details = load_testset(), []

    for row in rows:
        normalized = normalize(json.dumps({"text": row["iast"], "source": "typed"}))
        splits = json.loads(sandhi_split(normalized))["splits"]
        words = splits[0]["words"] if splits else []

        try:
            arm_a = hf_io.call_space(
                cfg["space_id"], cfg["api_name"], row["devanagari"], result_index=0
            )
            arm_b = hf_io.call_space(
                cfg["space_id"], cfg["interpret_api_name"],
                SYSTEM_PROMPT.format(words=" ".join(words)),
                "", row["devanagari"], 512, 0.7, 0.9, 1.1, False, 0, False,
                result_index=0,
            )
        except hf_io.FixtureMissing:
            print(f"{row['id']}: no fixture — run with WORKFLOW_IO_MODE=record")
            continue

        details.append({
            "id": row["id"], "reference": row["reference_translation"],
            "arm_a_verse_only": str(arm_a), "arm_b_with_padaccheda": str(arm_b),
        })
        print(f"\n=== {row['id']} ===")
        print(f"reference: {row['reference_translation']}")
        print(f"arm A (verse only):      {str(arm_a)[:300]}")
        print(f"arm B (with padaccheda): {str(arm_b)[:300]}")

    print(f"\n{SMALL_SAMPLE_NOTE}")
    print("Adequacy is scored by reading these pairs; at n=5 no automated "
          "judge would be meaningful.")
    write_results("meaning_ab", {"n": len(details), "details": details,
                                 "note": SMALL_SAMPLE_NOTE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
