"""Answer the spec's open question: does feeding padaccheda improve meaning?

Both arms call /interpret_sanskrit_verse (config["shlokartha"]["api_name"]) —
the actual commentary generator, confirmed live: its output index 6 holds the
full commentary. The old arm A called /sanskrit_commentator instead; we now
know that endpoint only returns a prompt template, not a commentary, so the
old A-vs-B comparison was meaningless. Both arms are re-run against the real
generator and differ ONLY in whether the padaccheda is injected into the
system prompt:

Arm A: plain system prompt, no word split.
Arm B: system prompt naming the padaccheda words, which is the only way this
       Space accepts them.

If arm B wins, wire the padaccheda into op_meaning's system_prompt in the
graph.

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

SYSTEM_PROMPT_PLAIN = (
    "You are a careful Sanskrit commentator. Separate literal meaning, "
    "grammar, translation, and cultural context. Do not invent textual "
    "details."
)

SYSTEM_PROMPT_WITH_PADACCHEDA = (
    "You are a careful Sanskrit commentator. Separate literal meaning, "
    "grammar, translation, and cultural context. Do not invent textual "
    "details. The verse has been split into these words (padaccheda): "
    "{words}. Use the split to resolve compounds before commenting."
)


def main() -> int:
    _utf8_stdout()
    cfg = hf_io.load_config()["shlokartha"]
    source = cfg["source"]
    result_index = cfg["meaning_output_index"]
    rows, details = load_testset(), []

    for row in rows:
        normalized = normalize(json.dumps({"text": row["iast"], "source": "typed"}))
        splits = json.loads(sandhi_split(normalized))["splits"]
        words = splits[0]["words"] if splits else []

        try:
            arm_a = hf_io.call_space(
                cfg["space_id"], cfg["api_name"],
                SYSTEM_PROMPT_PLAIN, source, row["devanagari"],
                256, 0.7, 0.95, 1.1, False, 42, False,
                result_index=result_index,
            )
            arm_b = hf_io.call_space(
                cfg["space_id"], cfg["api_name"],
                SYSTEM_PROMPT_WITH_PADACCHEDA.format(words=" ".join(words)),
                source, row["devanagari"],
                256, 0.7, 0.95, 1.1, False, 42, False,
                result_index=result_index,
            )
        except hf_io.FixtureMissing:
            print(f"{row['id']}: no fixture — run with WORKFLOW_IO_MODE=record")
            continue

        details.append({
            "id": row["id"], "reference": row["reference_translation"],
            "arm_a_no_padaccheda": str(arm_a), "arm_b_with_padaccheda": str(arm_b),
        })
        print(f"\n=== {row['id']} ===")
        print(f"reference: {row['reference_translation']}")
        print(f"arm A (no padaccheda):   {str(arm_a)[:300]}")
        print(f"arm B (with padaccheda): {str(arm_b)[:300]}")

    print(f"\n{SMALL_SAMPLE_NOTE}")
    print("Adequacy is scored by reading these pairs; at n=5 no automated "
          "judge would be meaningful.")
    write_results("meaning_ab", {"n": len(details), "details": details,
                                 "note": SMALL_SAMPLE_NOTE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
