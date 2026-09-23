"""Synthetic-reciter CER.

The spec's 50 verses by 3 human reciters do not exist. This synthesizes
5 verses x 3 voices with a public TTS Space, transcribes them, and reports
CER. Synthetic audio has none of the melisma or elongated vowels of real
chanting, so this number is a FLOOR, not field performance.

Costs nothing (public CPU Spaces) but makes ~30 network calls:
    WORKFLOW_IO_MODE=record python eval/run_asr_eval.py
Thereafter it replays for free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import SMALL_SAMPLE_NOTE, character_error_rate, load_testset, write_results
from nodes import hf_io
from nodes.hf_io import _utf8_stdout
from nodes.source import asr

VOICES = [
    "hi-IN-MadhurNeural - hi-IN (Male)",
    "hi-IN-SwaraNeural - hi-IN (Female)",
    "en-IN-PrabhatNeural - en-IN (Male)",
]


def main() -> int:
    _utf8_stdout()
    cfg = hf_io.load_config()
    rows, scores, details = load_testset(), [], []

    print("| id | voice | CER |")
    print("| --- | --- | --- |")
    for row in rows:
        for voice in VOICES:
            try:
                audio = hf_io.call_space(
                    cfg["tts"]["space_id"], cfg["tts"]["api_name"],
                    row["devanagari"], voice, 0, 0, result_index=0,
                )
                hypothesis = asr(audio)
            except hf_io.FixtureMissing:
                print(f"| {row['id']} | {voice} | no fixture — run with "
                      f"WORKFLOW_IO_MODE=record |")
                continue

            cer = character_error_rate(row["devanagari"], hypothesis)
            scores.append(cer)
            details.append({"id": row["id"], "voice": voice, "cer": cer,
                            "hypothesis": hypothesis})
            print(f"| {row['id']} | {voice} | {cer:.3f} |")

    if not scores:
        print("\nNo results. Re-run with WORKFLOW_IO_MODE=record to create fixtures.")
        return 1

    mean = sum(scores) / len(scores)
    print(f"\nMean synthetic-reciter CER: {mean:.3f} over {len(scores)} clips")
    print("Synthetic voices, not human reciters — this is a floor, not field CER.")
    print(SMALL_SAMPLE_NOTE)
    write_results("asr", {"mean_cer": mean, "n": len(scores), "details": details,
                          "note": SMALL_SAMPLE_NOTE, "audio": "synthetic"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
