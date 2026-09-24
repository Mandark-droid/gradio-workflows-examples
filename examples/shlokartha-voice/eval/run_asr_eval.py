"""Synthetic-reciter CER, measured on IAST.

The spec's 50 verses by 3 human reciters do not exist. This synthesizes
5 verses x 3 voices with a public TTS Space, transcribes them, and reports
character error rate on IAST (the ASR returns Devanagari, transliterated
here before comparison, to match the spec's evaluation table). Synthetic
audio has none of the melisma or elongated vowels of real chanting, so this
number is a FLOOR, not field performance.

IMPORTANT — verified limitation: as of last verification, the upstream
`/transcribe_sanskrit_audio` endpoint returns an EMPTY transcription for
every one of this harness's TTS-generated clips; synthetic speech is not
what that ASR was trained on. A CER computed from empty hypotheses measures
that upstream behaviour, not this pipeline's quality, so do not read a
number produced this way as a pipeline score. The harness is kept anyway
because it is not broken — it works unchanged against real recorded audio.
Drop real recitations into the same manifest shape (eval/data/gita_5.json)
and this script produces a meaningful CER with no code changes.

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
from nodes.normalize import DEVANAGARI, IAST, transliterate
from nodes.source import asr

VOICES = [
    "hi-IN-MadhurNeural - hi-IN (Male)",
    "hi-IN-SwaraNeural - hi-IN (Female)",
    "en-IN-PrabhatNeural - en-IN (Male)",
]


def _to_iast(devanagari_text: str) -> str:
    """The ASR returns Devanagari; the spec's CER row is on IAST."""
    text = devanagari_text.strip()
    if not text:
        return ""
    return transliterate(text, DEVANAGARI, IAST)


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

            hypothesis_iast = _to_iast(hypothesis)
            cer = character_error_rate(row["iast"], hypothesis_iast)
            scores.append(cer)
            details.append({"id": row["id"], "voice": voice, "cer": cer,
                            "hypothesis": hypothesis,
                            "hypothesis_iast": hypothesis_iast})
            print(f"| {row['id']} | {voice} | {cer:.3f} |")

    if not scores:
        print("\nNo results. Re-run with WORKFLOW_IO_MODE=record to create fixtures.")
        return 1

    if all(not d["hypothesis"].strip() for d in details):
        print(
            "\nAll transcriptions were empty. This measures the upstream "
            "ASR's behaviour on synthetic speech, not this pipeline. Supply "
            "real recitations to obtain a usable number."
        )
        write_results("asr", {"mean_cer": None, "n": len(scores), "details": details,
                              "note": SMALL_SAMPLE_NOTE, "audio": "synthetic",
                              "all_hypotheses_empty": True})
        return 0

    mean = sum(scores) / len(scores)
    print(f"\nMean synthetic-reciter CER: {mean:.3f} over {len(scores)} clips")
    print("Synthetic voices, not human reciters — this is a floor, not field CER.")
    print(SMALL_SAMPLE_NOTE)
    write_results("asr", {"mean_cer": mean, "n": len(scores), "details": details,
                          "note": SMALL_SAMPLE_NOTE, "audio": "synthetic"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
