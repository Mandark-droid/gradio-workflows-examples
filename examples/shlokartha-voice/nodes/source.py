"""ASR and source selection.

gr.Workflow has no conditional edges, so "typed text skips ASR" cannot be a
branch. Instead both references always feed select_source, which prefers
non-empty typed text — and asr returns "" for empty audio instead of raising,
because a raised exception here fails the entire run.
"""
from __future__ import annotations

import json

from nodes import hf_io


def asr(audio_path: str | None) -> str:
    """Transcribe audio. Returns "" for empty input and never raises."""
    if not audio_path or not str(audio_path).strip():
        return ""

    cfg = hf_io.load_config()["asr"]
    try:
        from gradio_client import handle_file

        result = hf_io.call_space(
            cfg["space_id"], cfg["api_name"], handle_file(str(audio_path)),
            result_index=0,
        )
    except Exception:
        # A dead Space must not fail the graph; downstream sees empty text and
        # the user can type the verse instead.
        return ""
    return str(result or "").strip()


NO_VERSE_MESSAGE = (
    "No verse was detected. Record again more clearly, or type the verse "
    "into the text box."
)


def shlokartha_meaning(verse: str) -> str:
    """Ask the Slokartha Space to interpret a verse.

    This is an fn node rather than a space node for one reason: gr.Workflow has
    no conditional edges, so this node runs on every call — including runs where
    ASR produced nothing. A space node handed an empty shloka raises and takes
    the whole endpoint down with it. Guarding here degrades to a readable
    message instead.
    """
    text = str(verse or "").strip()
    if not text:
        return NO_VERSE_MESSAGE

    cfg = hf_io.load_config()["shlokartha"]
    try:
        return str(
            hf_io.call_space(
                cfg["space_id"],
                cfg["api_name"],
                "You are a careful Sanskrit commentator. Separate literal "
                "meaning, grammar, translation, and cultural context. Do not "
                "invent textual details.",
                cfg.get("source", "bhagavad_gita"),
                text,
                256, 0.7, 0.95, 1.1, False, 42, False,
                result_index=int(cfg.get("meaning_output_index", 6)),
            )
            or ""
        ).strip() or NO_VERSE_MESSAGE
    except Exception as exc:
        return f"Meaning unavailable: {type(exc).__name__}"


def select_source(asr_text: str, typed_text: str) -> str:
    typed = str(typed_text or "").strip()
    transcribed = str(asr_text or "").strip()

    if typed:
        return json.dumps({"text": typed, "source": "typed"}, ensure_ascii=False)
    if transcribed:
        return json.dumps({"text": transcribed, "source": "asr"}, ensure_ascii=False)
    return json.dumps({"text": "", "source": "none"}, ensure_ascii=False)
