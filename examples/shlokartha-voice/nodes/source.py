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
        result = hf_io.call_space(
            cfg["space_id"], cfg["api_name"], str(audio_path), result_index=0
        )
    except Exception:
        # A dead Space must not fail the graph; downstream sees empty text and
        # the user can type the verse instead.
        return ""
    return str(result or "").strip()


def select_source(asr_text: str, typed_text: str) -> str:
    typed = str(typed_text or "").strip()
    transcribed = str(asr_text or "").strip()

    if typed:
        return json.dumps({"text": typed, "source": "typed"}, ensure_ascii=False)
    if transcribed:
        return json.dumps({"text": transcribed, "source": "asr"}, ensure_ascii=False)
    return json.dumps({"text": "", "source": "none"}, ensure_ascii=False)
