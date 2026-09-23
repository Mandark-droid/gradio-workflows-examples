"""ASR and source selection.

gr.Workflow has no conditional edges, so "typed text skips ASR" cannot be a
branch. Instead both references always feed select_source, which prefers
non-empty typed text — and asr returns "" for empty audio instead of raising,
because a raised exception here fails the entire run.
"""
from __future__ import annotations

import json

from nodes import hf_io


# asr() has no way to signal failure through a boolean or a second return
# value without changing its signature (every fn node here takes and returns
# plain strings). This sentinel is the second channel: a value select_source
# recognises and strips before anything downstream ever sees it. It starts
# with a NUL byte so it can never collide with genuine transcribed text.
ASR_ERROR_SENTINEL = "\x00ASR_ERROR"


def asr(audio_path: str | None) -> str:
    """Transcribe audio. Returns "" for empty input, ASR_ERROR_SENTINEL for a
    backend failure, and never raises."""
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
        # A dead Space must not fail the graph, but silently returning "" is
        # indistinguishable from silent audio and wrongly blames the user.
        # Report the failure via the sentinel so select_source can tell them
        # apart.
        return ASR_ERROR_SENTINEL
    return str(result or "").strip()


NO_VERSE_MESSAGE = (
    "No verse was detected. Record again more clearly, or type the verse "
    "into the text box."
)

ASR_ERROR_MESSAGE = (
    "The transcription service failed, so no verse could be read from your "
    "recording. This is not a problem with your audio — please type the "
    "verse into the text box instead."
)

LOW_TRUST_PREFIX = (
    "⚠️ Low confidence: the verse did not scan cleanly, so the "
    "transcription may be wrong and this reading may not be reliable.\n\n"
)


def shlokartha_meaning(verse: str, metre_json: str = "", normalized_json: str = "") -> str:
    """Ask the Slokartha Space to interpret a verse.

    This is an fn node rather than a space node for one reason: gr.Workflow has
    no conditional edges, so this node runs on every call — including runs where
    ASR produced nothing. A space node handed an empty shloka raises and takes
    the whole endpoint down with it. Guarding here degrades to a readable
    message instead.

    `metre_json` (from chandas_detect) drives the low-trust warning; when
    confidence is below the configured threshold the transcription likely
    drifted and the commentary is prefixed accordingly. `normalized_json`
    (from normalize) carries the source that produced this verse, so an ASR
    backend failure can be reported honestly instead of blaming the user.
    """
    text = str(verse or "").strip()
    if not text:
        try:
            source = json.loads(normalized_json or "{}").get("source", "")
        except (json.JSONDecodeError, TypeError):
            source = ""
        return ASR_ERROR_MESSAGE if source == "asr_error" else NO_VERSE_MESSAGE

    low_trust = False
    try:
        confidence = float(json.loads(metre_json or "{}").get("confidence", 1.0))
        low_trust = confidence < float(
            hf_io.load_config()["chandas"]["low_trust_below"]
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        low_trust = False

    cfg = hf_io.load_config()["shlokartha"]
    try:
        commentary = str(
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

    if commentary != NO_VERSE_MESSAGE and low_trust:
        commentary = LOW_TRUST_PREFIX + commentary
    return commentary


def select_source(asr_text: str, typed_text: str) -> str:
    typed = str(typed_text or "").strip()
    raw_asr = str(asr_text or "")
    asr_failed = raw_asr.startswith(ASR_ERROR_SENTINEL)
    transcribed = "" if asr_failed else raw_asr.strip()

    if typed:
        return json.dumps({"text": typed, "source": "typed"}, ensure_ascii=False)
    if transcribed:
        return json.dumps({"text": transcribed, "source": "asr"}, ensure_ascii=False)
    if asr_failed:
        return json.dumps({"text": "", "source": "asr_error"}, ensure_ascii=False)
    return json.dumps({"text": "", "source": "none"}, ensure_ascii=False)
