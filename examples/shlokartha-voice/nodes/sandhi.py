"""Padaccheda (word split) via sanskrit_parser, with a hard timeout.

Long compounds can blow up the search, so the parser runs in a worker thread
under a cap from config.yaml. On timeout or failure the verse is returned
unsplit and flagged, rather than failing the whole graph run.

Lexical scoring needs gensim and sentencepiece, which this CPU Space does not
carry. Scores are therefore rank-derived and labelled score_kind "rank".
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from indic_transliteration.sanscript import IAST, SLP1, transliterate

from nodes.hf_io import load_config

logging.getLogger("sanskrit_parser").setLevel(logging.ERROR)

_parser = None


def _silence_parser_logging() -> None:
    """The library's submodules set DEBUG on their own child loggers at import
    time, so setting only the parent's level has no effect — one split emitted
    roughly 392 KB of DEBUG output to stderr. Levels must be set on each child
    that exists once the import has happened."""
    for name in list(logging.root.manager.loggerDict):
        if name == "sanskrit_parser" or name.startswith("sanskrit_parser."):
            child = logging.getLogger(name)
            child.setLevel(logging.ERROR)
            child.propagate = False


def _get_parser():
    global _parser
    if _parser is None:
        from sanskrit_parser import Parser

        _silence_parser_logging()
        _parser = Parser()
    return _parser


def _run_parser(text: str, top_k: int) -> list[list[str]]:
    """Return up to top_k splits, each a list of IAST words."""
    results: list[list[str]] = []
    for split in _get_parser().split(text, limit=top_k):
        words = [
            transliterate(str(word), SLP1, IAST) for word in split.split
        ]
        if words:
            results.append(words)
    return results


def sandhi_split(normalized_json: str) -> str:
    cfg = load_config()["sandhi"]
    timeout = float(cfg["timeout_seconds"])
    top_k = int(cfg["top_k"])

    try:
        iast = str(json.loads(normalized_json).get("iast", "")).strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        iast = str(normalized_json).strip()

    if not iast:
        return json.dumps({"splits": [], "truncated": False}, ensure_ascii=False)

    # Not a context manager: __exit__ would shutdown(wait=True) and block until
    # a runaway parse finished, defeating the cap. A thread cannot be killed, so
    # an overrunning parse is abandoned to finish on its own while the caller
    # returns immediately with the unsplit fallback.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        candidates = pool.submit(_run_parser, iast, top_k).result(timeout=timeout)
    except Exception:
        pool.shutdown(wait=False)
        return json.dumps(
            {
                "splits": [
                    {"words": iast.split(), "score": 1.0, "score_kind": "rank"}
                ],
                "truncated": True,
            },
            ensure_ascii=False,
        )
    pool.shutdown(wait=False)

    if not candidates:
        return json.dumps(
            {
                "splits": [
                    {"words": iast.split(), "score": 1.0, "score_kind": "rank"}
                ],
                "truncated": True,
            },
            ensure_ascii=False,
        )

    total = len(candidates)
    splits = [
        {
            "words": words,
            "score": round((total - rank) / total, 3),
            "score_kind": "rank",
        }
        for rank, words in enumerate(candidates)
    ]
    return json.dumps({"splits": splits, "truncated": False}, ensure_ascii=False)
