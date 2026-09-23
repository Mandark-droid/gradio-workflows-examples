"""Prompt construction and gold normalization.

One shared template per task_type, applied identically to every candidate, so
a model is never advantaged by prompt wording. Chat templating happens inside
each candidate via the provider's chat endpoint.
"""
from __future__ import annotations

import json

TEMPLATES = {
    "exact": "{prompt}\n\nAnswer with the exact value only, no explanation.",
    "numeric": "{prompt}\n\nAnswer with the final number only, no units, no words.",
    "json": "{prompt}\n\nReply with a single valid JSON object only. No prose, no code fences.",
    "free-form": "{prompt}",
    "truncation": "{prompt}",
}

_FALLBACK = "{prompt}"


def prompt_builder(task_type: str, prompt: str) -> str:
    template = TEMPLATES.get(str(task_type).strip().lower(), _FALLBACK)
    return template.format(prompt=str(prompt or "").strip())


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def gold_extract(gold: str, task_type: str) -> str:
    answer_type = str(task_type or "").strip().lower()
    raw = str(gold or "").strip()

    if answer_type == "exact":
        value = _normalize(raw)
    elif answer_type == "numeric":
        value = raw
    elif answer_type == "json":
        try:
            value = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            value = raw
    else:
        value = raw

    return json.dumps({"gold": value, "answer_type": answer_type}, ensure_ascii=False)
