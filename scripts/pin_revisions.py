"""Report the Hub commit sha each configured model currently resolves to.

This is PROVENANCE, not a pin. Inference Providers expose no way to request a
specific revision — `InferenceClient` accepts no revision argument — so the
provider serves whatever weights it holds. Recording the sha lets you notice
afterwards that a model changed; it cannot make a run reproducible.

Deliberately writes a separate file rather than editing candidates.yaml, so it
can never strip that file's explanatory comments.

    python scripts/pin_revisions.py --example eval-arena
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _resolve_token() -> str | None:
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def report(example: str) -> int:
    config_path = ROOT / "examples" / example / "candidates.yaml"
    if not config_path.exists():
        print(f"No candidates.yaml at {config_path.relative_to(ROOT)}")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=_resolve_token())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    entries = list(config["candidates"]) + [config["judge"]]
    resolved: dict[str, str] = {}
    for entry in entries:
        model_id = entry["model_id"]
        try:
            resolved[model_id] = api.model_info(model_id).sha
            print(f"{model_id} -> {resolved[model_id]}")
        except Exception as exc:
            resolved[model_id] = ""
            print(f"{model_id}: could not resolve ({type(exc).__name__})")

    out = ROOT / "examples" / example / "model_provenance.json"
    out.write_text(
        json.dumps(
            {
                "note": (
                    "Commit shas the Hub reported when this file was written. "
                    "Inference Providers cannot be asked for a specific "
                    "revision, so these do NOT pin the served weights."
                ),
                "resolved": resolved,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="eval-arena")
    return report(parser.parse_args().example)


if __name__ == "__main__":
    sys.exit(main())
