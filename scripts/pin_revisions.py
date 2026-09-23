"""Resolve every model id in candidates.yaml to its current commit sha.

Pinning is reproducible and auditable rather than hand-transcribed.

    python scripts/pin_revisions.py --example eval-arena
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def pin(example: str) -> int:
    path = ROOT / "examples" / example / "candidates.yaml"
    if not path.exists():
        print(f"No candidates.yaml at {path.relative_to(ROOT)}")
        return 2

    import huggingface_hub
    from huggingface_hub import HfApi

    # HF_TOKEN is often unset on a dev machine even when a valid credential
    # is already stored (e.g. via `huggingface-cli login`). Prefer the env
    # var when present, else fall back to the stored credential rather than
    # requiring the env var to exist.
    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    api = HfApi(token=token)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    entries = list(config["candidates"]) + [config["judge"]]
    for entry in entries:
        model_id = entry["model_id"]
        try:
            info = api.model_info(model_id)
            entry["revision"] = info.sha
            print(f"{model_id} -> {info.sha}")
        except Exception as exc:
            print(f"{model_id}: could not resolve ({type(exc).__name__}) — left unpinned")

    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"\nWrote {path.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="eval-arena")
    return pin(parser.parse_args().example)


if __name__ == "__main__":
    sys.exit(main())
