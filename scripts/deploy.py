"""Upload an example directory to a Hugging Face Space.

The example directory IS the Space root, so this is a plain folder upload with
no build step. Refuses to run if the secret scanner finds anything, if the
example directory has uncommitted changes (running the app locally rewrites
workflow.json — see the "A note for contributors" section of the example's
README), or if no Hugging Face credential can be found.

    python scripts/deploy.py --example shlokartha-voice --space owner/name
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Allow `python scripts/deploy.py ...` (invoked directly, not via `-m`) by
# making the repo root importable so `scripts.check_no_secrets` resolves.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_no_secrets import load_deny_terms, scan_repo

EXAMPLES = {
    "shlokartha-voice": ROOT / "examples" / "shlokartha-voice",
    "eval-arena": ROOT / "examples" / "eval-arena",
}

SKIP_DIRS = {"tests", "fixtures", "__pycache__", ".pytest_cache", "results", "runs"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def build_file_list(example_dir: Path) -> list[Path]:
    files = []
    for path in sorted(example_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(example_dir).parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        files.append(path)
    return files


def _resolve_token() -> str | None:
    """HF_TOKEN if set, else the credential huggingface_hub already stores.

    Nothing is defaulted and no secret is ever printed — get_token() reads the
    credential the hub CLI wrote, so the token never passes through argv or logs.
    """
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def deploy(example: str, space_id: str, dry_run: bool) -> int:
    example_dir = EXAMPLES.get(example)
    if example_dir is None or not example_dir.exists():
        print(f"Unknown example: {example}. Known: {sorted(EXAMPLES)}")
        return 2

    findings = scan_repo(ROOT, load_deny_terms(ROOT))
    if findings:
        for path, finding in findings:
            print(f"{path.relative_to(ROOT)}:{finding.line_no}: {finding.kind}")
        print("\nRefusing to deploy: secret scanner found issues.")
        return 1

    files = build_file_list(example_dir)
    print(f"{len(files)} file(s) would be uploaded to {space_id}:")
    for path in files:
        print(f"  {path.relative_to(example_dir)}")

    if dry_run:
        print("\nDry run — nothing uploaded.")
        return 0

    token = _resolve_token()
    if not token:
        print(
            "\nNo Hugging Face credential found: HF_TOKEN is not set and "
            "huggingface_hub has no stored credential either. Run "
            "`huggingface-cli login` or export HF_TOKEN, then retry."
        )
        return 2

    # gr.Workflow rewrites workflow.json when the app runs locally (drops
    # geometry and some `required` flags — upstream beta behaviour). Checked
    # here, after the token is confirmed but before anything is uploaded, so
    # a machine-rewritten graph never ships silently.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(example_dir)],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("Refusing to deploy: uncommitted changes under the example.")
        print(dirty)
        print(
            "\nNote: running the app locally rewrites workflow.json. "
            "Commit deliberately, or `git checkout --` it, then retry."
        )
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"authenticated as: {api.whoami().get('name')}")
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    # Upload exactly the files the dry run previewed. `ignore_patterns` used
    # to be matched against the full relative path (fnmatch semantics), so a
    # pattern like "results/*" never matched "eval/results/x.json" — that
    # file was excluded from the preview above but uploaded anyway. Deriving
    # the allow-list from build_file_list keeps upload and preview identical.
    allow = [str(p.relative_to(example_dir)).replace("\\", "/") for p in files]
    api.upload_folder(
        folder_path=str(example_dir),
        repo_id=space_id,
        repo_type="space",
        allow_patterns=allow,
    )
    print(f"\nDeployed: https://hf.co/spaces/{space_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, choices=sorted(EXAMPLES))
    parser.add_argument("--space", required=True, help="owner/name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return deploy(args.example, args.space, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
