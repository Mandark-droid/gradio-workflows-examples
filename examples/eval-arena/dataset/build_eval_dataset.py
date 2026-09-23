"""Build and publish the 5-row public eval dataset.

One row per scorer branch, so the minimum data exercises every path:
exact match, numeric tolerance, JSON validity, free-form judging, and a row
engineered to run past max_new_tokens so the truncated failure tag is real.

The spec called for 500 rows; reduced to 5 to hold the project near zero
cost. Everything downstream reads its row count from the dataset, so
restoring 500 is a data change, not a code change.

    python dataset/build_eval_dataset.py --push owner/name
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.jsonl"

REQUIRED_COLUMNS = ["id", "task_type", "prompt", "gold", "meta"]
ANSWER_TYPES = ["exact", "numeric", "json", "free-form", "truncation"]


def load_rows() -> list[dict]:
    rows = []
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def push(repo_id: str) -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Export it and retry; it is never stored.")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(ROWS),
        path_in_repo="data/train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    card = (
        "---\nlicense: apache-2.0\nsize_categories:\n- n<1K\n---\n\n"
        "# gr.Workflow Eval Arena — sample rows\n\n"
        "Five rows, one per scorer branch, for the "
        "[gr.Workflow eval arena example]"
        "(https://github.com/Mandark-droid/gradio-workflows-examples).\n\n"
        "| column | meaning |\n| --- | --- |\n"
        "| `id` | stable row identifier |\n"
        "| `task_type` | exact, numeric, json, free-form, truncation |\n"
        "| `prompt` | the prompt sent to every candidate |\n"
        "| `gold` | reference answer |\n| `meta` | free-text note |\n\n"
        "Deliberately tiny: it exists to exercise every scoring path, not to "
        "rank models. Any conclusion about model quality drawn from five rows "
        "is not statistically meaningful.\n"
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed: https://hf.co/datasets/{repo_id}")
    print("Now set dataset.dataset_id in candidates.yaml to this id.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", metavar="REPO_ID")
    args = parser.parse_args()

    rows = load_rows()
    print(f"{len(rows)} rows, task types: {[r['task_type'] for r in rows]}")
    return push(args.push) if args.push else 0


if __name__ == "__main__":
    sys.exit(main())
