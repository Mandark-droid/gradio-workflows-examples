"""Publish a completed run to a public Hub dataset.

Each record ties results to the exact graph and candidate configuration that
produced them, via workflow_json_hash and candidate_config_hash.

    python dataset/push_results.py --checkpoint runs/x.ckpt.jsonl --repo owner/name
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_token() -> str | None:
    """HF_TOKEN if set, else the credential huggingface_hub already stores."""
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", default=time.strftime("run_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    source = Path(args.checkpoint)
    if not source.exists():
        print(f"No checkpoint at {source}")
        return 2

    token = _resolve_token()
    if not token:
        print(
            "No token found: HF_TOKEN is not set and huggingface_hub has no "
            "stored credential. Run `huggingface-cli login` or export "
            "HF_TOKEN and retry; it is never stored by this script."
        )
        return 2

    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = entry.get("payload", {})
        pairs = payload.get("pairs") or []
        for model_id, scored in (payload.get("per_model") or {}).items():
            # `pairs` survives the driver's merge of the three /scores
            # subjects but died here — the spec's schema names
            # judge_outcomes, and dropping every judge verdict at the last
            # step is what made a published run unauditable.
            judge_outcomes = []
            for pair in pairs:
                a, b = pair.get("a"), pair.get("b")
                if model_id not in (a, b):
                    continue
                opponent = b if model_id == a else a
                judge_outcomes.append({
                    "opponent": opponent,
                    "winner": pair.get("winner"),
                    "agreed": pair.get("agreed"),
                    "confidence": pair.get("confidence"),
                })
            records.append({
                "run_id": args.run_id,
                "row_id": payload.get("row_id", ""),
                "model_id": model_id,
                "score": scored.get("score"),
                "failure_tag": scored.get("failure_tag"),
                "latency_ms": (payload.get("latency") or {}).get(model_id, {}).get("latency_ms"),
                "tokens_out": (payload.get("latency") or {}).get(model_id, {}).get("tokens_out"),
                "wins": (payload.get("wins") or {}).get(model_id, 0),
                "judge_outcomes": judge_outcomes,
                "workflow_json_hash": payload.get("workflow_json_hash", ""),
                "candidate_config_hash": payload.get("candidate_config_hash", ""),
            })

    out = source.parent / f"{args.run_id}.results.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo=f"runs/{args.run_id}.jsonl",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Pushed {len(records)} records: https://hf.co/datasets/{args.repo}")
    print("Note: results from a 5-row run are not statistically meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
