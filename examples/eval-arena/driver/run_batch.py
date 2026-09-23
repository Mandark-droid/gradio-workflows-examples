"""Run every dataset row through the deployed graph.

Calls only /scores, because aggregate embeds verdict and latency — with one
endpoint per connected component that is the only available shape, not an
optimization.

    python driver/run_batch.py --space owner/name --rows 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driver.checkpoint import append, load_done
from nodes import arena_io
from gradio_client import Client

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"

CALLS_PER_ROW = 9          # 3 candidates + 6 position-swapped judge calls
USD_PER_CALL = 0.0015      # conservative upper bound for 8B-class models
MAX_RETRIES = 2


def project_cost(n_rows: int) -> tuple[int, float]:
    calls = n_rows * CALLS_PER_ROW
    return calls, round(calls * USD_PER_CALL, 4)


def workflow_hash() -> str:
    path = HERE.parent / "workflow.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _resolve_token() -> str | None:
    """HF_TOKEN if set, else the credential huggingface_hub already stores.

    Nothing is defaulted and no secret is printed — get_token() reads the
    credential the hub CLI wrote, so it never passes through argv or logs.
    """
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


def _as_dict(item) -> dict:
    """A subject arrives as a JSON string or an already-parsed object."""
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return item if isinstance(item, dict) else {}


def _merge_subjects(result) -> dict:
    """/scores returns three subjects and no single one carries everything.

    scores has `wins` but no latency or pairs; verdict has latency and pairs
    but no wins. Keeping only result[0] silently empties every latency and
    judge figure in the analysis, so the record is assembled from all three.
    """
    if not isinstance(result, (list, tuple)):
        result = [result]
    scores = _as_dict(result[0]) if len(result) > 0 else {}
    latency = _as_dict(result[1]) if len(result) > 1 else {}
    verdict = _as_dict(result[2]) if len(result) > 2 else {}

    merged = dict(scores)
    merged["latency"] = latency or verdict.get("latency") or {}
    merged["pairs"] = verdict.get("pairs") or []
    return merged


def _run_row(client, row_index: int, hf_token: str = "") -> dict:
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = client.predict(row_index, hf_token, api_name="/scores")
            return _merge_subjects(result)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    # Never drop a row silently — persist the failure instead.
    return {"row_index": row_index, "error": last_error, "failure_tag": "error"}


def run(
    space_id: str, n_rows: int, checkpoint_path: Path,
    max_calls: int, workers: int, assume_yes: bool,
) -> int:
    done = load_done(checkpoint_path)
    todo = [i for i in range(n_rows) if i not in done]

    calls, usd = project_cost(len(todo))
    print(f"Space:      {space_id}")
    print(f"Rows:       {len(todo)} to run, {len(done)} already checkpointed")
    print(f"Calls:      ~{calls} ({CALLS_PER_ROW} per row)")
    print(f"Estimate:   ~USD {usd}")
    print(f"Ceiling:    {max_calls}")

    if not todo:
        print("\nNothing to do — every row is checkpointed.")
        return 0

    if calls > max_calls:
        print(f"\nRefusing to run: {calls} calls exceeds --max-calls {max_calls}.")
        return 1

    if not assume_yes:
        if input("\nProceed? [y/N] ").strip().lower() != "y":
            print("Aborted. Nothing spent.")
            return 1

    token = _resolve_token()
    if not token:
        print(
            "No token found: HF_TOKEN is not set and huggingface_hub has no "
            "stored credential. Run `huggingface-cli login` or export "
            "HF_TOKEN and retry; it is never stored by this script."
        )
        return 2

    client = Client(space_id, token=token)
    wf_hash, cfg_hash = workflow_hash(), arena_io.config_hash()

    def _task(row_index: int) -> tuple[int, dict]:
        payload = _run_row(client, row_index, token)
        payload["workflow_json_hash"] = wf_hash
        payload["candidate_config_hash"] = cfg_hash
        return row_index, payload

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row_index, payload in pool.map(_task, todo):
            append(checkpoint_path, row_index, payload)
            state = "error" if payload.get("error") else "ok"
            print(f"  row {row_index}: {state}")

    print(f"\nCheckpoint: {checkpoint_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="owner/name")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        RUNS / f"run_{time.strftime('%Y%m%d_%H%M%S')}.ckpt.jsonl"
    )
    return run(args.space, args.rows, checkpoint, args.max_calls,
               args.workers, args.yes)


if __name__ == "__main__":
    sys.exit(main())
