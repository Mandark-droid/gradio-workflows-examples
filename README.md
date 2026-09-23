# Gradio Workflow Examples

Worked examples of [`gr.Workflow`](https://gradio.app/guides/workflows) Spaces:
graphs of nodes — local functions, Hugging Face Spaces, and hosted models —
wired together on a canvas, with the whole graph deployable as a single
Hugging Face Space.

## Examples

- **`examples/shlokartha-voice/`** — a Sanskrit verse pipeline. ASR or typed
  text in; normalization, sandhi split, metre detection, meaning, narration
  and illustration out. A second, cheap text-only endpoint exposes just the
  metre detector. See that directory's own README for the full mechanism and
  its Space card.
- **`examples/eval-arena/`** — a hot-swappable model eval arena. A dataset
  row runs through three candidate models in parallel, scored by a
  deterministic scorer plus a position-controlled pairwise judge, all
  exposed as one REST endpoint, `/scores`. Graph, batch driver, checkpointed
  resume, run analysis and results publisher are all built; the bundled
  public dataset is the 5-row
  [`kshitijthakkar/gradio-workflow-eval-arena`](https://hf.co/datasets/kshitijthakkar/gradio-workflow-eval-arena),
  which exists to exercise every scoring path rather than to rank models. See
  that directory's own README for the full mechanism, the swap procedure and
  its Space card.

## Live Spaces

Both examples are deployed and were validated against their real endpoints:

- [`kshitijthakkar/shlokartha-voice-workflow`](https://hf.co/spaces/kshitijthakkar/shlokartha-voice-workflow)
  — `/verse` returns all six outputs in about 26 s (most of it one CPU
  generation on an external Space); `/metre_only` returns in under 4 s.
- [`kshitijthakkar/eval-arena-workflow`](https://hf.co/spaces/kshitijthakkar/eval-arena-workflow)
  — `/scores` runs one dataset row through three candidates and the judge in
  about 10 s. Paste your own Hugging Face token into the canvas's **HF Token**
  box so the run bills your quota rather than the Space owner's.

A five-row batch through the arena, with results published to
[`kshitijthakkar/eval-arena-runs`](https://hf.co/datasets/kshitijthakkar/eval-arena-runs),
cost roughly USD 0.07. Any ranking drawn from five rows is not statistically
meaningful, and the analysis output says so on every run.

Each example directory **is** its Space root: the app, the workflow graph,
requirements and Space card all live directly under it, so deployment is a
plain folder upload with no build step.

## Running the tests

Each example defines its own top-level `nodes` package, so its tests have to
run from inside that example's own directory — collecting every example's
tests together from the repo root would make `import nodes` ambiguous between
examples. That's why the repo-root `pytest.ini` deliberately scopes to just
`tests/` (the shared scripts' tests), and each example carries its own
`pytest.ini` scoped to its own `tests/`.

Run every suite at once:

    python scripts/run_tests.py

or run one suite by hand:

    .venv/Scripts/python.exe -m pytest tests/ -v
    cd examples/shlokartha-voice && ../../.venv/Scripts/python.exe -m pytest tests/ -v

## Deploying an example

    python scripts/deploy.py --example shlokartha-voice --space owner/name

This uploads the example directory as-is to a Hugging Face Space (creating it
if it doesn't exist). It refuses to run if `scripts/check_no_secrets.py` finds
anything, or if the example directory has uncommitted changes — running the
app locally rewrites `workflow.json`, and this guard exists so a
machine-rewritten graph never ships in place of the hand-authored one. Add
`--dry-run` to see the file list without uploading anything.

Credentials come from `HF_TOKEN` if it's set in the environment, otherwise
whatever `huggingface_hub` already has stored locally (e.g. from
`huggingface-cli login`) — never a default, and never printed.

## Running the eval arena driver

Once `examples/eval-arena` is deployed as a Space, drive it from that
directory:

    cd examples/eval-arena
    ../../.venv/Scripts/python.exe driver/run_batch.py --space owner/name --rows 5 --max-calls 60
    ../../.venv/Scripts/python.exe driver/analyse.py --checkpoint runs/<file>.ckpt.jsonl
    ../../.venv/Scripts/python.exe dataset/push_results.py --checkpoint runs/<file>.ckpt.jsonl --repo owner/eval-arena-runs

`run_batch.py` checkpoints every row as it completes and refuses to spend
past `--max-calls`; re-running the same command with the same
`--checkpoint` does nothing once every row is done. `analyse.py` reports mean
score, bootstrap CI, Bradley-Terry ratings and latency percentiles, and
flags them as not statistically meaningful at the bundled dataset's 5 rows.
`push_results.py` publishes the scored records to a public Hub dataset,
tagged with the workflow and candidate config hashes that produced them.
