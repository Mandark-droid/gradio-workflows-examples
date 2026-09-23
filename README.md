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
- **`examples/eval-arena/`** — *in progress.* Planned as a Space that runs a
  row of data through several candidate models and scores them, exposed as a
  REST endpoint. Not yet built in this repo.

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
