---
title: Hot-Swappable Model Eval Arena
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
python_version: "3.11"
pinned: false
hf_oauth: true
---

# Hot-Swappable Model Eval Arena

One eval row, three candidate models, deterministic scoring plus a
position-controlled pairwise judge — with every intermediate output visible on
the canvas instead of buried in a JSONL dump.

Built with [`gr.Workflow`](https://gradio.app/guides/workflows).

## Bring your own token

The three candidates and the judge call Inference Providers, which needs a
Hugging Face token. Paste one into the **HF Token** box on the canvas and it is
used for that run, so you spend your own quota rather than the Space owner's.
A token with `inference.serverless.write` is enough. Without one, every
candidate returns an `error` failure tag and the row scores zero.

**Two honest caveats.** `gr.Workflow` port types have no password variant, so
the box renders as ordinary text — your token is visible on screen while you
type it. And the canvas autosaves the graph for anyone with write access, so if
you own this Space, do not save after entering a token: it would be written
into `workflow.json`. The committed file ships with that field empty and it
should stay that way.

## Swapping a candidate

Edit one line in `candidates.yaml`:

```yaml
candidates:
  - {slot: a, model_id: your/model, max_new_tokens: 512, greedy: true}
```

Nothing else changes. The scorer, judge and aggregate nodes never name a
model — they read slot letters — so a swap touches no graph node and no
downstream code.

## What the canvas shows

`row_idx` → dataset → prompt builder → three candidates in parallel →
deterministic scorer and pairwise judge → aggregate → Scores, Latency,
Verdict.

Every candidate emits the same envelope, so the scorer never special-cases a
model:

```json
{"model_id": "...", "output": "...", "latency_ms": 0, "tokens_out": 0, "reasoning_chars": 0, "error": null}
```

## Scoring

Deterministic scores are the primary signal, dispatched on the gold answer
type: exact match after normalization, numeric within 1e-3 relative tolerance,
JSON parse plus comparison, and a truncation check. Free-form rows defer to
the judge.

The judge sees each pair twice with the positions swapped. A pair counts as a
win only when both orderings name the same model; if the judge picks the same
*position* twice it is showing position bias, and the pair is recorded as a
tie. The judge is from a fourth model family, out-of-family from every
candidate.

## Reasoning models, and why the budget is 1024

Three of the four candidates are reasoning models: they emit chain-of-thought
into `reasoning_content` and return only the final answer in `content`. Probed
live on the prompt "what is 7 times 23", Qwen3-8B spent 345 completion tokens
before answering; Llama-3.1-8B, which is not a reasoning model, spent 2.

A 512-token budget would therefore cut a reasoning model off mid-thought on a
hard row, return empty `content`, and score it `unparseable` — penalising a
model for thinking rather than for being wrong, and quietly destroying the
"same size, three families, fair fight" premise. Hence 1024, and hence
`reasoning_chars` in the envelope: you can see that a model was thinking
rather than stalling.

Read the latency column with this in mind. A reasoning model is not slower
because it is worse.

## Three things worth knowing

- **One endpoint, not three.** `gr.Workflow` gives one endpoint per connected
  component, so `/scores` returns Scores, Latency and Verdict together. The
  driver calls it once per row.
- **Parallel on the canvas, sequential through the API.** Candidates genuinely
  run in parallel when you hit Run; through the generated API they run one
  after another. Per-row latency via the driver is roughly the sum.
- **The bundled dataset has five rows.** It exists to exercise every scoring
  path, not to rank models. Any conclusion drawn from it is not statistically
  meaningful, and the analysis output says so.

## Record, replay, live

Every network call routes through one small layer with three modes, set by
`ARENA_IO_MODE`: `replay` (the default) reads recorded fixtures and never
touches the network, `record` calls live and saves the response, and `live`
calls without saving. The default is `replay` so tests and local development
can never spend money by accident.

`app.py` sets the mode to `live` for exactly this reason: a deployed Space has
no fixtures, and inheriting the replay default would make every call fail with
`FixtureMissing` — quietly, since the failure is caught and reported as an
ordinary error result.
