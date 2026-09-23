# Fixtures

Recorded responses for the record/replay layer in `nodes/arena_io.py`.
`ARENA_IO_MODE` defaults to `replay`, so these fixtures are used
automatically — tests and local development never touch the network unless
you explicitly opt into `record` or `live`.

Each file is named after `arena_io.fixture_key(kind, target, args)`: a sha256
of the call's shape, so an unchanged call always resolves to the same file
and a changed one (different model, prompt, or decode parameter) misses and
raises `FixtureMissing` in replay mode rather than silently reusing a stale
response.

## What is covered

These fixtures were recorded from dataset row 0 only (`load_rows()[0]`),
covering:

- The three candidate `chat` calls (`candidate_a`, `candidate_b`,
  `candidate_c`) for that row's prompt.
- The judge's pairwise verdicts (`pairwise_judge`) for that row. All three
  candidates answered identically on this row, so the six position-swapped
  judge calls collapsed to a single distinct fixture — `fixture_key` hashes
  the literal prompt text, and all six judge prompts were byte-identical.

That is 4 fixture files for what would otherwise be 9 live calls.

## What is NOT covered

- Every other dataset row (only row 0 is recorded).
- Any run where a candidate or the judge model is swapped in
  `candidates.yaml`.
- Error paths (rate limits, timeouts, malformed judge output) — those are
  covered by handcrafted fixtures inside the unit tests themselves, not by
  files here.

## Recording more

```
cd examples/eval-arena
ARENA_IO_MODE=record ../../.venv/Scripts/python.exe -c "
import json
from dataset.build_eval_dataset import load_rows
from nodes.prompts import prompt_builder, gold_extract
from nodes.candidate import candidate_a, candidate_b, candidate_c
from nodes.judge import pairwise_judge
row = load_rows()[0]
p = prompt_builder(row['task_type'], row['prompt'])
a, b, c = candidate_a(p), candidate_b(p), candidate_c(p)
pairwise_judge(a, b, c, p)
print('recorded')
"
```

Swap `load_rows()[0]` for another index to record a different row. This
needs a Hugging Face token (`HF_TOKEN` or a stored `huggingface-cli login`
credential) and spends real Inference Providers quota, so it is a deliberate,
manual step — never run automatically.

Before committing a newly recorded fixture, read it: confirm it holds no
token-shaped string, no absolute path, and no private project name.
