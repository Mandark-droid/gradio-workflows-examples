# Fixtures

Recorded responses for the record/replay layer in `nodes/hf_io.py`.
`WORKFLOW_IO_MODE` defaults to `replay`, so these fixtures are used
automatically — tests and local development never touch the network unless
you explicitly opt into `record` or `live`.

Each file is named after `hf_io.fixture_key(kind, target, endpoint, args)`: a
sha256 of the call's shape, so an unchanged call always resolves to the same
file and a changed one misses and raises `FixtureMissing` in replay mode
rather than silently reusing a stale response. A fixture whose call returned
a binary payload (e.g. an image) also gets a matching `.bin` file next to its
`.json`; none of the fixtures here do.

## What is covered

Two calls, for a single verse (`धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः`):

- `nodes.source.shlokartha_meaning` — the commentary call to the Ślokārtha
  Space (`/interpret_sanskrit_verse`).
- `nodes.hf_io.call_space` for the TTS Space's `/tts_interface` endpoint,
  narrating the first 300 characters of that meaning.

The TTS fixture's `value` was hand-edited after recording: `gradio_client`
returns a local temp filepath for an Audio output, and the raw path recorded
here would have leaked this machine's username and directory layout. Nothing
in the repo reads that path's contents during replay, only the fact that a
filepath string comes back, so it was replaced with the placeholder
`"tts-output.mp3"`.

## What is NOT covered

- ASR (`/transcribe_sanskrit_audio`) and the chant Space (`/synthesize`) —
  not recorded.
- FLUX (`call_model_text_to_image`) — deliberately not recorded. Its fixture
  is a binary PNG of roughly 1 MB, which would bloat a public repo for little
  value. Calls to it still work locally in `record` or `live` mode; they just
  are not replayable from a fresh clone.
- Any verse other than the one above, and any config change (voice, rate,
  pitch, source).

## Recording more

```
cd examples/shlokartha-voice
WORKFLOW_IO_MODE=record ../../.venv/Scripts/python.exe -c "
from nodes.source import shlokartha_meaning
from nodes import hf_io
verse = 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः'
m = shlokartha_meaning(verse, '', '')
cfg = hf_io.load_config()['tts']
hf_io.call_space(cfg['space_id'], cfg['api_name'], m[:300], cfg['default_voice'], cfg['rate'], cfg['pitch'], result_index=0)
print('recorded')
"
```

Swap the `verse` string to record a different one. This needs a Hugging Face
token (`HF_TOKEN` or a stored `huggingface-cli login` credential) and spends
real quota against public Spaces, so it is a deliberate, manual step — never
run automatically.

Before committing a newly recorded fixture, read it: confirm it holds no
token-shaped string, no absolute path, and no private project name. A Space
call whose output is a file (audio, image) is especially worth checking —
`gradio_client` downloads such outputs to a local temp path and that path is
what gets written into the fixture unless you strip it by hand.
