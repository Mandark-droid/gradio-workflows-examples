---
title: Ślōkārtha Voice Pipeline
emoji: 🕉️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
python_version: "3.11"
pinned: false
hf_oauth: true
---

# Ślōkārtha Voice Pipeline

Chant or type a Sanskrit verse; get back the verse in Devanagari and IAST, its
padaccheda (word split), its metre, its meaning, a narration and an
illustration — with every stage visible on the canvas.

Built with [`gr.Workflow`](https://gradio.app/guides/workflows). Errors compound
across ASR, normalization, sandhi and meaning, so the point of a canvas is that
a wrong meaning can be traced to the stage that caused it.

## The graph

Two disconnected components share one `workflow.json`:

- **`/verse`** — the full pipeline. Returns six outputs: verse, padaccheda,
  metre, meaning, narration, illustration.
- **`/metre_only`** — a text-only metre spine. No ASR, no Space call, no billed
  inference. This is the cheap endpoint to use as a Sanskrit scansion utility.

`gr.Workflow` gives one endpoint per connected component, not per output, which
is why the cheap path is a second component rather than a second subject.

## Metre detection

Syllabifies IAST, marks each syllable guru if it has a long vowel, an anusvāra
or visarga, or is followed by a conjunct, then matches per-pāda weight strings
against a metre table covering Anuṣṭubh, Indravajrā, Upendravajrā, Upajāti,
Vasantatilakā, Mālinī, Śikhariṇī and Śārdūlavikrīḍita.

Confidence is the fraction of pādas that match. Below 0.75 the meaning is
flagged low-trust — a known Anuṣṭubh verse that fails scansion almost always
means the transcription drifted.

## Latency, honestly

`/metre_only` returns in well under a second — it is pure CPU and touches no
external service.

`/verse` takes roughly 25–30s warm. Almost all of that is one call to the
Ślōkārtha Space, which generates its commentary on CPU hardware (measured
23.1s). Fan-out also runs sequentially through the generated API even though
the canvas runs it in parallel, so the branches add up rather than overlap.
This is inherent to reusing a public CPU Space rather than something tuned
away.

## A note for contributors

Running the app locally causes `gr.Workflow` to rewrite `workflow.json` —
it drops the hand-authored `x`/`y` geometry and some `required` flags. That is
upstream beta behaviour, not a bug in this example. If `git status` shows
`workflow.json` modified after you ran the app and you did not mean to change
the graph, restore it with `git checkout -- workflow.json`.

## Record, replay, live

Every network call routes through one small layer with three modes, set by
`WORKFLOW_IO_MODE`: `replay` (the default) reads recorded fixtures and never
touches the network, `record` calls live and saves the response, and `live`
calls without saving. The default is `replay` so tests and local development
can never spend money by accident.

`app.py` sets the mode to `live` for exactly this reason: a deployed Space has
no fixtures, and inheriting the replay default would make every call fail with
`FixtureMissing` — quietly, since the failure is caught and reported as an
ordinary error result.

`tests/fixtures/` is committed, so a fresh clone can replay without a token or
a live call. It covers exactly two calls for one verse: the Ślokārtha meaning
call and the TTS narration call. ASR, the chant Space, and FLUX image
generation are not recorded — FLUX deliberately so, since its fixture is a
~1 MB binary that is not worth committing to a public repo. See
`tests/fixtures/README.md` for exactly what is and is not covered, and the
command to record more.

## Credits

Meaning comes from the `/interpret_sanskrit_verse` endpoint of
[Ślokārtha](https://hf.co/spaces/kshitijthakkar/shlokartha-playground); ASR
uses the same Space's `/transcribe_sanskrit_audio` endpoint. Ślokārtha serves
the public [`kshitijthakkar/shlokartha`](https://hf.co/kshitijthakkar/shlokartha)
checkpoint trained on
[`kshitijthakkar/shlokartha-sft`](https://hf.co/datasets/kshitijthakkar/shlokartha-sft).
Narration uses [Edge TTS](https://hf.co/spaces/innoai/Edge-TTS-Text-to-Speech);
illustration uses FLUX.1-schnell.

## A limitation worth knowing

The Sanskrit ASR is the `/transcribe_sanskrit_audio` endpoint of the Ślōkārtha
Space. In testing it returned an empty transcription for text-to-speech audio
of a verse — synthetic speech is not what it was trained on. Real recited audio
may fare better, but treat the audio path as the less reliable of the two
inputs and use the text box when transcription comes back empty.

When no verse is detected the pipeline does not fail: the meaning node returns
a short message saying so, and the remaining outputs are produced from whatever
text was available.
