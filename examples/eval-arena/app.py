"""Hot-swappable model eval arena — a gr.Workflow Space.

Run locally:      python app.py
Record fixtures:  ARENA_IO_MODE=record python app.py
"""
from __future__ import annotations

import os

# A deployed Space is the live context. The record/replay layer defaults to
# "replay" so development and tests never spend money, but that same default
# would make every call here fail with FixtureMissing. setdefault, not a plain
# assignment, so an operator can still force "record" when capturing fixtures.
os.environ.setdefault("ARENA_IO_MODE", "live")

import gradio as gr

from nodes.aggregate import aggregate
from nodes.candidate import candidate_a, candidate_b, candidate_c
from nodes.judge import pairwise_judge
from nodes.prompts import gold_extract, prompt_builder
from nodes.scorer import deterministic_scorer

# Keys must exactly match the "fn" value of each kind:"fn" operator in
# workflow.json, or the canvas cannot resolve the node.
BINDINGS = {
    "prompt_builder": prompt_builder,
    "gold_extract": gold_extract,
    "candidate_a": candidate_a,
    "candidate_b": candidate_b,
    "candidate_c": candidate_c,
    "deterministic_scorer": deterministic_scorer,
    "pairwise_judge": pairwise_judge,
    "aggregate": aggregate,
}

if __name__ == "__main__":
    gr.Workflow(graph="workflow.json", bind=BINDINGS).launch()
