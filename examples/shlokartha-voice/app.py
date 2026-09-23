"""Ślōkārtha voice pipeline — a gr.Workflow Space.

Run locally:      python app.py
Record fixtures:  WORKFLOW_IO_MODE=record python app.py
"""
from __future__ import annotations

import gradio as gr

from nodes.chandas import chandas_detect
from nodes.hf_io import _utf8_stdout
from nodes.illustration import illustration_prompt
from nodes.normalize import normalize
from nodes.sandhi import sandhi_split
from nodes.source import asr, select_source

# Keys must exactly match the "fn" value of each kind:"fn" operator in
# workflow.json, or the canvas cannot resolve the node.
BINDINGS = {
    "asr": asr,
    "select_source": select_source,
    "normalize": normalize,
    "sandhi_split": sandhi_split,
    "chandas_detect": chandas_detect,
    "illustration_prompt": illustration_prompt,
}

if __name__ == "__main__":
    _utf8_stdout()
    gr.Workflow(graph="workflow.json", bind=BINDINGS).launch()
