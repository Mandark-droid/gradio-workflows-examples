import json
from pathlib import Path
import pytest

GRAPH = Path(__file__).resolve().parent.parent / "workflow.json"


@pytest.fixture(scope="module")
def graph():
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def test_schema_version_is_2(graph):
    assert graph["schema_version"] == "2"


def test_every_edge_endpoint_exists(graph):
    nodes = {
        n["id"]: n
        for key in ("references", "operators", "subjects")
        for n in graph[key]
    }
    for edge in graph["edges"]:
        assert edge["from_node_id"] in nodes, edge
        assert edge["to_node_id"] in nodes, edge
        src = nodes[edge["from_node_id"]]
        dst = nodes[edge["to_node_id"]]
        assert any(p["id"] == edge["from_port_id"] for p in src["outputs"]), edge
        assert any(p["id"] == edge["to_port_id"] for p in dst["inputs"]), edge


def test_edge_types_match_both_ports(graph):
    ports = {
        (n["id"], p["id"]): p["type"]
        for key in ("references", "operators", "subjects")
        for n in graph[key]
        for p in n["inputs"] + n["outputs"]
    }
    for edge in graph["edges"]:
        src = ports[(edge["from_node_id"], edge["from_port_id"])]
        dst = ports[(edge["to_node_id"], edge["to_port_id"])]
        assert src == dst or "any" in (src, dst), edge


def test_node_ids_are_unique(graph):
    ids = [
        n["id"]
        for key in ("references", "operators", "subjects")
        for n in graph[key]
    ]
    assert len(ids) == len(set(ids))


def test_every_fn_operator_names_a_bound_function(graph):
    import app

    bound = set(app.BINDINGS)
    for node in graph["operators"]:
        if node.get("kind") == "fn":
            assert node["fn"] in bound, node["id"]


def test_there_are_exactly_two_subject_groups(graph):
    from gradio.workflow_api import WorkflowGraph, subject_groups

    groups = subject_groups(WorkflowGraph(graph))
    assert len(groups) == 2


def test_endpoint_names_are_verse_and_metre_only(graph):
    from gradio.workflow_api import WorkflowGraph, subject_groups, _group_slug_iter

    names = [
        name for _, name in _group_slug_iter(subject_groups(WorkflowGraph(graph)))
    ]
    assert names == ["verse", "metre_only"]


def test_metre_only_component_touches_no_space_or_model(graph):
    from gradio.workflow_api import (
        WorkflowGraph, subject_groups, upstream_node_ids,
    )

    wg = WorkflowGraph(graph)
    group_b = [g for g in subject_groups(wg) if g[0]["id"] == "sub_metre_only"][0]
    upstream = upstream_node_ids(wg, group_b[0]["id"])
    kinds = {
        n.get("kind")
        for n in graph["operators"]
        if n["id"] in upstream
    }
    assert kinds <= {"fn"}, f"cheap endpoint must be fn-only, got {kinds}"


def test_app_declares_live_mode_for_the_deployed_space():
    """The record/replay layer defaults to replay to protect the budget; a
    deployed Space must override that or every call fails FixtureMissing."""
    import os
    import importlib

    saved = os.environ.pop("WORKFLOW_IO_MODE", None)
    try:
        import app

        importlib.reload(app)
        assert os.environ.get("WORKFLOW_IO_MODE") == "live"
    finally:
        os.environ.pop("WORKFLOW_IO_MODE", None)
        if saved is not None:
            os.environ["WORKFLOW_IO_MODE"] = saved


def test_app_does_not_override_an_explicit_mode():
    import os
    import importlib

    saved = os.environ.get("WORKFLOW_IO_MODE")
    os.environ["WORKFLOW_IO_MODE"] = "record"
    try:
        import app

        importlib.reload(app)
        assert os.environ["WORKFLOW_IO_MODE"] == "record"
    finally:
        os.environ.pop("WORKFLOW_IO_MODE", None)
        if saved is not None:
            os.environ["WORKFLOW_IO_MODE"] = saved
