import json
from pathlib import Path
import pytest

GRAPH = Path(__file__).resolve().parent.parent / "workflow.json"


@pytest.fixture(scope="module")
def graph():
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def test_schema_version_is_2(graph):
    assert graph["schema_version"] == "2"


def test_every_edge_endpoint_and_port_exists(graph):
    nodes = {
        n["id"]: n
        for key in ("references", "operators", "subjects")
        for n in graph[key]
    }
    for edge in graph["edges"]:
        src, dst = nodes[edge["from_node_id"]], nodes[edge["to_node_id"]]
        assert any(p["id"] == edge["from_port_id"] for p in src["outputs"]), edge
        assert any(p["id"] == edge["to_port_id"] for p in dst["inputs"]), edge


def test_dataset_input_port_is_named_row_index(graph):
    node = next(n for n in graph["operators"] if n["id"] == "op_dataset")
    assert [p["id"] for p in node["inputs"]] == ["row_index"]


def test_dataset_output_labels_match_dataset_columns(graph):
    from dataset.build_eval_dataset import REQUIRED_COLUMNS

    node = next(n for n in graph["operators"] if n["id"] == "op_dataset")
    labels = [p["label"] for p in node["outputs"]]
    assert sorted(labels) == sorted(REQUIRED_COLUMNS)


def test_aggregate_declares_three_output_ports(graph):
    node = next(n for n in graph["operators"] if n["id"] == "op_aggregate")
    assert [p["output_index"] for p in node["outputs"]] == [0, 1, 2]


def test_no_candidate_is_a_model_kind_node(graph):
    for node_id in ("op_cand_a", "op_cand_b", "op_cand_c"):
        node = next(n for n in graph["operators"] if n["id"] == node_id)
        assert node["kind"] == "fn", "model nodes expose no latency or tokens"


def test_every_fn_operator_names_a_bound_function(graph):
    import app

    bound = set(app.BINDINGS)
    for node in graph["operators"]:
        if node.get("kind") == "fn":
            assert node["fn"] in bound, node["id"]


def test_there_is_exactly_one_subject_group(graph):
    from gradio.workflow_api import WorkflowGraph, subject_groups

    assert len(subject_groups(WorkflowGraph(graph))) == 1


def test_the_only_endpoint_is_scores(graph):
    from gradio.workflow_api import WorkflowGraph, subject_groups, _group_slug_iter

    names = [n for _, n in _group_slug_iter(subject_groups(WorkflowGraph(graph)))]
    assert names == ["scores"]


def test_row_idx_and_hf_token_are_the_only_endpoint_parameters(graph):
    from gradio.workflow_api import WorkflowGraph, subject_groups, group_free_inputs

    wg = WorkflowGraph(graph)
    frees = group_free_inputs(wg, subject_groups(wg)[0])
    assert [f["node"]["id"] for f in frees] == ["ref_row_idx", "ref_hf_token"]


def test_app_declares_live_mode_for_the_deployed_space():
    """The record/replay layer defaults to replay to protect the budget; a
    deployed Space must override that or every call fails FixtureMissing."""
    import os
    import importlib

    saved = os.environ.pop("ARENA_IO_MODE", None)
    try:
        import app

        importlib.reload(app)
        assert os.environ.get("ARENA_IO_MODE") == "live"
    finally:
        os.environ.pop("ARENA_IO_MODE", None)
        if saved is not None:
            os.environ["ARENA_IO_MODE"] = saved


def test_app_does_not_override_an_explicit_mode():
    import os
    import importlib

    saved = os.environ.get("ARENA_IO_MODE")
    os.environ["ARENA_IO_MODE"] = "record"
    try:
        import app

        importlib.reload(app)
        assert os.environ["ARENA_IO_MODE"] == "record"
    finally:
        os.environ.pop("ARENA_IO_MODE", None)
        if saved is not None:
            os.environ["ARENA_IO_MODE"] = saved
