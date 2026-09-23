from dataset.build_eval_dataset import load_rows, REQUIRED_COLUMNS, ANSWER_TYPES


def test_there_are_five_rows():
    assert len(load_rows()) == 5


def test_every_row_has_the_required_columns():
    for row in load_rows():
        assert set(row) == set(REQUIRED_COLUMNS), row.get("id")


def test_every_answer_type_is_exercised_exactly_once():
    types = sorted(row["task_type"] for row in load_rows())
    assert types == sorted(ANSWER_TYPES)


def test_ids_are_unique():
    ids = [row["id"] for row in load_rows()]
    assert len(ids) == len(set(ids))


def test_all_values_are_strings_so_they_survive_text_ports():
    for row in load_rows():
        for key, value in row.items():
            assert isinstance(value, str), (row["id"], key)


def test_numeric_row_gold_parses_as_a_number():
    row = next(r for r in load_rows() if r["task_type"] == "numeric")
    assert float(row["gold"])


def test_json_row_gold_parses_as_json():
    import json

    row = next(r for r in load_rows() if r["task_type"] == "json")
    assert isinstance(json.loads(row["gold"]), dict)
