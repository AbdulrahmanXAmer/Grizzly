from __future__ import annotations


def test_nested_dict_list_paths():
    import grizzly

    data = [
        {"user": {"id": 1, "name": "Ada"}, "items": [{"id": 10}, {"id": 11}]},
        {"user": {"id": 2, "name": None}, "items": []},
    ]

    cols = grizzly.detect_columns(data)
    assert "user.id" in cols
    assert "user.name" in cols
    assert "items[].id" in cols


def test_top_level_scalar():
    import grizzly

    schema = grizzly.detect_schema(123)
    paths = [c["path"] for c in schema["columns"]]
    assert paths == ["value"]


