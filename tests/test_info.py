from __future__ import annotations


def test_grizzly_info_prints_dtype_and_columns(capsys):
    import grizzly

    g = grizzly.Grizzly([{"a": 1, "b": {"c": "x"}, "items": [{"id": 2}, {"id": None}]}], sample_size=10)
    g.info()
    out = capsys.readouterr().out

    assert "Grizzly info:" in out
    assert "a" in out
    assert "b.c" in out
    assert "items[].id" in out
    # dtype hints
    assert "int" in out or "mixed" in out


