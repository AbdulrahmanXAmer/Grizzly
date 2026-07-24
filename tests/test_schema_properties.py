"""Property-based tests for schema inference.

Example-based tests only cover the shapes someone thought to write down. Schema
inference takes arbitrary nested Python data, so the interesting inputs are the
ones nobody would think of: a dict whose keys collide after path joining, a
list of heterogeneous scalars, an empty container in the middle of a structure.

Hypothesis generates those. The tests below assert invariants that must hold
for *any* input rather than checking specific outputs, and the most valuable
one is differential: the Rust extension and the pure-Python fallback in
``grizzly.fallback`` are two independent implementations of the same
specification, so they should agree.
"""

from __future__ import annotations

import pytest

import grizzly
from grizzly import fallback

hypothesis = pytest.importorskip("hypothesis", reason="requires hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Generous budget: schema inference builds real Python objects, so the default
# deadline occasionally trips on an unlucky large example rather than on a bug.
SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Keys are restricted to identifier-ish text. Path notation joins keys with
# ".", so a key containing a dot is genuinely ambiguous ({"a.b": 1} and
# {"a": {"b": 1}} both produce the path "a.b"). That ambiguity is a real
# property of the format, tested separately rather than mixed into every case.
keys = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=8,
)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=12),
)


def nested(max_leaves=12):
    """Arbitrary JSON-like structures: scalars, dicts, and lists thereof."""
    return st.recursive(
        scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(keys, children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


records = st.lists(st.dictionaries(keys, nested(6), max_size=4), min_size=1, max_size=8)


def paths_of(schema):
    return [column["path"] for column in schema["columns"]]


# ---------------------------------------------------------------------------
# structural invariants
# ---------------------------------------------------------------------------


@SETTINGS
@given(data=nested())
def test_schema_is_always_well_formed(data):
    """Every column carries the full set of fields, whatever the input."""
    schema = grizzly.detect_schema(data, sample_size=10_000)

    assert isinstance(schema["columns"], list)
    for column in schema["columns"]:
        assert set(column) >= {"path", "count", "null_count", "types", "inferred", "examples"}
        assert isinstance(column["path"], str)
        assert column["count"] >= 0
        assert column["null_count"] >= 0
        assert column["null_count"] <= column["count"], (
            f"{column['path']}: more nulls than observations"
        )
        assert isinstance(column["types"], list)
        assert column["inferred"]


@SETTINGS
@given(data=nested())
def test_paths_are_unique(data):
    """A path identifies one column; duplicates would make the schema unusable."""
    paths = paths_of(grizzly.detect_schema(data, sample_size=10_000))
    assert len(paths) == len(set(paths))


@SETTINGS
@given(data=nested())
def test_paths_are_sorted(data):
    """Output ordering is stable, so schemas can be diffed between runs."""
    paths = paths_of(grizzly.detect_schema(data, sample_size=10_000))
    assert paths == sorted(paths)


@SETTINGS
@given(data=nested())
def test_inference_is_deterministic(data):
    """The same input must always produce the same schema."""
    first = grizzly.detect_schema(data, sample_size=10_000)
    second = grizzly.detect_schema(data, sample_size=10_000)
    assert first == second


@SETTINGS
@given(data=nested())
def test_null_is_reported_alongside_the_real_type(data):
    """A column of ints with some Nones is an int column that has nulls."""
    schema = grizzly.detect_schema(data, sample_size=10_000)
    for column in schema["columns"]:
        if column["null_count"] > 0 and column["null_count"] < column["count"]:
            assert "null" in column["types"], (
                f"{column['path']} has nulls but does not list the null type"
            )
        if column["inferred"] != "null":
            non_null = [t for t in column["types"] if t != "null"]
            assert non_null, f"{column['path']} inferred {column['inferred']} from nulls only"


@SETTINGS
@given(data=nested())
def test_never_truncates_below_the_depth_cap(data):
    """Generated structures are shallow, so the depth guard must not fire."""
    schema = grizzly.detect_schema(data, sample_size=10_000)
    assert schema["max_depth_exceeded"] is False


# ---------------------------------------------------------------------------
# differential: native core vs pure-Python fallback
# ---------------------------------------------------------------------------


@SETTINGS
@given(data=records)
def test_native_and_fallback_agree_on_paths(data):
    """Two independent implementations of the same path notation.

    The fallback exists to keep Grizzly usable without the compiled extension,
    which only works if it produces the same schema. Any divergence is a bug in
    one of them.
    """
    native_paths = set(paths_of(grizzly.detect_schema(data, sample_size=10_000)))
    fallback_paths = set(paths_of(fallback.detect_schema(data, sample_size=10_000)))

    assert native_paths == fallback_paths, (
        f"only native: {sorted(native_paths - fallback_paths)}; "
        f"only fallback: {sorted(fallback_paths - native_paths)}"
    )


@SETTINGS
@given(data=records)
def test_native_and_fallback_agree_on_null_counts(data):
    """Counts must match too, not just the set of paths."""
    native = {c["path"]: c for c in grizzly.detect_schema(data, sample_size=10_000)["columns"]}
    fallback_cols = {
        c["path"]: c for c in fallback.detect_schema(data, sample_size=10_000)["columns"]
    }

    for path, column in native.items():
        assert column["count"] == fallback_cols[path]["count"], f"{path} count"
        assert column["null_count"] == fallback_cols[path]["null_count"], f"{path} null_count"


# ---------------------------------------------------------------------------
# path notation
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    outer=keys,
    inner=keys,
    value=st.integers(min_value=-1000, max_value=1000),
)
def test_nested_dicts_join_keys_with_a_dot(outer, inner, value):
    schema = grizzly.detect_schema({outer: {inner: value}}, sample_size=10_000)
    assert paths_of(schema) == [f"{outer}.{inner}"]


@SETTINGS
@given(key=keys, values=st.lists(st.integers(), min_size=1, max_size=5))
def test_lists_of_scalars_get_a_bracket_suffix(key, values):
    schema = grizzly.detect_schema({key: values}, sample_size=10_000)
    assert paths_of(schema) == [f"{key}[]"]


@SETTINGS
@given(key=keys, inner=keys, value=st.integers())
def test_lists_of_dicts_combine_both_notations(key, inner, value):
    schema = grizzly.detect_schema({key: [{inner: value}]}, sample_size=10_000)
    assert paths_of(schema) == [f"{key}[].{inner}"]


def test_dotted_keys_and_nesting_are_ambiguous_by_design():
    """A documented consequence of the path format, pinned so it stays known.

    `{"a.b": 1}` and `{"a": {"b": 1}}` both flatten to the path "a.b". The
    notation cannot distinguish them; this records that rather than leaving it
    to be discovered.
    """
    flat = grizzly.detect_schema({"a.b": 1}, sample_size=10_000)
    deep = grizzly.detect_schema({"a": {"b": 1}}, sample_size=10_000)
    assert paths_of(flat) == paths_of(deep) == ["a.b"]


# ---------------------------------------------------------------------------
# containers with no leaves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [{}, [], [[]], {"a": {}}, {"a": []}])
def test_empty_containers_produce_no_columns(empty):
    """An empty container has no values, so there is nothing to describe."""
    schema = grizzly.detect_schema(empty, sample_size=10_000)
    assert schema["columns"] == []


@SETTINGS
@given(value=scalars)
def test_a_bare_scalar_is_reported_as_value(value):
    """Top-level scalars have no key, so they are named "value"."""
    schema = grizzly.detect_schema(value, sample_size=10_000)
    assert paths_of(schema) == ["value"]
