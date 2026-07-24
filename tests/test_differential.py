"""Differential tests: Grizzly's answers must match reference implementations.

Unit tests confirm the internals behave as designed. These confirm the design
is *right*, by checking Grizzly's output against libraries whose correctness is
not in question -- pandas and polars for profiling, and a closed-form NumPy
solution for regression.

This is the layer that catches an entire class of bug the other tests cannot:
statistics that are self-consistent and stable but simply wrong. A merge that
double-counts, an off-by-one in a chunk boundary, or a variance formula with
the wrong denominator all produce output that looks perfectly reasonable in
isolation.

Every test skips cleanly when its reference library is not installed.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import random

import pytest

import grizzly

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_POLARS = importlib.util.find_spec("polars") is not None

# Grizzly is sampling-first: sample_size caps how many rows it reads, and its
# chunked parallel reads can stop slightly short of an exact request. Asking for
# far more rows than exist is what guarantees it sees the whole file, which is
# required for any comparison against a library that always reads everything.
FULL_COVERAGE = 10_000_000

# Absolute tolerance for float comparisons. Grizzly accumulates in a streaming
# fashion and merges across chunks, so its summation order differs from a
# vectorised library's; the values must agree to well within any tolerance that
# matters, but not bit-for-bit.
TOLERANCE = 1e-6


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def numeric_csv(tmp_path):
    """A deterministic numeric CSV with a categorical column and some nulls."""
    rng = random.Random(1234)
    header = ["x", "y", "z", "label", "opt"]
    rows = []
    for _ in range(5_000):
        rows.append(
            [
                f"{rng.gauss(0.0, 1.0):.6f}",
                f"{rng.gauss(100.0, 15.0):.6f}",
                str(rng.randint(-1000, 1000)),
                rng.choice(["alpha", "bravo", "charlie"]),
                "" if rng.random() < 0.1 else f"{rng.gauss(5.0, 2.0):.6f}",
            ]
        )
    return write_csv(tmp_path / "numeric.csv", header, rows)


def grizzly_columns(path):
    profile = grizzly.csv_profile(path, sample_size=FULL_COVERAGE, lite=False)
    assert profile["rows_sampled"] == 5_000, (
        "Grizzly did not read the whole file, so any comparison below would be "
        f"against a different number of rows (read {profile['rows_sampled']})"
    )
    return {col["name"]: col for col in profile["columns"]}


# ---------------------------------------------------------------------------
# profiling vs pandas
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_numeric_stats_match_pandas(numeric_csv):
    import pandas as pd

    columns = grizzly_columns(numeric_csv)
    df = pd.read_csv(numeric_csv)

    for name in ("x", "y", "z", "opt"):
        series = df[name]
        col = columns[name]

        assert col["min"] == pytest.approx(float(series.min()), abs=TOLERANCE), f"{name} min"
        assert col["max"] == pytest.approx(float(series.max()), abs=TOLERANCE), f"{name} max"
        assert col["mean"] == pytest.approx(float(series.mean()), abs=TOLERANCE), f"{name} mean"

        # Grizzly reports a population std; pandas defaults to the sample std.
        assert col["std"] == pytest.approx(float(series.std(ddof=0)), abs=1e-6), (
            f"{name} population std"
        )


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_null_counts_match_pandas(numeric_csv):
    import pandas as pd

    columns = grizzly_columns(numeric_csv)
    df = pd.read_csv(numeric_csv)

    for name in df.columns:
        expected_nulls = int(df[name].isna().sum())
        assert columns[name]["null_count"] == expected_nulls, f"{name} null_count"

        # Grizzly's `count` includes nulls, unlike pandas'. The row total is the
        # invariant that holds across both conventions.
        assert columns[name]["count"] == len(df), f"{name} observed rows"

    assert columns["opt"]["null_count"] > 0, "fixture should contain nulls to compare"


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_inferred_types_match_pandas_dtypes(numeric_csv):
    import pandas as pd

    columns = grizzly_columns(numeric_csv)
    df = pd.read_csv(numeric_csv)

    assert columns["z"]["inferred"] == "int"
    assert pd.api.types.is_integer_dtype(df["z"])

    for name in ("x", "y", "opt"):
        assert columns[name]["inferred"] == "float", name
        assert pd.api.types.is_float_dtype(df[name]), name

    assert columns["label"]["inferred"] == "string"
    assert not pd.api.types.is_numeric_dtype(df["label"])


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_mode_matches_pandas(numeric_csv):
    import pandas as pd

    columns = grizzly_columns(numeric_csv)
    df = pd.read_csv(numeric_csv)

    expected = df["label"].mode().iloc[0]
    expected_count = int((df["label"] == expected).sum())

    assert columns["label"]["mode"] == expected
    assert columns["label"]["mode_count"] == expected_count


# ---------------------------------------------------------------------------
# profiling vs polars
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
def test_numeric_stats_match_polars(numeric_csv):
    """A second independent reference, in case pandas and Grizzly share a bug."""
    import polars as pl

    columns = grizzly_columns(numeric_csv)
    df = pl.read_csv(numeric_csv)

    for name in ("x", "y", "z", "opt"):
        series = df[name]
        col = columns[name]

        assert col["min"] == pytest.approx(float(series.min()), abs=TOLERANCE), f"{name} min"
        assert col["max"] == pytest.approx(float(series.max()), abs=TOLERANCE), f"{name} max"
        assert col["mean"] == pytest.approx(float(series.mean()), abs=TOLERANCE), f"{name} mean"
        assert col["null_count"] == series.null_count(), f"{name} null_count"


# ---------------------------------------------------------------------------
# chunk-merge correctness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
@pytest.mark.parametrize("fast_csv", [True, False])
def test_parallel_and_sequential_paths_agree(numeric_csv, fast_csv):
    """fast_csv=True chunks and merges in parallel; False reads sequentially.

    Both must produce identical statistics. A merge bug would show up here as a
    difference between the two paths on the same file.
    """
    import pandas as pd

    profile = grizzly.csv_profile(
        numeric_csv, sample_size=FULL_COVERAGE, lite=False, fast_csv=fast_csv
    )
    columns = {c["name"]: c for c in profile["columns"]}
    df = pd.read_csv(numeric_csv)

    assert profile["rows_sampled"] == len(df)
    for name in ("x", "y", "z"):
        assert columns[name]["mean"] == pytest.approx(float(df[name].mean()), abs=TOLERANCE), (
            f"{name} mean under fast_csv={fast_csv}"
        )
        assert columns[name]["min"] == pytest.approx(float(df[name].min()), abs=TOLERANCE)
        assert columns[name]["max"] == pytest.approx(float(df[name].max()), abs=TOLERANCE)


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_statistics_are_stable_across_file_sizes(tmp_path):
    """Crossing the chunking threshold must not change the answer.

    Small files fit in a single chunk; larger ones are split across threads and
    merged. If the merge is wrong, the two disagree.
    """
    import pandas as pd

    rng = random.Random(99)
    for n_rows in (10, 1_000, 60_000):
        rows = [[f"{rng.gauss(0.0, 1.0):.6f}"] for _ in range(n_rows)]
        path = write_csv(tmp_path / f"sized_{n_rows}.csv", ["v"], rows)

        profile = grizzly.csv_profile(path, sample_size=FULL_COVERAGE, lite=False)
        col = profile["columns"][0]
        series = pd.read_csv(path)["v"]

        assert profile["rows_sampled"] == n_rows, f"{n_rows} rows"
        assert col["mean"] == pytest.approx(float(series.mean()), abs=TOLERANCE), f"{n_rows} mean"
        assert col["std"] == pytest.approx(float(series.std(ddof=0)), abs=1e-6), f"{n_rows} std"


# ---------------------------------------------------------------------------
# regression vs a closed-form solution
# ---------------------------------------------------------------------------


# csv_linear_regression requires train_frac in the open interval (0, 1), so the
# model is never fitted on every row. With shuffle=False the split is
# sequential -- the first floor(n_rows * train_frac) rows -- which is what lets
# the reference below be fitted on exactly the same rows Grizzly used.
REGRESSION_ROWS = 4_000
TRAIN_FRAC = 0.8
TRAIN_CUT = int(REGRESSION_ROWS * TRAIN_FRAC)


@pytest.fixture
def regression_csv(tmp_path):
    """A well-conditioned linear system with a known generating process."""
    rng = random.Random(7)
    n_features, n_rows = 4, REGRESSION_ROWS
    weights = [1.5, -2.0, 0.75, 3.25]
    bias = 0.5

    header = [f"f{i}" for i in range(n_features)] + ["target"]
    rows = []
    for _ in range(n_rows):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        y = sum(w * xi for w, xi in zip(weights, x, strict=True))
        y += bias + rng.gauss(0.0, 0.05)
        rows.append([f"{v:.8f}" for v in (*x, y)])

    return write_csv(tmp_path / "regression.csv", header, rows), weights, bias


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_regression_coefficients_match_closed_form(regression_csv):
    """Grizzly's Rust solver must match the ordinary least squares solution.

    The reference is NumPy's lstsq on the same rows, which is the definition of
    the answer rather than another approximation of it.
    """
    import numpy as np

    path, _, _ = regression_csv

    result = grizzly.csv_linear_regression(
        path,
        target="target",
        train_frac=TRAIN_FRAC,
        shuffle=False,
        sample_size=FULL_COVERAGE,
    )

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    # Fit the reference on the same rows Grizzly trained on: the leading
    # TRAIN_CUT, per the sequential split used when shuffle=False.
    train = raw[:TRAIN_CUT]
    X, y = train[:, :-1], train[:, -1]
    design = np.column_stack([X, np.ones(len(X))])
    reference, *_ = np.linalg.lstsq(design, y, rcond=None)

    expected_coef, expected_intercept = reference[:-1], reference[-1]

    assert len(result["coef"]) == len(expected_coef)
    for i, (got, want) in enumerate(zip(result["coef"], expected_coef, strict=True)):
        assert got == pytest.approx(want, abs=1e-6), f"coefficient {i}"
    assert result["intercept"] == pytest.approx(expected_intercept, abs=1e-6)


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_regression_recovers_the_generating_weights(regression_csv):
    """A sanity check on the whole pipeline, not just the solver.

    With low noise and a well-conditioned design, the fit should land close to
    the weights the data was actually generated from.
    """
    path, weights, bias = regression_csv

    result = grizzly.csv_linear_regression(
        path,
        target="target",
        train_frac=TRAIN_FRAC,
        shuffle=False,
        sample_size=FULL_COVERAGE,
    )

    for i, (got, want) in enumerate(zip(result["coef"], weights, strict=True)):
        assert got == pytest.approx(want, abs=0.01), f"weight {i}"
    assert result["intercept"] == pytest.approx(bias, abs=0.01)


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_r2_matches_the_definition(regression_csv):
    """R^2 must equal 1 - SS_res / SS_tot computed independently."""
    import numpy as np

    path, _, _ = regression_csv

    result = grizzly.csv_linear_regression(
        path,
        target="target",
        train_frac=TRAIN_FRAC,
        shuffle=False,
        sample_size=FULL_COVERAGE,
    )

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    # The reported r2 is a test-set score, so it must be reproduced on the
    # held-out rows rather than on everything.
    test = raw[TRAIN_CUT:]
    X, y = test[:, :-1], test[:, -1]
    predicted = X @ np.array(result["coef"]) + result["intercept"]

    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    expected_r2 = 1.0 - ss_res / ss_tot

    assert result["r2"] == pytest.approx(expected_r2, abs=1e-6)
    assert 0.9 < result["r2"] <= 1.0, "low-noise fit should explain nearly all variance"


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_ridge_penalty_shrinks_coefficients(regression_csv):
    """Ridge must behave like ridge: a larger penalty shrinks the solution."""
    path, _, _ = regression_csv

    def norm(ridge_lambda):
        result = grizzly.csv_linear_regression(
            path,
            target="target",
            train_frac=TRAIN_FRAC,
            shuffle=False,
            sample_size=FULL_COVERAGE,
            ridge_lambda=ridge_lambda,
        )
        return math.sqrt(sum(c * c for c in result["coef"]))

    unpenalised = norm(0.0)
    lightly = norm(10.0)
    heavily = norm(10_000.0)

    assert lightly < unpenalised, "a ridge penalty should shrink the coefficients"
    assert heavily < lightly, "a larger penalty should shrink them further"


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_train_test_split_partitions_the_rows(regression_csv):
    """The split must account for every row exactly once."""
    path, _, _ = regression_csv

    result = grizzly.csv_linear_regression(
        path,
        target="target",
        train_frac=0.8,
        seed=0,
        sample_size=FULL_COVERAGE,
        return_debug=True,
    )

    assert result["train_n"] + result["test_n"] == REGRESSION_ROWS
    # An 80/20 split, allowing for the randomised per-row assignment.
    assert 0.75 < result["train_n"] / REGRESSION_ROWS < 0.85


@pytest.mark.parametrize("shuffle", [True, False])
def test_split_is_relative_to_the_rows_present_not_the_sample_cap(regression_csv, shuffle):
    """A sample_size far larger than the file must not collapse the test set.

    Regression test. The sequential (shuffle=False) path used to derive its row
    count from sample_size rather than from the data, so train_cut became a
    fraction of the requested cap instead of of the file. With the default
    sample_size of 1,000,000, every file under 800,000 rows put all of its rows
    on the train side: the test set was empty and r2 came back as exactly 0.0
    for a model that fit almost perfectly.
    """
    path, _, _ = regression_csv

    result = grizzly.csv_linear_regression(
        path,
        target="target",
        train_frac=TRAIN_FRAC,
        shuffle=shuffle,
        sample_size=FULL_COVERAGE,
    )

    assert result["test_n"] > 0, "test set was empty, so r2 is meaningless"
    assert result["train_n"] + result["test_n"] == REGRESSION_ROWS
    assert result["train_n"] == pytest.approx(TRAIN_CUT, rel=0.1)
    assert result["r2"] > 0.9, "a near-perfect fit must not report r2 = 0.0"


def test_default_sample_size_does_not_break_the_split(regression_csv):
    """The same failure, reached through the documented defaults.

    sample_size defaults to 1,000,000, which is far larger than this 4,000-row
    fixture -- exactly the condition that used to silently empty the test set.
    """
    path, _, _ = regression_csv

    result = grizzly.csv_linear_regression(path, target="target", shuffle=False)

    assert result["test_n"] > 0
    assert result["r2"] > 0.9
