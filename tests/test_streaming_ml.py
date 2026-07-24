"""Tests for the streaming ML paths: standardization and SGD.

These are the capabilities that justify the "feature statistics for training
pipelines" framing rather than "faster DataFrame", so they are held to the same
differential standard as the rest: standardization must produce what a
reference implementation produces, and SGD must converge to the same answer as
the closed-form solver it exists to replace at scale.
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

FULL_COVERAGE = 10_000_000


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def read_column(path, name):
    with open(path, newline="") as fh:
        return [float(row[name]) for row in csv.DictReader(fh)]


@pytest.fixture
def scaled_csv(tmp_path):
    """Columns on deliberately different scales, plus a constant one."""
    rng = random.Random(11)
    header = ["small", "large", "constant", "label"]
    rows = []
    for _ in range(3_000):
        rows.append(
            [
                f"{rng.gauss(0.0, 0.001):.9f}",
                f"{rng.gauss(50_000.0, 12_000.0):.4f}",
                "7.0",
                rng.choice(["a", "b"]),
            ]
        )
    return write_csv(tmp_path / "scaled.csv", header, rows)


# ---------------------------------------------------------------------------
# streaming standardization
# ---------------------------------------------------------------------------


def test_standardize_params_match_the_profile(scaled_csv):
    """The scaler reuses the profile's moments rather than recomputing them."""
    params = grizzly.csv_standardize_params(scaled_csv, sample_size=FULL_COVERAGE)["params"]
    profile = grizzly.csv_profile(scaled_csv, sample_size=FULL_COVERAGE, lite=False)
    columns = {c["name"]: c for c in profile["columns"]}

    for name in ("small", "large", "constant"):
        assert params[name]["mean"] == pytest.approx(columns[name]["mean"])
        assert params[name]["std"] == pytest.approx(columns[name]["std"])


@pytest.mark.skipif(not HAS_PANDAS, reason="requires pandas")
def test_standardize_matches_pandas(scaled_csv, tmp_path):
    """Output must equal (x - mean) / std computed by a reference."""
    import pandas as pd

    params = grizzly.csv_standardize_params(scaled_csv, sample_size=FULL_COVERAGE)["params"]
    out = str(tmp_path / "standardized.csv")
    grizzly.csv_transform_standardize(scaled_csv, out, params)

    source = pd.read_csv(scaled_csv)
    result = pd.read_csv(out)

    for name in ("small", "large"):
        expected = (source[name] - source[name].mean()) / source[name].std(ddof=0)
        assert result[name].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-9), name


def test_standardized_columns_have_zero_mean_and_unit_variance(scaled_csv, tmp_path):
    """The defining property, checked on the output file itself."""
    params = grizzly.csv_standardize_params(scaled_csv, sample_size=FULL_COVERAGE)["params"]
    out = str(tmp_path / "standardized.csv")
    grizzly.csv_transform_standardize(scaled_csv, out, params)

    profile = grizzly.csv_profile(out, sample_size=FULL_COVERAGE, lite=False)
    columns = {c["name"]: c for c in profile["columns"]}

    for name in ("small", "large"):
        assert columns[name]["mean"] == pytest.approx(0.0, abs=1e-6), f"{name} mean"
        assert columns[name]["std"] == pytest.approx(1.0, abs=1e-6), f"{name} std"


def test_constant_column_becomes_zero_not_nan(scaled_csv, tmp_path):
    """A zero-variance column has nothing to scale by.

    Dividing by its std would emit NaN or infinity into the output file and
    poison every downstream consumer, so it is written as 0.0 instead.
    """
    params = grizzly.csv_standardize_params(scaled_csv, sample_size=FULL_COVERAGE)["params"]
    out = str(tmp_path / "standardized.csv")
    grizzly.csv_transform_standardize(scaled_csv, out, params)

    values = read_column(out, "constant")
    assert all(v == 0.0 for v in values)
    assert not any(math.isnan(v) or math.isinf(v) for v in values)


def test_standardize_preserves_rows_and_non_numeric_columns(scaled_csv, tmp_path):
    params = grizzly.csv_standardize_params(scaled_csv, sample_size=FULL_COVERAGE)["params"]
    out = str(tmp_path / "standardized.csv")
    result = grizzly.csv_transform_standardize(scaled_csv, out, params)

    assert result["rows_written"] == 3_000

    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3_000
    # The categorical column passes through untouched.
    assert {row["label"] for row in rows} == {"a", "b"}


# ---------------------------------------------------------------------------
# streaming SGD
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_csv(tmp_path):
    """A well-conditioned system whose features live on very different scales.

    The scale spread is the point: it is what a single global learning rate
    cannot handle without the on-the-fly standardization, so this fixture would
    fail if that were removed.
    """
    rng = random.Random(23)
    weights = [2.0, -0.0001, 45.0]
    bias = 3.0
    header = ["tiny", "huge", "mid", "target"]
    rows = []
    for _ in range(6_000):
        x = [rng.gauss(0.0, 1.0), rng.gauss(100_000.0, 20_000.0), rng.gauss(5.0, 2.0)]
        y = sum(w * xi for w, xi in zip(weights, x, strict=True)) + bias
        y += rng.gauss(0.0, 0.05)
        rows.append([f"{v:.8f}" for v in (*x, y)])
    return write_csv(tmp_path / "sgd.csv", header, rows), weights, bias


def test_sgd_fits_a_useful_model(regression_csv):
    path, _, _ = regression_csv
    result = grizzly.csv_sgd_regression(
        path, target="target", epochs=15, learning_rate=0.1, sample_size=FULL_COVERAGE
    )

    assert result["train_n"] > 0
    assert result["test_n"] > 0
    assert result["r2"] > 0.99, f"expected a near-perfect fit, got r2={result['r2']}"
    assert result["epochs"] == 15
    assert math.isfinite(result["final_train_mse"])


@pytest.mark.skipif(not HAS_NUMPY, reason="requires numpy")
def test_sgd_converges_to_the_closed_form_solution(regression_csv):
    """SGD exists to replace the closed-form solver at scale, so it has to agree.

    Coefficients are returned in the original feature space precisely so this
    comparison is possible.
    """
    import numpy as np

    path, _, _ = regression_csv

    sgd = grizzly.csv_sgd_regression(
        path,
        target="target",
        epochs=40,
        learning_rate=0.1,
        train_frac=0.8,
        seed=0,
        shuffle=False,
        sample_size=FULL_COVERAGE,
    )

    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    train = raw[: int(6_000 * 0.8)]
    X, y = train[:, :-1], train[:, -1]
    design = np.column_stack([X, np.ones(len(X))])
    reference, *_ = np.linalg.lstsq(design, y, rcond=None)

    for i, (got, want) in enumerate(zip(sgd["coef"], reference[:-1], strict=True)):
        # Relative tolerance: the coefficients span five orders of magnitude,
        # so a single absolute tolerance would be meaningless for all but one.
        assert got == pytest.approx(want, rel=0.05), (
            f"coefficient {i} ({sgd['features'][i]}): SGD {got} vs closed form {want}"
        )


def test_sgd_recovers_the_generating_weights(regression_csv):
    """End-to-end sanity check across five orders of magnitude in scale."""
    path, weights, bias = regression_csv
    result = grizzly.csv_sgd_regression(
        path, target="target", epochs=40, learning_rate=0.1, sample_size=FULL_COVERAGE
    )

    for i, (got, want) in enumerate(zip(result["coef"], weights, strict=True)):
        assert got == pytest.approx(want, rel=0.05), f"weight {i}"
    assert result["intercept"] == pytest.approx(bias, rel=0.1)


def test_sgd_improves_with_more_epochs(regression_csv):
    """Training error must actually decrease, or nothing is being learned."""
    path, _, _ = regression_csv

    def mse(epochs):
        return grizzly.csv_sgd_regression(
            path,
            target="target",
            epochs=epochs,
            learning_rate=0.1,
            sample_size=FULL_COVERAGE,
        )["final_train_mse"]

    assert mse(20) < mse(1), "more epochs should reduce training error"


def test_sgd_l2_shrinks_coefficients(regression_csv):
    path, _, _ = regression_csv

    def norm(l2):
        result = grizzly.csv_sgd_regression(
            path,
            target="target",
            epochs=10,
            learning_rate=0.1,
            l2=l2,
            sample_size=FULL_COVERAGE,
        )
        # Compare in standardized space via the largest coefficient's feature,
        # since raw magnitudes differ by orders of magnitude.
        return abs(result["coef"][2])

    assert norm(0.5) < norm(0.0), "an L2 penalty should shrink the fit"


def test_sgd_reports_divergence_instead_of_returning_nan(regression_csv):
    """An absurd learning rate must fail loudly, not return NaN coefficients."""
    path, _, _ = regression_csv

    with pytest.raises(ValueError, match="diverged"):
        grizzly.csv_sgd_regression(
            path,
            target="target",
            epochs=5,
            learning_rate=1e6,
            sample_size=FULL_COVERAGE,
        )


def test_sgd_rejects_invalid_arguments(regression_csv):
    path, _, _ = regression_csv

    with pytest.raises(ValueError, match="train_frac"):
        grizzly.csv_sgd_regression(path, target="target", train_frac=1.0)
    with pytest.raises(ValueError, match="epochs"):
        grizzly.csv_sgd_regression(path, target="target", epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        grizzly.csv_sgd_regression(path, target="target", learning_rate=0.0)


def test_sgd_split_uses_rows_present(regression_csv):
    """Same invariant as the closed-form solver: the split follows the data."""
    path, _, _ = regression_csv
    result = grizzly.csv_sgd_regression(path, target="target", epochs=1, sample_size=FULL_COVERAGE)

    assert result["train_n"] + result["test_n"] == 6_000
    assert 0.75 < result["train_n"] / 6_000 < 0.85
