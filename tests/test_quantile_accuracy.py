"""Quantify the approximation error in Grizzly's t-digest quantiles.

Grizzly computes percentiles from a t-digest rather than by sorting, which
makes them approximate. "Approximate" is not a useful thing to tell a user, so
these tests pin down *how* approximate, and in which direction the guarantee
runs.

The distinction that matters:

**Rank error** is what a t-digest actually bounds. For an estimate `v` returned
for quantile `q`, it is how far the true fraction of data at or below `v` sits
from `q`. Measured worst case across every distribution below: **0.16%**.

**Value error** is how far `v` is from the exact quantile, and it is *not*
bounded by the algorithm. On smooth distributions it stays under 0.21% of the
data range. On a distribution with a sharp discontinuity it can be much larger
while the rank stays exactly right -- if 95% of a column is exactly zero, the
true p95 sits on the jump, and a rank-correct answer can be far from the
interpolated value. Zero-inflated columns (counts, spend, sparse features) are
common in real data and p95/p99 on them is exactly what gets used for outlier
thresholds, so this is worth knowing rather than discovering.

Bounds asserted here are deliberately looser than the measured values, so the
tests catch a regression rather than fail on ordinary float noise.
"""

from __future__ import annotations

import bisect
import csv
import random

import pytest

import grizzly

FULL_COVERAGE = 10_000_000

QUANTILES = (("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95))

# Measured worst case is 0.16%; assert a 0.5% ceiling.
MAX_RANK_ERROR = 0.005

# Measured worst case on smooth distributions is 0.21% of range; allow 1%.
MAX_VALUE_ERROR_FRACTION = 0.01


def write_column(tmp_path, name, values):
    path = tmp_path / f"{name}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["v"])
        for value in values:
            writer.writerow([f"{value:.9f}"])
    return str(path)


def profile_column(path, expected_rows):
    profile = grizzly.csv_profile(path, sample_size=FULL_COVERAGE, lite=False)
    assert profile["rows_sampled"] == expected_rows, "did not read the whole column"
    return profile["columns"][0]


def exact_quantile(ordered, q):
    """Exact quantile by linear interpolation, matching NumPy's default."""
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def rank_error(ordered, value, q):
    """How far `q` is from the range of quantiles `value` legitimately answers.

    With repeated values a single number is the correct answer for a whole band
    of quantiles, so the comparison is against the interval
    [bisect_left/n, bisect_right/n] rather than a single point. Without this,
    ties alone would register as enormous error.
    """
    n = len(ordered)
    low = bisect.bisect_left(ordered, value) / n
    high = bisect.bisect_right(ordered, value) / n
    if low <= q <= high:
        return 0.0
    return low - q if q < low else q - high


def smooth_distributions():
    rng = random.Random(4242)
    return {
        "uniform": [rng.uniform(0, 1000) for _ in range(10_000)],
        "gaussian": [rng.gauss(0, 1) for _ in range(50_000)],
        "ramp": [float(i) for i in range(50_000)],
        "lognormal": [rng.lognormvariate(0, 1) for _ in range(20_000)],
        "pareto": [rng.paretovariate(1.5) for _ in range(20_000)],
        "small": [rng.gauss(50, 10) for _ in range(1_000)],
    }


@pytest.mark.parametrize("name", sorted(smooth_distributions()))
def test_rank_error_is_bounded(tmp_path, name):
    """The guarantee a t-digest actually makes, across distribution shapes."""
    values = smooth_distributions()[name]
    ordered = sorted(values)
    column = profile_column(write_column(tmp_path, name, values), len(values))

    for key, q in QUANTILES:
        error = rank_error(ordered, column[key], q)
        assert error <= MAX_RANK_ERROR, (
            f"{name} {key}: rank error {error:.5f} exceeds {MAX_RANK_ERROR}"
        )


@pytest.mark.parametrize("name", sorted(smooth_distributions()))
def test_value_error_is_small_on_smooth_distributions(tmp_path, name):
    """Without a discontinuity, the returned value is close to exact too."""
    values = smooth_distributions()[name]
    ordered = sorted(values)
    data_range = ordered[-1] - ordered[0]
    column = profile_column(write_column(tmp_path, name, values), len(values))

    for key, q in QUANTILES:
        error = abs(column[key] - exact_quantile(ordered, q)) / data_range
        assert error <= MAX_VALUE_ERROR_FRACTION, (
            f"{name} {key}: value error {error:.4%} of range exceeds {MAX_VALUE_ERROR_FRACTION:.0%}"
        )


def zero_inflated():
    """95% exact zeros with a disjoint tail: the worst case for value error."""
    rng = random.Random(4242)
    values = [0.0] * 19_000 + [rng.uniform(100, 1000) for _ in range(1_000)]
    rng.shuffle(values)
    return values


def test_zero_inflated_column_keeps_rank_exact(tmp_path):
    """Rank stays exact even where value error is at its worst."""
    values = zero_inflated()
    ordered = sorted(values)
    column = profile_column(write_column(tmp_path, "zero_inflated", values), len(values))

    for key, q in QUANTILES:
        error = rank_error(ordered, column[key], q)
        assert error == 0.0, f"{key}: rank error {error:.5f} on a zero-inflated column"


def test_zero_inflated_value_error_is_documented_not_silent(tmp_path):
    """Pin the known worst case so a change in behaviour is visible.

    p95 falls exactly on the jump from the zeros to the tail. The estimate is
    rank-correct but lands well above the interpolated exact value; this test
    records that rather than pretending it does not happen.
    """
    values = zero_inflated()
    ordered = sorted(values)
    column = profile_column(write_column(tmp_path, "zero_inflated", values), len(values))

    # Below the discontinuity every quantile is exactly zero and exactly right.
    for key in ("p25", "median", "p75", "p90"):
        assert column[key] == 0.0, f"{key} should be exactly zero"

    exact_p95 = exact_quantile(ordered, 0.95)
    data_range = ordered[-1] - ordered[0]
    value_error = abs(column["p95"] - exact_p95) / data_range

    # Substantially worse than the smooth-distribution bound, and expected to
    # be. If this ever drops below the smooth bound the docs should be updated.
    assert value_error > MAX_VALUE_ERROR_FRACTION, (
        "zero-inflated p95 value error is now small; the documented caveat and "
        "these bounds should be revisited"
    )
    assert value_error < 0.20, (
        f"zero-inflated p95 value error {value_error:.2%} of range is worse "
        "than previously measured (~7.8%)"
    )


@pytest.mark.parametrize("name", sorted(smooth_distributions()))
def test_quantiles_are_monotonic_and_within_range(tmp_path, name):
    """Structural invariants that must hold regardless of approximation."""
    values = smooth_distributions()[name]
    column = profile_column(write_column(tmp_path, name, values), len(values))

    estimates = [column[key] for key, _ in QUANTILES]
    assert estimates == sorted(estimates), f"{name}: quantiles are not monotonic"
    assert column["min"] <= estimates[0], f"{name}: p25 below the observed minimum"
    assert estimates[-1] <= column["max"], f"{name}: p95 above the observed maximum"


def test_sampling_is_stratified_not_a_prefix(tmp_path):
    """A small sample spreads across the file rather than reading its head.

    The profiler splits the file into per-thread chunks and each chunk consumes
    rows from its own region, so `sample_size` does not mean "the first N rows".
    That is what makes sampling usable on data with meaningful row order --
    sorted, time-series, partitioned by date -- where `head -n` or
    `pandas.read_csv(nrows=...)` would give a badly biased answer.

    Strictly ascending values make the distinction unmissable: a prefix would
    report a maximum near zero.
    """
    n = 200_000
    values = [float(i) for i in range(n)]
    path = write_column(tmp_path, "ascending", values)

    profile = grizzly.csv_profile(path, sample_size=256, lite=False)
    column = profile["columns"][0]

    assert profile["rows_sampled"] < n / 100, "expected a small sample"
    # A prefix of 256 ascending rows would top out around 255.
    assert column["max"] > 0.9 * (n - 1), (
        f"max {column['max']} suggests a prefix was read rather than a spread"
    )
    # And the median should land near the true middle, not near the start.
    assert column["median"] == pytest.approx((n - 1) / 2, rel=0.15)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("constant", [7.5] * 10_000),
        ("two_values", [0.0] * 5_000 + [1.0] * 5_000),
        ("single_row", [42.0]),
    ],
)
def test_degenerate_columns(tmp_path, name, values):
    """Columns with no spread must not produce nonsense."""
    column = profile_column(write_column(tmp_path, name, values), len(values))

    for key, _ in QUANTILES:
        assert column["min"] <= column[key] <= column["max"], f"{name} {key} out of range"

    if name == "constant":
        for key, _ in QUANTILES:
            assert column[key] == pytest.approx(7.5), f"{name} {key}"
