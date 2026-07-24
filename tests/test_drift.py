"""Tests for drift detection against a reference profile.

Drift detection is the capability that distinguishes Grizzly from a faster
DataFrame: pandas and polars will happily compute these statistics, but neither
tells you that today's batch no longer resembles the data a model was trained
on. The comparison works on profiles rather than raw data, so the reference
training set does not need to be kept around.

The tests below construct known drift and assert it is caught, and -- just as
importantly -- construct *no* drift and assert nothing fires. A detector that
alerts on everything is as useless as one that alerts on nothing.
"""

from __future__ import annotations

import csv
import random

import pytest

import grizzly
from grizzly import drift

FULL_COVERAGE = 10_000_000


def write_numeric(path, values_by_column, n_rows):
    header = list(values_by_column)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(n_rows):
            writer.writerow([values_by_column[name][i] for name in header])
    return str(path)


def profile_of(path):
    return grizzly.csv_profile(path, sample_size=FULL_COVERAGE, lite=False)


@pytest.fixture
def baseline(tmp_path):
    rng = random.Random(101)
    n = 5_000
    columns = {
        "stable": [f"{rng.gauss(10.0, 2.0):.6f}" for _ in range(n)],
        "shifting": [f"{rng.gauss(0.0, 1.0):.6f}" for _ in range(n)],
        "nullable": [f"{rng.gauss(5.0, 1.0):.6f}" for _ in range(n)],
    }
    return profile_of(write_numeric(tmp_path / "baseline.csv", columns, n))


# ---------------------------------------------------------------------------
# the negative case
# ---------------------------------------------------------------------------


def test_same_distribution_reports_no_drift(tmp_path):
    """Two independent samples from one distribution must not raise an alert."""
    rng = random.Random(7)
    n = 5_000
    reference = profile_of(
        write_numeric(tmp_path / "a.csv", {"x": [f"{rng.gauss(0, 1):.6f}" for _ in range(n)]}, n)
    )
    current = profile_of(
        write_numeric(tmp_path / "b.csv", {"x": [f"{rng.gauss(0, 1):.6f}" for _ in range(n)]}, n)
    )

    report = drift.compare_profiles(reference, current)

    assert report["verdict"] == "stable", report["columns"][0]["reasons"]
    column = report["columns"][0]
    assert column["psi"] < drift.PSI_MODERATE
    assert column["reasons"] == []


def test_identical_profile_has_zero_psi(baseline):
    """A profile compared against itself is the degenerate zero case."""
    report = drift.compare_profiles(baseline, baseline)

    assert report["verdict"] == "stable"
    for column in report["columns"]:
        assert column["psi"] == pytest.approx(0.0, abs=1e-9), column["column"]
        assert column["mean_shift_in_std"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# the positive cases
# ---------------------------------------------------------------------------


def test_mean_shift_is_detected(tmp_path):
    """A distribution that slides bodily is the classic covariate shift."""
    rng = random.Random(3)
    n = 5_000
    reference = profile_of(
        write_numeric(tmp_path / "ref.csv", {"x": [f"{rng.gauss(0, 1):.6f}" for _ in range(n)]}, n)
    )
    current = profile_of(
        write_numeric(tmp_path / "cur.csv", {"x": [f"{rng.gauss(3, 1):.6f}" for _ in range(n)]}, n)
    )

    report = drift.compare_profiles(reference, current)
    column = report["columns"][0]

    assert report["verdict"] == "significant"
    assert column["psi"] > drift.PSI_SIGNIFICANT
    assert column["mean_shift_in_std"] == pytest.approx(3.0, rel=0.15)
    assert any("mean moved" in r for r in column["reasons"])


def test_variance_change_is_detected(tmp_path):
    """A distribution that keeps its mean but widens still drifts."""
    rng = random.Random(5)
    n = 5_000
    reference = profile_of(
        write_numeric(tmp_path / "ref.csv", {"x": [f"{rng.gauss(0, 1):.6f}" for _ in range(n)]}, n)
    )
    current = profile_of(
        write_numeric(tmp_path / "cur.csv", {"x": [f"{rng.gauss(0, 5):.6f}" for _ in range(n)]}, n)
    )

    report = drift.compare_profiles(reference, current)
    column = report["columns"][0]

    # The mean is unchanged, so only a distribution-shape metric can catch this.
    assert abs(column["mean_shift_in_std"]) < 0.25
    assert column["psi"] > drift.PSI_MODERATE
    assert report["verdict"] in {"moderate", "significant"}


def test_null_rate_increase_is_detected(tmp_path):
    """A feature that quietly stops being populated.

    One of the most common production failures, and invisible to a mean or a
    quantile because the missing rows simply are not there to move them.
    """
    rng = random.Random(9)
    n = 4_000
    # Two columns, not one: in a single-column file an empty value is just a
    # blank line, which the parser skips rather than counting as a null. The
    # keeper column makes the missing field unambiguous.
    reference = profile_of(
        write_numeric(
            tmp_path / "ref.csv",
            {
                "x": [f"{rng.gauss(0, 1):.6f}" for _ in range(n)],
                "keeper": ["1"] * n,
            },
            n,
        )
    )
    # Same distribution, but 30% of the values have gone missing.
    values = ["" if rng.random() < 0.30 else f"{rng.gauss(0, 1):.6f}" for _ in range(n)]
    current = profile_of(write_numeric(tmp_path / "cur.csv", {"x": values, "keeper": ["1"] * n}, n))

    report = drift.compare_profiles(reference, current)
    column = next(c for c in report["columns"] if c["column"] == "x")

    assert column["null_rate_change"] == pytest.approx(0.30, abs=0.05)
    assert any("missing-value rate" in r for r in column["reasons"])
    assert report["verdict"] in {"moderate", "significant"}


def test_type_change_is_reported_and_suppresses_numeric_metrics(tmp_path):
    """A column that stops being numeric is a schema failure, not drift.

    PSI on a column whose type changed would be meaningless, so it is not
    computed; the type change itself is reported as significant.
    """
    n = 1_000
    reference = profile_of(
        write_numeric(tmp_path / "ref.csv", {"x": [str(i) for i in range(n)]}, n)
    )
    current = profile_of(
        write_numeric(tmp_path / "cur.csv", {"x": [f"id_{i}" for i in range(n)]}, n)
    )

    report = drift.compare_profiles(reference, current)
    column = report["columns"][0]

    assert column["type_changed"] is True
    assert column["psi"] is None, "PSI must not be computed across a type change"
    assert report["verdict"] == "significant"
    assert any("type changed" in r for r in column["reasons"])


# ---------------------------------------------------------------------------
# schema changes
# ---------------------------------------------------------------------------


def test_missing_column_is_significant(tmp_path):
    """A disappeared feature is more urgent than a drifted one."""
    n = 500
    reference = profile_of(
        write_numeric(
            tmp_path / "ref.csv",
            {"a": [str(i) for i in range(n)], "b": [str(i) for i in range(n)]},
            n,
        )
    )
    current = profile_of(write_numeric(tmp_path / "cur.csv", {"a": [str(i) for i in range(n)]}, n))

    report = drift.compare_profiles(reference, current)

    assert report["missing_columns"] == ["b"]
    assert report["verdict"] == "significant"


def test_new_column_is_reported_but_not_fatal(tmp_path):
    n = 500
    reference = profile_of(
        write_numeric(tmp_path / "ref.csv", {"a": [str(i) for i in range(n)]}, n)
    )
    current = profile_of(
        write_numeric(
            tmp_path / "cur.csv",
            {"a": [str(i) for i in range(n)], "extra": [str(i) for i in range(n)]},
            n,
        )
    )

    report = drift.compare_profiles(reference, current)

    assert report["new_columns"] == ["extra"]
    assert report["verdict"] == "moderate"


# ---------------------------------------------------------------------------
# round trip and reporting
# ---------------------------------------------------------------------------


def test_reference_profile_round_trips_through_disk(baseline, tmp_path):
    """A reference is saved next to a model and reloaded much later."""
    path = tmp_path / "reference" / "profile.json"
    drift.save_reference(baseline, path)

    assert path.exists()
    loaded = drift.load_reference(path)
    report = drift.compare_profiles(loaded, baseline)

    assert report["verdict"] == "stable"


def test_detect_drift_profiles_the_file_itself(baseline, tmp_path):
    """The end-to-end entry point: reference on disk, CSV to check."""
    reference_path = tmp_path / "profile.json"
    drift.save_reference(baseline, reference_path)

    rng = random.Random(77)
    n = 5_000
    current_csv = write_numeric(
        tmp_path / "current.csv",
        {
            "stable": [f"{rng.gauss(10.0, 2.0):.6f}" for _ in range(n)],
            "shifting": [f"{rng.gauss(4.0, 1.0):.6f}" for _ in range(n)],
            "nullable": [f"{rng.gauss(5.0, 1.0):.6f}" for _ in range(n)],
        },
        n,
    )

    report = drift.detect_drift(current_csv, reference_path, sample_size=FULL_COVERAGE)

    assert report["current_path"] == current_csv
    by_name = {c["column"]: c for c in report["columns"]}
    assert by_name["shifting"]["severity"] == "significant"
    assert by_name["stable"]["severity"] == "stable"


def test_report_sorts_worst_first(baseline, tmp_path):
    rng = random.Random(31)
    n = 5_000
    current = profile_of(
        write_numeric(
            tmp_path / "cur.csv",
            {
                "stable": [f"{rng.gauss(10.0, 2.0):.6f}" for _ in range(n)],
                "shifting": [f"{rng.gauss(6.0, 1.0):.6f}" for _ in range(n)],
                "nullable": [f"{rng.gauss(5.0, 1.0):.6f}" for _ in range(n)],
            },
            n,
        )
    )

    report = drift.compare_profiles(baseline, current)
    severities = [c["severity"] for c in report["columns"]]
    order = {"significant": 0, "moderate": 1, "stable": 2}
    assert severities == sorted(severities, key=lambda s: order[s])


def test_format_report_is_readable(baseline, tmp_path):
    rng = random.Random(41)
    n = 5_000
    current = profile_of(
        write_numeric(
            tmp_path / "cur.csv",
            {
                "stable": [f"{rng.gauss(10.0, 2.0):.6f}" for _ in range(n)],
                "shifting": [f"{rng.gauss(5.0, 1.0):.6f}" for _ in range(n)],
                "nullable": [f"{rng.gauss(5.0, 1.0):.6f}" for _ in range(n)],
            },
            n,
        )
    )

    text = drift.format_report(drift.compare_profiles(baseline, current))

    assert "Drift verdict:" in text
    assert "shifting" in text
    assert "PSI" in text
