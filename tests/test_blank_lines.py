"""A blank line must not truncate the file.

Regression tests. Three separate readers each hand-rolled the same line scan
and each got the same detail wrong: an empty line made the scan stop rather
than skip, so everything after the first blank line was silently dropped.

The failure was invisible from the outside. On a 1,000-row file with one stray
blank line after row 500, `csv_linear_regression` and `csv_sgd_regression`
trained on 500 rows and reported `train_n + test_n == 500` with no warning, and
`csv_profile` lost whichever chunk contained the gap. Nothing raised, nothing
logged; the model was simply fitted on half the data.

Blank lines are ordinary in exported CSVs -- trailing newlines, a separator
between logical sections, a partially written append. All three readers now
share `FastLineIter`, so they cannot disagree about what a row is again.
"""

from __future__ import annotations

import csv

import pytest

import grizzly

FULL_COVERAGE = 10_000_000
ROWS = 1_000
GAP_AFTER = 500


@pytest.fixture
def csv_with_gap(tmp_path):
    """1,000 data rows with a single blank line in the middle."""
    path = tmp_path / "gap.csv"
    with open(path, "w", newline="") as fh:
        fh.write("x,y\n")
        for i in range(GAP_AFTER):
            fh.write(f"{i},{i * 2}\n")
        fh.write("\n")
        for i in range(GAP_AFTER, ROWS):
            fh.write(f"{i},{i * 2}\n")
    return str(path)


@pytest.fixture
def csv_with_many_gaps(tmp_path):
    """Blank lines scattered throughout, including consecutive ones."""
    path = tmp_path / "gaps.csv"
    with open(path, "w", newline="") as fh:
        fh.write("x,y\n")
        for i in range(ROWS):
            fh.write(f"{i},{i * 2}\n")
            if i % 97 == 0:
                fh.write("\n")
            if i % 313 == 0:
                fh.write("\n\n\n")
    return str(path)


def test_profile_reads_every_row(csv_with_gap):
    profile = grizzly.csv_profile(csv_with_gap, sample_size=FULL_COVERAGE, lite=False)

    assert profile["rows_sampled"] == ROWS
    column = {c["name"]: c for c in profile["columns"]}["x"]
    assert column["max"] == pytest.approx(ROWS - 1), "rows after the gap were dropped"
    assert column["count"] == ROWS


def test_profile_handles_many_gaps(csv_with_many_gaps):
    profile = grizzly.csv_profile(csv_with_many_gaps, sample_size=FULL_COVERAGE, lite=False)
    assert profile["rows_sampled"] == ROWS


@pytest.mark.parametrize("fast_csv", [True, False])
def test_profile_agrees_across_both_read_paths(csv_with_gap, fast_csv):
    profile = grizzly.csv_profile(
        csv_with_gap, sample_size=FULL_COVERAGE, lite=False, fast_csv=fast_csv
    )
    assert profile["rows_sampled"] == ROWS, f"fast_csv={fast_csv}"


def test_linear_regression_trains_on_every_row(csv_with_gap):
    result = grizzly.csv_linear_regression(
        csv_with_gap, target="y", sample_size=FULL_COVERAGE, shuffle=False
    )

    assert result["train_n"] + result["test_n"] == ROWS, "the model was fitted on a truncated file"
    # y = 2x exactly, so a correct fit is essentially perfect.
    assert result["r2"] > 0.999
    assert result["coef"][0] == pytest.approx(2.0, abs=1e-6)


def test_sgd_trains_on_every_row(csv_with_gap):
    result = grizzly.csv_sgd_regression(
        csv_with_gap, target="y", epochs=1, sample_size=FULL_COVERAGE
    )
    assert result["train_n"] + result["test_n"] == ROWS


def test_transform_writes_every_row(csv_with_gap, tmp_path):
    out = str(tmp_path / "scaled.csv")
    params = grizzly.csv_minmax_params(csv_with_gap, sample_size=FULL_COVERAGE)["params"]

    result = grizzly.csv_transform_minmax(csv_with_gap, out, params)
    assert result["rows_written"] == ROWS

    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) - 1 == ROWS, "rows missing from the written file"


def test_trailing_blank_lines_are_not_rows(tmp_path):
    """A file ending in blank lines has no phantom trailing rows."""
    path = tmp_path / "trailing.csv"
    path.write_text("x,y\n1,2\n3,4\n\n\n\n")

    profile = grizzly.csv_profile(str(path), sample_size=FULL_COVERAGE, lite=False)
    assert profile["rows_sampled"] == 2


def test_file_that_is_only_blank_lines(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("x,y\n\n\n\n")

    profile = grizzly.csv_profile(str(path), sample_size=FULL_COVERAGE, lite=False)
    assert profile["rows_sampled"] == 0
