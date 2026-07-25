"""Tests for csv_classification_metrics: a predictions file in, metrics out.

Differential against scikit-learn computing the same metrics from the same
arrays. Accuracy and log-loss are exact computations and must match to
floating-point noise; only ROC-AUC carries a real tolerance, for the same
4096-bin streaming histogram the fits use.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import random

import pytest

import grizzly

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

FULL_COVERAGE = 10_000_000

requires_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")


def write_predictions_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y_true", "y_score", "other"])
        writer.writerows(rows)
    return str(path)


def make_scored(n=5_000, seed=0, sharpness=2.0):
    """Labels drawn from the scores themselves, so the scores are informative
    but imperfect -- the metrics land strictly between coin-flip and perfect."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = rng.random()
        # Push scores toward the extremes so accuracy is meaningfully high.
        s = s**sharpness if rng.random() < 0.5 else 1 - (1 - s) ** sharpness
        y = 1 if rng.random() < s else 0
        rows.append([y, f"{s:.9f}", "x"])
    return rows


@pytest.fixture
def predictions_csv(tmp_path):
    return write_predictions_csv(tmp_path / "preds.csv", make_scored())


def test_reports_counts_and_bounded_metrics(predictions_csv):
    result = grizzly.csv_classification_metrics(
        predictions_csv, label="y_true", score="y_score", sample_size=FULL_COVERAGE
    )
    assert result["n"] == 5_000
    assert result["n_skipped"] == 0
    assert 0.5 < result["accuracy"] < 1.0
    assert 0.5 < result["roc_auc"] < 1.0
    assert result["log_loss"] > 0.0
    cm = result["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == result["n"]


def test_skips_bad_rows_and_counts_them(tmp_path):
    rows = make_scored(100)
    rows[10][1] = ""  # missing score
    rows[20][1] = "not-a-number"
    rows[30][1] = "inf"  # parseable but not finite
    rows[40][0] = ""  # missing label
    path = write_predictions_csv(tmp_path / "gappy.csv", rows)

    result = grizzly.csv_classification_metrics(
        path, label="y_true", score="y_score", sample_size=FULL_COVERAGE
    )
    assert result["n"] == 96
    assert result["n_skipped"] == 4


def test_rejects_labels_outside_zero_and_one(tmp_path):
    rows = make_scored(50)
    rows[25][0] = 2
    path = write_predictions_csv(tmp_path / "bad_label.csv", rows)

    with pytest.raises(ValueError, match="only 0 and 1"):
        grizzly.csv_classification_metrics(
            path, label="y_true", score="y_score", sample_size=FULL_COVERAGE
        )


def test_unknown_column_is_an_error(predictions_csv):
    with pytest.raises(ValueError, match="not found"):
        grizzly.csv_classification_metrics(
            predictions_csv, label="y_true", score="nope", sample_size=FULL_COVERAGE
        )


def test_saturated_scores_do_not_destroy_log_loss(tmp_path):
    """A confidently wrong probability of exactly 0 or 1 must clamp, not -inf."""
    rows = [[1, "0.0", "x"], [0, "1.0", "x"], [1, "0.9", "x"], [0, "0.1", "x"]]
    path = write_predictions_csv(tmp_path / "saturated.csv", rows)

    result = grizzly.csv_classification_metrics(
        path, label="y_true", score="y_score", sample_size=FULL_COVERAGE
    )
    assert math.isfinite(result["log_loss"])
    assert result["accuracy"] == 0.5


@requires_sklearn
def test_matches_sklearn_on_the_same_file(predictions_csv):
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    result = grizzly.csv_classification_metrics(
        predictions_csv, label="y_true", score="y_score", sample_size=FULL_COVERAGE
    )

    with open(predictions_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    y = np.asarray([float(r["y_true"]) for r in rows])
    s = np.asarray([float(r["y_score"]) for r in rows])

    assert result["accuracy"] == pytest.approx(accuracy_score(y, s >= 0.5), abs=1e-12)
    assert result["log_loss"] == pytest.approx(log_loss(y, s), rel=1e-9)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-3)


@requires_sklearn
def test_agrees_with_a_fits_own_reported_metrics(tmp_path):
    """Closing the loop: score a fit's predictions through this function and
    land on the numbers the fit itself reported.

    With shuffle=False the held-out rows are simply the file's tail, so the
    predictions CSV can be built from the returned model exactly.
    """
    import numpy as np

    # Reuse the logistic test generator's data shape without importing it.
    rng = random.Random(4)
    n, p = 8_000, 4
    weights = [rng.uniform(-1.5, 1.5) for _ in range(p)]
    train_path = tmp_path / "train.csv"
    with open(train_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"f_{j}" for j in range(p)] + ["target"])
        for _ in range(n):
            x = [rng.gauss(0, 1) for _ in range(p)]
            z = sum(w * v for w, v in zip(weights, x))
            prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            writer.writerow([f"{v:.6f}" for v in x] + [1 if rng.random() < prob else 0])

    fit = grizzly.csv_logistic_regression(
        str(train_path), target="target", epochs=20, shuffle=False, sample_size=FULL_COVERAGE
    )

    with open(train_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    test_rows = rows[fit["train_n"] :]
    coef = np.asarray(fit["coef"])
    preds_path = tmp_path / "preds.csv"
    with open(preds_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y_true", "y_score"])
        for r in test_rows:
            x = np.asarray([float(r[f"f_{j}"]) for j in range(p)])
            score = 1.0 / (1.0 + np.exp(-(x @ coef + fit["intercept"])))
            writer.writerow([r["target"], f"{score:.12f}"])

    scored = grizzly.csv_classification_metrics(
        str(preds_path), label="y_true", score="y_score", sample_size=FULL_COVERAGE
    )
    assert scored["n"] == fit["test_n"]
    assert scored["accuracy"] == pytest.approx(fit["accuracy"], abs=1e-9)
    assert scored["log_loss"] == pytest.approx(fit["log_loss"], rel=1e-6)
    assert scored["roc_auc"] == pytest.approx(fit["roc_auc"], abs=1e-3)
