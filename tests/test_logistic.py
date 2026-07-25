"""Tests for the streaming logistic classifier.

Held to the same differential standard as the rest of the ML surface: a
classifier that is fast and wrong is worthless, so the metrics it reports are
checked against scikit-learn on the same data, and the parts that are genuinely
approximations -- the binned ROC-AUC, the SGD coefficients -- are pinned to how
approximate they are allowed to be.

The distinction that matters throughout: grizzly's split and sklearn's split are
equally sized but not identical row-for-row, so two independent fits differ by
sampling noise *plus* optimizer error. Where a test needs to isolate one of
those, it reconstructs grizzly's own held-out rows and scores both sides on
exactly those, rather than comparing two fits and calling the difference small.
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


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def write_binary_csv(path, n_rows=20_000, n_features=6, seed=0, sharpness=1.0):
    """A CSV whose label is drawn from a logistic model of the features.

    Sampled rather than thresholded: a hard threshold makes the classes
    perfectly separable, and on separable data the logistic likelihood has no
    finite maximum -- coefficients run to infinity and any two implementations
    "agree" only in both diverging. Overlapping classes give a well-posed
    optimum to compare against.
    """
    rng = random.Random(seed)
    weights = [rng.uniform(-1.5, 1.5) for _ in range(n_features)]
    bias = rng.uniform(-0.5, 0.5)
    header = [f"f_{i}" for i in range(n_features)] + ["target"]
    rows = []
    for _ in range(n_rows):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        z = sharpness * (sum(w * xi for w, xi in zip(weights, x)) + bias)
        rows.append([f"{v:.6f}" for v in x] + [1 if rng.random() < sigmoid(z) else 0])
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path), weights, bias


@pytest.fixture
def binary_csv(tmp_path):
    path, _, _ = write_binary_csv(tmp_path / "binary.csv")
    return path


# ---------------------------------------------------------------------------
# basic contract
# ---------------------------------------------------------------------------


def test_reports_every_documented_metric(binary_csv):
    result = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=20, learning_rate=0.05, sample_size=FULL_COVERAGE
    )

    for key in (
        "coef",
        "intercept",
        "train_n",
        "test_n",
        "epochs",
        "accuracy",
        "log_loss",
        "roc_auc",
        "positive_rate",
        "confusion_matrix",
        "final_train_loss",
    ):
        assert key in result, f"missing {key}"

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["log_loss"] > 0.0

    cm = result["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == result["test_n"]
    # Accuracy and the confusion matrix are accumulated in the same pass; if
    # they disagree, one of them is being counted wrongly.
    assert cm["tp"] + cm["tn"] == pytest.approx(result["accuracy"] * result["test_n"])


def test_uses_every_row_across_the_split(binary_csv):
    result = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=3, train_frac=0.8, sample_size=FULL_COVERAGE
    )
    assert result["train_n"] + result["test_n"] == 20_000
    # The split is honoured, not merely non-empty.
    assert result["train_n"] == pytest.approx(16_000, abs=1)
    assert result["test_n"] == pytest.approx(4_000, abs=1)


def test_learns_something_better_than_guessing(binary_csv):
    result = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=20, learning_rate=0.05, sample_size=FULL_COVERAGE
    )
    # The generating model is noisy, so perfect accuracy is impossible; the
    # point is that it lands well clear of the majority-class baseline.
    baseline = max(result["positive_rate"], 1.0 - result["positive_rate"])
    assert result["accuracy"] > baseline + 0.1
    assert result["roc_auc"] > 0.75
    # An untrained model would sit at log(2) = 0.693.
    assert result["log_loss"] < 0.6


def test_recovers_the_generating_weights(tmp_path):
    """The sharpest end-to-end check: the fit must find the model that made the
    data, not merely some model that scores well."""
    path, weights, _ = write_binary_csv(
        tmp_path / "recover.csv", n_rows=40_000, n_features=5, seed=7
    )
    result = grizzly.csv_logistic_regression(
        path, target="target", epochs=60, learning_rate=0.05, sample_size=FULL_COVERAGE
    )
    scale = max(abs(w) for w in weights)
    for name, got, want in zip(result["features"], result["coef"], weights):
        assert got == pytest.approx(want, abs=0.25 * scale), f"{name}: {got} vs {want}"


def test_is_deterministic_for_a_given_seed(binary_csv):
    kwargs = dict(target="target", epochs=10, learning_rate=0.04, seed=42, sample_size=FULL_COVERAGE)
    first = grizzly.csv_logistic_regression(binary_csv, **kwargs)
    second = grizzly.csv_logistic_regression(binary_csv, **kwargs)
    assert first["coef"] == second["coef"]
    assert first["intercept"] == second["intercept"]
    assert first["roc_auc"] == second["roc_auc"]


def test_caching_does_not_change_the_answer(binary_csv):
    """Cached replay must be bit-identical to streaming, as for regression.

    The cache exists only to avoid re-parsing; if it changed the arithmetic it
    would be a silently different model depending on available memory.
    """
    kwargs = dict(target="target", epochs=8, learning_rate=0.04, sample_size=FULL_COVERAGE)
    cached = grizzly.csv_logistic_regression(binary_csv, cache_budget_mb=512, **kwargs)
    streamed = grizzly.csv_logistic_regression(binary_csv, cache_budget_mb=0, **kwargs)
    assert cached["coef"] == streamed["coef"]
    assert cached["intercept"] == streamed["intercept"]
    assert cached["log_loss"] == streamed["log_loss"]


# ---------------------------------------------------------------------------
# label validation
# ---------------------------------------------------------------------------


def test_rejects_labels_outside_zero_and_one(tmp_path):
    path = tmp_path / "bad_labels.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f_0", "target"])
        for i in range(100):
            writer.writerow([f"{i * 0.01:.4f}", 2 if i == 50 else 0])

    with pytest.raises(ValueError, match="only 0 and 1"):
        grizzly.csv_logistic_regression(str(path), target="target", sample_size=FULL_COVERAGE)


def test_accepts_a_single_class_without_dividing_by_zero(tmp_path):
    """Degenerate but real: a filtered dataset can end up all one class."""
    path = tmp_path / "one_class.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f_0", "f_1", "target"])
        rng = random.Random(3)
        for _ in range(500):
            writer.writerow([f"{rng.gauss(0, 1):.4f}", f"{rng.gauss(0, 1):.4f}", 1])

    result = grizzly.csv_logistic_regression(
        str(path), target="target", epochs=5, sample_size=FULL_COVERAGE
    )
    # AUC is undefined with one class; 0.5 is the honest answer, not NaN.
    assert result["roc_auc"] == 0.5
    assert math.isfinite(result["log_loss"])
    assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# differential: scikit-learn
# ---------------------------------------------------------------------------


def splitmix64(x):
    """Python mirror of the Rust `splitmix64`, for reconstructing the split."""
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def held_out_indices(n_rows, train_frac, seed):
    """Reproduce exactly which rows grizzly holds out.

    Mirrors the Fisher-Yates shuffle in `fit_sgd`: same PRNG, same traversal
    order, same cut. If this ever drifts from the Rust the test below fails
    loudly, rather than quietly comparing two different sets of rows.
    """
    perm = list(range(n_rows))
    for i in range(n_rows - 1, 0, -1):
        j = splitmix64(seed ^ i) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    train = set(perm[: int(n_rows * train_frac)])
    return [i for i in range(n_rows) if i not in train]


@requires_sklearn
def test_metrics_match_sklearn_exactly_on_identical_rows(binary_csv):
    """Isolates grizzly's metric code from every other source of difference.

    Both sides score the *same* held-out rows with the *same* fitted model, so
    a disagreement here is grizzly's evaluation arithmetic and nothing else --
    not sampling noise, not optimizer error. Accuracy and log-loss are exact
    computations and must match to floating-point noise; only ROC-AUC is
    approximated, and this is the test that justifies binning it rather than
    sorting the held-out scores.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    result = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=20, learning_rate=0.05, sample_size=FULL_COVERAGE
    )

    with open(binary_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    test_idx = held_out_indices(len(rows), 0.8, 0)
    # If the reconstruction is wrong then nothing below means anything, so it
    # is checked against what the fit itself reported.
    assert len(test_idx) == result["test_n"]

    names = result["features"]
    coef = np.asarray(result["coef"])
    X = np.asarray([[float(rows[i][n]) for n in names] for i in test_idx])
    y = np.asarray([float(rows[i]["target"]) for i in test_idx])
    scores = 1.0 / (1.0 + np.exp(-(X @ coef + result["intercept"])))

    assert accuracy_score(y, scores >= 0.5) == pytest.approx(result["accuracy"], abs=1e-12)
    assert log_loss(y, scores) == pytest.approx(result["log_loss"], rel=1e-9)
    assert roc_auc_score(y, scores) == pytest.approx(result["roc_auc"], abs=1e-3)


@requires_sklearn
def test_metrics_agree_with_sklearns_own_fit(binary_csv):
    """The headline claim: grizzly's classifier is as good as sklearn's.

    Two independent fits on equally-sized but not identical splits, compared on
    the metrics a classifier is actually judged by. These converge much faster
    than the coefficients do, which is why they can carry a tight tolerance
    while the coefficient test below cannot.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    result = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=40, learning_rate=0.05, sample_size=FULL_COVERAGE
    )

    with open(binary_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = result["features"]
    X = np.asarray([[float(r[n]) for n in names] for r in rows])
    y = np.asarray([float(r["target"]) for r in rows])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    model = LogisticRegression(C=np.inf, max_iter=2000).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    assert result["accuracy"] == pytest.approx(
        accuracy_score(y_test, model.predict(X_test)), abs=0.02
    )
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_test, proba), abs=0.02)
    assert result["log_loss"] == pytest.approx(log_loss(y_test, proba), abs=0.02)


@requires_sklearn
def test_coefficients_converge_toward_the_sklearn_optimum(binary_csv):
    """SGD coefficients approach the MLE; more epochs must get closer.

    A fixed tolerance alone cannot tell "converging slowly" from "converged to
    the wrong place", so this asserts the *direction* as well: deviation from
    sklearn's optimum shrinks as epochs increase. That is the property which
    distinguishes optimizer error from a bug in the gradient.

    It shrinks to a floor, not to zero: the two sides train on different rows,
    so their maximum-likelihood coefficients genuinely differ, and past a few
    tens of epochs the remaining gap is that sampling difference rather than
    anything more training can remove. Hence the wide epoch gap below -- closer
    together, this would be measuring noise around the floor and would flake.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    with open(binary_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    probe = grizzly.csv_logistic_regression(
        binary_csv, target="target", epochs=1, sample_size=FULL_COVERAGE
    )
    names = probe["features"]
    X = np.asarray([[float(r[n]) for n in names] for r in rows])
    y = np.asarray([float(r["target"]) for r in rows])
    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=0)
    reference = LogisticRegression(C=np.inf, max_iter=2000).fit(X_train, y_train).coef_[0]
    scale = float(np.max(np.abs(reference)))

    def deviation(epochs):
        fit = grizzly.csv_logistic_regression(
            binary_csv,
            target="target",
            epochs=epochs,
            learning_rate=0.05,
            sample_size=FULL_COVERAGE,
        )
        return float(np.max(np.abs(np.asarray(fit["coef"]) - reference))) / scale

    few, many = deviation(5), deviation(120)
    assert many < few, f"more epochs did not help: {few:.4f} -> {many:.4f}"
    assert many < 0.10, f"coefficients {many:.4f} of scale away from the optimum"
