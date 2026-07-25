from __future__ import annotations

from typing import Any


def _load_native():
    try:
        from . import _grizzly as native  # type: ignore

        return native
    except Exception:  # pragma: no cover
        return None


def is_native() -> bool:
    """True if the Rust extension module (grizzly._grizzly) is importable."""
    return _load_native() is not None


def native_module():
    """Return the native module object if available, else None."""
    return _load_native()


def detect_schema(
    data: Any,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    normalize_input: bool = True,
) -> dict[str, Any]:
    """
    Infer a flattened schema from arbitrary Python data.

    Path notation:
    - Dict keys are joined with '.':   user.name
    - List/tuple nesting uses '[]':    items[].id, matrix[][].value
    """
    if normalize_input:
        from .normalize import normalize

        data = normalize(data, sample_size=sample_size)

    native = _load_native()
    if native is not None:
        return native.detect_schema(data, sample_size=sample_size, max_examples=max_examples)

    # Fallback: minimal, slower Python implementation (kept intentionally small).
    from .fallback import detect_schema as py_detect_schema

    return py_detect_schema(data, sample_size=sample_size, max_examples=max_examples)


def detect_columns(data: Any, *, sample_size: int = 1000) -> list[str]:
    """Convenience wrapper returning just the sorted column paths."""
    schema = detect_schema(data, sample_size=sample_size)
    return [c["path"] for c in schema.get("columns", [])]


def info(
    data: Any,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    normalize_input: bool = True,
    show_examples: bool = False,
    max_cols: int | None = None,
    file=None,
) -> None:
    """Module-level convenience wrapper: grizzly.info(data) -> prints summary."""
    from .grizzly import Grizzly

    Grizzly(
        data,
        sample_size=sample_size,
        max_examples=max_examples,
        normalize_input=normalize_input,
    ).info(file=file, show_examples=show_examples, max_cols=max_cols)


def csv_profile(
    path: str,
    *,
    sample_size: int = 1000,
    max_examples: int = 5,
    fast_csv: bool = True,
    lite: bool = False,
    track_freq: bool = True,
    collect_examples: bool = True,
) -> dict[str, Any]:
    """
    Rust-accelerated CSV profiling: delimiter/header sniff + dtype + basic stats per column.

    Args:
        path: Path to CSV file (supports .csv.gz)
        sample_size: Maximum number of rows to sample
        max_examples: Maximum examples to collect per column
        fast_csv: If True, uses parallel byte chunking (assumes no quoted newlines).
                  If False, uses sequential reading (correct for any CSV).
                  Default: True for speed.
        lite: If True, only compute numeric stats (min/max/mean/std/quantiles).
              Skips type inference, examples, and frequency tracking.
              Use this for Polars-equivalent benchmarking speed.
        track_freq: If True, track value frequency for mode calculation.
        collect_examples: If True, collect example values per column.

    Returns: Profile dict with columns, stats, delimiter info, etc.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_profile requires the native Rust extension; build with `maturin develop`."
        )
    return native.csv_profile(
        path,
        sample_size=sample_size,
        max_examples=max_examples,
        fast_csv=fast_csv,
        lite=lite,
        track_freq=track_freq,
        collect_examples=collect_examples,
    )


def csv_minmax_params(path: str, *, sample_size: int = 1000) -> dict[str, Any]:
    """
    Return min/max per numeric column (sampled), suitable for min-max scaling.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_minmax_params requires the native Rust extension; build with `maturin develop`."
        )
    return native.csv_minmax_params(path, sample_size=sample_size)


def csv_transform_minmax(
    input_path: str,
    output_path: str,
    params: dict[str, dict[str, float]],
    *,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> dict[str, Any]:
    """
    Transform a CSV by applying min-max scaling to numeric columns.

    Args:
        input_path: Path to input CSV (can be .csv.gz)
        output_path: Path to output CSV
        params: Dict of {col_name: {"min": ..., "max": ...}, ...}
        delimiter: Optional delimiter (None = auto-detect)
        has_header: Whether file has header row (None = auto-detect)

    Returns: {
        "input_path": ...,
        "output_path": ...,
        "rows_written": ...,
        "numeric_cols_scaled": ...,
        "has_header": ...
    }
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_transform_minmax requires the native Rust extension; build with `maturin develop`."
        )
    return native.csv_transform_minmax(input_path, output_path, params, delimiter, has_header)


def csv_standardize_params(path: str, *, sample_size: int = 1000) -> dict[str, Any]:
    """Return mean/std per numeric column, suitable for standardization.

    Computed from the same single streaming pass as the profile, so the cost is
    one read of the file rather than a materialised DataFrame.

    Note the standard deviation is a *population* std, matching the rest of
    Grizzly's statistics; pandas and polars default to the sample std.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_standardize_params requires the native Rust extension; "
            "build with `maturin develop`."
        )
    return native.csv_standardize_params(path, sample_size=sample_size)


def csv_transform_standardize(
    input_path: str,
    output_path: str,
    params: dict[str, dict[str, float]],
    *,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> dict[str, Any]:
    """Standardize numeric columns to zero mean and unit variance, streaming.

    Rows are read, transformed, and written in chunks, so peak memory is bounded
    by the chunk size rather than by the size of the file.

    Args:
        input_path: Path to input CSV (can be .csv.gz)
        output_path: Path to output CSV
        params: Dict of {col_name: {"mean": ..., "std": ...}, ...}
        delimiter: Optional delimiter (None = auto-detect)
        has_header: Whether file has header row (None = auto-detect)

    A column whose std is zero or non-finite is written as 0.0 rather than
    NaN: a constant column carries no signal to scale, and NaN would propagate
    into everything downstream.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_transform_standardize requires the native Rust extension; "
            "build with `maturin develop`."
        )
    return native.csv_transform_standardize(input_path, output_path, params, delimiter, has_header)


def csv_sgd_regression(
    path: str,
    *,
    target: str,
    features: list[str] | None = None,
    epochs: int = 5,
    learning_rate: float = 0.05,
    l2: float = 0.0,
    train_frac: float = 0.8,
    seed: int = 0,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
    shuffle: bool = True,
    grad_clip: float = 10.0,
    cache_budget_mb: int = 512,
) -> dict[str, Any]:
    """Fit a linear model by SGD, streaming from CSV in bounded memory.

    Use this instead of :func:`csv_linear_regression` when the feature count is
    large. The closed-form solver accumulates an X'X matrix, costing O(p^2)
    memory and O(n p^2) time; this holds only the weight vector, so memory is
    O(p) and each epoch is O(n p). It never builds a design matrix.

    Features are standardized on the fly, from moments computed in the same
    parallel pass that reads the data, because a single global learning rate
    cannot suit features on very different scales.
    Coefficients are returned in the original feature space, so they are
    directly comparable with the closed-form solver's.

    Rows are visited in file order within each epoch: shuffling a stream would
    require buffering it, which would give up the bounded memory that is the
    point. Shuffle on disk first if row order carries meaning.

    `grad_clip` bounds the influence of any single row. Real data contains
    outliers that survive standardization -- a mis-metered taxi trip recorded
    as 100,000 miles sits far out even after scaling -- and one such row is
    enough to diverge an unclipped fit. Pass `math.inf` to disable.

    Clipping guarantees a finite result, not a good one. An extreme outlier
    also inflates the standard deviation used to standardize, compressing every
    ordinary value toward zero and leaving little signal to learn from. That is
    a property of z-score scaling rather than of SGD; winsorize such columns
    before fitting.

    `cache_budget_mb` trades memory for speed. Parsing is most of the cost of
    a fit, and it otherwise happens once per epoch plus once for evaluation;
    when the standardized matrix fits under the budget, the file is parsed
    exactly once instead — in parallel, into a cache that every epoch and the
    evaluation pass then replay from memory. The fitted weights are
    bit-identical either way, because the fill computes the exact values
    streaming would have recomputed, in the same per-row order, through the
    same update. Set 0 to always stream in O(p) memory; over-budget data
    falls back to streaming on its own.

    Returns a dict with `coef`, `intercept`, `r2` (test set), `train_n`,
    `test_n`, `epochs`, and `final_train_mse`.

    Raises:
        ValueError: if the fit diverges, which usually means `learning_rate`
            is too high for the data.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_sgd_regression requires the native Rust extension; build with `maturin develop`."
        )
    return native.csv_sgd_regression(
        path,
        target=target,
        features=features,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        train_frac=train_frac,
        seed=seed,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
        shuffle=shuffle,
        grad_clip=grad_clip,
        cache_budget_mb=cache_budget_mb,
    )


def csv_logistic_regression(
    path: str,
    *,
    target: str,
    features: list[str] | None = None,
    epochs: int = 5,
    learning_rate: float = 0.05,
    l2: float = 0.0,
    train_frac: float = 0.8,
    seed: int = 0,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
    shuffle: bool = True,
    grad_clip: float = 10.0,
    cache_budget_mb: int = 512,
) -> dict[str, Any]:
    """Fit a binary logistic classifier by SGD, streaming from CSV.

    The classification counterpart to :func:`csv_sgd_regression`, sharing all of
    its machinery -- on-the-fly standardization, the deterministic split, the
    cached-replay optimisation, gradient clipping -- with log-loss in place of
    squared error. Memory is O(p) in the feature count, independent of rows.

    The target column must contain only 0 and 1. A parseable label outside that
    set raises rather than being coerced, because a classifier quietly trained
    on labels of 2 and 7 returns a plausible-looking model that means nothing.

    `coef` and `intercept` are returned in the original feature space and
    parameterise a logit: ``p = sigmoid(intercept + coef . x)``. They are
    directly comparable with scikit-learn's
    ``LogisticRegression(penalty=None)``.

    See :func:`csv_sgd_regression` for `grad_clip`, `cache_budget_mb`, and the
    row-order caveat, which behave identically here.

    Returns a dict with `coef`, `intercept`, `train_n`, `test_n`, `epochs`,
    `final_train_loss` (mean training log-loss), and held-out `accuracy`,
    `log_loss`, `roc_auc`, `positive_rate`, and `confusion_matrix` (a dict of
    `tp`/`fp`/`tn`/`fn`). `r2` is present for symmetry with the regression
    path but means little for a classifier.

    ROC-AUC is computed from a 4096-bin histogram of the held-out scores rather
    than by sorting them, which keeps evaluation within the same bounded memory
    as the fit; it agrees with scikit-learn to about 1e-3.

    Raises:
        ValueError: if the target contains a label outside {0, 1}, or if the
            fit diverges, which usually means `learning_rate` is too high.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_logistic_regression requires the native Rust extension; "
            "build with `maturin develop`."
        )
    return native.csv_logistic_regression(
        path,
        target=target,
        features=features,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        train_frac=train_frac,
        seed=seed,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
        shuffle=shuffle,
        grad_clip=grad_clip,
        cache_budget_mb=cache_budget_mb,
    )


def csv_gaussian_nb(
    path: str,
    *,
    target: str,
    features: list[str] | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
    shuffle: bool = True,
    var_smoothing: float = 1e-9,
) -> dict[str, Any]:
    """Fit a binary Gaussian Naive Bayes classifier from CSV.

    Naive Bayes is the model whose fitted parameters *are* summary statistics:
    a class prior and a per-(class, feature) mean and variance. Training is one
    chunk-parallel grouped-moments pass over the file and evaluation a second,
    so there are no epochs, no learning rate, and no cache — memory is
    O(classes × features) regardless of file size, at full parallel speed.
    This is the strongest bounded-memory guarantee of any model here: even
    `cache_budget_mb` does not exist because nothing needs caching.

    The target column must contain only 0 and 1, as for
    :func:`csv_logistic_regression`; a parseable label outside that set raises.

    `var_smoothing` matches scikit-learn's ``GaussianNB`` exactly — the largest
    per-feature variance of the pooled training data, times this factor, is
    added to every class variance so a constant feature cannot inject a
    division by zero.

    Returns a dict with `priors`, `theta` (per-class means), `var` (per-class
    smoothed variances) — directly comparable with scikit-learn's ``theta_``
    and ``var_`` — plus `class_counts`, `train_n`, `test_n`, and held-out
    `accuracy`, `log_loss`, `roc_auc`, `positive_rate`, and
    `confusion_matrix`.

    Raises:
        ValueError: if the target contains a parseable label outside {0, 1}.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_gaussian_nb requires the native Rust extension; "
            "build with `maturin develop`."
        )
    return native.csv_gaussian_nb(
        path,
        target=target,
        features=features,
        train_frac=train_frac,
        seed=seed,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
        shuffle=shuffle,
        var_smoothing=var_smoothing,
    )


def csv_classification_metrics(
    path: str,
    *,
    label: str,
    score: str,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
) -> dict[str, Any]:
    """Score a predictions file: labels and probabilities in, metrics out.

    The fits report metrics on their own held-out split; this covers everything
    else — a model trained elsewhere, an older grizzly model reapplied, a
    vendor's scores — where the predictions already exist as two CSV columns
    and the question is how good they are. One chunk-parallel pass in O(1)
    memory per row, same accumulator the fits use, so the numbers are
    directly comparable with theirs.

    `label` must contain only 0 and 1 where parseable; a parseable value
    outside that set raises. Rows where either column is missing, unparseable,
    or the score is not finite are skipped and counted in `n_skipped` — never
    silently dropped. ROC-AUC uses the same 4096-bin streaming estimate as the
    fits (agrees with scikit-learn to ~1e-3); accuracy and log-loss are exact.

    Returns a dict with `n`, `n_skipped`, `accuracy`, `log_loss`, `roc_auc`,
    `positive_rate`, and `confusion_matrix`.

    Raises:
        ValueError: if the label column contains a parseable value outside
            {0, 1}, or a named column does not exist.
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_classification_metrics requires the native Rust extension; "
            "build with `maturin develop`."
        )
    return native.csv_classification_metrics(
        path,
        label=label,
        score=score,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
    )


def csv_linear_regression(
    path: str,
    *,
    target: str,
    features: list[str] | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    sample_size: int = 1_000_000,
    delimiter: str | None = None,
    has_header: bool | None = None,
    fast_csv: bool = True,
    shuffle: bool = True,
    ridge_lambda: float = 0.0,
    return_debug: bool = False,
) -> dict[str, Any]:
    """
    Rust-native linear regression on CSV/CSV.GZ (no numpy required).

    Returns:
      { "r2": ..., "coef": [...], "intercept": ..., "train_n": ..., "test_n": ... }
    """
    native = _load_native()
    if native is None:
        raise RuntimeError(
            "csv_linear_regression requires the native Rust extension; build with `maturin develop`."
        )
    return native.csv_linear_regression(
        path,
        target=target,
        features=features,
        train_frac=train_frac,
        seed=seed,
        sample_size=sample_size,
        delimiter=delimiter,
        has_header=has_header,
        fast_csv=fast_csv,
        shuffle=shuffle,
        ridge_lambda=ridge_lambda,
        return_debug=return_debug,
    )
