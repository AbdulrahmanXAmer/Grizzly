"""Detect distribution drift by comparing a profile against a reference.

The premise: a profile is small, cheap, and comparable. Grizzly can summarise a
dataset in a single streaming pass, so a profile taken at training time can be
saved alongside the model and every later batch compared against it -- without
keeping the training data, and without loading either dataset into memory.

That is the part a faster DataFrame does not give you. `pandas` and `polars`
will both compute these statistics; neither will tell you that today's batch no
longer looks like the data your model was fitted on.

Metrics computed per column:

``psi``
    Population Stability Index, from the reference quantile bins. The standard
    rule of thumb in credit risk and ad-tech: below 0.1 is stable, 0.1-0.25 is
    a moderate shift worth investigating, above 0.25 is a significant shift.
    Bins come from the reference profile's percentiles, so no raw data is
    needed on either side.

``mean_shift``
    Change in mean, expressed in reference standard deviations. Robust to the
    unit a feature is measured in, which raw differences are not.

``null_rate_change``
    Absolute change in the fraction of missing values. A feature that silently
    stops being populated is one of the most common production failures and
    does not show up in a mean or a quantile.

``new_categories`` / ``dropped_categories``
    Modal-value changes for non-numeric columns. Grizzly tracks a mode, not a
    full frequency table, so this detects a change in the dominant level rather
    than the full categorical distribution.

``type_changed``
    The inferred type differs. Usually a schema or upstream-parsing failure
    rather than drift, and worth surfacing separately because the numeric
    metrics become meaningless when it happens.

PSI is computed from quantiles, which Grizzly estimates with a t-digest. That
approximation is bounded in *rank* (see ``tests/test_quantile_accuracy.py``),
which is the right guarantee here: PSI depends on how much probability mass
falls in each bin, and rank error is exactly the error in that mass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .api import csv_profile

# Quantile keys Grizzly reports, in ascending order. These define the PSI bin
# edges, giving 6 bins: (-inf, p25], (p25, p50], ... , (p95, +inf).
QUANTILE_KEYS = ("p25", "median", "p75", "p90", "p95")

# Standard PSI interpretation thresholds.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# A mean shift beyond this many reference standard deviations is called out
# even when PSI stays low, which happens when a distribution slides bodily
# rather than changing shape.
MEAN_SHIFT_SIGNIFICANT = 0.25

# A change in missing-data rate beyond this is reported regardless of PSI.
NULL_RATE_SIGNIFICANT = 0.05

# Guards the log in the PSI sum against an empty bin.
_EPSILON = 1e-6


def save_reference(profile: dict[str, Any], path: str | Path) -> Path:
    """Persist a profile to JSON so later batches can be compared against it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2, default=str))
    return target


def load_reference(path: str | Path) -> dict[str, Any]:
    """Load a profile previously saved by :func:`save_reference`."""
    return json.loads(Path(path).read_text())


def _columns_by_name(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in profile.get("columns", [])}


def _bin_edges(column: dict[str, Any]) -> list[float] | None:
    """Ascending, strictly increasing bin edges from a column's quantiles."""
    values = [column.get(k) for k in QUANTILE_KEYS]
    if any(v is None or not isinstance(v, (int, float)) or math.isnan(v) for v in values):
        return None

    edges: list[float] = []
    for v in values:
        value = float(v)  # type: ignore[arg-type]
        # Duplicate edges mean an empty bin, which makes PSI undefined rather
        # than large. Collapsing them keeps the bin count honest.
        if not edges or value > edges[-1]:
            edges.append(value)
    return edges if len(edges) >= 2 else None


def _bin_proportions(column: dict[str, Any], edges: list[float]) -> list[float] | None:
    """Estimate the mass in each bin from a column's own quantiles.

    Both sides are summarised, not raw, so the mass in a bin is estimated by
    asking where the reference edges fall within the other distribution's
    quantile ladder and interpolating between them.
    """
    ladder: list[tuple[float, float]] = []
    quantile_levels = (0.25, 0.50, 0.75, 0.90, 0.95)
    for key, level in zip(QUANTILE_KEYS, quantile_levels, strict=True):
        value = column.get(key)
        if value is None or not isinstance(value, (int, float)) or math.isnan(value):
            return None
        ladder.append((float(value), level))

    minimum, maximum = column.get("min"), column.get("max")
    if minimum is None or maximum is None:
        return None
    points = [(float(minimum), 0.0), *ladder, (float(maximum), 1.0)]
    points.sort(key=lambda p: p[0])

    def cumulative(x: float) -> float:
        """Fraction of the column at or below x, interpolated on the ladder."""
        if x <= points[0][0]:
            return 0.0
        if x >= points[-1][0]:
            return 1.0
        for (x0, c0), (x1, c1) in zip(points, points[1:], strict=False):
            if x0 <= x <= x1:
                if x1 - x0 < 1e-12:
                    return c1
                return c0 + (c1 - c0) * (x - x0) / (x1 - x0)
        return 1.0

    cuts = [cumulative(e) for e in edges]
    proportions = [cuts[0]]
    for previous, current in zip(cuts, cuts[1:], strict=False):
        proportions.append(max(0.0, current - previous))
    proportions.append(max(0.0, 1.0 - cuts[-1]))

    total = sum(proportions)
    if total <= 0:
        return None
    return [p / total for p in proportions]


def population_stability_index(reference: dict[str, Any], current: dict[str, Any]) -> float | None:
    """PSI between two profiled columns, or None if it cannot be computed.

    sum over bins of (current - reference) * ln(current / reference).
    """
    edges = _bin_edges(reference)
    if edges is None:
        return None

    reference_bins = _bin_proportions(reference, edges)
    current_bins = _bin_proportions(current, edges)
    if reference_bins is None or current_bins is None:
        return None

    psi = 0.0
    for ref_p, cur_p in zip(reference_bins, current_bins, strict=True):
        ref_p = max(ref_p, _EPSILON)
        cur_p = max(cur_p, _EPSILON)
        psi += (cur_p - ref_p) * math.log(cur_p / ref_p)
    return psi


def _null_rate(column: dict[str, Any]) -> float:
    count = column.get("count") or 0
    if not count:
        return 0.0
    # Grizzly's `count` includes nulls, so this is a true rate.
    return float(column.get("null_count") or 0) / float(count)


def compare_column(name: str, reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Drift metrics for one column, with a severity verdict."""
    reasons: list[str] = []

    type_changed = reference.get("inferred") != current.get("inferred")
    if type_changed:
        reasons.append(f"type changed: {reference.get('inferred')} -> {current.get('inferred')}")

    null_change = _null_rate(current) - _null_rate(reference)
    if abs(null_change) > NULL_RATE_SIGNIFICANT:
        reasons.append(f"missing-value rate moved by {null_change:+.1%}")

    psi = None if type_changed else population_stability_index(reference, current)
    if psi is not None and psi >= PSI_SIGNIFICANT:
        reasons.append(f"PSI {psi:.3f} indicates a significant distribution shift")
    elif psi is not None and psi >= PSI_MODERATE:
        reasons.append(f"PSI {psi:.3f} indicates a moderate distribution shift")

    mean_shift = None
    ref_mean, cur_mean = reference.get("mean"), current.get("mean")
    ref_std = reference.get("std")
    if all(isinstance(v, (int, float)) for v in (ref_mean, cur_mean, ref_std)):
        if ref_std and float(ref_std) > 1e-12:
            mean_shift = (float(cur_mean) - float(ref_mean)) / float(ref_std)  # type: ignore[arg-type]
            if abs(mean_shift) > MEAN_SHIFT_SIGNIFICANT:
                reasons.append(f"mean moved by {mean_shift:+.2f} reference std")

    mode_changed = False
    ref_mode, cur_mode = reference.get("mode"), current.get("mode")
    if ref_mode is not None and cur_mode is not None and ref_mode != cur_mode:
        mode_changed = True
        reasons.append(f"most common value changed: {ref_mode!r} -> {cur_mode!r}")

    if type_changed or (psi is not None and psi >= PSI_SIGNIFICANT):
        severity = "significant"
    elif reasons:
        severity = "moderate"
    else:
        severity = "stable"

    return {
        "column": name,
        "severity": severity,
        "psi": psi,
        "mean_shift_in_std": mean_shift,
        "null_rate_change": null_change,
        "type_changed": type_changed,
        "mode_changed": mode_changed,
        "reference_type": reference.get("inferred"),
        "current_type": current.get("inferred"),
        "reasons": reasons,
    }


def compare_profiles(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare two profiles column by column.

    Columns present in only one side are reported separately rather than being
    silently ignored: a disappeared feature is a more urgent problem than a
    drifted one, and a training pipeline that quietly fills it with nulls will
    produce a model that looks fine and predicts badly.
    """
    reference_columns = _columns_by_name(reference)
    current_columns = _columns_by_name(current)

    shared = sorted(set(reference_columns) & set(current_columns))
    missing = sorted(set(reference_columns) - set(current_columns))
    added = sorted(set(current_columns) - set(reference_columns))

    results = [
        compare_column(name, reference_columns[name], current_columns[name]) for name in shared
    ]
    results.sort(
        key=lambda r: (
            {"significant": 0, "moderate": 1, "stable": 2}[r["severity"]],
            -(r["psi"] or 0.0),
        )
    )

    counts = dict.fromkeys(("significant", "moderate", "stable"), 0)
    for result in results:
        counts[result["severity"]] += 1

    if missing or counts["significant"]:
        verdict = "significant"
    elif counts["moderate"] or added:
        verdict = "moderate"
    else:
        verdict = "stable"

    return {
        "verdict": verdict,
        "counts": counts,
        "columns": results,
        "missing_columns": missing,
        "new_columns": added,
        "reference_rows": reference.get("rows_sampled"),
        "current_rows": current.get("rows_sampled"),
    }


def detect_drift(
    current_path: str,
    reference: str | Path | dict[str, Any],
    *,
    sample_size: int = 1_000_000,
) -> dict[str, Any]:
    """Profile a CSV and compare it against a reference profile.

    Args:
        current_path: CSV to profile now.
        reference: A profile dict, or a path to one saved by
            :func:`save_reference`.
        sample_size: Rows to read. Grizzly is sampling-first, so this bounds the
            work; pass a value above the row count for full coverage.
    """
    reference_profile = reference if isinstance(reference, dict) else load_reference(reference)
    current_profile = csv_profile(current_path, sample_size=sample_size, lite=False)
    report = compare_profiles(reference_profile, current_profile)
    report["current_path"] = current_path
    return report


def format_report(report: dict[str, Any], *, max_columns: int = 20) -> str:
    """Render a drift report as plain text for a terminal or a CI log."""
    lines: list[str] = []
    counts = report["counts"]
    lines.append(
        f"Drift verdict: {report['verdict'].upper()}  "
        f"({counts['significant']} significant, {counts['moderate']} moderate, "
        f"{counts['stable']} stable)"
    )
    if report.get("reference_rows") and report.get("current_rows"):
        lines.append(
            f"  reference rows: {report['reference_rows']:,}   "
            f"current rows: {report['current_rows']:,}"
        )

    if report["missing_columns"]:
        lines.append(f"  MISSING columns: {', '.join(report['missing_columns'])}")
    if report["new_columns"]:
        lines.append(f"  new columns: {', '.join(report['new_columns'])}")

    lines.append("")
    header = f"  {'column':<24} {'severity':<12} {'PSI':>8} {'mean shift':>12} {'null Δ':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for result in report["columns"][:max_columns]:
        psi = "n/a" if result["psi"] is None else f"{result['psi']:.4f}"
        shift = (
            "n/a" if result["mean_shift_in_std"] is None else f"{result['mean_shift_in_std']:+.2f}"
        )
        lines.append(
            f"  {result['column']:<24} {result['severity']:<12} {psi:>8} "
            f"{shift:>12} {result['null_rate_change']:>+8.1%}"
        )

    flagged = [r for r in report["columns"] if r["reasons"]]
    if flagged:
        lines.append("")
        lines.append("  Why:")
        for result in flagged[:max_columns]:
            for reason in result["reasons"]:
                lines.append(f"    {result['column']}: {reason}")

    return "\n".join(lines)
