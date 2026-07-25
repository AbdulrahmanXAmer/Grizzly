"""Render ``benches/results/results.json`` into the README benchmark section.

The README never contains hand-typed performance numbers. This script owns the
region between the ``BENCH:START`` and ``BENCH:END`` markers, and CI runs it
with ``--check`` so that a stale or edited table fails the build. That is the
whole point: a published number and a measured number cannot drift apart.

Usage::

    python -m benches.render --write     # regenerate the README section
    python -m benches.render --check     # exit 1 if the README is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "benches" / "results" / "results.json"
DEFAULT_README = REPO_ROOT / "README.md"

START_MARKER = "<!-- BENCH:START -->"
END_MARKER = "<!-- BENCH:END -->"

STUDY_START_MARKER = "<!-- STUDY:START -->"
STUDY_END_MARKER = "<!-- STUDY:END -->"

DEFAULT_STUDY = REPO_ROOT / "benches" / "results" / "sampling_study.json"
DEFAULT_MEMORY = REPO_ROOT / "benches" / "results" / "memory_study.json"
DEFAULT_FIT = REPO_ROOT / "benches" / "results" / "fit_results.json"

MEMORY_START_MARKER = "<!-- MEMORY:START -->"
MEMORY_END_MARKER = "<!-- MEMORY:END -->"

FIT_START_MARKER = "<!-- FIT:START -->"
FIT_END_MARKER = "<!-- FIT:END -->"

DEFAULT_CLASSIFY = REPO_ROOT / "benches" / "results" / "classify_results.json"

CLASSIFY_START_MARKER = "<!-- CLASSIFY:START -->"
CLASSIFY_END_MARKER = "<!-- CLASSIFY:END -->"

CLASSIFY_METHOD_LABELS = {
    "grizzly_logistic": "**Grizzly** logistic SGD (10 epochs)",
    "pandas_sklearn": "pandas → sklearn `LogisticRegression`",
    "polars_sklearn": "polars → sklearn `LogisticRegression`",
    "pandas_sgd": "pandas → sklearn `SGDClassifier` (10 epochs)",
}

HIGHLIGHTS_START_MARKER = "<!-- HIGHLIGHTS:START -->"
HIGHLIGHTS_END_MARKER = "<!-- HIGHLIGHTS:END -->"

FIT_METHOD_LABELS = {
    "grizzly_exact": "**Grizzly** closed-form",
    "grizzly_sgd": "**Grizzly** SGD (10 epochs)",
    "pandas_sklearn": "pandas → sklearn `LinearRegression`",
    "polars_sklearn": "polars → sklearn `LinearRegression`",
    "pandas_sgd": "pandas → sklearn `SGDRegressor` (10 epochs)",
}

BASELINES = ("pandas", "polars")

# A cell whose standard deviation exceeds this fraction of its median was
# measured on a machine too busy to trust. Flagging it is the point: a noisy
# benchmark that looks clean is worse than one that admits it is noisy.
NOISE_THRESHOLD = 0.20

WORKLOAD_TITLES = {
    "profile": "Profile",
    "transform": "Transform",
}


def _human_bytes(n: int | None) -> str:
    if not n:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0  # type: ignore[assignment]
    return "unknown"


def _ratio(grizzly_ms: float, other_ms: float) -> str:
    """Describe Grizzly's median relative to another library's."""
    if grizzly_ms <= 0 or other_ms <= 0:
        return "n/a"
    if other_ms >= grizzly_ms:
        return f"{other_ms / grizzly_ms:.2f}x faster"
    return f"{grizzly_ms / other_ms:.2f}x slower"


def _dataset_label(datasets: dict[str, Any], key: str) -> str:
    meta = datasets[key]
    return f"{meta['shape']}, {meta['rows']:,} rows ({_human_bytes(meta['bytes'])})"


def render_environment(report: dict[str, Any]) -> list[str]:
    env = report["environment"]
    method = report["methodology"]

    versions = {}
    for measurement in report["measurements"]:
        for library, result in measurement["results"].items():
            versions[library] = result["library_version"]
    version_line = ", ".join(f"{lib} {ver}" for lib, ver in sorted(versions.items()))

    commit = env["git"]["commit"]
    commit_display = commit[:12] + (" (dirty working tree)" if env["git"]["dirty"] else "")

    return [
        "<details>",
        "<summary><strong>Measurement environment and methodology</strong></summary>",
        "",
        "| | |",
        "|---|---|",
        f"| CPU | {env['cpu']} ({env['cpu_count']} cores) |",
        f"| Memory | {_human_bytes(env['memory_bytes'])} |",
        f"| Platform | {env['platform']} |",
        f"| Python | {env['python']} ({env['python_implementation']}) |",
        f"| Rust | {env['rustc']} |",
        f"| Cargo profile | {env['cargo_profile']} |",
        f"| Libraries | {version_line} |",
        f"| Grizzly commit | `{commit_display}` |",
        f"| Measured | {env['timestamp_utc']} |",
        "",
        f"- **Repetitions:** {method['repetitions']} timed runs per cell, "
        f"{method['warmup_runs_discarded']} warmup run discarded; "
        f"headline figure is the {method['headline_statistic']}.",
        f"- **Isolation:** {method['process_isolation']}.",
        f"- **Timed region:** {method['timed_region']}.",
        "- **Equivalence:** every library's per-column output is fingerprinted and "
        "compared; a run where implementations disagree is reported as a mismatch "
        "rather than a speedup.",
        f"- **Sampling:** Grizzly's `sample_size` is set to {method['grizzly_sample_size_margin']}x "
        "the row count and full row coverage is asserted, so it is not credited for "
        "reading less data than the libraries it is compared against.",
        "",
        "Reproduce with `python -m benches.bench --strict`. See "
        "[`benches/README.md`](benches/README.md) for the full methodology, "
        "including known limitations.",
        "",
        "</details>",
    ]


def _is_noisy(result: dict[str, Any]) -> bool:
    """True if this cell's spread is too wide for its median to mean much."""
    timing = result["timing"]
    median = timing["median_ms"]
    return median > 0 and timing["stdev_ms"] / median > NOISE_THRESHOLD


def noisy_cells(report: dict[str, Any]) -> list[str]:
    """Labels of every measurement containing at least one noisy library."""
    noisy = []
    for measurement in report["measurements"]:
        offenders = sorted(
            library for library, result in measurement["results"].items() if _is_noisy(result)
        )
        if offenders:
            title = WORKLOAD_TITLES.get(measurement["workload"], measurement["workload"])
            noisy.append(f"{title} / {measurement['dataset']} ({', '.join(offenders)})")
    return noisy


def render_summary(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Workload | Dataset | Grizzly | vs pandas | vs polars | Peak memory |",
        "|----------|---------|--------:|-----------|-----------|-------------|",
    ]
    for measurement in report["measurements"]:
        results = measurement["results"]
        grizzly_ms = results["grizzly"]["timing"]["median_ms"]
        title = WORKLOAD_TITLES.get(measurement["workload"], measurement["workload"])
        label = _dataset_label(report["datasets"], measurement["dataset"])
        cells = [
            _ratio(grizzly_ms, results[b]["timing"]["median_ms"]) if b in results else "n/a"
            for b in BASELINES
        ]

        # Memory belongs next to time, not buried in a details block: a library
        # that wins on wall-clock by holding the dataset in RAM has made a
        # trade, and the reader should see both sides of it.
        grizzly_rss = results["grizzly"]["peak_rss_bytes"]
        polars_rss = results.get("polars", {}).get("peak_rss_bytes")
        if polars_rss:
            memory = (
                f"{_human_bytes(grizzly_rss)} "
                f"<sub>({polars_rss / grizzly_rss:.1f}x less than polars)</sub>"
            )
        else:
            memory = _human_bytes(grizzly_rss)

        flag = "" if measurement["equivalence"]["status"] == "ok" else " ⚠️"
        if any(_is_noisy(result) for result in results.values()):
            flag += " \U0001f4c9"
        lines.append(
            f"| **{title}**{flag} | {label} | {grizzly_ms:.1f} ms | "
            f"{cells[0]} | {cells[1]} | {memory} |"
        )

    noisy = noisy_cells(report)
    if noisy:
        lines += [
            "",
            f"> \U0001f4c9 **Noisy measurement.** In the cells below, at least one "
            f"library's standard deviation exceeded {NOISE_THRESHOLD:.0%} of its "
            "median, so the ratios are indicative rather than precise. Running "
            "`make docker-bench` pins the software environment and the CPU/memory "
            "allocation, which removes most of this; the rest is whatever else the "
            "host is doing, and only an idle machine fixes that.",
            ">",
        ]
        lines += [f"> - {cell}" for cell in noisy]
    return lines


def render_detail(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for measurement in report["measurements"]:
        title = WORKLOAD_TITLES.get(measurement["workload"], measurement["workload"])
        label = _dataset_label(report["datasets"], measurement["dataset"])
        lines += [
            "",
            f"**{title} — {label}**",
            "",
            "| Library | Median | Std dev | Min | Peak RSS |",
            "|---------|-------:|--------:|----:|---------:|",
        ]
        ordered = sorted(
            measurement["results"].items(),
            key=lambda kv: kv[1]["timing"]["median_ms"],
        )
        for library, result in ordered:
            timing = result["timing"]
            name = f"**{library}**" if library == "grizzly" else library
            lines.append(
                f"| {name} | {timing['median_ms']:.1f} ms | "
                f"{timing['stdev_ms']:.1f} ms | {timing['min_ms']:.1f} ms | "
                f"{_human_bytes(result['peak_rss_bytes'])} |"
            )
        if measurement["equivalence"]["status"] != "ok":
            lines += [
                "",
                "> ⚠️ **Equivalence check failed for this cell — the libraries did not "
                "produce the same result, so the timings are not comparable:**",
            ]
            lines += [f"> - {p}" for p in measurement["equivalence"]["problems"]]
    return lines


def render_section(report: dict[str, Any]) -> str:
    lines = [
        START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->",
        "",
        *render_summary(report),
        "",
        *render_environment(report),
        "",
        "<details>",
        "<summary><strong>Per-cell detail</strong></summary>",
        *render_detail(report),
        "",
        "</details>",
        "",
        END_MARKER,
    ]
    return "\n".join(lines)


def render_study(report: dict[str, Any]) -> str:
    """Render the sampling accuracy-vs-speed study.

    The point of the table is the shape of the curve, not any single row: how
    much accuracy each additional row of input actually buys.
    """
    dataset = report["dataset"]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in report["results"]:
        by_variant.setdefault(row["distribution"], []).append(row)

    lines = [
        STUDY_START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->",
        "",
        f"Dataset: {dataset['rows']:,} rows drawn from {dataset['distribution']}, "
        f"seed {dataset['seed']}.",
        "",
    ]

    shuffled = by_variant.get("shuffled", [])
    if shuffled:
        full = max(r["seconds"] for r in shuffled)
        lines += [
            "| Rows read | Coverage | Time | vs full scan | Worst rank error | Worst value error |",
            "|----------:|---------:|-----:|-------------:|-----------------:|------------------:|",
        ]
        for row in shuffled:
            share = row["seconds"] / full if full else 1.0
            lines.append(
                f"| {row['rows_actually_read']:,} | {row['coverage']:.1%} | "
                f"{row['seconds'] * 1000:.1f} ms | {share:.0%} | "
                f"{row['worst_rank_error']:.5f} | "
                f"{row['worst_value_error_fraction']:.4%} |"
            )

    sorted_rows = by_variant.get("sorted", [])
    if sorted_rows and shuffled:
        full_shuffled = shuffled[-1]["worst_rank_error"]
        full_sorted = sorted_rows[-1]["worst_rank_error"]
        lines += [
            "",
            "<details>",
            "<summary><strong>The same sweep on pre-sorted input</strong></summary>",
            "",
            "Rows are read in per-thread chunks, each consuming from its own region "
            "of the file, so a small sample is spread across the whole file rather "
            "than taken from the front. Sampling therefore stays usable on data with "
            "meaningful row order, where `head -n` or `pandas.read_csv(nrows=...)` "
            "would give a badly biased answer.",
            "",
            "What sorted input does cost is t-digest accuracy, which is sensitive to "
            f"insertion order: at full coverage it lands at {full_sorted:.5f} rank "
            f"error against {full_shuffled:.5f} for shuffled input.",
            "",
            "| Rows read | Coverage | Time | Worst rank error | Worst value error |",
            "|----------:|---------:|-----:|-----------------:|------------------:|",
        ]
        for row in sorted_rows:
            lines.append(
                f"| {row['rows_actually_read']:,} | {row['coverage']:.1%} | "
                f"{row['seconds'] * 1000:.1f} ms | "
                f"{row['worst_rank_error']:.5f} | "
                f"{row['worst_value_error_fraction']:.4%} |"
            )
        lines += ["", "</details>"]

    lines += [
        "",
        "Reproduce with `python -m benches.study_sampling`.",
        "",
        STUDY_END_MARKER,
    ]
    return "\n".join(lines)


def render_memory(report: dict[str, Any]) -> str:
    """Render the memory-ceiling study.

    The interesting cell is not a timing but a survival: below some limit one
    library stops finishing at all.
    """
    dataset = report["dataset"]
    input_mb = dataset["input_bytes"] / 1e6

    by_cap: dict[str, dict[str, dict[str, Any]]] = {}
    for row in report["results"]:
        by_cap.setdefault(row["cap"], {})[row["library"]] = row

    output_mb = next(
        (r["output_bytes"] / 1e6 for r in report["results"] if r["status"] == "ok"),
        0.0,
    )

    lines = [
        MEMORY_START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->",
        "",
        f"Transforming a **{input_mb:,.0f} MB** input into a **{output_mb:,.0f} MB** output "
        f"({dataset['rows']:,} rows x {dataset['feature_columns'] + 1} columns), "
        f"under a container memory limit, on {report['methodology']['cpus']} CPUs.",
        "",
        "| Memory limit | Grizzly | polars |",
        "|---|---|---|",
    ]

    def cell(entry: dict[str, Any] | None) -> str:
        if entry is None:
            return "—"
        if entry["status"] == "ok":
            return f"{entry['seconds']:.1f}s &nbsp; <sub>peak {entry['peak_rss_mb']:.0f} MB</sub>"
        if entry["status"] == "oom_killed":
            return "**OOM-killed**"
        return f"failed (exit {entry['exit_code']})"

    for cap, entries in by_cap.items():
        lines.append(f"| {cap} | {cell(entries.get('grizzly'))} | {cell(entries.get('polars'))} |")

    survivors = {
        lib: [r["cap"] for r in report["results"] if r["library"] == lib and r["status"] == "ok"]
        for lib in ("grizzly", "polars")
    }
    grizzly_floor = survivors["grizzly"][-1] if survivors["grizzly"] else "n/a"
    polars_floor = survivors["polars"][-1] if survivors["polars"] else "n/a"

    lines += [
        "",
        f"Grizzly completes at **{grizzly_floor}**, transforming a file "
        f"{input_mb / _cap_to_mb(grizzly_floor):.1f}x larger than the memory it was given "
        f"and writing {output_mb / _cap_to_mb(grizzly_floor):.1f}x more than that. polars "
        f"stops finishing below {polars_floor}: it materialises the frame, so the ceiling "
        "is the dataset rather than the working set.",
        "",
        "This is the axis that decides whether a nightly job on a small worker runs at "
        "all, and no amount of wall-clock advantage substitutes for it. Where both fit "
        "in memory, the timings above are close and sometimes favour polars.",
        "",
        "Reproduce with `make docker-memory-study` (needs Docker: a container limit is "
        "the only honest ceiling, since `ulimit -v` bounds address space rather than "
        "resident memory and an mmap reader passes straight through it).",
        "",
        MEMORY_END_MARKER,
    ]
    return "\n".join(lines)


def render_fit(report: dict[str, Any]) -> str:
    """Render the model-fitting benchmark: CSV on disk to fitted model."""
    dataset = report["dataset"]
    by_method = {r["method"]: r for r in report["results"]}
    grizzly_exact_ms = by_method["grizzly_exact"]["timing"]["median_ms"]
    grizzly_sgd_ms = by_method["grizzly_sgd"]["timing"]["median_ms"]

    def row(method: str, baseline_ms: float) -> str:
        r = by_method[method]
        ms = r["timing"]["median_ms"]
        if method.startswith("grizzly"):
            versus = "—"
        elif ms >= baseline_ms:
            versus = f"{ms / baseline_ms:.1f}x slower"
        else:
            versus = f"{baseline_ms / ms:.1f}x faster"
        return (
            f"| {FIT_METHOD_LABELS[method]} | {ms:.1f} ms | {versus} | "
            f"{r['r2']:.4f} | {_human_bytes(r['peak_rss_bytes'])} |"
        )

    lines = [
        FIT_START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->",
        "",
        f"Workload: **CSV on disk → 80/20 split → fitted model → held-out R²**, "
        f"on {dataset['rows']:,} rows × {dataset['feature_columns']} features "
        f"({_human_bytes(dataset['bytes'])}). Reading — and scaling, where the "
        "method needs it — is inside the timed region, because that is what "
        "training from a file costs.",
        "",
        "| Method | Time | vs Grizzly | R² | Peak memory |",
        "|--------|-----:|-----------|---:|------------:|",
        row("grizzly_exact", grizzly_exact_ms),
        row("pandas_sklearn", grizzly_exact_ms),
        row("polars_sklearn", grizzly_exact_ms),
        row("grizzly_sgd", grizzly_sgd_ms),
        row("pandas_sgd", grizzly_sgd_ms),
    ]

    agreement = report.get("agreement", {})
    if agreement.get("status") == "ok":
        worst = max(r.get("max_coef_deviation", 0.0) for r in report["results"])
        lines += [
            "",
            f"Every method's coefficients agree with the exact-OLS consensus "
            f"(worst deviation {worst:.2%} of the coefficient scale), so these "
            "are timings of the same model, not five different ones.",
        ]
    else:
        lines += [
            "",
            "> ⚠️ **The methods did not agree on the fitted model in this run:**",
        ]
        lines += [f"> - {p}" for p in agreement.get("problems", [])]

    lines += [
        "",
        "Reproduce with `python -m benches.bench_fit --strict`.",
        "",
        FIT_END_MARKER,
    ]
    return "\n".join(lines)


def render_classify(report: dict[str, Any]) -> str:
    """Render the classification benchmark: CSV on disk to a scored classifier."""
    dataset = report["dataset"]
    by_method = {r["method"]: r for r in report["results"]}
    baseline_ms = by_method["grizzly_logistic"]["timing"]["median_ms"]

    def row(method: str) -> str:
        r = by_method[method]
        ms = r["timing"]["median_ms"]
        if method.startswith("grizzly"):
            versus = "—"
        elif ms >= baseline_ms:
            versus = f"{ms / baseline_ms:.1f}x slower"
        else:
            versus = f"{baseline_ms / ms:.1f}x faster"
        # Same standard as the main table: a median whose spread is this wide
        # is not a measurement, and saying so is better than printing it clean.
        marker = " ⚠️" if _is_noisy(r) else ""
        return (
            f"| {CLASSIFY_METHOD_LABELS[method]} | {ms:.1f} ms{marker} | {versus} | "
            f"{r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
            f"{_human_bytes(r['peak_rss_bytes'])} |"
        )

    lines = [
        CLASSIFY_START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->",
        "",
        f"Workload: **CSV on disk → 80/20 split → fitted classifier → held-out "
        f"metrics**, on {dataset['rows']:,} rows × "
        f"{dataset['feature_columns']} features "
        f"({_human_bytes(dataset['bytes'])}). Labels are *sampled* from "
        "`Bernoulli(sigmoid(w·x + b))` rather than thresholded, so the classes "
        "overlap — on perfectly separable data the logistic likelihood has no "
        "finite maximum and every implementation ‘agrees’ only in diverging.",
        "",
        "| Method | Time | vs Grizzly | Accuracy | ROC-AUC | Peak memory |",
        "|--------|-----:|-----------|---------:|--------:|------------:|",
        row("grizzly_logistic"),
        row("pandas_sklearn"),
        row("polars_sklearn"),
        row("pandas_sgd"),
    ]

    # State plainly where grizzly stands, computed rather than asserted, so it
    # cannot quietly go stale into a claim the numbers no longer support.
    grizzly = by_method["grizzly_logistic"]
    rivals = [r for r in report["results"] if not r["method"].startswith("grizzly")]
    fastest = min(rivals, key=lambda r: r["timing"]["median_ms"])
    if fastest["timing"]["median_ms"] < baseline_ms:
        ratio = baseline_ms / fastest["timing"]["median_ms"]
        mem = fastest["peak_rss_bytes"] / grizzly["peak_rss_bytes"]
        lines += [
            "",
            f"**Grizzly does not win this one on wall-clock.** "
            f"{CLASSIFY_METHOD_LABELS[fastest['method']]} fits {ratio:.1f}x "
            f"faster, because a logistic epoch is real arithmetic per row — an "
            f"`exp` and a `ln` — where the regression path accumulates a matrix "
            f"once and solves it. What grizzly keeps is the memory profile: "
            f"{mem:.1f}x less peak RSS, bounded by the feature count rather "
            f"than the file, which is what decides whether the job runs at all "
            f"on a file larger than memory.",
        ]

    agreement = report.get("agreement", {})
    if agreement.get("status") == "ok":
        acc = [r["accuracy"] for r in report["results"]]
        auc = [r["roc_auc"] for r in report["results"]]
        lines += [
            "",
            f"Held-out metrics agree across every method (accuracy within "
            f"{max(acc) - min(acc):.4f}, ROC-AUC within {max(auc) - min(auc):.4f}), "
            "so these are timings of equally good classifiers rather than four "
            "different ones.",
            "",
            "Unlike the regression table above, the gate is on metrics rather "
            "than coefficients — logistic regression has no closed form, so a "
            "converged L-BFGS fit and a ten-epoch SGD fit legitimately land on "
            "different coefficients while classifying about equally well.",
        ]
    else:
        lines += [
            "",
            "> ⚠️ **The methods did not agree on the fitted classifier in this run:**",
        ]
        lines += [f"> - {p}" for p in agreement.get("problems", [])]

    if any(_is_noisy(r) for r in report["results"]):
        lines += [
            "",
            f"⚠️ marks a cell whose standard deviation exceeded "
            f"{NOISE_THRESHOLD:.0%} of its median — the machine was busy and "
            "that number should not be read as a measurement. Re-run on an "
            "idle host, or use `make docker-bench-classify` for a pinned one.",
        ]

    lines += [
        "",
        "Reproduce with `python -m benches.bench_classify --strict`.",
        "",
        CLASSIFY_END_MARKER,
    ]
    return "\n".join(lines)


def render_highlights(
    bench: dict[str, Any] | None,
    fit: dict[str, Any] | None,
    memory: dict[str, Any] | None,
) -> str:
    """The hero strip: four headline claims, every one derived from committed
    results so the prettiest part of the README is held to the same standard
    as the tables it summarises."""

    def tile(big: str, small: str) -> str:
        return f'<td align="center"><h3>{big}</h3><sub>{small}</sub></td>'

    tiles: list[str] = []

    if bench:
        profile_ratios = []
        memory_ratios = []
        for m in bench["measurements"]:
            results = m["results"]
            if "polars" not in results:
                continue
            g, p = results["grizzly"], results["polars"]
            if m["workload"] == "profile":
                profile_ratios.append(p["timing"]["median_ms"] / g["timing"]["median_ms"])
            memory_ratios.append(p["peak_rss_bytes"] / g["peak_rss_bytes"])
        if profile_ratios:
            tiles.append(
                tile(
                    f"{max(profile_ratios):.1f}×",
                    "faster profiling<br>than polars",
                )
            )
        if memory_ratios:
            tiles.append(
                tile(
                    f"{min(memory_ratios):.1f}–{max(memory_ratios):.1f}×",
                    "less peak memory<br>than polars",
                )
            )

    if fit and fit.get("agreement", {}).get("status") == "ok":
        by_method = {r["method"]: r for r in fit["results"]}
        ratio = (
            by_method["pandas_sklearn"]["timing"]["median_ms"]
            / by_method["grizzly_exact"]["timing"]["median_ms"]
        )
        tiles.append(tile(f"{ratio:.0f}×", "faster model fits than<br>pandas + sklearn"))

    if memory:
        survived = [
            r["cap"] for r in memory["results"] if r["library"] == "grizzly" and r["status"] == "ok"
        ]
        polars_died = any(
            r["library"] == "polars" and r["status"] == "oom_killed" for r in memory["results"]
        )
        if survived and polars_died:
            input_mb = memory["dataset"]["input_bytes"] / 1e6
            tiles.append(
                tile(
                    f"{input_mb:,.0f} MB in {survived[-1]}",
                    "transforms where polars<br>is OOM-killed",
                )
            )

    lines = [
        HIGHLIGHTS_START_MARKER,
        "",
        "<!-- Generated by `python -m benches.render --write` from committed",
        "     benchmark results. The hero numbers are held to the same standard",
        "     as the tables: measured, never typed. -->",
        "",
        "<table>",
        "<tr>",
        *tiles,
        "</tr>",
        "</table>",
        "",
        "<sub>All measured in a pinned container · [methodology](benches/README.md) · regenerated by `make render`</sub>",
        "",
        HIGHLIGHTS_END_MARKER,
    ]
    return "\n".join(lines)


def _cap_to_mb(cap: str) -> float:
    """Parse a Docker memory string like '250m' or '2g' into megabytes."""
    cap = cap.strip().lower()
    if cap.endswith("g"):
        return float(cap[:-1]) * 1024
    if cap.endswith("m"):
        return float(cap[:-1])
    return float(cap) / 1e6


def _splice_between(readme_text: str, section: str, start_marker: str, end_marker: str) -> str:
    start = readme_text.find(start_marker)
    end = readme_text.find(end_marker)
    if start == -1 or end == -1:
        raise SystemExit(
            f"README is missing the {start_marker} / {end_marker} markers; "
            "add them around the section first."
        )
    if end < start:
        raise SystemExit(f"{end_marker} appears before {start_marker} in the README.")
    return readme_text[:start] + section + readme_text[end + len(end_marker) :]


def splice(readme_text: str, section: str) -> str:
    return _splice_between(readme_text, section, START_MARKER, END_MARKER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--classify", type=Path, default=DEFAULT_CLASSIFY)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="update the README in place")
    group.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the README does not match results.json",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"{args.results} not found. Run `python -m benches.bench` first.")

    report = json.loads(args.results.read_text())
    current = args.readme.read_text()
    updated = splice(current, render_section(report))

    # The studies are optional: the benchmark tables should still render in a
    # checkout where they have not been run.
    if args.study.exists():
        study = json.loads(args.study.read_text())
        updated = _splice_between(
            updated, render_study(study), STUDY_START_MARKER, STUDY_END_MARKER
        )
    memory = json.loads(args.memory.read_text()) if args.memory.exists() else None
    if memory is not None:
        updated = _splice_between(
            updated, render_memory(memory), MEMORY_START_MARKER, MEMORY_END_MARKER
        )
    fit = json.loads(args.fit.read_text()) if args.fit.exists() else None
    if fit is not None:
        updated = _splice_between(updated, render_fit(fit), FIT_START_MARKER, FIT_END_MARKER)
    if args.classify.exists():
        classify = json.loads(args.classify.read_text())
        updated = _splice_between(
            updated, render_classify(classify), CLASSIFY_START_MARKER, CLASSIFY_END_MARKER
        )

    # The hero strip summarises whichever result sets exist; markers may be
    # absent in a stripped-down README, in which case skip quietly.
    if HIGHLIGHTS_START_MARKER in updated and HIGHLIGHTS_END_MARKER in updated:
        updated = _splice_between(
            updated,
            render_highlights(report, fit, memory),
            HIGHLIGHTS_START_MARKER,
            HIGHLIGHTS_END_MARKER,
        )

    if args.check:
        if current != updated:
            print(
                "README generated sections are out of sync with "
                f"{args.results.relative_to(REPO_ROOT)} or "
                f"{args.study.relative_to(REPO_ROOT)}.\n"
                "Run: python -m benches.render --write",
                file=sys.stderr,
            )
            return 1
        print("README generated sections are in sync.")
        return 0

    args.readme.write_text(updated)
    print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
