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
        "| Workload | Dataset | Grizzly | vs pandas | vs polars |",
        "|----------|---------|--------:|-----------|-----------|",
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
        flag = "" if measurement["equivalence"]["status"] == "ok" else " ⚠️"
        if any(_is_noisy(result) for result in results.values()):
            flag += " \U0001f4c9"
        lines.append(
            f"| **{title}**{flag} | {label} | {grizzly_ms:.1f} ms | {cells[0]} | {cells[1]} |"
        )

    noisy = noisy_cells(report)
    if noisy:
        lines += [
            "",
            f"> \U0001f4c9 **These measurements are too noisy to publish.** In the cells below, "
            f"at least one library's standard deviation exceeded "
            f"{NOISE_THRESHOLD:.0%} of its median, which means the measuring machine was "
            "busy and the ratios above are not reliable. Re-run "
            "`python -m benches.bench --strict && python -m benches.render --write` "
            "on an idle machine before quoting these numbers.",
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


def splice(readme_text: str, section: str) -> str:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1:
        raise SystemExit(
            f"README is missing the {START_MARKER} / {END_MARKER} markers; "
            "add them around the benchmark section first."
        )
    if end < start:
        raise SystemExit(f"{END_MARKER} appears before {START_MARKER} in the README.")
    return readme_text[:start] + section + readme_text[end + len(END_MARKER) :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
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
    section = render_section(report)
    current = args.readme.read_text()
    updated = splice(current, section)

    if args.check:
        if current != updated:
            print(
                "README benchmark tables are out of sync with "
                f"{args.results.relative_to(REPO_ROOT)}.\n"
                "Run: python -m benches.render --write",
                file=sys.stderr,
            )
            return 1
        print("README benchmark tables are in sync.")
        return 0

    args.readme.write_text(updated)
    print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
