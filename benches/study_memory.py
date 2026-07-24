"""Memory-ceiling study: how small a machine can transform how large a file?

Wall-clock benchmarks answer "how fast", which is only interesting once the
answer to "does it run at all" is yes. For a training pipeline the second
question is often the binding one: a nightly job on a 2 GB worker either
processes the day's data or it does not.

This study transforms one fixed file under progressively tighter container
memory limits and records, for each library, whether it finished and what it
peaked at. A library that materialises the dataset gets killed by the OOM
killer (SIGKILL, exit 137); one that streams keeps going.

Docker is required, because a container memory limit is the only honest way to
impose a ceiling -- `ulimit -v` bounds address space rather than resident
memory, which mmap-based readers sail straight through.

Usage::

    python -m benches.study_memory --rows 3000000
    python -m benches.study_memory --caps 900m 500m 250m
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from benches.gen_data import write_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent

# Descending, so the point where each library stops coping is obvious.
DEFAULT_CAPS = ("900m", "700m", "500m", "350m", "250m")

LIBRARIES = ("grizzly", "polars")

# Under a container memory limit the kernel sends SIGKILL, which Docker reports
# as 128 + 9.
OOM_EXIT_CODE = 137


def build_image(image: str) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout.strip()
    subprocess.run(
        [
            "docker",
            "build",
            "--quiet",
            "--build-arg",
            f"GIT_COMMIT={commit or 'unknown'}",
            "-t",
            image,
            ".",
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_one(image: str, library: str, cap: str, cpus: int, data_rel: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            f"--cpus={cpus}",
            f"--memory={cap}",
            "-v",
            f"{REPO_ROOT / 'data'}:/app/data",
            image,
            "python",
            "benches/_memory_runner.py",
            library,
            data_rel,
            f"data/out_{library}.csv",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    if proc.returncode == 0:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"status": "ok", "exit_code": 0, **payload}
    if proc.returncode == OOM_EXIT_CODE:
        return {"status": "oom_killed", "exit_code": proc.returncode}
    return {
        "status": "failed",
        "exit_code": proc.returncode,
        "stderr": proc.stderr.strip()[-400:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--caps", nargs="+", default=list(DEFAULT_CAPS))
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--image", default="grizzly:local")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benches" / "results" / "memory_study.json",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data"
    dataset = data_dir / f"memory_study_{args.rows}.csv"
    if not dataset.exists():
        print(f"generating {dataset.name} ...")
        write_dataset(dataset, shape="numeric", n_rows=args.rows, n_features=args.features, seed=0)
    input_mb = dataset.stat().st_size / 1e6
    print(f"input: {dataset.name}  {input_mb:.1f} MB\n")

    print("building image ...")
    build_image(args.image)

    results: list[dict[str, Any]] = []
    print(f"\n{'cap':<8} {'library':<9} {'result':<12} {'time':>8} {'peak RSS':>11}")
    print("-" * 54)

    for cap in args.caps:
        for library in LIBRARIES:
            outcome = run_one(
                args.image, library, cap, args.cpus, str(dataset.relative_to(REPO_ROOT))
            )
            outcome |= {"cap": cap, "library": library}
            results.append(outcome)

            if outcome["status"] == "ok":
                print(
                    f"{cap:<8} {library:<9} {'OK':<12} "
                    f"{outcome['seconds']:7.2f}s {outcome['peak_rss_mb']:10.1f} MB"
                )
            elif outcome["status"] == "oom_killed":
                print(f"{cap:<8} {library:<9} {'OOM-KILLED':<12} {'—':>8} {'—':>11}")
            else:
                print(f"{cap:<8} {library:<9} {'FAILED':<12}  exit={outcome['exit_code']}")

    report = {
        "schema_version": 1,
        "dataset": {
            "rows": args.rows,
            "feature_columns": args.features,
            "input_bytes": dataset.stat().st_size,
        },
        "methodology": {
            "cpus": args.cpus,
            "oom_exit_code": OOM_EXIT_CODE,
            "note": (
                "Container memory limits are used because ulimit -v bounds "
                "address space rather than resident memory, which an mmap-based "
                "reader passes straight through."
            ),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    survived = {
        lib: [r["cap"] for r in results if r["library"] == lib and r["status"] == "ok"]
        for lib in LIBRARIES
    }
    for lib, caps in survived.items():
        floor = caps[-1] if caps else "none"
        print(f"  {lib:<9} completed down to: {floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
