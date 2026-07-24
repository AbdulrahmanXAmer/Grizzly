"""Assert the demo actually detected drift, for CI.

The offline fixture injects known drift -- higher fares, longer trips, more
generous tipping. A run that reports none means a stage of the pipeline
silently stopped working, which a zero exit code from the demo alone would not
reveal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: assert_drift.py <demo-json>", file=sys.stderr)
        return 2

    report = json.loads(Path(sys.argv[1]).read_text())["drift"]
    verdict = report["verdict"]
    counts = report["counts"]

    print(f"source:  {json.loads(Path(sys.argv[1]).read_text())['source']}")
    print(f"verdict: {verdict}  {counts}")

    if verdict != "significant":
        print(
            f"FAIL: expected the injected drift to be reported as significant, got {verdict!r}",
            file=sys.stderr,
        )
        return 1

    if counts["significant"] < 1:
        print("FAIL: verdict was significant but no column was flagged", file=sys.stderr)
        return 1

    # A detector that flags everything is as useless as one that flags nothing;
    # the fixture leaves passenger_count deliberately unchanged.
    if counts["stable"] < 1:
        print(
            "FAIL: every column was flagged, including ones the fixture did not "
            "change -- the detector is not discriminating",
            file=sys.stderr,
        )
        return 1

    print("OK: injected drift detected, unchanged columns left alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
