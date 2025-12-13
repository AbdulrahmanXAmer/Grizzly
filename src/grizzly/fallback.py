from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple


def _is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))


def _type_name(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int) and not isinstance(x, bool):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "string"
    if isinstance(x, (bytes, bytearray)):
        return "bytes"
    return type(x).__name__


@dataclass
class _Col:
    count: int = 0
    null_count: int = 0
    types: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)


def _add_example(col: _Col, v: Any, max_examples: int) -> None:
    if len(col.examples) >= max_examples:
        return
    try:
        s = repr(v)
    except Exception:
        s = "<unrepr>"
    col.examples.append(s)


def _flatten(
    out: MutableMapping[str, _Col],
    path: str,
    v: Any,
    *,
    max_examples: int,
    budget: List[int],
) -> None:
    # budget[0] is remaining nodes to traverse
    if budget[0] <= 0:
        return
    budget[0] -= 1

    if isinstance(v, Mapping):
        for k, vv in v.items():
            kk = str(k)
            p = f"{path}.{kk}" if path else kk
            _flatten(out, p, vv, max_examples=max_examples, budget=budget)
        return

    if _is_seq(v):
        p = f"{path}[]" if path else "[]"
        # sample a prefix only (budget already bounds this)
        for vv in v[: min(len(v), 50)]:
            _flatten(out, p, vv, max_examples=max_examples, budget=budget)
        return

    col = out.setdefault(path or "value", _Col())
    col.count += 1
    if v is None:
        col.null_count += 1
    col.types.add(_type_name(v))
    _add_example(col, v, max_examples)


def detect_schema(data: Any, *, sample_size: int = 1000, max_examples: int = 5) -> Dict[str, Any]:
    cols: Dict[str, _Col] = {}
    budget = [max(1, sample_size)]

    # treat top-level list/tuple as "rows" if it looks like a batch
    if _is_seq(data):
        for item in data[: min(len(data), sample_size)]:
            _flatten(cols, "", item, max_examples=max_examples, budget=budget)
    else:
        _flatten(cols, "", data, max_examples=max_examples, budget=budget)

    columns = []
    for path, c in cols.items():
        inferred = next(iter(sorted(c.types)), "unknown")
        columns.append(
            {
                "path": path,
                "inferred": inferred,
                "types": sorted(c.types),
                "count": c.count,
                "null_count": c.null_count,
                "examples": c.examples,
            }
        )

    columns.sort(key=lambda x: x["path"])
    return {"columns": columns, "sample_size": sample_size}


