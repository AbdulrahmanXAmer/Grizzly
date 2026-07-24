<div align="center">

# Grizzly

<p align="center">
  <strong>Rust-powered data profiling and schema inference for Python</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyO3-FFD43B?style=for-the-badge&logo=python&logoColor=black" alt="PyO3">
</p>

<p align="center">
  <a href="../../actions/workflows/ci.yml"><img src="../../actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="#performance"><img src="https://img.shields.io/badge/Benchmarks-reproducible-success?style=flat-square" alt="Reproducible benchmarks"></a>
  <img src="https://img.shields.io/badge/Wheels-abi3--py310-blue?style=flat-square" alt="abi3 wheels">
  <img src="https://img.shields.io/badge/No_NumPy-Required-orange?style=flat-square" alt="No NumPy">
  <img src="https://img.shields.io/badge/Native_ML-Linear_Regression-purple?style=flat-square" alt="Native ML">
</p>

---

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#features">Features</a> •
  <a href="#performance">Performance</a> •
  <a href="#ml-workflows">ML Workflows</a> •
  <a href="#api-reference">API</a> •
  <a href="#development">Development</a>
</p>

</div>

---

## Overview

Grizzly is a Python package with a **Rust (PyO3) core** for high-performance data operations:

<table>
<tr>
<td width="50%">

### What It Does

- Schema/column detection from any Python data
- CSV/CSV.GZ profiling with stats & percentiles
- Fast min-max scaling transforms
- Rust-native linear regression (no NumPy!)

</td>
<td width="50%">

### Why It's Fast

- Rust core via PyO3 bindings
- Sampling-first design
- Parallel chunked CSV processing
- Zero-copy where possible

</td>
</tr>
</table>

---

## Features

<table>
<tr>
<td align="center" width="25%">
<br>
<strong>Schema Inference</strong>
<br><br>
<sub>Detect types from nested dicts, lists, iterables</sub>
</td>
<td align="center" width="25%">
<br>
<strong>CSV Profiling</strong>
<br><br>
<sub>Stats, percentiles, outliers, missing data</sub>
</td>
<td align="center" width="25%">
<br>
<strong>Fast Transforms</strong>
<br><br>
<sub>Streaming min-max scaling, CSV in to CSV out</sub>
</td>
<td align="center" width="25%">
<br>
<strong>Native ML</strong>
<br><br>
<sub>Linear regression without NumPy</sub>
</td>
</tr>
</table>

<details>
<summary><strong>Path Notation (Schema Inference)</strong></summary>
<br>

| Pattern | Example | Description |
|---------|---------|-------------|
| Dict keys | `user.name` | Keys joined with `.` |
| Arrays | `items[].id` | Arrays add `[]` |
| Nested | `matrix[][].value` | Multi-dimensional |

</details>

---

## Quickstart

### Requirements

```
Python >= 3.10
Rust toolchain (rustup recommended, source builds only)
```

### Installation

```bash
# Clone and setup
cd grizzly
python3 -m venv .venv
source .venv/bin/activate

# Install build tools
python -m pip install -U pip maturin pytest

# Build native extension
maturin develop --release

# Verify installation
python -c "import grizzly; print('native:', grizzly.is_native())"
```

<details>
<summary><strong>Optional: Install all extras</strong></summary>

```bash
python -m pip install ".[all]"
```

</details>

---

## API Reference

### 1. Schema Inference

```python
import grizzly

data = [
    {"user": {"id": 1, "name": "Ada"}, "items": [{"id": 10}, {"id": 11}]},
    {"user": {"id": 2, "name": None}, "items": []},
]

schema = grizzly.detect_schema(data, sample_size=1000)
cols = grizzly.detect_columns(data)
grizzly.info(data, show_examples=True)
```

<details>
<summary><strong>Normalize various data sources</strong></summary>

```python
import grizzly

# Works with pandas, numpy, pyarrow, CSV paths
records = grizzly.normalize("data.parquet", sample_size=1000)
schema = grizzly.detect_schema(records)
```

</details>

### 2. CSV Profiling / EDA

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=100_000)

# Full profile (types + examples + mode)
prof = g.csv_profile(lite=False)

# Fast EDA report
rep = g.eda(lite=True, return_json=True)
print(rep["dataset"])
print(rep["missing"][:3])
print(rep["numeric"][:1])
```

### 3. Min-Max Scaling

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000)
scaler = g.fit_minmax()
scaler.transform("data_scaled.csv")
```

<details>
<summary><strong>Lower-level API</strong></summary>

```python
import grizzly

params = grizzly.csv_minmax_params("data.csv.gz", sample_size=100_000)["params"]
grizzly.csv_transform_minmax("data.csv.gz", "data_scaled.csv", params, delimiter=None)
```

</details>

### 4. Linear Regression (Rust-Native)

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000)
res = g.fit_linear_regression(target="col_9", train_frac=0.8, seed=0)

print(f"R²: {res['r2']}")
print(f"Coefficients: {len(res['coef'])}")
print(f"Intercept: {res['intercept']}")
```

---

## ML Workflows

Grizzly focuses on **fast linear models** with a pragmatic API:

<table>
<tr>
<th width="50%">Rust-Native (No NumPy)</th>
<th width="50%">NumPy-Based</th>
</tr>
<tr>
<td>

Train directly from CSV/CSV.GZ
- Fastest path to baseline model
- No array conversion overhead
- Built-in train/test split

</td>
<td>

Convert to arrays first
- Sklearn-style API
- More preprocessing options
- Ridge regression included

</td>
</tr>
</table>

### Rust-Native Regression

```python
import grizzly

g = grizzly.Grizzly("data.csv.gz", sample_size=1_000_000, fast_csv=True)

# Optional: select specific columns
g = g.select(["col_0", "col_3", "col_9"])

res = g.fit_linear_regression(
    target="col_9",
    features=["col_0", "col_3"],  # default: all except target
    train_frac=0.8,
    seed=0,
    shuffle=True,
    ridge_lambda=0.0,
    return_debug=False,
)

print(f"R²: {res['r2']:.4f}")
print(f"Train: {res['train_n']}, Test: {res['test_n']}")
```

<details>
<summary><strong>Return Values</strong></summary>

| Key | Type | Description |
|-----|------|-------------|
| `r2` | float | Test-set R² |
| `coef` | list | Feature coefficients |
| `intercept` | float | Model intercept |
| `train_n` | int | Training rows used |
| `test_n` | int | Test rows used |

With `return_debug=True`:
- `test_n_assigned`, `ss_res`, `ss_tot`, `y_mean_test`

</details>

### NumPy Regression

```python
import grizzly
from grizzly.ml import LinearRegression, RidgeRegression

g = grizzly.Grizzly("data.csv.gz", sample_size=200_000, fast_csv=True)
X, y = g.to_numpy(sampled=True, dtype="float32", target="col_9")

lr = LinearRegression().fit(X, y)
print(f"LR R²: {lr.score(X, y):.4f}")

ridge = RidgeRegression(alpha=1.0).fit(X, y)
print(f"Ridge R²: {ridge.score(X, y):.4f}")
```

---

## Performance

All numbers below are generated by the benchmark suite in
[`benches/`](benches/README.md) and written into this README by
`python -m benches.render --write`. They are never typed by hand, and CI fails
if this section drifts from the committed `benches/results/results.json`.

Reproduce with:

```bash
python -m pip install -r benches/requirements.txt
maturin develop --release
python -m benches.bench --strict
```

<!-- BENCH:START -->

<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->

| Workload | Dataset | Grizzly | vs pandas | vs polars |
|----------|---------|--------:|-----------|-----------|
| **Profile** 📉 | numeric, 100,000 rows (19.0 MB) | 27.8 ms | 13.65x faster | 3.09x faster |
| **Profile** 📉 | numeric, 500,000 rows (95.2 MB) | 227.9 ms | 9.58x faster | 1.57x faster |
| **Profile** 📉 | mixed, 100,000 rows (14.2 MB) | 88.4 ms | 4.56x faster | 1.12x slower |
| **Profile** 📉 | mixed, 500,000 rows (70.9 MB) | 360.8 ms | 4.53x faster | 1.39x slower |
| **Transform** 📉 | numeric, 100,000 rows (19.0 MB) | 94.5 ms | 22.47x faster | 1.02x slower |
| **Transform** 📉 | numeric, 500,000 rows (95.2 MB) | 1470.4 ms | 15.95x faster | 1.10x faster |

> 📉 **These measurements are too noisy to publish.** In the cells below, at least one library's standard deviation exceeded 20% of its median, which means the measuring machine was busy and the ratios above are not reliable. Re-run `python -m benches.bench --strict && python -m benches.render --write` on an idle machine before quoting these numbers.
>
> - Profile / numeric_100000 (grizzly, pandas, polars)
> - Profile / numeric_500000 (grizzly, polars)
> - Profile / mixed_100000 (polars)
> - Profile / mixed_500000 (grizzly, pandas, polars)
> - Transform / numeric_100000 (grizzly, polars)
> - Transform / numeric_500000 (grizzly, polars)

<details>
<summary><strong>Measurement environment and methodology</strong></summary>

| | |
|---|---|
| CPU | Apple M1 (8 cores) |
| Memory | 8.0 GB |
| Platform | macOS-15.7.7-arm64-arm-64bit-Mach-O |
| Python | 3.14.6 (CPython) |
| Rust | rustc 1.97.1 (8bab26f4f 2026-07-14) |
| Cargo profile | release (lto=true, codegen-units=1, opt-level=3) |
| Libraries | grizzly 0.1.0, pandas 3.0.5, polars 1.43.0 |
| Grizzly commit | `606711d52c42 (dirty working tree)` |
| Measured | 2026-07-24T18:29:15+00:00 |

- **Repetitions:** 7 timed runs per cell, 1 warmup run discarded; headline figure is the median.
- **Isolation:** one fresh interpreter per repetition.
- **Timed region:** profile: read CSV + compute per-column stats; transform: read CSV + min-max scale numeric columns + write CSV.
- **Equivalence:** every library's per-column output is fingerprinted and compared; a run where implementations disagree is reported as a mismatch rather than a speedup.
- **Sampling:** Grizzly's `sample_size` is set to 4x the row count and full row coverage is asserted, so it is not credited for reading less data than the libraries it is compared against.

Reproduce with `python -m benches.bench --strict`. See [`benches/README.md`](benches/README.md) for the full methodology, including known limitations.

</details>

<details>
<summary><strong>Per-cell detail</strong></summary>

**Profile — numeric, 100,000 rows (19.0 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 27.8 ms | 15.1 ms | 27.2 ms | 51.8 MB |
| polars | 85.8 ms | 47.4 ms | 60.5 ms | 95.6 MB |
| pandas | 379.0 ms | 187.2 ms | 242.0 ms | 146.7 MB |

**Profile — numeric, 500,000 rows (95.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 227.9 ms | 95.3 ms | 164.2 ms | 129.5 MB |
| polars | 357.5 ms | 85.3 ms | 291.9 ms | 236.2 MB |
| pandas | 2183.0 ms | 390.7 ms | 1571.6 ms | 236.0 MB |

**Profile — mixed, 100,000 rows (14.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| polars | 78.6 ms | 38.8 ms | 59.1 ms | 104.3 MB |
| **grizzly** | 88.4 ms | 9.3 ms | 79.2 ms | 73.3 MB |
| pandas | 403.5 ms | 74.2 ms | 374.4 ms | 141.7 MB |

**Profile — mixed, 500,000 rows (70.9 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| polars | 260.2 ms | 87.9 ms | 162.5 ms | 261.8 MB |
| **grizzly** | 360.8 ms | 112.5 ms | 331.3 ms | 162.9 MB |
| pandas | 1633.1 ms | 801.9 ms | 1368.4 ms | 254.4 MB |

**Transform — numeric, 100,000 rows (19.0 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| polars | 92.3 ms | 18.6 ms | 68.7 ms | 114.2 MB |
| **grizzly** | 94.5 ms | 36.9 ms | 80.4 ms | 101.5 MB |
| pandas | 2123.4 ms | 124.9 ms | 1950.3 ms | 192.8 MB |

**Transform — numeric, 500,000 rows (95.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 1470.4 ms | 501.2 ms | 712.9 ms | 325.1 MB |
| polars | 1613.4 ms | 821.9 ms | 1287.4 ms | 258.5 MB |
| pandas | 23458.4 ms | 3141.8 ms | 20315.6 ms | 405.7 MB |

</details>

<!-- BENCH:END -->

---

## Performance Knobs

<table>
<tr>
<td width="50%">

### `sample_size`

Grizzly is **sampling-first** by design. Many operations stop after `sample_size` rows.

```python
g = grizzly.Grizzly("big.csv", sample_size=100_000)
```

</td>
<td width="50%">

### `fast_csv`

| Mode | Speed | Compatibility |
|------|-------|---------------|
| `True` | Faster (parallel) | Simple CSVs |
| `False` | Slower | Quoted newlines, tricky CSVs |

</td>
</tr>
</table>

---

## Column Naming

| File Type | Column Names |
|-----------|--------------|
| **With header** | From header row |
| **No header** | `col_0`, `col_1`, ... `col_{n-1}` |

This is why headerless datasets use targets like `col_9`.

> **Note on `count`.** Grizzly's per-column `count` is the number of rows
> *observed*, including nulls. pandas, polars, and SQL all use `count` for the
> non-null tally instead. Use `count - null_count` if you want the
> pandas-compatible meaning.

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r benches/requirements.txt
maturin develop --release
```

Everything CI enforces, runnable locally:

| Command | Gate |
|---------|------|
| `cargo fmt --all -- --check` | Rust formatting |
| `cargo clippy --all-targets -- -D warnings` | Rust lints, warnings are errors |
| `ruff check . && ruff format --check .` | Python lints and formatting |
| `mypy src/grizzly benches` | Type checking |
| `pytest tests` | Test suite |
| `python -m benches.bench --strict` | Benchmarks + cross-library equivalence |
| `python -m benches.render --check` | README numbers match `results.json` |

The test suite runs twice in CI, once without the optional dependencies
(numpy, pandas, pyarrow) and once with, because `normalize()` changes behaviour
depending on what is installed.

### Project status

Alpha. Specific things worth knowing before depending on it:

- **Quantiles are approximate**, and measurably so. Percentiles come from a
  t-digest; `min`, `max`, `mean`, and `std` are exact. Measured by
  [`tests/test_quantile_accuracy.py`](tests/test_quantile_accuracy.py):

  | Metric | Worst measured | What it means |
  |--------|---------------:|---------------|
  | Rank error | **0.16%** | The returned value sits within 0.16 percentage points of the requested quantile's true position. This is the guarantee a t-digest actually makes. |
  | Value error (smooth data) | **0.21% of range** | How far the number itself is from the exact quantile. |
  | Value error (zero-inflated) | **7.8% of range** | Not bounded by the algorithm. See below. |

  The last row is the case to know about. If a column is mostly one value --
  95% zeros, say, which is normal for counts, spend, and sparse features --
  then a quantile landing on the jump is *rank-correct but far from the exact
  value*. For that column the p95 came back as 83.0 where the exact answer was
  5.0, while its rank was exactly right. If you threshold outliers on p95/p99
  of a zero-inflated column, compute those exactly rather than from the
  profile.
- **`std` is a population standard deviation**, where pandas and polars default
  to the sample standard deviation (`ddof=1`).
- **`fast_csv=True` assumes no quoted newlines.** Pass `fast_csv=False` for
  arbitrary CSV.
- **Rust panics abort the process.** The release profile sets `panic = "abort"`,
  so a panic in the extension terminates the interpreter rather than raising a
  Python exception.

---

## Generate Synthetic Data

The benchmark suite ships a deterministic generator, so there is no need to
paste one out of this README:

```bash
# Two shapes: all-float ("numeric") and mixed float/int/categorical/nullable
python -m benches.gen_data --out data/ --rows 100000 --shapes numeric mixed
```

A given `(shape, rows, seed)` always produces a byte-identical file, which is
what makes the benchmark results comparable across machines. See
[`benches/gen_data.py`](benches/gen_data.py).

<details>
<summary><strong>End-to-End Example</strong></summary>

```python
import grizzly

path = "data/synth_100k.csv.gz"
g = grizzly.Grizzly(path, sample_size=1_000_000, fast_csv=True)

# Quick EDA
rep = g.eda(lite=True, return_json=True)
print(rep["dataset"])
print("top missing:", rep["missing"][:3])

# Train model directly from CSV
res = g.fit_linear_regression(target="col_20", train_frac=0.8, seed=0)
print(f"R²: {res['r2']:.4f}, coefficients: {len(res['coef'])}")
```

</details>

---

## Troubleshooting

<details>
<summary><strong>"native: False" or extension not loaded</strong></summary>

1. Ensure Python 3.10 or newer: `python --version`
2. Rebuild: `maturin develop --release`
3. Verify: `python -c "import grizzly; print(grizzly.is_native())"`

</details>

<details>
<summary><strong>KeyError / "target not found" for headerless CSVs</strong></summary>

Use synthetic column names: `col_0`, `col_1`, ... `col_{n-1}`

```python
res = g.fit_linear_regression(target="col_9")  # Not "target" or custom name
```

</details>

<details>
<summary><strong>ModuleNotFoundError: No module named 'grizzly'</strong></summary>

You're using system Python instead of venv. Use explicit paths:

```bash
.venv/bin/python -c "import grizzly; print(grizzly.is_native())"
.venv/bin/python your_script.py
```

</details>

---

<div align="center">

## License

See `LICENSE` file.

---

<sub>Built with Rust and PyO3</sub>

</div>
