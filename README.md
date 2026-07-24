<div align="center">

# Grizzly

<p align="center">
  <strong>Streaming feature statistics for training pipelines</strong>
  <br>
  <sub>Profile, scale, fit, and detect drift on datasets larger than memory — one pass, bounded memory, Rust core</sub>
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

Grizzly summarises, scales, and fits models over CSV data in a **single
streaming pass**, with a Rust (PyO3) core. Memory stays bounded by the chunk
size rather than the file size, so the input can be larger than RAM.

The point is not to be another DataFrame. It is the layer *underneath* one —
the part of a training pipeline that has to answer "what does this data look
like, is it still the same as last month, and can I fit a baseline on it"
without materialising anything.

<table>
<tr>
<td width="50%">

### What it does

- **Profile** — types, null rates, moments, and approximate quantiles in one pass
- **Standardize / scale** — z-score or min-max, CSV in to CSV out, streaming
- **Fit** — closed-form or streaming SGD, no design matrix
- **Detect drift** — compare a batch against a reference profile saved at training time

</td>
<td width="50%">

### Why it holds up

- Sampling is **stratified across the file**, not a prefix — usable on ordered data
- Quantile error is **measured**, not asserted ([study below](#performance))
- SGD is **O(p) memory**, so feature count is not bounded by RAM
- Every benchmark number here is [regenerated from committed code](benches/README.md)

</td>
</tr>
</table>

### Where it fits

| You want to… | Use |
|---|---|
| Query, join, reshape | polars / DuckDB — Grizzly does not do this |
| Load a dataset that fits in RAM and explore it | pandas / polars |
| Summarise a file **larger than RAM** in one pass | **Grizzly** |
| Know whether today's batch matches training data | **Grizzly** (`grizzly.drift`) |
| Fit a baseline without materialising a matrix | **Grizzly** (`csv_sgd_regression`) |

Grizzly is not trying to beat polars at being polars — on full-scan profiling
they are close, and the numbers below say so plainly. The difference is what
happens next: nothing in a DataFrame library tells you that a feature stopped
being populated last Tuesday.

---

## Quick example

```python
import grizzly
from grizzly import drift

# One streaming pass: types, nulls, moments, quantiles.
profile = grizzly.csv_profile("january.csv", sample_size=1_000_000)

# Save it next to the model. The training data itself is not needed again.
drift.save_reference(profile, "reference.json")

# Standardize for training, streaming CSV to CSV.
params = grizzly.csv_standardize_params("january.csv")["params"]
grizzly.csv_transform_standardize("january.csv", "scaled.csv", params)

# Fit in bounded memory: only the weight vector is held.
model = grizzly.csv_sgd_regression("scaled.csv", target="tip_amount", epochs=10)
print(model["r2"], model["coef"])

# Months later, against the same reference.
report = drift.detect_drift("june.csv", "reference.json")
print(drift.format_report(report))
```

```text
Drift verdict: SIGNIFICANT  (1 significant, 0 moderate, 5 stable)

  column                   severity          PSI   mean shift    null Δ
  ---------------------------------------------------------------------
  tip_amount               significant    0.3483        +0.06    +0.0%
  passenger_count          stable         0.0438        -0.07    +0.0%
  fare_amount              stable         0.0081        +0.01    +0.0%
```

<sub>Real output from `make demo` on NYC yellow taxi trips, January vs June 2024.</sub>

---

## Features

| Capability | Function | Memory |
|---|---|---|
| **Profile** a CSV — types, nulls, moments, quantiles | `csv_profile` | bounded by chunk |
| **Infer a schema** from nested Python data | `detect_schema` | bounded by sample |
| **Standardize** to zero mean / unit variance | `csv_standardize_params` + `csv_transform_standardize` | bounded by chunk |
| **Min-max scale** to [0, 1] | `csv_minmax_params` + `csv_transform_minmax` | bounded by chunk |
| **Fit** exactly (normal equations) | `csv_linear_regression` | O(p²) |
| **Fit** by streaming SGD | `csv_sgd_regression` | **O(p)** |
| **Detect drift** against a saved profile | `grizzly.drift` | profile-only |

The two fit paths return coefficients in the same space, so they are directly
comparable — the SGD one exists for when `p` is large enough that an X'X matrix
stops being an option.

<details>
<summary><strong>What drift detection reports</strong></summary>
<br>

| Metric | Catches |
|---|---|
| `psi` | Distribution shape change, binned on the reference quantiles |
| `mean_shift_in_std` | A distribution sliding bodily, in reference std units |
| `null_rate_change` | A feature that quietly stopped being populated |
| `type_changed` | Upstream schema or parsing failure |
| `mode_changed` | The dominant categorical level changing |
| `missing_columns` | A feature that disappeared entirely |

PSI is suppressed when a column's type changed, where it would be meaningless.
Comparison is profile-to-profile, so the training data does not need to be kept.

</details>

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

# Or, for a comparison that is not at the mercy of whatever else your laptop
# is doing, run it in a container pinned to a fixed CPU and memory allocation:
make docker-bench
```

### Full-scan comparison

<!-- BENCH:START -->

<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->

| Workload | Dataset | Grizzly | vs pandas | vs polars |
|----------|---------|--------:|-----------|-----------|
| **Profile** 📉 | numeric, 100,000 rows (19.0 MB) | 65.3 ms | 4.83x faster | 1.43x faster |
| **Profile** 📉 | numeric, 500,000 rows (95.2 MB) | 297.3 ms | 6.68x faster | 1.22x faster |
| **Profile** | mixed, 100,000 rows (14.2 MB) | 52.0 ms | 4.49x faster | 1.27x faster |
| **Profile** 📉 | mixed, 500,000 rows (70.9 MB) | 222.3 ms | 6.24x faster | 1.84x faster |
| **Transform** 📉 | numeric, 100,000 rows (19.0 MB) | 174.0 ms | 14.48x faster | 1.35x slower |
| **Transform** 📉 | numeric, 500,000 rows (95.2 MB) | 928.2 ms | 14.10x faster | 1.08x slower |

> 📉 **Noisy measurement.** In the cells below, at least one library's standard deviation exceeded 20% of its median, so the ratios are indicative rather than precise. Running `make docker-bench` pins the software environment and the CPU/memory allocation, which removes most of this; the rest is whatever else the host is doing, and only an idle machine fixes that.
>
> - Profile / numeric_100000 (polars)
> - Profile / numeric_500000 (grizzly)
> - Profile / mixed_500000 (pandas, polars)
> - Transform / numeric_100000 (polars)
> - Transform / numeric_500000 (grizzly)

<details>
<summary><strong>Measurement environment and methodology</strong></summary>

| | |
|---|---|
| CPU | unknown (8 cores) |
| Memory | 3.8 GB |
| Platform | Linux-6.12.76-linuxkit-aarch64-with-glibc2.36 |
| Python | 3.12.13 (CPython) |
| Rust | rustc 1.90.0 (1159e78c4 2025-09-14) |
| Cargo profile | release (lto=true, codegen-units=1, opt-level=3) |
| Libraries | grizzly 0.1.0, pandas 3.0.5, polars 1.43.0 |
| Grizzly commit | `unknown` |
| Measured | 2026-07-24T20:11:03+00:00 |

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
| **grizzly** | 65.3 ms | 11.7 ms | 49.5 ms | 39.1 MB |
| polars | 93.4 ms | 123.2 ms | 81.8 ms | 87.3 MB |
| pandas | 315.0 ms | 50.7 ms | 261.4 ms | 130.6 MB |

**Profile — numeric, 500,000 rows (95.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 297.3 ms | 132.7 ms | 256.6 ms | 115.2 MB |
| polars | 362.3 ms | 21.5 ms | 354.2 ms | 227.6 MB |
| pandas | 1984.8 ms | 228.5 ms | 1501.3 ms | 234.7 MB |

**Profile — mixed, 100,000 rows (14.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 52.0 ms | 6.3 ms | 49.8 ms | 50.0 MB |
| polars | 65.8 ms | 11.1 ms | 61.1 ms | 101.0 MB |
| pandas | 233.0 ms | 33.0 ms | 220.3 ms | 137.2 MB |

**Profile — mixed, 500,000 rows (70.9 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| **grizzly** | 222.3 ms | 10.5 ms | 213.3 ms | 133.5 MB |
| polars | 409.5 ms | 123.0 ms | 391.7 ms | 271.1 MB |
| pandas | 1386.2 ms | 295.9 ms | 1098.3 ms | 276.7 MB |

**Transform — numeric, 100,000 rows (19.0 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| polars | 128.8 ms | 27.5 ms | 109.9 ms | 145.7 MB |
| **grizzly** | 174.0 ms | 20.6 ms | 142.8 ms | 79.8 MB |
| pandas | 2520.3 ms | 326.7 ms | 2412.5 ms | 169.1 MB |

**Transform — numeric, 500,000 rows (95.2 MB)**

| Library | Median | Std dev | Min | Peak RSS |
|---------|-------:|--------:|----:|---------:|
| polars | 855.9 ms | 135.2 ms | 780.7 ms | 275.8 MB |
| **grizzly** | 928.2 ms | 242.3 ms | 850.2 ms | 312.0 MB |
| pandas | 13082.9 ms | 297.4 ms | 12563.5 ms | 432.1 MB |

</details>

<!-- BENCH:END -->

### What sampling actually costs

A speed comparison answers the wrong question if the tool gets to read less
data than the thing it is compared against. Grizzly's `sample_size` is exactly
that lever, so here is the tradeoff curve it buys — accuracy against time, on
the same file.

<!-- STUDY:START -->

<!-- Generated by `python -m benches.render --write`. Do not edit by hand. -->

Dataset: 2,000,000 rows drawn from lognormal(0, 1), seed 1234.

| Rows read | Coverage | Time | vs full scan | Worst rank error | Worst value error |
|----------:|---------:|-----:|-------------:|-----------------:|------------------:|
| 2,048 | 0.1% | 1.4 ms | 5% | 0.01085 | 0.0425% |
| 10,112 | 0.5% | 1.4 ms | 5% | 0.00682 | 0.0578% |
| 20,096 | 1.0% | 1.6 ms | 6% | 0.00362 | 0.0402% |
| 100,096 | 5.0% | 2.6 ms | 9% | 0.00120 | 0.0165% |
| 200,064 | 10.0% | 5.3 ms | 19% | 0.00108 | 0.0079% |
| 500,096 | 25.0% | 8.0 ms | 28% | 0.00067 | 0.0035% |
| 1,000,064 | 50.0% | 14.1 ms | 50% | 0.00067 | 0.0046% |
| 1,999,980 | 100.0% | 26.8 ms | 95% | 0.00023 | 0.0090% |
| 2,000,000 | 100.0% | 28.1 ms | 100% | 0.00023 | 0.0090% |

<details>
<summary><strong>The same sweep on pre-sorted input</strong></summary>

Rows are read in per-thread chunks, each consuming from its own region of the file, so a small sample is spread across the whole file rather than taken from the front. Sampling therefore stays usable on data with meaningful row order, where `head -n` or `pandas.read_csv(nrows=...)` would give a badly biased answer.

What sorted input does cost is t-digest accuracy, which is sensitive to insertion order: at full coverage it lands at 0.00110 rank error against 0.00023 for shuffled input.

| Rows read | Coverage | Time | Worst rank error | Worst value error |
|----------:|---------:|-----:|-----------------:|------------------:|
| 2,048 | 0.1% | 1.3 ms | 0.00418 | 0.1257% |
| 10,112 | 0.5% | 1.4 ms | 0.00417 | 0.1234% |
| 20,096 | 1.0% | 1.5 ms | 0.00433 | 0.1221% |
| 100,096 | 5.0% | 2.4 ms | 0.00437 | 0.1160% |
| 200,064 | 10.0% | 3.7 ms | 0.00343 | 0.1096% |
| 500,096 | 25.0% | 6.3 ms | 0.00337 | 0.0935% |
| 1,000,064 | 50.0% | 11.8 ms | 0.00253 | 0.0515% |
| 1,998,488 | 99.9% | 21.7 ms | 0.00108 | 0.0145% |
| 2,000,000 | 100.0% | 20.5 ms | 0.00110 | 0.0169% |

</details>

Reproduce with `python -m benches.study_sampling`.

<!-- STUDY:END -->

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
| `cargo test` | Rust unit tests for the parser and statistics internals |
| `ruff check . && ruff format --check .` | Python lints and formatting |
| `mypy src/grizzly benches` | Type checking |
| `pytest tests` | Python test suite |
| `python -m benches.bench --strict` | Benchmarks + cross-library equivalence |
| `python -m benches.render --check` | README numbers match `results.json` |

### How the tests are layered

| Layer | Where | What it catches |
|-------|-------|-----------------|
| Rust unit tests | [`src/tests.rs`](src/tests.rs) | Internals in isolation, especially that per-chunk merges equal a single pass. A wrong merge yields plausible but incorrect statistics. |
| Differential | [`tests/test_differential.py`](tests/test_differential.py) | Disagreement with pandas, polars, and a closed-form NumPy least-squares solution. Found the train/test split bug. |
| Property-based | [`tests/test_schema_properties.py`](tests/test_schema_properties.py) | Invariants over Hypothesis-generated nested data, plus native-vs-fallback agreement. Found the path-detection bug. |
| Accuracy | [`tests/test_quantile_accuracy.py`](tests/test_quantile_accuracy.py) | Quantile error drifting beyond its measured bounds. |
| Fuzzing | [`fuzz/`](fuzz/) | Panics, out-of-bounds reads, and broken chunk partitioning on arbitrary bytes. |
| Crash regression | [`tests/test_deep_nesting.py`](tests/test_deep_nesting.py) | The stack overflow that used to kill the interpreter. Runs in a subprocess. |

The Python suite runs twice in CI, once without the optional dependencies and
once with, because `normalize()` changes behaviour depending on what is
installed and because the differential and property suites skip themselves
when their reference libraries are missing.

Two suites need a non-default build:

```bash
# Panic propagation: requires the `testing` feature, which exposes _force_panic
maturin develop --release --features testing && pytest tests/test_panic_propagation.py

# Fuzzing: requires nightly and cargo-fuzz
cargo +nightly fuzz run parse_csv fuzz/corpus/parse_csv fuzz/seeds/parse_csv -- -max_total_time=60
```

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
