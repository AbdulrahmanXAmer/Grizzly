//! Grizzly: High-Performance Data Profiling in Rust
//!
//! Optimizations:
//! - Bounded gzip decompression (stops after sample_size rows)
//! - Memory mapping (memmap2) for zero-copy I/O on raw files
//! - Hash-based string frequency tracking (no global interning)
//! - atoi-simd for SIMD-accelerated integer parsing (same as Polars)
//! - T-Digest streaming quantiles (O(1) query)
//! - SIMD delimiter/quote detection (memchr2/memchr3)
//! - Rayon parallelization + GIL release
//! - Per-chunk budget (no atomics in profiling hot path)
//! - Fast delimiter-only parsing for transform (bypass csv crate)
//! - maybe_float_bytes fast-reject for string columns
//! - Indexed Vec for chunk ordering (no sort)

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyMapping, PySequence, PyString, PyTuple};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;

use ahash::AHashMap;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use memchr::memchr;
use memmap2::Mmap;
use rayon::prelude::*;
use tdigest::TDigest;

// Type tracking with a bitmask (keeps hot-path allocations low).

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum DType {
    Null = 1,
    Bool = 2,
    Int = 4,
    Float = 8,
    String = 16,
}

fn mask_to_types(mask: u8) -> Vec<&'static str> {
    let mut out = Vec::with_capacity(5);
    if mask & DType::Null as u8 != 0 {
        out.push("null");
    }
    if mask & DType::Bool as u8 != 0 {
        out.push("bool");
    }
    if mask & DType::Int as u8 != 0 {
        out.push("int");
    }
    if mask & DType::Float as u8 != 0 {
        out.push("float");
    }
    if mask & DType::String as u8 != 0 {
        out.push("string");
    }
    out
}

fn infer_from_mask(mask: u8) -> &'static str {
    let non_null = mask & !(DType::Null as u8);
    if non_null == 0 {
        return "null";
    }
    if non_null == DType::Int as u8 {
        return "int";
    }
    if non_null == DType::Float as u8 {
        return "float";
    }
    if non_null == DType::Bool as u8 {
        return "bool";
    }
    if non_null == DType::String as u8 {
        return "string";
    }
    if non_null == (DType::Int as u8 | DType::Float as u8) {
        return "float";
    }
    "mixed"
}

// Streaming numeric stats with T-Digest quantiles.

#[derive(Clone)]
struct NumStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    digest: TDigest,
    pending: Vec<f64>,
}

impl Default for NumStats {
    fn default() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            digest: TDigest::new_with_size(100),
            pending: Vec::with_capacity(1024),
        }
    }
}

impl NumStats {
    #[inline(always)]
    fn push(&mut self, x: f64) {
        if self.n == 0 {
            self.min = x;
            self.max = x;
        } else {
            if x < self.min {
                self.min = x;
            }
            if x > self.max {
                self.max = x;
            }
        }
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        self.pending.push(x);
        if self.pending.len() >= 1024 {
            let batch = std::mem::take(&mut self.pending);
            self.digest = self.digest.merge_unsorted(batch);
            self.pending = Vec::with_capacity(1024);
        }
    }

    #[inline(always)]
    fn finalize(&mut self) {
        if !self.pending.is_empty() {
            let batch = std::mem::take(&mut self.pending);
            self.digest = self.digest.merge_unsorted(batch);
        }
    }

    fn merge(&mut self, other: &mut NumStats) {
        // Finalize both before merging
        self.finalize();
        other.finalize();

        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }

        let n_combined = self.n + other.n;
        let delta = other.mean - self.mean;
        let new_mean = self.mean + delta * (other.n as f64 / n_combined as f64);
        let new_m2 = self.m2
            + other.m2
            + delta * delta * (self.n as f64) * (other.n as f64) / (n_combined as f64);

        self.mean = new_mean;
        self.m2 = new_m2;
        self.n = n_combined;

        // Use std::mem::take to avoid cloning digests.
        let self_digest = std::mem::take(&mut self.digest);
        let other_digest = std::mem::take(&mut other.digest);
        self.digest = TDigest::merge_digests(vec![self_digest, other_digest]);
    }

    fn std_pop(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some((self.m2 / self.n as f64).sqrt())
        }
    }

    fn quantile(&self, q: f64) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.digest.estimate_quantile(q))
        }
    }
}

// Hash-based frequency tracking (no global interning in the hot path).

/// Interned string frequency table: keyed by (hash, len), each bucket holds the
/// collision candidates as (bytes, count) pairs.
type StringFreqMap = AHashMap<(u64, usize), Vec<(Vec<u8>, u64)>>;

#[derive(Clone)]
struct FreqTracker {
    ints: AHashMap<i64, u64>,
    strings: StringFreqMap,
    max_strings: usize,
    total_unique: usize,
    hash_state: ahash::RandomState,
}

impl Default for FreqTracker {
    fn default() -> Self {
        Self {
            ints: AHashMap::new(),
            strings: AHashMap::new(),
            max_strings: 5000,
            total_unique: 0,
            hash_state: ahash::RandomState::with_seeds(0, 0, 0, 0),
        }
    }
}

impl FreqTracker {
    fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn push_int(&mut self, v: i64) {
        *self.ints.entry(v).or_insert(0) += 1;
    }

    #[inline(always)]
    fn push_string_bytes(&mut self, bytes: &[u8]) {
        // FIX: Skip very long strings (reduces memory traffic for high-cardinality columns)
        const MAX_TRACKED_STRING_LEN: usize = 64;

        if self.total_unique >= self.max_strings {
            return;
        }
        if bytes.len() > MAX_TRACKED_STRING_LEN {
            return;
        }

        let hash = self.hash_state.hash_one(bytes);
        let key = (hash, bytes.len());

        // Check collision candidates for exact match
        let candidates = self.strings.entry(key).or_default();

        for (existing_bytes, count) in candidates.iter_mut() {
            if existing_bytes == bytes {
                *count += 1;
                return;
            }
        }

        // New unique string
        candidates.push((bytes.to_vec(), 1));
        self.total_unique += 1;
    }

    fn merge(&mut self, other: &FreqTracker) {
        for (&k, &v) in &other.ints {
            *self.ints.entry(k).or_insert(0) += v;
        }
        for (key, other_candidates) in &other.strings {
            if self.total_unique >= self.max_strings {
                break;
            }

            let candidates = self.strings.entry(*key).or_default();
            for (other_bytes, other_count) in other_candidates {
                let mut found = false;
                for (existing_bytes, count) in candidates.iter_mut() {
                    if existing_bytes == other_bytes {
                        *count += other_count;
                        found = true;
                        break;
                    }
                }
                if !found && self.total_unique < self.max_strings {
                    candidates.push((other_bytes.clone(), *other_count));
                    self.total_unique += 1;
                }
            }
        }
    }

    fn mode_int(&self) -> Option<(i64, u64)> {
        self.ints
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(&k, &v)| (k, v))
    }

    fn mode_string(&self) -> Option<(String, u64)> {
        // Find mode across all candidates.
        self.strings
            .values()
            .flat_map(|candidates| candidates.iter())
            .max_by_key(|(_, count)| count)
            .map(|(bytes, count)| (String::from_utf8_lossy(bytes).into_owned(), *count))
    }
}

// Per-column statistics.

#[derive(Clone, Default)]
struct ColStats {
    count: u64,
    null_count: u64,
    type_mask: u8,
    examples: Vec<String>,
    num: NumStats,
    freq: FreqTracker,
}

impl ColStats {
    fn merge(&mut self, other: &mut ColStats) {
        self.count += other.count;
        self.null_count += other.null_count;
        self.type_mask |= other.type_mask;
        for ex in &other.examples {
            if self.examples.len() < 5 {
                self.examples.push(ex.clone());
            }
        }
        self.num.merge(&mut other.num);
        self.freq.merge(&other.freq);
    }

    fn finalize(&mut self) {
        self.num.finalize();
    }
}

// Cell processing (SIMD int parsing + fast-float).

/// Full cell processing with type inference, examples, and frequency tracking
#[inline(always)]
fn process_cell(
    bytes: &[u8],
    stats: &mut ColStats,
    max_examples: usize,
    track_freq: bool,
    collect_examples: bool,
) {
    stats.count += 1;
    let trimmed = trim_bytes(bytes);

    if trimmed.is_empty() {
        stats.null_count += 1;
        stats.type_mask |= DType::Null as u8;
        return;
    }

    // FIX: Use atoi-simd (same as Polars) for SIMD-accelerated integer parsing
    if is_integer_bytes(trimmed) {
        if let Ok(i) = atoi_simd::parse::<i64>(trimmed) {
            stats.type_mask |= DType::Int as u8;
            if track_freq {
                stats.freq.push_int(i);
            }
            stats.num.push(i as f64);
            if collect_examples && stats.examples.len() < max_examples {
                if let Ok(s) = std::str::from_utf8(trimmed) {
                    stats.examples.push(s.to_string());
                }
            }
            return;
        }
    }

    // Only attempt float parse if bytes look numeric.
    if maybe_float_bytes(trimmed) {
        if let Ok(val) = fast_float::parse::<f64, _>(trimmed) {
            stats.type_mask |= DType::Float as u8;
            stats.num.push(val);
            if collect_examples && stats.examples.len() < max_examples {
                if let Ok(s) = std::str::from_utf8(trimmed) {
                    stats.examples.push(s.to_string());
                }
            }
            return;
        }
    }

    // Bool check
    if trimmed.eq_ignore_ascii_case(b"true") || trimmed.eq_ignore_ascii_case(b"false") {
        stats.type_mask |= DType::Bool as u8;
        if track_freq {
            stats.freq.push_string_bytes(trimmed);
        }
        if collect_examples && stats.examples.len() < max_examples {
            if let Ok(s) = std::str::from_utf8(trimmed) {
                stats.examples.push(s.to_string());
            }
        }
        return;
    }

    // String fallback
    stats.type_mask |= DType::String as u8;
    if track_freq {
        stats.freq.push_string_bytes(trimmed);
    }
    if collect_examples && stats.examples.len() < max_examples {
        if let Ok(s) = std::str::from_utf8(trimmed) {
            stats.examples.push(s.to_string());
        }
    }
}

/// LITE cell processing - just numeric stats (min/max/mean/std/quantiles)
/// No type inference, no examples, no frequency tracking.
/// This is what you use when you just want Polars-equivalent profiling speed.
#[inline(always)]
fn process_cell_lite(bytes: &[u8], stats: &mut ColStats) {
    stats.count += 1;
    let trimmed = trim_bytes(bytes);

    if trimmed.is_empty() {
        stats.null_count += 1;
        return;
    }

    // Try int first (common case)
    if is_integer_bytes(trimmed) {
        if let Ok(i) = atoi_simd::parse::<i64>(trimmed) {
            stats.num.push(i as f64);
            return;
        }
    }

    // Try float
    if let Ok(val) = fast_float::parse::<f64, _>(trimmed) {
        stats.num.push(val);
    }
    // Non-numeric: just count, no other work
}

// File I/O (mmap for raw files, bounded gzip for profiling).

enum FileData {
    Mmap(Mmap),
    Buffered(Vec<u8>),
}

/// Bounded gzip decompression: stop after ~`max_rows` newlines for sampling-first workflows.
fn load_gz_bounded(path: &str, max_rows: usize) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);

    // Estimate: ~200 bytes per row on average, plus some buffer
    let estimated_size = (max_rows + 100) * 256;
    let mut out = Vec::with_capacity(estimated_size.min(64 << 20)); // Cap at 64MB initial

    let mut buf = [0u8; 64 * 1024]; // 64KB read buffer
    let mut newline_count = 0usize;

    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        } // EOF

        // Count newlines in this chunk
        for &b in &buf[..n] {
            if b == b'\n' {
                newline_count += 1;
            }
        }

        out.extend_from_slice(&buf[..n]);

        // Stop if we have enough rows (+ header + some margin)
        if newline_count >= max_rows + 10 {
            break;
        }
    }

    Ok(out)
}

fn load_file_data(path: &str) -> std::io::Result<FileData> {
    if path.to_lowercase().ends_with(".gz") {
        // Transform needs full data for output, so decompress fully.
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;
        Ok(FileData::Buffered(data))
    } else {
        // For raw files, memory map (zero-copy)
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileData::Mmap(mmap))
    }
}

/// Load file for profiling (bounded gzip if applicable).
fn load_file_for_profile(path: &str, sample_size: usize) -> std::io::Result<FileData> {
    if path.to_lowercase().ends_with(".gz") {
        // Only decompress until we have enough rows.
        let data = load_gz_bounded(path, sample_size)?;
        Ok(FileData::Buffered(data))
    } else {
        // For raw files, memory map (zero-copy)
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(FileData::Mmap(mmap))
    }
}

impl FileData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            FileData::Mmap(m) => m.as_ref(),
            FileData::Buffered(v) => v.as_slice(),
        }
    }
}

// Parallel CSV profiling.

/// Result of scanning a CSV for profiling, before conversion into a Python dict:
/// (column names, per-column stats, rows scanned, detected delimiter, header present).
type ProfileScan = (Vec<String>, Vec<ColStats>, usize, Option<u8>, bool);

/// Result of fitting a linear model, before conversion into a Python dict:
/// (feature names, coefficients, intercept, train rows, test rows used, r2,
/// test rows assigned, residual sum of squares, total sum of squares, mean y).
///
/// TODO: this tuple is load-bearing across a long function and should become a
/// named struct; the alias documents the positions in the meantime.
type RegressionFit = (
    Vec<String>,
    Vec<f64>,
    f64,
    usize,
    usize,
    f64,
    usize,
    f64,
    f64,
    f64,
);

/// Profile a CSV file.
///
/// # Arguments
/// * `path` - Path to CSV file (supports .gz)
/// * `sample_size` - Max rows to sample
/// * `max_examples` - Max examples per column
/// * `fast_csv` - If true, uses parallel byte chunking (assumes no quoted newlines).
///                If false, uses sequential reading (correct for any CSV).
/// * `lite` - If true, only compute numeric stats (min/max/mean/std/quantiles).
///            Skips type inference, examples, and frequency tracking for speed.
/// * `track_freq` - If true, track frequency for mode calculation
/// * `collect_examples` - If true, collect example values
// Argument count mirrors the public Python signature, which is the API contract;
// collapsing it into a config struct would only move the arity into the caller.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (path, sample_size=1000, max_examples=5, fast_csv=true, lite=false, track_freq=true, collect_examples=true))]
fn csv_profile(
    py: Python<'_>,
    path: String,
    sample_size: usize,
    max_examples: usize,
    fast_csv: bool,
    lite: bool,
    track_freq: bool,
    collect_examples: bool,
) -> PyResult<PyObject> {
    let result = py.allow_threads(|| -> Result<ProfileScan, String> {
        // Profiling is sampling-first: use bounded gzip when applicable.
        let file_data = load_file_for_profile(&path, sample_size).map_err(|e| e.to_string())?;
        let bytes = file_data.as_bytes();

        if bytes.is_empty() {
            return Ok((vec![], vec![], 0, None, false));
        }

        // Find first line for sniffing
        let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
        let first_line = &bytes[..first_newline];

        // Determine split mode (delimiter or whitespace)
        let delimiter = sniff_delimiter_simd(first_line);
        let split_mode = match delimiter {
            Some(d) => SplitMode::Delim(d),
            None => SplitMode::Whitespace,
        };

        // For header detection and csv crate compat, we need a delimiter byte
        let delim_byte_for_detection = match split_mode {
            SplitMode::Delim(d) => d,
            SplitMode::Whitespace => b' ', // Use space as proxy for detection
        };

        // Header detection heuristic: compare numeric rate of row 0 vs subsequent rows.
        let has_header = detect_header_smart(bytes, delim_byte_for_detection, 5);

        // Get column names from first line (fast path - no csv crate)
        let header_line = if first_line.ends_with(b"\r") {
            &first_line[..first_line.len() - 1]
        } else {
            first_line
        };

        let col_names: Vec<String> = if has_header {
            get_fields(header_line, split_mode)
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let trimmed = trim_bytes(f);
                    if trimmed.is_empty() {
                        format!("col_{i}")
                    } else {
                        String::from_utf8_lossy(trimmed).into_owned()
                    }
                })
                .collect()
        } else {
            get_fields(header_line, split_mode)
                .iter()
                .enumerate()
                .map(|(i, _)| format!("col_{i}"))
                .collect()
        };

        let ncols = col_names.len();
        if ncols == 0 {
            return Ok((vec![], vec![], 0, delimiter, has_header));
        }

        // Skip header bytes if present
        let data_start = if has_header { first_newline + 1 } else { 0 };
        let data_bytes = &bytes[data_start..];

        let (merged_stats, total_rows): (Vec<ColStats>, usize) = if fast_csv {
            // Fast path: direct byte splitting (assumes no quoted newlines in fields).
            let num_threads = rayon::current_num_threads();
            let byte_chunks = chunk_bytes_aligned(data_bytes, num_threads);
            let n_chunks = byte_chunks.len();

            // Per-chunk row budget (avoid shared atomics).
            let per_chunk_budget = sample_size
                .checked_div(n_chunks)
                .map_or(sample_size, |budget| budget + 1);

            byte_chunks
                .into_par_iter()
                .map(|chunk_bytes| {
                    let mut local_stats: Vec<ColStats> = (0..ncols)
                        .map(|_| ColStats {
                            freq: FreqTracker::new(),
                            ..Default::default()
                        })
                        .collect();

                    let mut rows_in_chunk = 0usize;

                    // Shares FastLineIter with every other reader. The
                    // hand-rolled loop this replaces treated a blank line as
                    // end-of-chunk rather than skipping it, so a stray empty
                    // line silently dropped the rest of its chunk from the
                    // profile. Fields are visited in order, so no per-row Vec
                    // is allocated either.
                    for line in FastLineIter::new(chunk_bytes) {
                        if rows_in_chunk >= per_chunk_budget {
                            break;
                        }

                        for_each_field(line, split_mode, |i, field| {
                            if i >= ncols {
                                return;
                            }
                            if lite {
                                // LITE MODE: Just numeric stats
                                process_cell_lite(field, &mut local_stats[i]);
                            } else {
                                // FULL MODE: Type inference + examples + freq
                                process_cell(
                                    field,
                                    &mut local_stats[i],
                                    max_examples,
                                    track_freq,
                                    collect_examples,
                                );
                            }
                        });

                        rows_in_chunk += 1;
                    }

                    for stat in &mut local_stats {
                        stat.finalize();
                    }

                    (local_stats, rows_in_chunk)
                })
                .reduce(
                    || {
                        let empty_stats: Vec<ColStats> = (0..ncols)
                            .map(|_| ColStats {
                                freq: FreqTracker::new(),
                                ..Default::default()
                            })
                            .collect();
                        (empty_stats, 0)
                    },
                    |(mut a_stats, a_rows), (mut b_stats, b_rows)| {
                        for (i, b_stat) in b_stats.iter_mut().enumerate() {
                            if i < a_stats.len() {
                                a_stats[i].merge(b_stat);
                            }
                        }
                        (a_stats, a_rows + b_rows)
                    },
                )
        } else {
            // SAFE PATH: Sequential reading (correct for any CSV including quoted newlines)
            // Use csv crate when we need full CSV correctness (handles quoted newlines, etc.)
            let mut stats: Vec<ColStats> = (0..ncols)
                .map(|_| ColStats {
                    freq: FreqTracker::new(),
                    ..Default::default()
                })
                .collect();

            let mut reader = ReaderBuilder::new()
                .has_headers(has_header)
                .delimiter(delim_byte_for_detection)
                .flexible(true)
                .from_reader(bytes);

            let mut total_rows = 0usize;

            for record in reader.byte_records().take(sample_size).flatten() {
                for (i, field) in record.iter().enumerate() {
                    if i < ncols {
                        if lite {
                            process_cell_lite(field, &mut stats[i]);
                        } else {
                            process_cell(
                                field,
                                &mut stats[i],
                                max_examples,
                                track_freq,
                                collect_examples,
                            );
                        }
                    }
                }
                total_rows += 1;
            }

            for stat in &mut stats {
                stat.finalize();
            }

            (stats, total_rows)
        };

        Ok((col_names, merged_stats, total_rows, delimiter, has_header))
    });

    let (col_names, stats, rows_sampled, delimiter, has_header) =
        result.map_err(pyo3::exceptions::PyIOError::new_err)?;

    // Build Python output
    let out = PyDict::new(py);
    out.set_item("path", &path)?;
    out.set_item("rows_sampled", rows_sampled)?;
    // Report delimiter correctly based on split mode
    let delimiter_str = match delimiter {
        Some(d) => (d as char).to_string(),
        None => "whitespace".to_string(),
    };
    out.set_item("delimiter", delimiter_str)?;
    out.set_item("has_header", has_header)?;

    let col_list = PyList::empty(py);
    for (i, name) in col_names.iter().enumerate() {
        if i >= stats.len() {
            break;
        }
        let s = &stats[i];

        let d = PyDict::new(py);
        d.set_item("name", name)?;
        d.set_item("index", i)?;
        d.set_item("count", s.count)?;
        d.set_item("null_count", s.null_count)?;
        d.set_item("types", mask_to_types(s.type_mask))?;
        d.set_item("inferred", infer_from_mask(s.type_mask))?;

        d.set_item("examples", &s.examples)?;

        if s.num.n > 0 {
            d.set_item("min", s.num.min)?;
            d.set_item("max", s.num.max)?;
            d.set_item("mean", s.num.mean)?;
            d.set_item("std", s.num.std_pop().unwrap_or(0.0))?;
            d.set_item("median", s.num.quantile(0.5).unwrap_or(0.0))?;
            d.set_item("p25", s.num.quantile(0.25).unwrap_or(0.0))?;
            d.set_item("p75", s.num.quantile(0.75).unwrap_or(0.0))?;
            d.set_item("p90", s.num.quantile(0.90).unwrap_or(0.0))?;
            d.set_item("p95", s.num.quantile(0.95).unwrap_or(0.0))?;
            d.set_item("outliers_3sigma", 0)?;
        } else {
            d.set_item("min", py.None())?;
            d.set_item("max", py.None())?;
            d.set_item("mean", py.None())?;
            d.set_item("std", py.None())?;
            d.set_item("median", py.None())?;
            d.set_item("p25", py.None())?;
            d.set_item("p75", py.None())?;
            d.set_item("p90", py.None())?;
            d.set_item("p95", py.None())?;
            d.set_item("outliers_3sigma", 0)?;
        }

        // Mode from hash-based tracker
        if let Some((val, count)) = s.freq.mode_int() {
            d.set_item("mode", val.to_string())?;
            d.set_item("mode_count", count)?;
        } else if let Some((val, count)) = s.freq.mode_string() {
            d.set_item("mode", &val)?;
            d.set_item("mode_count", count)?;
        } else {
            d.set_item("mode", py.None())?;
            d.set_item("mode_count", 0)?;
        }

        col_list.append(d)?;
    }

    out.set_item("columns", col_list)?;
    Ok(out.into())
}

// Min-max params extraction.

#[pyfunction]
fn csv_minmax_params(py: Python<'_>, path: String, sample_size: usize) -> PyResult<PyObject> {
    // Use lite mode - we only need min/max, not type inference or examples
    let prof_obj = csv_profile(py, path.clone(), sample_size, 0, true, true, false, false)?;
    let prof = prof_obj.bind(py).downcast::<PyDict>()?;
    let out = PyDict::new(py);
    out.set_item("path", path)?;
    let params = PyDict::new(py);

    let cols_any = prof
        .get_item("columns")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing columns"))?;
    let cols_list = cols_any.downcast::<PyList>()?;

    for c in cols_list.iter() {
        let cd = c.downcast::<PyDict>()?;
        let name: String = cd
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing name"))?
            .extract()?;
        let min_opt = cd.get_item("min")?;
        let max_opt = cd.get_item("max")?;

        if let (Some(min_v), Some(max_v)) = (min_opt, max_opt) {
            if !min_v.is_none() && !max_v.is_none() {
                let d = PyDict::new(py);
                d.set_item("min", &min_v)?;
                d.set_item("max", &max_v)?;
                params.set_item(name, d)?;
            }
        }
    }

    out.set_item("params", params)?;
    Ok(out.into())
}

/// Write `buf` at an absolute file offset, without moving a shared cursor.
///
/// Positioned writes are what allow the output chunks to be written
/// concurrently: each knows exactly where it belongs, so no writer has to wait
/// for another to finish appending. `File::write_all_at` is Unix-only and
/// `seek_write` is the Windows equivalent, hence the split.
#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut written = 0usize;
    while written < buf.len() {
        // seek_write is not guaranteed to consume the whole buffer.
        let n = file.seek_write(&buf[written..], offset + written as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "positioned write made no progress",
            ));
        }
        written += n;
    }
    Ok(())
}

// Parallel streaming transforms.

/// Apply a per-column affine transform `(x - offset) / scale`, streaming.
///
/// Min-max scaling and standardization differ only in how `(offset, scale)` is
/// derived -- `(min, max - min)` versus `(mean, std)` -- so they share this one
/// parallel writer instead of two near-identical copies of it. Rows are read,
/// transformed, and written in chunks, so memory stays bounded by the chunk
/// size rather than the file size.
///
/// A non-finite or effectively-zero scale maps the column to 0.0: a constant
/// column has no spread to scale by, and emitting NaN or infinity would poison
/// every downstream consumer of the output file.
fn transform_affine(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params_map: AHashMap<String, (f64, f64)>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    let input_clone = input_path.clone();
    let output_clone = output_path.clone();

    let (rows_written, numeric_cols_scaled, has_header_detected) = py
        .allow_threads(move || -> Result<(usize, usize, bool), String> {
            let file_data = load_file_data(&input_clone).map_err(|e| e.to_string())?;
            let bytes = file_data.as_bytes();

            if bytes.is_empty() {
                return Ok((0, 0, false));
            }

            // Determine split mode (delimiter or whitespace)
            let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
            let first_line = &bytes[..first_newline];

            let split_mode = if let Some(d) = delimiter.and_then(|d| d.bytes().next()) {
                SplitMode::Delim(d) // User specified delimiter
            } else {
                // Auto-detect
                match sniff_delimiter_simd(first_line) {
                    Some(d) => SplitMode::Delim(d),
                    None => SplitMode::Whitespace, // FIX: Use whitespace mode!
                }
            };

            // For header detection with csv crate
            let delim_byte_for_detection = match split_mode {
                SplitMode::Delim(d) => d,
                SplitMode::Whitespace => b' ',
            };

            // Determine if file has header (auto-detect if not provided)
            let has_header_actual = has_header
                .unwrap_or_else(|| detect_header_smart(bytes, delim_byte_for_detection, 5));

            // Get first line fields for column count and naming
            let first_line_clean = if first_line.ends_with(b"\r") {
                &first_line[..first_line.len() - 1]
            } else {
                first_line
            };
            let first_fields = get_fields(first_line_clean, split_mode);
            let ncols = first_fields.len();

            if ncols == 0 {
                return Err("No columns detected in CSV".to_string());
            }

            // Column names based on header presence.
            let col_names: Vec<String> = if has_header_actual {
                // Use header values as names.
                first_fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let trimmed = trim_bytes(f);
                        if trimmed.is_empty() {
                            format!("col_{i}")
                        } else {
                            String::from_utf8_lossy(trimmed).into_owned()
                        }
                    })
                    .collect()
            } else {
                // No header: generate col_0, col_1, ...
                (0..ncols).map(|i| format!("col_{i}")).collect()
            };

            // Build per-column params once (None for columns not being scaled).
            let col_params: Vec<Option<(f64, f64)>> = col_names
                .iter()
                .map(|name| params_map.get(name).copied())
                .collect();

            // Count how many columns will actually be scaled (for validation)
            let numeric_cols_to_scale = col_params.iter().filter(|p| p.is_some()).count();

            // Split data into chunks for parallel processing.
            let data_start = if has_header_actual {
                first_newline + 1
            } else {
                0
            };
            let data_bytes = &bytes[data_start..];

            let num_threads = rayon::current_num_threads();
            let byte_chunks = chunk_bytes_aligned(data_bytes, num_threads);
            let _n_chunks = byte_chunks.len();

            // Output delimiter.
            let output_delim = match split_mode {
                SplitMode::Delim(d) => d,
                SplitMode::Whitespace => b' ',
            };

            // Collect per-chunk output buffers.
            // Output capacity estimate. Scaling a value written with a handful
            // of decimals produces a full round-trip double -- "3.141593"
            // becomes "0.4724761615149732" -- so a scaled file is roughly twice
            // its input. Reserving the old 1.25x guaranteed a reallocation and
            // a full copy of a multi-megabyte buffer in every chunk.
            let growth_numerator = if numeric_cols_to_scale > 0 { 9 } else { 5 };
            let estimate_capacity = |len: usize| len.saturating_mul(growth_numerator) / 4 + 64;

            // Build the header first: its length is the starting file offset,
            // and every chunk's destination is derived from it.
            let mut header = Vec::new();
            for (i, name) in col_names.iter().enumerate() {
                if i > 0 {
                    header.push(output_delim);
                }
                write_field_csv(&mut header, name.as_bytes(), output_delim);
            }
            header.push(b'\n');

            let file = File::create(&output_clone).map_err(|e| e.to_string())?;
            write_at(&file, &header, 0).map_err(|e| e.to_string())?;

            // Chunks are processed in batches rather than all at once.
            //
            // Formatting every chunk up front and only then writing meant the
            // whole output existed in memory simultaneously: a 100 MB input
            // produced 200 MB of live buffers and a 327 MB peak RSS, which also
            // made the "memory bounded by chunk size" claim in the docs false.
            // Batching bounds live output to one batch, and lets the allocator
            // reuse the same blocks for every batch instead of faulting in
            // fresh pages for each of ~128 chunks.
            let batch_size = (rayon::current_num_threads() * 2).max(1);

            let mut total_rows = 0usize;
            let mut cursor = header.len() as u64;

            for batch in byte_chunks.chunks(batch_size) {
                let batch_results: Vec<(Vec<u8>, usize)> = batch
                    .par_iter()
                    .map(|&chunk_bytes| {
                        let mut output_buf =
                            Vec::with_capacity(estimate_capacity(chunk_bytes.len()));
                        let mut ryu_buf = ryu::Buffer::new();
                        let mut rows = 0usize;

                        // Fields are visited in order and written as they are seen,
                        // so there is no reason to collect them into a Vec first.
                        for line in FastLineIter::new(chunk_bytes) {
                            for_each_field(line, split_mode, |i, field| {
                                if i > 0 {
                                    output_buf.push(output_delim);
                                }

                                let trimmed = trim_bytes(field);

                                if let Some(Some((offset, scale))) = col_params.get(i) {
                                    // Only parse columns with params (known numeric).
                                    if let Ok(x) = fast_float::parse::<f64, _>(trimmed) {
                                        let scaled = if scale.is_finite() && scale.abs() > 1e-12 {
                                            (x - offset) / scale
                                        } else {
                                            0.0
                                        };
                                        // format(), not format_finite(): fast_float
                                        // parses "inf" and "nan", so `scaled` is not
                                        // guaranteed finite and format_finite's
                                        // output would be unspecified for those.
                                        let s = ryu_buf.format(scaled);
                                        output_buf.extend_from_slice(s.as_bytes());
                                    } else {
                                        // Non-numeric value in numeric column, pass through
                                        write_field_csv(&mut output_buf, trimmed, output_delim);
                                    }
                                } else {
                                    // No transform, fast path for non-numeric columns
                                    write_field_csv(&mut output_buf, trimmed, output_delim);
                                }
                            });
                            output_buf.push(b'\n');
                            rows += 1;
                        }

                        (output_buf, rows)
                    })
                    .collect();

                // Offsets within the batch are known once it is formatted, so
                // the batch is written in parallel too: each buffer knows
                // exactly where it belongs and no writer waits on another.
                let mut offsets = Vec::with_capacity(batch_results.len());
                for (buffer, rows) in &batch_results {
                    offsets.push(cursor);
                    cursor += buffer.len() as u64;
                    total_rows += rows;
                }

                batch_results
                    .par_iter()
                    .zip(offsets.par_iter())
                    .try_for_each(|((buffer, _), &offset)| {
                        write_at(&file, buffer, offset).map_err(|e| e.to_string())
                    })?;
            }

            // Deliberately no fsync. The previous implementation flushed a
            // BufWriter, which only pushes into the page cache; forcing durable
            // media writes would be a different guarantee and a much slower one,
            // and it is not what the libraries this is measured against do.
            Ok((total_rows, numeric_cols_to_scale, has_header_actual))
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("input_path", input_path)?;
    out.set_item("output_path", output_path)?;
    out.set_item("rows_written", rows_written)?;
    out.set_item("numeric_cols_scaled", numeric_cols_scaled)?;
    out.set_item("has_header", has_header_detected)?;
    Ok(out.into())
}

/// Read `params` as a mapping of column name to two named f64 fields.
fn affine_params_from_dict(
    params: &Bound<'_, PyDict>,
    offset_key: &str,
    scale_key: &str,
) -> PyResult<AHashMap<String, (f64, f64)>> {
    let mut out: AHashMap<String, (f64, f64)> = AHashMap::new();
    for (k, v) in params.iter() {
        let col_name: String = k.extract()?;
        let v_dict = v.downcast::<PyDict>()?;
        let offset: f64 = v_dict
            .get_item(offset_key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "column {col_name:?} is missing {offset_key:?}"
                ))
            })?
            .extract()?;
        let scale: f64 = v_dict
            .get_item(scale_key)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "column {col_name:?} is missing {scale_key:?}"
                ))
            })?
            .extract()?;
        out.insert(col_name, (offset, scale));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_path, output_path, params, delimiter=None, has_header=None))]
fn csv_transform_minmax(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params: Bound<'_, PyDict>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    // (x - min) / (max - min)
    let mut params_map = affine_params_from_dict(&params, "min", "max")?;
    for value in params_map.values_mut() {
        value.1 -= value.0;
    }
    transform_affine(
        py,
        input_path,
        output_path,
        params_map,
        delimiter,
        has_header,
    )
}

/// Mean and population standard deviation per numeric column, from one pass.
#[pyfunction]
fn csv_standardize_params(py: Python<'_>, path: String, sample_size: usize) -> PyResult<PyObject> {
    // lite mode: mean and std come from the streaming accumulator, so no type
    // inference, examples, or frequency tracking is needed.
    let prof_obj = csv_profile(py, path.clone(), sample_size, 0, true, true, false, false)?;
    let prof = prof_obj.bind(py).downcast::<PyDict>()?;
    let out = PyDict::new(py);
    out.set_item("path", path)?;
    let params = PyDict::new(py);

    let cols_any = prof
        .get_item("columns")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing columns"))?;
    let cols_list = cols_any.downcast::<PyList>()?;

    for c in cols_list.iter() {
        let cd = c.downcast::<PyDict>()?;
        let name: String = cd
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing name"))?
            .extract()?;

        let (Some(mean_v), Some(std_v)) = (cd.get_item("mean")?, cd.get_item("std")?) else {
            continue;
        };
        if mean_v.is_none() || std_v.is_none() {
            continue;
        }
        let d = PyDict::new(py);
        d.set_item("mean", &mean_v)?;
        d.set_item("std", &std_v)?;
        params.set_item(name, d)?;
    }

    out.set_item("params", params)?;
    Ok(out.into())
}

/// Standardize numeric columns to zero mean and unit variance, streaming.
#[pyfunction]
#[pyo3(signature = (input_path, output_path, params, delimiter=None, has_header=None))]
fn csv_transform_standardize(
    py: Python<'_>,
    input_path: String,
    output_path: String,
    params: Bound<'_, PyDict>,
    delimiter: Option<String>,
    has_header: Option<bool>,
) -> PyResult<PyObject> {
    // (x - mean) / std
    let params_map = affine_params_from_dict(&params, "mean", "std")?;
    transform_affine(
        py,
        input_path,
        output_path,
        params_map,
        delimiter,
        has_header,
    )
}

// Python object schema detection (compatibility path).

#[derive(Default, Clone)]
struct PyColStats {
    count: u64,
    null_count: u64,
    types: BTreeSet<String>,
    examples: Vec<String>,
}

fn type_name(v: &Bound<'_, PyAny>) -> String {
    if v.is_none() {
        return "null".to_string();
    }
    if v.extract::<bool>().is_ok() {
        return "bool".to_string();
    }
    if v.extract::<i64>().is_ok() {
        return "int".to_string();
    }
    if v.extract::<f64>().is_ok() {
        return "float".to_string();
    }
    if v.downcast::<PyString>().is_ok() {
        return "string".to_string();
    }
    match v.get_type().name() {
        Ok(name) => name.to_string(),
        Err(_) => "unknown".to_string(),
    }
}

fn is_string_like(v: &Bound<'_, PyAny>) -> bool {
    v.is_instance_of::<PyString>()
}

fn is_bytes_like(v: &Bound<'_, PyAny>) -> bool {
    v.extract::<&[u8]>().is_ok() || v.downcast::<pyo3::types::PyByteArray>().is_ok()
}

fn is_sequence_like(v: &Bound<'_, PyAny>) -> bool {
    if is_string_like(v) || is_bytes_like(v) {
        return false;
    }
    v.downcast::<PyList>().is_ok()
        || v.downcast::<PyTuple>().is_ok()
        || v.downcast::<PySequence>().is_ok()
}

fn add_example(_py: Python<'_>, col: &mut PyColStats, v: &Bound<'_, PyAny>, max_examples: usize) {
    if col.examples.len() >= max_examples {
        return;
    }
    if let Ok(r) = v.repr() {
        if let Ok(s) = r.extract::<String>() {
            col.examples.push(s);
        }
    }
}

/// Maximum nesting depth `flatten_value` will descend into.
///
/// This is a stack-safety bound, not a stylistic one. `flatten_value` recurses
/// once per level of nesting, so without a cap a sufficiently deep input
/// overflows the Rust stack and takes the whole interpreter down with SIGSEGV
/// -- not a catchable Python exception. The observed crash threshold is around
/// 20k-30k levels; 512 is far above any real-world nested record and far below
/// the danger zone.
const MAX_NESTING_DEPTH: usize = 512;

#[allow(clippy::too_many_arguments)]
fn flatten_value(
    py: Python<'_>,
    cols: &mut BTreeMap<String, PyColStats>,
    path: &str,
    v: &Bound<'_, PyAny>,
    sample_budget: &mut i64,
    max_examples: usize,
    depth: usize,
    depth_exceeded: &mut bool,
) -> PyResult<()> {
    if *sample_budget <= 0 {
        return Ok(());
    }
    if depth >= MAX_NESTING_DEPTH {
        // Stop descending rather than overflow the stack. The caller reports
        // this to Python so the truncation is visible instead of silent.
        *depth_exceeded = true;
        *sample_budget -= 1;
        // Still record the node, so the caller sees where truncation happened
        // rather than an empty schema. Deliberately does NOT call add_example:
        // repr() of a deeply nested object recurses inside CPython and would
        // reintroduce the very stack overflow this guard exists to prevent.
        let col_key = if path.is_empty() {
            "value".to_string()
        } else {
            path.to_string()
        };
        let entry = cols.entry(col_key).or_default();
        entry.count += 1;
        if v.is_none() {
            entry.null_count += 1;
        }
        entry.types.insert(type_name(v));
        return Ok(());
    }
    *sample_budget -= 1;

    if let Ok(mapping) = v.downcast::<PyDict>() {
        for (k, vv) in mapping.iter() {
            let key = k.str()?.to_str()?.to_string();
            let p = if path.is_empty() {
                key
            } else {
                format!("{path}.{key}")
            };
            flatten_value(
                py,
                cols,
                &p,
                &vv,
                sample_budget,
                max_examples,
                depth + 1,
                depth_exceeded,
            )?;
        }
        return Ok(());
    }

    if let Ok(mapping) = v.downcast::<PyMapping>() {
        let items = mapping.items()?;
        for item in items.iter() {
            let tup = item.downcast::<PyTuple>()?;
            if tup.len() != 2 {
                continue;
            }
            let k = tup.get_item(0)?;
            let vv = tup.get_item(1)?;
            let key = k.str()?.to_str()?.to_string();
            let p = if path.is_empty() {
                key
            } else {
                format!("{path}.{key}")
            };
            flatten_value(
                py,
                cols,
                &p,
                &vv,
                sample_budget,
                max_examples,
                depth + 1,
                depth_exceeded,
            )?;
        }
        return Ok(());
    }

    if is_sequence_like(v) {
        let p = if path.is_empty() {
            "[]".to_string()
        } else {
            format!("{path}[]")
        };
        if let Ok(list) = v.downcast::<PyList>() {
            for i in 0..list.len().min(50) {
                let item = list.get_item(i)?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
        if let Ok(tup) = v.downcast::<PyTuple>() {
            for i in 0..tup.len().min(50) {
                let item = tup.get_item(i)?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
        if let Ok(iter) = PyIterator::from_object(v) {
            for (idx, item) in iter.enumerate() {
                if idx >= 50 {
                    break;
                }
                let item = item?;
                flatten_value(
                    py,
                    cols,
                    &p,
                    &item,
                    sample_budget,
                    max_examples,
                    depth + 1,
                    depth_exceeded,
                )?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
    }

    let col_key = if path.is_empty() {
        "value".to_string()
    } else {
        path.to_string()
    };
    let entry = cols.entry(col_key).or_default();
    entry.count += 1;
    if v.is_none() {
        entry.null_count += 1;
    }
    entry.types.insert(type_name(v));
    add_example(py, entry, v, max_examples);
    Ok(())
}

fn infer_best_type(types: &BTreeSet<String>) -> String {
    if types.is_empty() {
        return "unknown".to_string();
    }
    let non_null: Vec<&str> = types
        .iter()
        .map(|s| s.as_str())
        .filter(|x| *x != "null")
        .collect();
    if non_null.is_empty() {
        return "null".to_string();
    }
    if non_null.len() == 1 {
        return non_null[0].to_string();
    }
    let set: BTreeSet<&str> = non_null.iter().copied().collect();
    if set.iter().all(|x| *x == "int" || *x == "float") {
        return "float".to_string();
    }
    "mixed".to_string()
}

#[pyfunction]
fn detect_schema(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    sample_size: usize,
    max_examples: usize,
) -> PyResult<PyObject> {
    let mut cols: BTreeMap<String, PyColStats> = BTreeMap::new();
    let mut budget = sample_size as i64;
    let mut depth_exceeded = false;

    if is_sequence_like(&data) {
        if let Ok(list) = data.downcast::<PyList>() {
            for i in 0..list.len().min(sample_size) {
                let item = list.get_item(i)?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else if let Ok(tup) = data.downcast::<PyTuple>() {
            for i in 0..tup.len().min(sample_size) {
                let item = tup.get_item(i)?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else if let Ok(iter) = PyIterator::from_object(&data) {
            for (idx, item) in iter.enumerate() {
                if idx >= sample_size {
                    break;
                }
                let item = item?;
                flatten_value(
                    py,
                    &mut cols,
                    "",
                    &item,
                    &mut budget,
                    max_examples,
                    0,
                    &mut depth_exceeded,
                )?;
                if budget <= 0 {
                    break;
                }
            }
        } else {
            flatten_value(
                py,
                &mut cols,
                "",
                &data,
                &mut budget,
                max_examples,
                0,
                &mut depth_exceeded,
            )?;
        }
    } else {
        flatten_value(
            py,
            &mut cols,
            "",
            &data,
            &mut budget,
            max_examples,
            0,
            &mut depth_exceeded,
        )?;
    }

    let out = PyDict::new(py);
    let col_list = PyList::empty(py);
    for (path, stats) in cols.iter() {
        let d = PyDict::new(py);
        d.set_item("path", path)?;
        d.set_item("count", stats.count)?;
        d.set_item("null_count", stats.null_count)?;
        d.set_item("types", stats.types.iter().cloned().collect::<Vec<_>>())?;
        d.set_item("inferred", infer_best_type(&stats.types))?;
        d.set_item("examples", &stats.examples)?;
        col_list.append(d)?;
    }
    out.set_item("columns", col_list)?;
    out.set_item("sample_size", sample_size)?;
    // Report truncation rather than silently returning a partial schema.
    out.set_item("max_depth_exceeded", depth_exceeded)?;
    out.set_item("max_depth", MAX_NESTING_DEPTH)?;
    Ok(out.into())
}

mod parse;
use parse::*;

// Streaming SGD regression.

/// Where the data starts and what the columns are called.
///
/// Shared setup for anything that streams a CSV: delimiter, header presence,
/// column names, and the byte offset of the first data row.
struct CsvLayout {
    split_mode: SplitMode,
    col_names: Vec<String>,
    data_start: usize,
}

fn csv_layout(
    bytes: &[u8],
    delimiter: Option<u8>,
    has_header: Option<bool>,
) -> Result<CsvLayout, String> {
    if bytes.is_empty() {
        return Err("Empty file".to_string());
    }
    let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
    let first_line = &bytes[..first_newline];

    let split_mode = match delimiter {
        Some(d) => SplitMode::Delim(d),
        None => match sniff_delimiter_simd(first_line) {
            Some(d) => SplitMode::Delim(d),
            None => SplitMode::Whitespace,
        },
    };
    let delim_for_detection = match split_mode {
        SplitMode::Delim(d) => d,
        SplitMode::Whitespace => b' ',
    };
    let has_header_actual =
        has_header.unwrap_or_else(|| detect_header_smart(bytes, delim_for_detection, 5));

    let first_line_clean = if first_line.ends_with(b"\r") {
        &first_line[..first_line.len() - 1]
    } else {
        first_line
    };
    let first_fields = get_fields(first_line_clean, split_mode);
    if first_fields.is_empty() {
        return Err("No columns detected in CSV".to_string());
    }

    let col_names: Vec<String> = if has_header_actual {
        first_fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let t = trim_bytes(f);
                if t.is_empty() {
                    format!("col_{i}")
                } else {
                    String::from_utf8_lossy(t).into_owned()
                }
            })
            .collect()
    } else {
        (0..first_fields.len())
            .map(|i| format!("col_{i}"))
            .collect()
    };

    let data_start = if has_header_actual {
        first_newline + 1
    } else {
        0
    };

    Ok(CsvLayout {
        split_mode,
        col_names,
        data_start,
    })
}

/// Outcome of a streaming SGD fit, before conversion into a Python dict:
/// (feature names, coefficients, intercept, train rows, test rows, test r2,
/// epochs run, final training MSE).
type SgdFit = (Vec<String>, Vec<f64>, f64, usize, usize, f64, usize, f64);

/// Fit a linear model by stochastic gradient descent, streaming from CSV.
///
/// The closed-form solver in `csv_linear_regression` accumulates an X'X matrix,
/// which costs O(p^2) memory and O(n p^2) time -- fine for tens of features,
/// hopeless for the wide sparse data typical of ad-tech or recommender
/// training sets. This path holds only the weight vector, so memory is O(p)
/// and each epoch is O(n p), and it never materialises a design matrix.
///
/// Features are standardized on the fly using the mean and standard deviation
/// from a prior profiling pass. That is not cosmetic: raw features on wildly
/// different scales make a single global learning rate either diverge on the
/// large ones or crawl on the small ones. Coefficients are converted back to
/// the original feature space before returning, so they are directly
/// comparable with the closed-form solver's.
///
/// Rows are visited in file order within each epoch. Shuffling a stream would
/// mean buffering it, which would give up the bounded memory that is the point
/// of this path; data with meaningful row order should be shuffled on disk
/// first.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (path, target, features=None, epochs=5, learning_rate=0.05, l2=0.0, train_frac=0.8, seed=0_u64, sample_size=1_000_000, delimiter=None, has_header=None, shuffle=true, grad_clip=10.0))]
fn csv_sgd_regression(
    py: Python<'_>,
    path: String,
    target: String,
    features: Option<Vec<String>>,
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    train_frac: f64,
    seed: u64,
    sample_size: usize,
    delimiter: Option<String>,
    has_header: Option<bool>,
    shuffle: bool,
    grad_clip: f64,
) -> PyResult<PyObject> {
    if !(0.0 < train_frac && train_frac < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "train_frac must be in (0,1)",
        ));
    }
    if epochs == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "epochs must be >= 1",
        ));
    }
    if !(learning_rate.is_finite() && learning_rate > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "learning_rate must be positive and finite",
        ));
    }
    // Infinity is a legitimate value here: it disables clipping.
    if grad_clip.is_nan() || grad_clip <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad_clip must be positive (use infinity to disable clipping)",
        ));
    }

    // Standardization statistics, from one profiling pass over the same file.
    let stats_obj = csv_standardize_params(py, path.clone(), sample_size)?;
    let stats_dict = stats_obj.bind(py).downcast::<PyDict>()?.clone();
    let params_any = stats_dict
        .get_item("params")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing params"))?;
    let params = params_any.downcast::<PyDict>()?;
    let mut moments: AHashMap<String, (f64, f64)> = AHashMap::new();
    for (k, v) in params.iter() {
        let name: String = k.extract()?;
        let d = v.downcast::<PyDict>()?;
        let mean: f64 = d
            .get_item("mean")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing mean"))?
            .extract()?;
        let std: f64 = d
            .get_item("std")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing std"))?
            .extract()?;
        moments.insert(name, (mean, std));
    }

    let delim_byte = delimiter.and_then(|d| d.bytes().next());
    let path_for_thread = path.clone();
    let target_for_thread = target.clone();

    let result = py.allow_threads(move || -> Result<SgdFit, String> {
        let target = target_for_thread;
        let file_data = load_file_data(&path_for_thread).map_err(|e| e.to_string())?;
        let bytes = file_data.as_bytes();
        let layout = csv_layout(bytes, delim_byte, has_header)?;
        let col_names = &layout.col_names;

        let target_idx = col_names
            .iter()
            .position(|c| c == &target)
            .ok_or_else(|| format!("target column {target:?} not found"))?;

        let feature_idx: Vec<usize> = match &features {
            Some(names) => names
                .iter()
                .map(|n| {
                    col_names
                        .iter()
                        .position(|c| c == n)
                        .ok_or_else(|| format!("feature column {n:?} not found"))
                })
                .collect::<Result<_, _>>()?,
            None => (0..col_names.len()).filter(|&i| i != target_idx).collect(),
        };
        if feature_idx.is_empty() {
            return Err("no feature columns selected".to_string());
        }

        // Per-feature standardization constants. A feature with no usable
        // spread is centred but not divided, which leaves it contributing
        // nothing rather than producing NaN.
        let p = feature_idx.len();
        let mut mean = vec![0.0f64; p];
        let mut scale = vec![1.0f64; p];
        for (j, &idx) in feature_idx.iter().enumerate() {
            if let Some((m, s)) = moments.get(&col_names[idx]) {
                mean[j] = *m;
                if s.is_finite() && *s > 1e-12 {
                    scale[j] = *s;
                }
            }
        }

        let data_bytes = &bytes[layout.data_start..];
        let split_mode = layout.split_mode;

        // Row count, for a split defined against the data rather than the cap.
        let mut counted = 0usize;
        {
            let mut it = FastRowIter::new(data_bytes, split_mode);
            while it.next().is_some() && counted < sample_size {
                counted += 1;
            }
        }
        let n_rows = counted.max(1);
        let train_cut = ((n_rows as f64) * train_frac).floor() as usize;

        let is_train_mask: Option<Vec<bool>> = if shuffle {
            let mut perm: Vec<usize> = (0..n_rows).collect();
            for i in (1..n_rows).rev() {
                let r = splitmix64(seed ^ (i as u64));
                let j = (r % ((i + 1) as u64)) as usize;
                perm.swap(i, j);
            }
            let mut mask = vec![false; n_rows];
            for &idx in perm.iter().take(train_cut) {
                mask[idx] = true;
            }
            Some(mask)
        } else {
            None
        };

        let is_train = |row_idx: usize| -> bool {
            match &is_train_mask {
                Some(mask) => mask[row_idx],
                None => row_idx < train_cut,
            }
        };

        // Weights live in standardized space; O(p) total.
        let mut w = vec![0.0f64; p];
        let mut b = 0.0f64;
        let mut x = vec![0.0f64; p];

        let mut train_n = 0usize;
        let mut final_mse = 0.0f64;

        for epoch in 0..epochs {
            // Decay the step size each epoch so later passes refine rather
            // than bounce around the optimum.
            let lr = learning_rate / (1.0 + epoch as f64);
            let mut sq_err = 0.0f64;
            let mut seen = 0usize;

            for (this_row, fields) in FastRowIter::new(data_bytes, split_mode).enumerate() {
                if this_row >= n_rows {
                    break;
                }
                if fields.len() <= target_idx || !is_train(this_row) {
                    continue;
                }

                let Some(y) = parse_f64_opt(fields[target_idx]) else {
                    continue;
                };
                let mut usable = true;
                for (j, &idx) in feature_idx.iter().enumerate() {
                    match fields.get(idx).and_then(|f| parse_f64_opt(f)) {
                        Some(v) => x[j] = (v - mean[j]) / scale[j],
                        None => {
                            usable = false;
                            break;
                        }
                    }
                }
                if !usable {
                    continue;
                }

                let mut pred = b;
                for j in 0..p {
                    pred += w[j] * x[j];
                }
                let err = pred - y;
                sq_err += err * err;
                seen += 1;

                // Clip the per-sample gradient. Real data has outliers that
                // survive standardization -- a mis-metered taxi trip of 100,000
                // miles sits a thousand standard deviations out -- and a single
                // such row produces a step large enough to diverge the fit.
                // Clipping bounds any one row's influence without needing the
                // caller to hand-tune the learning rate per dataset.
                // intercept gradient is err * 1, hence the leading 1.0
                let grad_sq = 1.0 + x.iter().map(|v| v * v).sum::<f64>();
                let grad_norm = err.abs() * grad_sq.sqrt();
                let step = if grad_norm > grad_clip && grad_norm > 0.0 {
                    lr * (grad_clip / grad_norm)
                } else {
                    lr
                };

                for j in 0..p {
                    w[j] -= step * (err * x[j] + l2 * w[j]);
                }
                b -= step * err;

                if epoch == 0 {
                    train_n += 1;
                }
            }

            if seen > 0 {
                final_mse = sq_err / seen as f64;
            }
            if !b.is_finite() || w.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "SGD diverged at epoch {epoch}; lower learning_rate (was {learning_rate})"
                ));
            }
        }

        // Convert weights back to the original feature space so they can be
        // compared with the closed-form solver:
        //   y = b + sum_j w_j * (x_j - mean_j) / scale_j
        //     = (b - sum_j w_j * mean_j / scale_j) + sum_j (w_j / scale_j) x_j
        let coef: Vec<f64> = (0..p).map(|j| w[j] / scale[j]).collect();
        let intercept = b - (0..p).map(|j| w[j] * mean[j] / scale[j]).sum::<f64>();

        // Test-set R2, evaluated in the original space.
        let mut ss_res = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut test_n = 0usize;
        for (this_row, fields) in FastRowIter::new(data_bytes, split_mode).enumerate() {
            if this_row >= n_rows {
                break;
            }
            if fields.len() <= target_idx || is_train(this_row) {
                continue;
            }
            let Some(y) = parse_f64_opt(fields[target_idx]) else {
                continue;
            };
            let mut pred = intercept;
            let mut usable = true;
            for (j, &idx) in feature_idx.iter().enumerate() {
                match fields.get(idx).and_then(|f| parse_f64_opt(f)) {
                    Some(v) => pred += coef[j] * v,
                    None => {
                        usable = false;
                        break;
                    }
                }
            }
            if !usable {
                continue;
            }
            let d = y - pred;
            ss_res += d * d;
            sum_y += y;
            sum_y2 += y * y;
            test_n += 1;
        }

        let r2 = if test_n > 1 {
            let mean_y = sum_y / test_n as f64;
            let ss_tot = sum_y2 - 2.0 * mean_y * sum_y + (test_n as f64) * mean_y * mean_y;
            if ss_tot > 1e-12 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            }
        } else {
            0.0
        };

        let feature_names: Vec<String> =
            feature_idx.iter().map(|&i| col_names[i].clone()).collect();

        Ok((
            feature_names,
            coef,
            intercept,
            train_n,
            test_n,
            r2,
            epochs,
            final_mse,
        ))
    });

    let (feature_names, coef, intercept, train_n, test_n, r2, epochs_run, final_mse) =
        result.map_err(pyo3::exceptions::PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("path", path)?;
    out.set_item("target", target)?;
    out.set_item("features", feature_names)?;
    out.set_item("coef", coef)?;
    out.set_item("intercept", intercept)?;
    out.set_item("train_n", train_n)?;
    out.set_item("test_n", test_n)?;
    out.set_item("r2", r2)?;
    out.set_item("epochs", epochs_run)?;
    out.set_item("final_train_mse", final_mse)?;
    Ok(out.into())
}

#[cfg(test)]
mod tests;

// Module registration.

/// Panic on purpose, to verify how panics cross the FFI boundary.
///
/// PyO3 wraps every #[pyfunction] body in catch_unwind and converts a caught
/// panic into a Python PanicException. That only works if panics unwind: under
/// `panic = "abort"` the process dies instead, and no Python-level handler ever
/// runs. Gated behind the `testing` feature so it never ships.
#[cfg(feature = "testing")]
#[pyfunction]
fn _force_panic() {
    panic!("deliberate panic: verifying panic propagation across the FFI boundary");
}

#[pymodule]
fn _grizzly(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "testing")]
    m.add_function(wrap_pyfunction!(_force_panic, m)?)?;
    m.add_function(wrap_pyfunction!(detect_schema, m)?)?;
    m.add_function(wrap_pyfunction!(csv_profile, m)?)?;
    m.add_function(wrap_pyfunction!(csv_minmax_params, m)?)?;
    m.add_function(wrap_pyfunction!(csv_transform_minmax, m)?)?;
    m.add_function(wrap_pyfunction!(csv_standardize_params, m)?)?;
    m.add_function(wrap_pyfunction!(csv_transform_standardize, m)?)?;
    m.add_function(wrap_pyfunction!(csv_linear_regression, m)?)?;
    m.add_function(wrap_pyfunction!(csv_sgd_regression, m)?)?;
    Ok(())
}

// Rust-native ML: Linear Regression on CSV (no NumPy).

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    // Deterministic pseudo-random hash (fast, decent quality)
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn parse_f64_opt(bytes: &[u8]) -> Option<f64> {
    let t = trim_bytes(bytes);
    if t.is_empty() {
        return None;
    }
    fast_float::parse::<f64, _>(t).ok()
}

fn gaussian_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    // Solve Ax=b with partial pivoting. a is row-major n*n.
    for i in 0..n {
        // Pivot row
        let mut piv = i;
        let mut piv_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > piv_val {
                piv_val = v;
                piv = r;
            }
        }
        if piv_val == 0.0 || !piv_val.is_finite() {
            return None;
        }
        if piv != i {
            // swap rows in A
            for c in 0..n {
                a.swap(i * n + c, piv * n + c);
            }
            b.swap(i, piv);
        }

        // Eliminate below
        let diag = a[i * n + i];
        for r in (i + 1)..n {
            let f = a[r * n + i] / diag;
            if f == 0.0 {
                continue;
            }
            a[r * n + i] = 0.0;
            for c in (i + 1)..n {
                a[r * n + c] -= f * a[i * n + c];
            }
            b[r] -= f * b[i];
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i * n + c] * x[c];
        }
        let diag = a[i * n + i];
        if diag == 0.0 || !diag.is_finite() {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

// Argument count mirrors the public Python signature (see csv_profile above).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (path, target, features=None, train_frac=0.8, seed=0_u64, sample_size=1_000_000, delimiter=None, has_header=None, fast_csv=true, shuffle=true, ridge_lambda=0.0, return_debug=false))]
fn csv_linear_regression(
    py: Python<'_>,
    path: String,
    target: String,
    features: Option<Vec<String>>,
    train_frac: f64,
    seed: u64,
    sample_size: usize,
    delimiter: Option<String>,
    has_header: Option<bool>,
    fast_csv: bool,
    shuffle: bool,
    ridge_lambda: f64,
    return_debug: bool,
) -> PyResult<PyObject> {
    if !(0.0 < train_frac && train_frac < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "train_frac must be in (0,1)",
        ));
    }

    let result = py
        .allow_threads(|| -> Result<RegressionFit, String> {
            let file_data = load_file_data(&path).map_err(|e| e.to_string())?;
            let bytes = file_data.as_bytes();
            if bytes.is_empty() {
                return Err("Empty file".to_string());
            }

            // Determine split mode
            let first_newline = memchr(b'\n', bytes).unwrap_or(bytes.len());
            let first_line = &bytes[..first_newline];
            let split_mode = if let Some(d) = delimiter.and_then(|d| d.bytes().next()) {
                SplitMode::Delim(d)
            } else {
                match sniff_delimiter_simd(first_line) {
                    Some(d) => SplitMode::Delim(d),
                    None => SplitMode::Whitespace,
                }
            };
            let delim_byte_for_detection = match split_mode {
                SplitMode::Delim(d) => d,
                SplitMode::Whitespace => b' ',
            };
            let has_header_actual = has_header
                .unwrap_or_else(|| detect_header_smart(bytes, delim_byte_for_detection, 5));

            // Column names
            let first_line_clean = if first_line.ends_with(b"\r") {
                &first_line[..first_line.len() - 1]
            } else {
                first_line
            };
            let first_fields = get_fields(first_line_clean, split_mode);
            let ncols = first_fields.len();
            if ncols == 0 {
                return Err("No columns detected".to_string());
            }
            let col_names: Vec<String> = if has_header_actual {
                first_fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let t = trim_bytes(f);
                        if t.is_empty() {
                            format!("col_{i}")
                        } else {
                            String::from_utf8_lossy(t).into_owned()
                        }
                    })
                    .collect()
            } else {
                (0..ncols).map(|i| format!("col_{i}")).collect()
            };

            let target_idx = col_names
                .iter()
                .position(|c| c == &target)
                .ok_or_else(|| format!("target not found: {target}. Available: {:?}", col_names))?;

            let feature_names: Vec<String> = if let Some(fs) = features {
                fs.into_iter().filter(|c| c != &target).collect()
            } else {
                col_names
                    .iter()
                    .filter(|c| *c != &target)
                    .cloned()
                    .collect()
            };
            if feature_names.is_empty() {
                return Err("No feature columns selected".to_string());
            }
            let mut feature_idx: Vec<usize> = Vec::with_capacity(feature_names.len());
            for f in &feature_names {
                let idx = col_names
                    .iter()
                    .position(|c| c == f)
                    .ok_or_else(|| format!("feature not found: {f}. Available: {:?}", col_names))?;
                feature_idx.push(idx);
            }

            let p = feature_idx.len();
            let dim = p + 1; // + intercept
            let mut xtx = vec![0.0f64; dim * dim];
            let mut xty = vec![0.0f64; dim];

            // Second pass to evaluate test R2 after solving
            // We'll do two passes over bytes (fast for in-memory).

            // Helper to iterate rows (fast path) – reuse existing FastRowIter
            let data_start = if has_header_actual {
                first_newline + 1
            } else {
                0
            };
            let data_bytes = &bytes[data_start..];

            // Build a stable split that is consistent across both passes.
            // If shuffle=true, use a Fisher–Yates permutation (parity with Python rng.permutation).
            // If shuffle=false, use a simple sequential split (first train_cut rows).
            // Pre-count the rows actually present, capped at sample_size.
            //
            // This must happen for both split strategies. Deriving n_rows from
            // sample_size instead (as the sequential branch previously did)
            // makes train_cut a fraction of the *requested* cap rather than of
            // the data: with the default sample_size of 1,000,000, every file
            // under 800,000 rows put every row on the train side, leaving the
            // test set empty and reporting r2 = 0.0 for a perfectly good model.
            let mut counted = 0usize;
            if fast_csv {
                let mut it = FastRowIter::new(data_bytes, split_mode);
                while it.next().is_some() && counted < sample_size {
                    counted += 1;
                }
            } else {
                let mut reader = ReaderBuilder::new()
                    .has_headers(has_header_actual)
                    .delimiter(delim_byte_for_detection)
                    .flexible(true)
                    .from_reader(bytes);
                for _ in reader.byte_records().take(sample_size) {
                    counted += 1;
                }
            }
            let n_rows = counted.max(1);
            let train_cut = ((n_rows as f64) * train_frac).floor() as usize;

            let mut is_train_mask: Option<Vec<bool>> = None;
            if shuffle {
                let mut perm: Vec<usize> = (0..n_rows).collect();
                // Fisher–Yates using splitmix64
                for i in (1..n_rows).rev() {
                    let r = splitmix64(seed ^ (i as u64));
                    let j = (r % ((i + 1) as u64)) as usize;
                    perm.swap(i, j);
                }
                let mut mask = vec![false; n_rows];
                for &idx in perm.iter().take(train_cut) {
                    mask[idx] = true;
                }
                is_train_mask = Some(mask);
            }

            let mut rows_seen = 0usize;
            let mut train_n = 0usize;
            let mut test_assigned = 0usize;

            // PASS 1: accumulate XtX/Xty on TRAIN
            if fast_csv {
                let iter = FastRowIter::new(data_bytes, split_mode);
                for fields in iter {
                    if rows_seen >= n_rows {
                        break;
                    }
                    let row_idx = rows_seen;
                    rows_seen += 1;

                    if fields.len() < ncols {
                        continue;
                    }

                    // Split assignment (stable across passes)
                    let is_train = if let Some(mask) = &is_train_mask {
                        mask[row_idx]
                    } else {
                        row_idx < train_cut
                    };

                    let yv = match parse_f64_opt(fields[target_idx]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut x = vec![0.0f64; dim];
                    for (j, &idx) in feature_idx.iter().enumerate() {
                        let v = match parse_f64_opt(fields[idx]) {
                            Some(v) => v,
                            None => {
                                x.clear();
                                break;
                            }
                        };
                        x[j] = v;
                    }
                    if x.is_empty() {
                        continue;
                    }
                    x[p] = 1.0; // intercept

                    if is_train {
                        // xtx += x x^T
                        for r in 0..dim {
                            let xr = x[r];
                            for c in 0..dim {
                                xtx[r * dim + c] += xr * x[c];
                            }
                            xty[r] += xr * yv;
                        }
                        train_n += 1;
                    } else {
                        test_assigned += 1;
                    }
                }
            } else {
                // Safe path: fall back to csv crate parsing (quoted newlines, etc.)
                let mut reader = ReaderBuilder::new()
                    .has_headers(has_header_actual)
                    .delimiter(delim_byte_for_detection)
                    .flexible(true)
                    .from_reader(bytes);

                for result in reader.byte_records().take(n_rows) {
                    let record = match result {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let row_idx = rows_seen;
                    rows_seen += 1;
                    if record.len() < ncols {
                        continue;
                    }
                    let is_train = if let Some(mask) = &is_train_mask {
                        mask[row_idx]
                    } else {
                        row_idx < train_cut
                    };

                    let yv = match parse_f64_opt(record.get(target_idx).unwrap_or(&[])) {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut x = vec![0.0f64; dim];
                    for (j, &idx) in feature_idx.iter().enumerate() {
                        let v = match record.get(idx).and_then(parse_f64_opt) {
                            Some(v) => v,
                            None => {
                                x.clear();
                                break;
                            }
                        };
                        x[j] = v;
                    }
                    if x.is_empty() {
                        continue;
                    }
                    x[p] = 1.0;
                    if is_train {
                        for r in 0..dim {
                            let xr = x[r];
                            for c in 0..dim {
                                xtx[r * dim + c] += xr * x[c];
                            }
                            xty[r] += xr * yv;
                        }
                        train_n += 1;
                    } else {
                        test_assigned += 1;
                    }
                }
            }

            if train_n < dim {
                return Err(format!(
                    "Not enough training rows to fit model (train_n={train_n}, params={dim})"
                ));
            }

            // Optional ridge for stability: XtX += λI
            if ridge_lambda > 0.0 {
                for i in 0..dim {
                    xtx[i * dim + i] += ridge_lambda;
                }
            }

            let w = gaussian_solve(xtx.clone(), xty.clone(), dim).ok_or_else(|| {
                "Failed to solve linear system (singular/ill-conditioned)".to_string()
            })?;
            let coef = w[..p].to_vec();
            let intercept = w[p];

            // PASS 2: compute test R^2
            let mut ss_res = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_y2 = 0.0f64;
            let mut test_used = 0usize;
            let mut rows_seen2 = 0usize;

            if fast_csv {
                let iter = FastRowIter::new(data_bytes, split_mode);
                for fields in iter {
                    if rows_seen2 >= n_rows {
                        break;
                    }
                    let row_idx = rows_seen2;
                    rows_seen2 += 1;
                    if fields.len() < ncols {
                        continue;
                    }
                    let is_train = if let Some(mask) = &is_train_mask {
                        mask[row_idx]
                    } else {
                        row_idx < train_cut
                    };
                    if is_train {
                        continue;
                    }

                    let yv = match parse_f64_opt(fields[target_idx]) {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut pred = intercept;
                    for (j, &idx) in feature_idx.iter().enumerate() {
                        let xv = match parse_f64_opt(fields[idx]) {
                            Some(v) => v,
                            None => {
                                pred = f64::NAN;
                                break;
                            }
                        };
                        pred += coef[j] * xv;
                    }
                    if !pred.is_finite() {
                        continue;
                    }
                    let r = yv - pred;
                    ss_res += r * r;
                    sum_y += yv;
                    sum_y2 += yv * yv;
                    test_used += 1;
                }
            } else {
                let mut reader = ReaderBuilder::new()
                    .has_headers(has_header_actual)
                    .delimiter(delim_byte_for_detection)
                    .flexible(true)
                    .from_reader(bytes);
                for result in reader.byte_records().take(n_rows) {
                    let record = match result {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    let row_idx = rows_seen2;
                    rows_seen2 += 1;
                    if record.len() < ncols {
                        continue;
                    }
                    let is_train = if let Some(mask) = &is_train_mask {
                        mask[row_idx]
                    } else {
                        row_idx < train_cut
                    };
                    if is_train {
                        continue;
                    }
                    let yv = match record.get(target_idx).and_then(parse_f64_opt) {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut pred = intercept;
                    for (j, &idx) in feature_idx.iter().enumerate() {
                        let xv = match record.get(idx).and_then(parse_f64_opt) {
                            Some(v) => v,
                            None => {
                                pred = f64::NAN;
                                break;
                            }
                        };
                        pred += coef[j] * xv;
                    }
                    if !pred.is_finite() {
                        continue;
                    }
                    let r = yv - pred;
                    ss_res += r * r;
                    sum_y += yv;
                    sum_y2 += yv * yv;
                    test_used += 1;
                }
            }

            // IMPORTANT: R^2 must be computed on the same set of rows that were actually scored.
            // Using the split-assigned `test_n` while skipping parse-failed rows corrupts mean_y/ss_tot.
            let n = test_used.max(1) as f64;
            let mean_y = sum_y / n;
            let ss_tot = sum_y2 - n * mean_y * mean_y;
            let r2 = if ss_tot > 0.0 {
                1.0 - (ss_res / ss_tot)
            } else {
                0.0
            };

            // Return test_used to reflect actual evaluated rows
            // Return debug-friendly counts
            if return_debug {
                Ok((
                    feature_names,
                    coef,
                    intercept,
                    train_n,
                    test_used,
                    r2,
                    test_assigned,
                    ss_res,
                    ss_tot,
                    mean_y,
                ))
            } else {
                // Keep tuple shape stable even when not returning debug fields
                Ok((
                    feature_names,
                    coef,
                    intercept,
                    train_n,
                    test_used,
                    r2,
                    0usize,
                    0.0f64,
                    0.0f64,
                    0.0f64,
                ))
            }
        })
        .map_err(pyo3::exceptions::PyIOError::new_err)?;

    let (
        feature_names,
        coef,
        intercept,
        train_n,
        test_n,
        r2,
        test_assigned,
        ss_res,
        ss_tot,
        mean_y,
    ) = result;
    let out = PyDict::new(py);
    out.set_item("path", path)?;
    out.set_item("target", target)?;
    out.set_item("features", feature_names)?;
    out.set_item("coef", coef)?;
    out.set_item("intercept", intercept)?;
    out.set_item("train_n", train_n)?;
    out.set_item("test_n", test_n)?;
    out.set_item("r2", r2)?;
    if return_debug {
        out.set_item("test_n_assigned", test_assigned)?;
        out.set_item("ss_res", ss_res)?;
        out.set_item("ss_tot", ss_tot)?;
        out.set_item("y_mean_test", mean_y)?;
    }
    Ok(out.into())
}
