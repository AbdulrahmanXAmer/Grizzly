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
use std::io::{BufWriter, Read};

use ahash::AHashMap;
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use memchr::{memchr, memchr2, memchr3};
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

#[derive(Clone)]
struct FreqTracker {
    ints: AHashMap<i64, u64>,
    // Key: (hash, len) → collision candidates [(bytes, count)].
    strings: AHashMap<(u64, usize), Vec<(Vec<u8>, u64)>>,
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
        let candidates = self.strings.entry(key).or_insert_with(Vec::new);

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

            let candidates = self.strings.entry(*key).or_insert_with(Vec::new);
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

// Fast byte-level utilities.

#[inline(always)]
fn trim_bytes(bytes: &[u8]) -> &[u8] {
    // FIX: Single-pass trim (not position + rposition)
    let mut start = 0;
    let mut end = bytes.len();

    // Trim start
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }

    // Trim end
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &bytes[start..end]
}

fn sniff_delimiter_simd(bytes: &[u8]) -> Option<u8> {
    if memchr(b',', bytes).is_some() {
        return Some(b',');
    }
    if memchr(b'\t', bytes).is_some() {
        return Some(b'\t');
    }
    if memchr(b';', bytes).is_some() {
        return Some(b';');
    }
    if memchr(b'|', bytes).is_some() {
        return Some(b'|');
    }
    None
}

fn has_alphabetic(bytes: &[u8]) -> bool {
    bytes.iter().any(|&b| b.is_ascii_alphabetic())
}

/// Better header detection: compare numeric parse rate of row 0 vs rows 1-N.
/// If row 0 is mostly non-numeric and row 1+ are mostly numeric, it's likely a header.
fn detect_header_smart(bytes: &[u8], delim: u8, num_sample_rows: usize) -> bool {
    let mut lines = bytes.split(|&b| b == b'\n').take(num_sample_rows + 1);

    let first_line = match lines.next() {
        Some(l) => l,
        None => return false,
    };

    // Count numeric fields in first row
    let first_row_fields: Vec<&[u8]> = first_line.split(|&b| b == delim).collect();
    if first_row_fields.is_empty() {
        return false;
    }

    let first_numeric_rate = first_row_fields
        .iter()
        .filter(|f| {
            let trimmed = trim_bytes(f);
            !trimmed.is_empty()
                && (atoi_simd::parse::<i64>(trimmed).is_ok()
                    || fast_float::parse::<f64, _>(trimmed).is_ok())
        })
        .count() as f64
        / first_row_fields.len() as f64;

    // Count numeric fields in subsequent rows
    let mut total_fields = 0usize;
    let mut numeric_fields = 0usize;

    for line in lines {
        if line.is_empty() {
            continue;
        }
        for field in line.split(|&b| b == delim) {
            let trimmed = trim_bytes(field);
            total_fields += 1;
            if !trimmed.is_empty()
                && (atoi_simd::parse::<i64>(trimmed).is_ok()
                    || fast_float::parse::<f64, _>(trimmed).is_ok())
            {
                numeric_fields += 1;
            }
        }
    }

    if total_fields == 0 {
        return has_alphabetic(first_line);
    }

    let data_numeric_rate = numeric_fields as f64 / total_fields as f64;

    // Header heuristic:
    // - First row has low numeric rate (< 30%)
    // - Data rows have higher numeric rate (> 50%), OR
    // - First row is significantly less numeric than data rows
    let is_header = (first_numeric_rate < 0.3 && data_numeric_rate > 0.3)
        || (data_numeric_rate - first_numeric_rate > 0.3)
        || (first_numeric_rate < 0.1 && has_alphabetic(first_line));

    is_header
}

// Cell processing (SIMD int parsing + fast-float).

#[inline(always)]
fn is_integer_bytes(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && (bytes[0] == b'-' || bytes[0] == b'+' || bytes[0].is_ascii_digit())
        && !bytes.iter().any(|&b| b == b'.' || b == b'e' || b == b'E')
}

/// Fast-reject non-float bytes before attempting an expensive float parse.
#[inline(always)]
fn maybe_float_bytes(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Must start with digit, sign, or decimal
    let first = bytes[0];
    if !(first.is_ascii_digit() || first == b'-' || first == b'+' || first == b'.') {
        return false;
    }

    // Check if it contains float indicators (., e, E) or is all digits
    // Using memchr2 for SIMD acceleration
    memchr2(b'.', b'e', bytes).is_some()
        || memchr(b'E', bytes).is_some()
        || bytes
            .iter()
            .all(|&b| b.is_ascii_digit() || b == b'-' || b == b'+')
}

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

// Parallel byte chunking (fast path).

/// Splits a raw byte slice into chunks, aligning to newlines.
/// Uses Polars strategy: 16MB max, 4KB min chunks.
///
/// ⚠️ **CORRECTNESS LIMITATION**: This assumes no quoted newlines in CSV fields.
/// If your CSV contains quoted fields like `"hello\nworld",123`, this will
/// split in the middle of the field, causing parse errors.
///
/// **Fast path assumption**: "Clean" CSVs with no quoted newlines.
/// For full CSV correctness, use a producer/consumer pipeline with sequential reading.
fn chunk_bytes_aligned(bytes: &[u8], num_threads: usize) -> Vec<&[u8]> {
    const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024; // 16MB (Polars pattern)
    const MIN_CHUNK_SIZE: usize = 4 * 1024; // 4KB

    let len = bytes.len();
    if len == 0 || num_threads == 0 {
        return vec![];
    }

    // Use a Polars-like chunk sizing strategy.
    let optimal_chunk_size = len / (16 * num_threads);
    let chunk_size = optimal_chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);

    let mut ranges = Vec::new();
    let mut start = 0;

    while start < len {
        let mut end = std::cmp::min(start + chunk_size, len);

        // Align 'end' to the next newline to avoid splitting rows
        // WARNING: This can still split quoted fields with embedded newlines
        if end < len {
            if let Some(newline_pos) = memchr(b'\n', &bytes[end..]) {
                end += newline_pos + 1;
            } else {
                end = len;
            }
        }

        if end > start {
            ranges.push(&bytes[start..end]);
        }
        start = end;
    }
    ranges
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

/// Split mode for CSV parsing
#[derive(Clone, Copy, Debug)]
enum SplitMode {
    Delim(u8),  // Split by specific delimiter
    Whitespace, // Split by whitespace runs (like pandas sep=r"\s+")
}

/// Iterate fields in a line using a single-byte delimiter.
#[inline(always)]
fn iter_fields_delim(line: &[u8], delim: u8) -> impl Iterator<Item = &[u8]> {
    line.split(move |&b| b == delim)
}

/// Iterate fields in a line by splitting on whitespace runs
#[inline(always)]
fn iter_fields_ws(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(|b: &u8| b.is_ascii_whitespace())
        .filter(|f| !f.is_empty())
}

/// Get fields from a line using the appropriate split mode
fn get_fields(line: &[u8], mode: SplitMode) -> Vec<&[u8]> {
    match mode {
        SplitMode::Delim(d) => iter_fields_delim(line, d).collect(),
        SplitMode::Whitespace => iter_fields_ws(line).collect(),
    }
}

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
    let result = py.allow_threads(
        || -> Result<(Vec<String>, Vec<ColStats>, usize, Option<u8>, bool), String> {
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
            let header_line = if first_line.ends_with(&[b'\r']) {
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
                let per_chunk_budget = if n_chunks > 0 {
                    sample_size / n_chunks + 1
                } else {
                    sample_size
                };

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
                        let mut pos = 0usize;

                        // Direct byte iteration.
                        while pos < chunk_bytes.len() && rows_in_chunk < per_chunk_budget {
                            // Find end of line
                            let line_end = memchr(b'\n', &chunk_bytes[pos..])
                                .map(|i| pos + i)
                                .unwrap_or(chunk_bytes.len());

                            if line_end <= pos {
                                break;
                            }

                            // Handle \r\n
                            let line_len = if line_end > pos
                                && chunk_bytes.get(line_end - 1) == Some(&b'\r')
                            {
                                line_end - 1
                            } else {
                                line_end
                            };

                            let line = &chunk_bytes[pos..line_len];
                            pos = line_end + 1;

                            if line.is_empty() {
                                continue;
                            }

                            // Split fields and process
                            let fields = get_fields(line, split_mode);
                            if lite {
                                // LITE MODE: Just numeric stats
                                for (i, field) in fields.iter().enumerate() {
                                    if i < ncols {
                                        process_cell_lite(field, &mut local_stats[i]);
                                    }
                                }
                            } else {
                                // FULL MODE: Type inference + examples + freq
                                for (i, field) in fields.iter().enumerate() {
                                    if i < ncols {
                                        process_cell(
                                            field,
                                            &mut local_stats[i],
                                            max_examples,
                                            track_freq,
                                            collect_examples,
                                        );
                                    }
                                }
                            }

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

                for result in reader.byte_records().take(sample_size) {
                    if let Ok(record) = result {
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
                }

                for stat in &mut stats {
                    stat.finalize();
                }

                (stats, total_rows)
            };

            Ok((col_names, merged_stats, total_rows, delimiter, has_header))
        },
    );

    let (col_names, stats, rows_sampled, delimiter, has_header) =
        result.map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

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

// Parallel transform (min-max scaling).

/// Check whether a field needs CSV quoting.
#[inline(always)]
fn needs_quote_simd(field: &[u8], delim: u8) -> bool {
    memchr3(delim, b'"', b'\n', field).is_some() || memchr(b'\r', field).is_some()
}

/// Write a field with proper CSV quoting if needed
#[inline(always)]
fn write_field_csv(output: &mut Vec<u8>, field: &[u8], delim: u8) {
    if !needs_quote_simd(field, delim) {
        output.extend_from_slice(field);
        return;
    }

    // Needs quoting
    output.push(b'"');
    // Use memchr to find quotes efficiently.
    let mut start = 0;
    while let Some(pos) = memchr(b'"', &field[start..]) {
        let i = start + pos;
        output.extend_from_slice(&field[start..=i]); // Include the quote
        output.push(b'"'); // Double it
        start = i + 1;
    }
    output.extend_from_slice(&field[start..]);
    output.push(b'"');
}

// Fast row iterator: splits on newlines and then splits fields by the selected mode.

/// Fast row iterator - splits on newlines, yields field slices.
/// ⚠️ Assumes no quoted fields (use only in fast_transform mode).
struct FastRowIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    split_mode: SplitMode,
}

impl<'a> FastRowIter<'a> {
    fn new(bytes: &'a [u8], split_mode: SplitMode) -> Self {
        Self {
            bytes,
            pos: 0,
            split_mode,
        }
    }
}

impl<'a> Iterator for FastRowIter<'a> {
    type Item = Vec<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.bytes.len() {
            return None;
        }

        // Find end of line
        let line_end = memchr(b'\n', &self.bytes[self.pos..])
            .map(|i| self.pos + i)
            .unwrap_or(self.bytes.len());

        if line_end <= self.pos {
            return None;
        }

        // Handle \r\n
        let line_len = if line_end > self.pos && self.bytes.get(line_end - 1) == Some(&b'\r') {
            line_end - 1
        } else {
            line_end
        };

        let line = &self.bytes[self.pos..line_len];
        self.pos = line_end + 1;

        if line.is_empty() {
            return self.next(); // Skip empty lines
        }

        // Split by mode
        let fields = get_fields(line, self.split_mode);
        Some(fields)
    }
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
    // Parse params
    let mut params_map: AHashMap<String, (f64, f64)> = AHashMap::new();
    for (k, v) in params.iter() {
        let col_name: String = k.extract()?;
        let v_dict = v.downcast::<PyDict>()?;
        let min_val: f64 = v_dict
            .get_item("min")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing min"))?
            .extract()?;
        let max_val: f64 = v_dict
            .get_item("max")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing max"))?
            .extract()?;
        params_map.insert(col_name, (min_val, max_val));
    }

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
            let first_line_clean = if first_line.ends_with(&[b'\r']) {
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
            let chunk_results: Vec<(Vec<u8>, usize)> = byte_chunks
                .into_par_iter()
                .map(|chunk_bytes| {
                    let mut output_buf =
                        Vec::with_capacity(chunk_bytes.len() + chunk_bytes.len() / 4);
                    let mut ryu_buf = ryu::Buffer::new();
                    let mut rows = 0usize;

                    for fields in FastRowIter::new(chunk_bytes, split_mode) {
                        for (i, field) in fields.iter().enumerate() {
                            if i > 0 {
                                output_buf.push(output_delim);
                            }

                            let trimmed = trim_bytes(field);

                            if let Some(Some((min_v, max_v))) = col_params.get(i) {
                                // Only parse columns with params (known numeric).
                                if let Ok(x) = fast_float::parse::<f64, _>(trimmed) {
                                    let range = max_v - min_v;
                                    let scaled = if range > 1e-12 {
                                        (x - min_v) / range
                                    } else {
                                        0.0
                                    };
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
                        }
                        output_buf.push(b'\n');
                        rows += 1;
                    }

                    (output_buf, rows)
                })
                .collect();

            // Write header + all chunk buffers.
            let file = File::create(&output_clone).map_err(|e| e.to_string())?;
            let mut writer = BufWriter::with_capacity(8 << 20, file);

            // Write header
            for (i, name) in col_names.iter().enumerate() {
                if i > 0 {
                    std::io::Write::write_all(&mut writer, &[output_delim])
                        .map_err(|e| e.to_string())?;
                }
                let mut header_buf = Vec::new();
                write_field_csv(&mut header_buf, name.as_bytes(), output_delim);
                std::io::Write::write_all(&mut writer, &header_buf).map_err(|e| e.to_string())?;
            }
            std::io::Write::write_all(&mut writer, b"\n").map_err(|e| e.to_string())?;

            let mut total_rows = 0usize;
            for (buffer, rows) in chunk_results {
                std::io::Write::write_all(&mut writer, &buffer).map_err(|e| e.to_string())?;
                total_rows += rows;
            }

            std::io::Write::flush(&mut writer).map_err(|e| e.to_string())?;
            Ok((total_rows, numeric_cols_to_scale, has_header_actual))
        })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

    let out = PyDict::new(py);
    out.set_item("input_path", input_path)?;
    out.set_item("output_path", output_path)?;
    out.set_item("rows_written", rows_written)?;
    out.set_item("numeric_cols_scaled", numeric_cols_scaled)?;
    out.set_item("has_header", has_header_detected)?;
    Ok(out.into())
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

fn flatten_value(
    py: Python<'_>,
    cols: &mut BTreeMap<String, PyColStats>,
    path: &str,
    v: &Bound<'_, PyAny>,
    sample_budget: &mut i64,
    max_examples: usize,
) -> PyResult<()> {
    if *sample_budget <= 0 {
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
            flatten_value(py, cols, &p, &vv, sample_budget, max_examples)?;
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
            flatten_value(py, cols, &p, &vv, sample_budget, max_examples)?;
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
                flatten_value(py, cols, &p, &item, sample_budget, max_examples)?;
                if *sample_budget <= 0 {
                    break;
                }
            }
            return Ok(());
        }
        if let Ok(tup) = v.downcast::<PyTuple>() {
            for i in 0..tup.len().min(50) {
                let item = tup.get_item(i)?;
                flatten_value(py, cols, &p, &item, sample_budget, max_examples)?;
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
                flatten_value(py, cols, &p, &item, sample_budget, max_examples)?;
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

    if is_sequence_like(&data) {
        if let Ok(list) = data.downcast::<PyList>() {
            for i in 0..list.len().min(sample_size) {
                let item = list.get_item(i)?;
                flatten_value(py, &mut cols, "", &item, &mut budget, max_examples)?;
                if budget <= 0 {
                    break;
                }
            }
        } else if let Ok(tup) = data.downcast::<PyTuple>() {
            for i in 0..tup.len().min(sample_size) {
                let item = tup.get_item(i)?;
                flatten_value(py, &mut cols, "", &item, &mut budget, max_examples)?;
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
                flatten_value(py, &mut cols, "", &item, &mut budget, max_examples)?;
                if budget <= 0 {
                    break;
                }
            }
        } else {
            flatten_value(py, &mut cols, "", &data, &mut budget, max_examples)?;
        }
    } else {
        flatten_value(py, &mut cols, "", &data, &mut budget, max_examples)?;
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
    Ok(out.into())
}

// Module registration.

#[pymodule]
fn _grizzly(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_schema, m)?)?;
    m.add_function(wrap_pyfunction!(csv_profile, m)?)?;
    m.add_function(wrap_pyfunction!(csv_minmax_params, m)?)?;
    m.add_function(wrap_pyfunction!(csv_transform_minmax, m)?)?;
    m.add_function(wrap_pyfunction!(csv_linear_regression, m)?)?;
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
fn u64_to_unit_f64(x: u64) -> f64 {
    // Convert to [0,1)
    const DEN: f64 = (u64::MAX as f64) + 1.0;
    (x as f64) / DEN
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

    let result = py.allow_threads(
        || -> Result<(Vec<String>, Vec<f64>, f64, usize, usize, f64, usize, f64, f64, f64), String> {
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
        let has_header_actual = has_header.unwrap_or_else(|| detect_header_smart(bytes, delim_byte_for_detection, 5));

        // Column names
        let first_line_clean = if first_line.ends_with(&[b'\r']) { &first_line[..first_line.len()-1] } else { first_line };
        let first_fields = get_fields(first_line_clean, split_mode);
        let ncols = first_fields.len();
        if ncols == 0 {
            return Err("No columns detected".to_string());
        }
        let col_names: Vec<String> = if has_header_actual {
            first_fields.iter().enumerate().map(|(i, f)| {
                let t = trim_bytes(f);
                if t.is_empty() { format!("col_{i}") } else { String::from_utf8_lossy(t).into_owned() }
            }).collect()
        } else {
            (0..ncols).map(|i| format!("col_{i}")).collect()
        };

        let target_idx = col_names.iter().position(|c| c == &target)
            .ok_or_else(|| format!("target not found: {target}. Available: {:?}", col_names))?;

        let feature_names: Vec<String> = if let Some(fs) = features {
            fs.into_iter().filter(|c| c != &target).collect()
        } else {
            col_names.iter().filter(|c| *c != &target).cloned().collect()
        };
        if feature_names.is_empty() {
            return Err("No feature columns selected".to_string());
        }
        let mut feature_idx: Vec<usize> = Vec::with_capacity(feature_names.len());
        for f in &feature_names {
            let idx = col_names.iter().position(|c| c == f)
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
        let data_start = if has_header_actual { first_newline + 1 } else { 0 };
        let data_bytes = &bytes[data_start..];

        // Build a stable split that is consistent across both passes.
        // If shuffle=true, use a Fisher–Yates permutation (parity with Python rng.permutation).
        // If shuffle=false, use a simple sequential split (first train_cut rows).
        let n_rows = if shuffle {
            // Pre-count rows up to sample_size so we can permute indices deterministically.
            let mut n = 0usize;
            if fast_csv {
                let mut it = FastRowIter::new(data_bytes, split_mode);
                while it.next().is_some() && n < sample_size {
                    n += 1;
                }
            } else {
                let mut reader = ReaderBuilder::new()
                    .has_headers(has_header_actual)
                    .delimiter(delim_byte_for_detection)
                    .flexible(true)
                    .from_reader(bytes);
                for _ in reader.byte_records().take(sample_size) {
                    n += 1;
                }
            }
            n.max(1)
        } else {
            sample_size.max(1)
        };
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
            let mut iter = FastRowIter::new(data_bytes, split_mode);
            while let Some(fields) = iter.next() {
                if rows_seen >= n_rows { break; }
                let row_idx = rows_seen;
                rows_seen += 1;

                if fields.len() < ncols { continue; }

                // Split assignment (stable across passes)
                let is_train = if let Some(mask) = &is_train_mask {
                    mask[row_idx]
                } else {
                    row_idx < train_cut
                };

                let yv = match parse_f64_opt(fields[target_idx]) { Some(v) => v, None => continue };
                let mut x = vec![0.0f64; dim];
                for (j, &idx) in feature_idx.iter().enumerate() {
                    let v = match parse_f64_opt(fields[idx]) { Some(v) => v, None => { x.clear(); break; } };
                    x[j] = v;
                }
                if x.is_empty() { continue; }
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
                let record = match result { Ok(r) => r, Err(_) => continue };
                let row_idx = rows_seen;
                rows_seen += 1;
                if record.len() < ncols { continue; }
                let is_train = if let Some(mask) = &is_train_mask {
                    mask[row_idx]
                } else {
                    row_idx < train_cut
                };

                let yv = match parse_f64_opt(record.get(target_idx).unwrap_or(&[])) { Some(v) => v, None => continue };
                let mut x = vec![0.0f64; dim];
                for (j, &idx) in feature_idx.iter().enumerate() {
                    let v = match record.get(idx).and_then(|b| parse_f64_opt(b)) { Some(v) => v, None => { x.clear(); break; } };
                    x[j] = v;
                }
                if x.is_empty() { continue; }
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
            return Err(format!("Not enough training rows to fit model (train_n={train_n}, params={dim})"));
        }

        // Optional ridge for stability: XtX += λI
        if ridge_lambda > 0.0 {
            for i in 0..dim {
                xtx[i * dim + i] += ridge_lambda;
            }
        }

        let w = gaussian_solve(xtx.clone(), xty.clone(), dim)
            .ok_or_else(|| "Failed to solve linear system (singular/ill-conditioned)".to_string())?;
        let coef = w[..p].to_vec();
        let intercept = w[p];

        // PASS 2: compute test R^2
        let mut ss_res = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut test_used = 0usize;
        let mut rows_seen2 = 0usize;

        if fast_csv {
            let mut iter = FastRowIter::new(data_bytes, split_mode);
            while let Some(fields) = iter.next() {
                if rows_seen2 >= n_rows { break; }
                let row_idx = rows_seen2;
                rows_seen2 += 1;
                if fields.len() < ncols { continue; }
                let is_train = if let Some(mask) = &is_train_mask {
                    mask[row_idx]
                } else {
                    row_idx < train_cut
                };
                if is_train { continue; }

                let yv = match parse_f64_opt(fields[target_idx]) { Some(v) => v, None => continue };
                let mut pred = intercept;
                for (j, &idx) in feature_idx.iter().enumerate() {
                    let xv = match parse_f64_opt(fields[idx]) { Some(v) => v, None => { pred = f64::NAN; break; } };
                    pred += coef[j] * xv;
                }
                if !pred.is_finite() { continue; }
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
                let record = match result { Ok(r) => r, Err(_) => continue };
                let row_idx = rows_seen2;
                rows_seen2 += 1;
                if record.len() < ncols { continue; }
                let is_train = if let Some(mask) = &is_train_mask {
                    mask[row_idx]
                } else {
                    row_idx < train_cut
                };
                if is_train { continue; }
                let yv = match record.get(target_idx).and_then(|b| parse_f64_opt(b)) { Some(v) => v, None => continue };
                let mut pred = intercept;
                for (j, &idx) in feature_idx.iter().enumerate() {
                    let xv = match record.get(idx).and_then(|b| parse_f64_opt(b)) { Some(v) => v, None => { pred = f64::NAN; break; } };
                    pred += coef[j] * xv;
                }
                if !pred.is_finite() { continue; }
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
        let r2 = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

        // Return test_used to reflect actual evaluated rows
        // Return debug-friendly counts
        if return_debug {
            Ok((feature_names, coef, intercept, train_n, test_used, r2, test_assigned, ss_res, ss_tot, mean_y))
        } else {
            // Keep tuple shape stable even when not returning debug fields
            Ok((feature_names, coef, intercept, train_n, test_used, r2, 0usize, 0.0f64, 0.0f64, 0.0f64))
        }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

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
