//! Streaming statistics accumulators.
//!
//! Everything here is pure Rust with no PyO3 dependency: the per-column
//! accumulators (`NumStats`, `FreqTracker`, `ColStats`), the type-inference
//! bitmask, and the per-cell processing functions that feed them. This is the
//! profiler's hot path -- every cell of every sampled row goes through
//! `process_cell` or one of its leaner siblings -- so it is also the code most
//! worth micro-benchmarking, which the separation makes possible.
//!
//! The correctness-critical invariant, enforced by tests: merging per-chunk
//! accumulators must equal a single pass over the same values. The parallel
//! profiler depends on it.

use ahash::AHashMap;
use tdigest::TDigest;

use crate::parse::{is_integer_bytes, maybe_float_bytes, trim_bytes};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum DType {
    Null = 1,
    Bool = 2,
    Int = 4,
    Float = 8,
    String = 16,
}

pub fn mask_to_types(mask: u8) -> Vec<&'static str> {
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

pub fn infer_from_mask(mask: u8) -> &'static str {
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

/// What a profiling scan computes per numeric cell.
///
/// The distinction earns its keep in the hot path: the t-digest costs ~8x the
/// moments it rides along with (measured by `bench_numstats_digest_overhead`),
/// and the scaler-params scans only ever read min/max/mean/std.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    /// Type inference, examples, frequency tracking, moments, and quantiles.
    Full,
    /// Moments and quantiles only.
    Lite,
    /// Moments only: min/max/mean/std, no t-digest. What min-max scaling and
    /// standardization need, at a fraction of the per-cell cost.
    Moments,
}

#[derive(Clone)]
pub struct NumStats {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub min: f64,
    pub max: f64,
    pub digest: TDigest,
    pub pending: Vec<f64>,
    /// Values fed to the digest path. Tracked separately from `n` so that
    /// `quantile()` can answer honestly when a moments-only scan populated the
    /// moments but never fed the digest: the alternative is returning whatever
    /// an empty digest estimates, which is a number that looks like an answer.
    pub quant_n: u64,
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
            quant_n: 0,
        }
    }
}

impl NumStats {
    /// Update count, min, max, mean, and m2 — and nothing else.
    ///
    /// This is the shared moments arithmetic for both push paths, kept in one
    /// place so a moments-only scan is bit-identical to the moments a full
    /// scan would have produced.
    #[inline(always)]
    pub fn push_moments(&mut self, x: f64) {
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
    }

    #[inline(always)]
    pub fn push(&mut self, x: f64) {
        self.push_moments(x);
        self.quant_n += 1;

        self.pending.push(x);
        if self.pending.len() >= 1024 {
            let batch = std::mem::take(&mut self.pending);
            self.digest = self.digest.merge_unsorted(batch);
            self.pending = Vec::with_capacity(1024);
        }
    }

    #[inline(always)]
    pub fn finalize(&mut self) {
        if !self.pending.is_empty() {
            let batch = std::mem::take(&mut self.pending);
            self.digest = self.digest.merge_unsorted(batch);
        }
    }

    pub fn merge(&mut self, other: &mut NumStats) {
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
        self.quant_n += other.quant_n;

        // Use std::mem::take to avoid cloning digests.
        let self_digest = std::mem::take(&mut self.digest);
        let other_digest = std::mem::take(&mut other.digest);
        self.digest = TDigest::merge_digests(vec![self_digest, other_digest]);
    }

    pub fn std_pop(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some((self.m2 / self.n as f64).sqrt())
        }
    }

    pub fn quantile(&self, q: f64) -> Option<f64> {
        // Keyed off quant_n, not n: a moments-only scan has n > 0 but never
        // fed the digest, and an empty digest's estimate is not an answer.
        if self.quant_n == 0 {
            None
        } else {
            Some(self.digest.estimate_quantile(q))
        }
    }
}

/// Interned string frequency table: keyed by (hash, len), each bucket holds the
/// collision candidates as (bytes, count) pairs.
pub type StringFreqMap = AHashMap<(u64, usize), Vec<(Vec<u8>, u64)>>;

#[derive(Clone)]
pub struct FreqTracker {
    pub ints: AHashMap<i64, u64>,
    pub strings: StringFreqMap,
    pub max_strings: usize,
    pub total_unique: usize,
    pub hash_state: ahash::RandomState,
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
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn push_int(&mut self, v: i64) {
        *self.ints.entry(v).or_insert(0) += 1;
    }

    #[inline(always)]
    pub fn push_string_bytes(&mut self, bytes: &[u8]) {
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

    pub fn merge(&mut self, other: &FreqTracker) {
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

    pub fn mode_int(&self) -> Option<(i64, u64)> {
        self.ints
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(&k, &v)| (k, v))
    }

    pub fn mode_string(&self) -> Option<(String, u64)> {
        // Find mode across all candidates.
        self.strings
            .values()
            .flat_map(|candidates| candidates.iter())
            .max_by_key(|(_, count)| count)
            .map(|(bytes, count)| (String::from_utf8_lossy(bytes).into_owned(), *count))
    }
}

#[derive(Clone, Default)]
pub struct ColStats {
    pub count: u64,
    pub null_count: u64,
    pub type_mask: u8,
    pub examples: Vec<String>,
    pub num: NumStats,
    pub freq: FreqTracker,
}

impl ColStats {
    pub fn merge(&mut self, other: &mut ColStats) {
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

    pub fn finalize(&mut self) {
        self.num.finalize();
    }
}

/// Full cell processing with type inference, examples, and frequency tracking
#[inline(always)]
pub fn process_cell(
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
pub fn process_cell_lite(bytes: &[u8], stats: &mut ColStats) {
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

/// MOMENTS cell processing - min/max/mean/std only, no t-digest.
///
/// The scaler-params scans (min-max, standardization) read exactly these four
/// statistics and nothing else, and the digest they would otherwise feed costs
/// ~8x the moments themselves. Mirrors `process_cell_lite`'s classification
/// exactly — same trim, same int-then-float parse order — so the moments it
/// produces are bit-identical to what a lite scan would have reported.
#[inline(always)]
pub fn process_cell_moments(bytes: &[u8], stats: &mut ColStats) {
    stats.count += 1;
    let trimmed = trim_bytes(bytes);

    if trimmed.is_empty() {
        stats.null_count += 1;
        return;
    }

    if is_integer_bytes(trimmed) {
        if let Ok(i) = atoi_simd::parse::<i64>(trimmed) {
            stats.num.push_moments(i as f64);
            return;
        }
    }

    if let Ok(val) = fast_float::parse::<f64, _>(trimmed) {
        stats.num.push_moments(val);
    }
}
