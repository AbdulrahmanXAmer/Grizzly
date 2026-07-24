//! Model fitting: linear regression, and the shared machinery every model
//! after it reuses.
//!
//! Unlike `parse` and `stats`, this module does depend on PyO3 — these are the
//! `#[pyfunction]` entry points. It is separated anyway because it is where
//! every additional model goes, and lib.rs had already grown past the point
//! where adding one more was reasonable.
//!
//! Three pieces here are load-bearing for anything added later:
//!
//! * [`sgd_step`] — the per-sample update. Shared verbatim by the streaming
//!   and cached-replay paths so the two cannot drift; new losses plug in here.
//! * The XtX accumulation in [`csv_linear_regression`] — chunk-parallel
//!   sufficient statistics with prefix-summed row indices, folded in chunk
//!   order for determinism. Any model that fits by accumulating a fixed-size
//!   statistic over rows reuses this shape.
//! * [`csv_layout`] and the split/mask machinery — deterministic train/test
//!   assignment that every fit and eval pass agrees on.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use ahash::AHashMap;
use csv::ReaderBuilder;
use memchr::memchr;
use rayon::prelude::*;

use crate::parse::{
    chunk_bytes_aligned, detect_header_smart, for_each_field, get_fields, sniff_delimiter_simd,
    trim_bytes, FastLineIter, SplitMode,
};
use crate::{csv_standardize_params, load_file_data};

/// Result of fitting a linear model, before conversion into a Python dict:
/// (feature names, coefficients, intercept, train rows, test rows used, r2,
/// test rows assigned, residual sum of squares, total sum of squares, mean y).
///
/// TODO: this tuple is load-bearing across a long function and should become a
/// named struct; the alias documents the positions in the meantime.
pub type RegressionFit = (
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

/// Where the data starts and what the columns are called.
///
/// Shared setup for anything that streams a CSV: delimiter, header presence,
/// column names, and the byte offset of the first data row.
pub struct CsvLayout {
    pub split_mode: SplitMode,
    pub col_names: Vec<String>,
    pub data_start: usize,
}

pub fn csv_layout(
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

/// Outcome of a streaming SGD fit, before conversion into a Python dict.
///
/// A struct rather than the ten-element tuple this used to be: classification
/// adds five more fields, and positional access was already the wrong shape.
pub struct SgdOutcome {
    pub feature_names: Vec<String>,
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub train_n: usize,
    pub test_n: usize,
    pub epochs: usize,
    /// Mean per-sample value of the objective on the final training epoch.
    pub final_train_loss: f64,
    /// Held-out R^2. Meaningful for regression losses only.
    pub r2: f64,
    /// Present when the loss is a classification objective.
    pub classification: Option<ClassificationMetrics>,
}

/// Held-out classification metrics.
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub log_loss: f64,
    pub roc_auc: f64,
    pub true_positives: u64,
    pub false_positives: u64,
    pub true_negatives: u64,
    pub false_negatives: u64,
    pub positive_rate: f64,
}

/// Number of score bins used for the streaming ROC-AUC estimate.
///
/// AUC is a rank statistic, so the direct computation sorts every score --
/// which means holding the whole test set in memory and would give up the
/// bounded-memory property that is the point of this path. Binning the scores
/// instead makes it O(bins): with 4096 bins over [0,1] the estimate is within
/// ~1e-3 of exact, which the tests assert against scikit-learn.
const AUC_BINS: usize = 4096;

/// Streaming accumulator for binary classification metrics.
///
/// Fixed size regardless of how many rows it sees.
pub struct BinaryEval {
    pos_bins: Vec<u64>,
    neg_bins: Vec<u64>,
    correct: u64,
    total: u64,
    log_loss_sum: f64,
    tp: u64,
    fp: u64,
    tn: u64,
    r#fn: u64,
}

impl Default for BinaryEval {
    fn default() -> Self {
        Self {
            pos_bins: vec![0; AUC_BINS],
            neg_bins: vec![0; AUC_BINS],
            correct: 0,
            total: 0,
            log_loss_sum: 0.0,
            tp: 0,
            fp: 0,
            tn: 0,
            r#fn: 0,
        }
    }
}

impl BinaryEval {
    /// Record one held-out row. `score` is a probability in [0, 1].
    pub fn push(&mut self, score: f64, label: f64) {
        let positive = label > 0.5;
        let predicted = score >= 0.5;

        self.total += 1;
        if predicted == positive {
            self.correct += 1;
        }
        match (positive, predicted) {
            (true, true) => self.tp += 1,
            (false, true) => self.fp += 1,
            (false, false) => self.tn += 1,
            (true, false) => self.r#fn += 1,
        }

        // Clamped so a saturated probability cannot contribute -inf and
        // destroy the whole metric; 1e-15 matches scikit-learn's eps.
        let p = score.clamp(1e-15, 1.0 - 1e-15);
        self.log_loss_sum -= if positive { p.ln() } else { (1.0 - p).ln() };

        let bin = ((score.clamp(0.0, 1.0)) * (AUC_BINS - 1) as f64).round() as usize;
        if positive {
            self.pos_bins[bin] += 1;
        } else {
            self.neg_bins[bin] += 1;
        }
    }

    /// Mann-Whitney U over the binned scores.
    ///
    /// Walking bins in ascending score order, each positive outranks every
    /// negative in a lower bin and ties with those in its own; the 0.5 credit
    /// for ties is what makes this agree with scikit-learn rather than
    /// systematically over- or under-counting.
    fn roc_auc(&self) -> f64 {
        let n_pos: u64 = self.pos_bins.iter().sum();
        let n_neg: u64 = self.neg_bins.iter().sum();
        if n_pos == 0 || n_neg == 0 {
            // Undefined with only one class present. 0.5 is the honest
            // "no discrimination measurable" answer.
            return 0.5;
        }

        let mut negatives_below = 0u64;
        let mut rank_sum = 0.0f64;
        for bin in 0..AUC_BINS {
            let pos = self.pos_bins[bin] as f64;
            let neg = self.neg_bins[bin] as f64;
            if pos > 0.0 {
                rank_sum += pos * (negatives_below as f64 + 0.5 * neg);
            }
            negatives_below += self.neg_bins[bin];
        }
        rank_sum / (n_pos as f64 * n_neg as f64)
    }

    pub fn finish(&self) -> ClassificationMetrics {
        let total = self.total.max(1) as f64;
        ClassificationMetrics {
            accuracy: self.correct as f64 / total,
            log_loss: self.log_loss_sum / total,
            roc_auc: self.roc_auc(),
            true_positives: self.tp,
            false_positives: self.fp,
            true_negatives: self.tn,
            false_negatives: self.r#fn,
            positive_rate: (self.tp + self.r#fn) as f64 / total,
        }
    }
}

/// The objective an SGD fit minimises.
///
/// Every one of these has the same gradient *shape* with respect to the
/// weights -- `dL/dw_j = g * x_j` for a scalar `g` that depends only on the
/// prediction and the label. So a single update kernel serves all of them, and
/// adding a linear model means adding a variant here rather than another
/// training loop -- hinge (linear SVM) and Huber both fit this shape and drop
/// in without touching [`sgd_step`]. `residual()` returns that `g`; `loss()`
/// returns the value being minimised, for the reported training metric.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Loss {
    /// Ordinary least squares. Labels are real-valued.
    Squared,
    /// Binary logistic (log-loss). Labels must be 0 or 1; `pred` is a logit
    /// and the reported probability is its sigmoid.
    Logistic,
}

/// Numerically stable logistic function.
///
/// The naive `1/(1+exp(-z))` overflows to inf for `z` around -750 and then
/// produces NaN. Branching on the sign keeps the exponent negative, which is
/// the standard trick and matters here because an early SGD iterate can
/// easily produce a logit that large before the weights settle.
#[inline(always)]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Loss {
    /// The scalar `g` in `dL/dw_j = g * x_j`.
    #[inline(always)]
    pub fn residual(&self, pred: f64, y: f64) -> f64 {
        match *self {
            Loss::Squared => pred - y,
            // d/dz of -[y log s(z) + (1-y) log(1-s(z))] is exactly s(z) - y,
            // which is why logistic costs no more per row than squared.
            Loss::Logistic => sigmoid(pred) - y,
        }
    }

    /// The value being minimised, accumulated as the reported training metric.
    #[inline(always)]
    pub fn loss(&self, pred: f64, y: f64) -> f64 {
        match *self {
            Loss::Squared => {
                let r = pred - y;
                r * r
            }
            Loss::Logistic => {
                // log(1 + exp(-|z|)) + max(z,0) - y*z, the stable form of
                // -[y log s + (1-y) log(1-s)]: never takes log of an
                // underflowed sigmoid, which would be -inf.
                let z = pred;
                (1.0 + (-z.abs()).exp()).ln() + z.max(0.0) - y * z
            }
        }
    }

    /// Whether labels are expected to be 0/1.
    pub fn is_classification(&self) -> bool {
        matches!(self, Loss::Logistic)
    }
}

/// One SGD update, shared verbatim by the streaming and replay paths.
///
/// Kept as a single function so the cached-epoch replay cannot drift from the
/// streaming arithmetic: identical inputs through identical operations is what
/// makes the two paths bit-identical, and the tests assert exactly that. The
/// `loss` parameter is what lets regression and classification share it:
/// only the scalar residual differs.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn sgd_step(
    x: &[f64],
    y: f64,
    w: &mut [f64],
    b: &mut f64,
    lr: f64,
    l2: f64,
    grad_clip: f64,
    loss: Loss,
    sq_err: &mut f64,
    seen: &mut usize,
) {
    let p = w.len();
    let mut pred = *b;
    for j in 0..p {
        pred += w[j] * x[j];
    }
    let err = loss.residual(pred, y);
    *sq_err += loss.loss(pred, y);
    *seen += 1;

    // Clip the per-sample gradient. Real data has outliers that survive
    // standardization -- a mis-metered taxi trip of 100,000 miles sits a
    // thousand standard deviations out -- and a single such row produces a
    // step large enough to diverge the fit. Clipping bounds any one row's
    // influence without the caller hand-tuning the learning rate per dataset.
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
    *b -= step * err;
}

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
#[pyo3(signature = (path, target, features=None, epochs=5, learning_rate=0.05, l2=0.0, train_frac=0.8, seed=0_u64, sample_size=1_000_000, delimiter=None, has_header=None, shuffle=true, grad_clip=10.0, cache_budget_mb=512))]
pub fn csv_sgd_regression(
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
    cache_budget_mb: usize,
) -> PyResult<PyObject> {
    fit_sgd(
        py,
        path,
        target,
        features,
        epochs,
        learning_rate,
        l2,
        train_frac,
        seed,
        sample_size,
        delimiter,
        has_header,
        shuffle,
        grad_clip,
        cache_budget_mb,
        Loss::Squared,
    )
}

/// Fit a binary logistic classifier by SGD, streaming from CSV.
///
/// Identical machinery to [`csv_sgd_regression`] -- same standardization, same
/// deterministic split, same cached-replay optimisation -- with the squared
/// loss swapped for log-loss. The target column must contain only 0 and 1.
///
/// The reported metrics are the ones a classifier is actually judged on:
/// held-out accuracy, log-loss, ROC-AUC, and the confusion matrix. R^2 is
/// returned as well but means little here, so it is not the headline.
///
/// `coef` and `intercept` are in the original feature space and parameterise a
/// logit: `p = sigmoid(intercept + coef . x)`. They are directly comparable
/// with scikit-learn's `LogisticRegression(penalty=None)` fitted on the same
/// data, which the differential tests assert.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (path, target, features=None, epochs=5, learning_rate=0.05, l2=0.0, train_frac=0.8, seed=0_u64, sample_size=1_000_000, delimiter=None, has_header=None, shuffle=true, grad_clip=10.0, cache_budget_mb=512))]
pub fn csv_logistic_regression(
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
    cache_budget_mb: usize,
) -> PyResult<PyObject> {
    fit_sgd(
        py,
        path,
        target,
        features,
        epochs,
        learning_rate,
        l2,
        train_frac,
        seed,
        sample_size,
        delimiter,
        has_header,
        shuffle,
        grad_clip,
        cache_budget_mb,
        Loss::Logistic,
    )
}

/// The shared body behind every SGD entry point.
///
/// Only `loss` differs between regression and classification: the streaming,
/// splitting, standardization, caching, and divergence checks are the same
/// code for all of them, which is the whole reason [`Loss`] exists.
#[allow(clippy::too_many_arguments)]
fn fit_sgd(
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
    cache_budget_mb: usize,
    loss: Loss,
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

    let result = py.allow_threads(move || -> Result<SgdOutcome, String> {
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
        let counted = FastLineIter::new(data_bytes).take(sample_size).count();
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

        // Multi-epoch training re-reads the file per epoch, and parsing is
        // most of an epoch's cost. When the standardized training matrix fits
        // under the caller's budget, epoch 0 fills a cache while it streams
        // and later epochs replay from memory: parse once, train N times.
        //
        // The budget decision uses train_cut (an upper bound on train rows),
        // so it is made before any parsing. Over budget, or with the budget
        // set to 0, every epoch streams — the bounded-memory behavior is the
        // caller's to keep. Replay feeds the exact f64s streaming would have
        // recomputed through the same sgd_step, so the fitted weights are
        // bit-identical either way; the tests assert that.
        let cache_bytes = train_cut.saturating_mul(p + 1).saturating_mul(8);
        let use_cache = epochs > 1
            && cache_budget_mb > 0
            && cache_bytes <= cache_budget_mb.saturating_mul(1024 * 1024);
        let mut cache_x: Vec<f64> = Vec::new();
        let mut cache_y: Vec<f64> = Vec::new();
        if use_cache {
            cache_x.reserve_exact(train_cut.saturating_mul(p));
            cache_y.reserve_exact(train_cut);
        }

        let mut fields: Vec<&[u8]> = Vec::with_capacity(col_names.len());

        for epoch in 0..epochs {
            // Decay the step size each epoch so later passes refine rather
            // than bounce around the optimum.
            let lr = learning_rate / (1.0 + epoch as f64);
            let mut sq_err = 0.0f64;
            let mut seen = 0usize;

            if epoch > 0 && use_cache {
                // Replay from the cache filled during epoch 0.
                for (xrow, &y) in cache_x.chunks_exact(p).zip(cache_y.iter()) {
                    sgd_step(
                        xrow,
                        y,
                        &mut w,
                        &mut b,
                        lr,
                        l2,
                        grad_clip,
                        loss,
                        &mut sq_err,
                        &mut seen,
                    );
                }
            } else {
                for (this_row, line) in FastLineIter::new(data_bytes).enumerate() {
                    if this_row >= n_rows {
                        break;
                    }
                    // Split first: nothing is needed from a test row here, so
                    // it is not worth splitting into fields.
                    if !is_train(this_row) {
                        continue;
                    }

                    fields.clear();
                    for_each_field(line, split_mode, |_, f| fields.push(f));
                    if fields.len() <= target_idx {
                        continue;
                    }

                    let Some(y) = parse_f64_opt(fields[target_idx]) else {
                        continue;
                    };
                    // A classifier silently trained on labels of 2 and 7 would
                    // return a plausible-looking model that means nothing, so
                    // this is an error rather than a coercion. Rows with an
                    // unparseable label are skipped above like any other bad
                    // row; a *parseable* label outside {0,1} is a mistake.
                    if loss.is_classification() && y != 0.0 && y != 1.0 {
                        return Err(format!(
                            "target column {target:?} must contain only 0 and 1 for \
                             classification; found {y}"
                        ));
                    }
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

                    sgd_step(
                        &x,
                        y,
                        &mut w,
                        &mut b,
                        lr,
                        l2,
                        grad_clip,
                        loss,
                        &mut sq_err,
                        &mut seen,
                    );

                    if epoch == 0 {
                        train_n += 1;
                        if use_cache {
                            cache_x.extend_from_slice(&x);
                            cache_y.push(y);
                        }
                    }
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

        // Held-out evaluation, in the original feature space. Regression
        // metrics accumulate for every loss (they cost four adds a row);
        // classification metrics only when the objective is one.
        let mut ss_res = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut test_n = 0usize;
        let mut eval = loss.is_classification().then(BinaryEval::default);
        for (this_row, line) in FastLineIter::new(data_bytes).enumerate() {
            if this_row >= n_rows {
                break;
            }
            if is_train(this_row) {
                continue;
            }
            fields.clear();
            for_each_field(line, split_mode, |_, f| fields.push(f));
            if fields.len() <= target_idx {
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
            if let Some(ev) = eval.as_mut() {
                // `pred` is a logit here; the metrics want a probability.
                ev.push(sigmoid(pred), y);
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

        Ok(SgdOutcome {
            feature_names,
            coef,
            intercept,
            train_n,
            test_n,
            epochs,
            final_train_loss: final_mse,
            r2,
            classification: eval.map(|ev| ev.finish()),
        })
    });

    let fit = result.map_err(pyo3::exceptions::PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("path", path)?;
    out.set_item("target", target)?;
    out.set_item("features", fit.feature_names)?;
    out.set_item("coef", fit.coef)?;
    out.set_item("intercept", fit.intercept)?;
    out.set_item("train_n", fit.train_n)?;
    out.set_item("test_n", fit.test_n)?;
    out.set_item("r2", fit.r2)?;
    out.set_item("epochs", fit.epochs)?;
    // Kept under its original name for the regression path, whose callers and
    // README already reference it; for classification it is the mean log-loss,
    // which is what `final_train_loss` says without lying about the metric.
    out.set_item("final_train_mse", fit.final_train_loss)?;
    out.set_item("final_train_loss", fit.final_train_loss)?;

    if let Some(m) = fit.classification {
        out.set_item("accuracy", m.accuracy)?;
        out.set_item("log_loss", m.log_loss)?;
        out.set_item("roc_auc", m.roc_auc)?;
        out.set_item("positive_rate", m.positive_rate)?;
        let cm = PyDict::new(py);
        cm.set_item("tp", m.true_positives)?;
        cm.set_item("fp", m.false_positives)?;
        cm.set_item("tn", m.true_negatives)?;
        cm.set_item("fn", m.false_negatives)?;
        out.set_item("confusion_matrix", cm)?;
    }
    Ok(out.into())
}

#[inline(always)]
pub fn splitmix64(mut x: u64) -> u64 {
    // Deterministic pseudo-random hash (fast, decent quality)
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline(always)]
pub fn parse_f64_opt(bytes: &[u8]) -> Option<f64> {
    let t = trim_bytes(bytes);
    if t.is_empty() {
        return None;
    }
    fast_float::parse::<f64, _>(t).ok()
}

pub fn gaussian_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
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
pub fn csv_linear_regression(
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
                             // XtX is symmetric, so only the upper triangle (row-major, r <= c)
                             // is accumulated: dim*(dim+1)/2 slots instead of dim^2, which also
                             // halves the multiply-adds per row in the hot loop. Mirrored into a
                             // full matrix just before the solve.
            let tri = dim * (dim + 1) / 2;
            let mut xtx_tri = vec![0.0f64; tri];
            let mut xty = vec![0.0f64; dim];

            // Two passes over the mmapped bytes: accumulate on train, then
            // evaluate on test with the solved coefficients.

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
            // Chunk the data once; the same chunks serve the row count, the
            // training pass, and the evaluation pass. Counting is a parallel
            // newline scan per chunk, and the per-chunk counts prefix-sum into
            // each chunk's global starting row index — which is what lets the
            // accumulation passes run chunk-parallel while agreeing with the
            // split about which global row is which.
            let num_threads = rayon::current_num_threads();
            let byte_chunks: Vec<&[u8]> = if fast_csv {
                chunk_bytes_aligned(data_bytes, num_threads)
            } else {
                Vec::new()
            };
            let chunk_rows: Vec<usize> = byte_chunks
                .par_iter()
                .map(|c| FastLineIter::new(c).count())
                .collect();
            let chunk_base: Vec<usize> = {
                let mut bases = Vec::with_capacity(chunk_rows.len());
                let mut acc = 0usize;
                for &r in &chunk_rows {
                    bases.push(acc);
                    acc += r;
                }
                bases
            };

            let counted = if fast_csv {
                chunk_rows.iter().sum::<usize>().min(sample_size)
            } else {
                let mut reader = ReaderBuilder::new()
                    .has_headers(has_header_actual)
                    .delimiter(delim_byte_for_detection)
                    .flexible(true)
                    .from_reader(bytes);
                reader.byte_records().take(sample_size).count()
            };
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

            let mut train_n = 0usize;
            let mut test_assigned = 0usize;

            // PASS 1: accumulate XtX/Xty on TRAIN
            if fast_csv {
                // Chunk-parallel. Each chunk accumulates its own triangle and
                // counters; partials are folded in chunk order afterwards, so
                // the summation order — and therefore the result — is
                // deterministic for a given thread count.
                struct TrainPartial {
                    xtx_tri: Vec<f64>,
                    xty: Vec<f64>,
                    train_n: usize,
                    test_assigned: usize,
                }

                let partials: Vec<TrainPartial> = byte_chunks
                    .par_iter()
                    .zip(chunk_base.par_iter())
                    .map(|(&chunk, &base)| {
                        let mut part = TrainPartial {
                            xtx_tri: vec![0.0f64; tri],
                            xty: vec![0.0f64; dim],
                            train_n: 0,
                            test_assigned: 0,
                        };
                        if base >= n_rows {
                            return part;
                        }
                        // Scratch reused across rows: no per-row allocation.
                        let mut fields: Vec<&[u8]> = Vec::with_capacity(ncols);
                        let mut x = vec![0.0f64; dim];

                        for (local, line) in FastLineIter::new(chunk).enumerate() {
                            let row_idx = base + local;
                            if row_idx >= n_rows {
                                break;
                            }
                            // Split assignment first: pass 1 needs nothing
                            // from a test row, so it is not worth parsing.
                            let is_train = if let Some(mask) = &is_train_mask {
                                mask[row_idx]
                            } else {
                                row_idx < train_cut
                            };
                            if !is_train {
                                part.test_assigned += 1;
                                continue;
                            }

                            fields.clear();
                            for_each_field(line, split_mode, |_, f| fields.push(f));
                            if fields.len() < ncols {
                                continue;
                            }

                            let yv = match parse_f64_opt(fields[target_idx]) {
                                Some(v) => v,
                                None => continue,
                            };
                            let mut ok = true;
                            for (j, &idx) in feature_idx.iter().enumerate() {
                                match parse_f64_opt(fields[idx]) {
                                    Some(v) => x[j] = v,
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if !ok {
                                continue;
                            }
                            x[p] = 1.0; // intercept

                            // Upper triangle only: xtx[r][c] for r <= c.
                            let mut k = 0usize;
                            for r in 0..dim {
                                let xr = x[r];
                                part.xty[r] += xr * yv;
                                for &xc in &x[r..dim] {
                                    part.xtx_tri[k] += xr * xc;
                                    k += 1;
                                }
                            }
                            part.train_n += 1;
                        }
                        part
                    })
                    .collect();

                for part in partials {
                    for (a, b) in xtx_tri.iter_mut().zip(&part.xtx_tri) {
                        *a += b;
                    }
                    for (a, b) in xty.iter_mut().zip(&part.xty) {
                        *a += b;
                    }
                    train_n += part.train_n;
                    test_assigned += part.test_assigned;
                }
            } else {
                let mut rows_seen = 0usize;
                let mut x_scratch = vec![0.0f64; dim];
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
                    // Same shape as the fast path: split first, parse only
                    // train rows, accumulate the upper triangle.
                    let is_train = if let Some(mask) = &is_train_mask {
                        mask[row_idx]
                    } else {
                        row_idx < train_cut
                    };
                    if !is_train {
                        test_assigned += 1;
                        continue;
                    }
                    if record.len() < ncols {
                        continue;
                    }

                    let yv = match parse_f64_opt(record.get(target_idx).unwrap_or(&[])) {
                        Some(v) => v,
                        None => continue,
                    };
                    let mut ok = true;
                    for (j, &idx) in feature_idx.iter().enumerate() {
                        match record.get(idx).and_then(parse_f64_opt) {
                            Some(v) => x_scratch[j] = v,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    x_scratch[p] = 1.0;
                    let mut k = 0usize;
                    for r in 0..dim {
                        let xr = x_scratch[r];
                        xty[r] += xr * yv;
                        for &xc in &x_scratch[r..dim] {
                            xtx_tri[k] += xr * xc;
                            k += 1;
                        }
                    }
                    train_n += 1;
                }
            }

            if train_n < dim {
                return Err(format!(
                    "Not enough training rows to fit model (train_n={train_n}, params={dim})"
                ));
            }

            // Mirror the accumulated upper triangle into the full symmetric
            // matrix the solver expects.
            let mut xtx = vec![0.0f64; dim * dim];
            {
                let mut k = 0usize;
                for r in 0..dim {
                    for c in r..dim {
                        xtx[r * dim + c] = xtx_tri[k];
                        xtx[c * dim + r] = xtx_tri[k];
                        k += 1;
                    }
                }
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

            if fast_csv {
                // Same chunk-parallel shape as pass 1, same deterministic
                // in-order fold of the partial sums.
                #[derive(Default)]
                struct EvalPartial {
                    ss_res: f64,
                    sum_y: f64,
                    sum_y2: f64,
                    test_used: usize,
                }

                let partials: Vec<EvalPartial> = byte_chunks
                    .par_iter()
                    .zip(chunk_base.par_iter())
                    .map(|(&chunk, &base)| {
                        let mut part = EvalPartial::default();
                        if base >= n_rows {
                            return part;
                        }
                        let mut fields: Vec<&[u8]> = Vec::with_capacity(ncols);

                        for (local, line) in FastLineIter::new(chunk).enumerate() {
                            let row_idx = base + local;
                            if row_idx >= n_rows {
                                break;
                            }
                            let is_train = if let Some(mask) = &is_train_mask {
                                mask[row_idx]
                            } else {
                                row_idx < train_cut
                            };
                            if is_train {
                                continue;
                            }

                            fields.clear();
                            for_each_field(line, split_mode, |_, f| fields.push(f));
                            if fields.len() < ncols {
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
                            part.ss_res += r * r;
                            part.sum_y += yv;
                            part.sum_y2 += yv * yv;
                            part.test_used += 1;
                        }
                        part
                    })
                    .collect();

                for part in partials {
                    ss_res += part.ss_res;
                    sum_y += part.sum_y;
                    sum_y2 += part.sum_y2;
                    test_used += part.test_used;
                }
            } else {
                let mut rows_seen2 = 0usize;
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
