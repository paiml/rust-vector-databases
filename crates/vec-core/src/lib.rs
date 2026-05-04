//! Distance metrics, vector types, and shared error types for the
//! **Vector Databases with Rust** course (RDE c13).
//!
//! This crate is the bottom of the workspace stack — every other
//! `vec-*` crate depends on it. It only carries pure-Rust types and
//! arithmetic; no IO, no async, no external services.
//!
//! # Distance metrics
//!
//! The three workhorse metrics for vector retrieval, all over `&[f32]`:
//!
//! * [`cosine_similarity`] — angle-only, robust to magnitude;
//!   the default for normalized embeddings (most production stacks).
//! * [`dot_product`] — magnitude-aware; the right score when the
//!   embedder *encodes* importance into vector length, like OpenAI v3.
//! * [`l2_distance`] — Euclidean. Lower is better — opposite of the
//!   other two. Use when you actually want spatial distance.
//!
//! # Errors
//!
//! Length mismatches surface as [`MetricError::DimensionMismatch`].
//! Empty vectors surface as [`MetricError::EmptyVector`]. The metric
//! functions return [`MetricResult`] (`Result<f32, MetricError>`).
//!
//! # Example
//!
//! ```
//! use vec_core::{cosine_similarity, dot_product, l2_distance};
//! let a = [1.0_f32, 0.0, 0.0];
//! let b = [1.0_f32, 0.0, 0.0];
//! assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 1e-6);
//! assert!((dot_product(&a, &b).unwrap() - 1.0).abs() < 1e-6);
//! assert!(l2_distance(&a, &b).unwrap().abs() < 1e-6);
//! ```

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use thiserror::Error;

/// Result alias for metric calculations.
pub type MetricResult = Result<f32, MetricError>;

/// Errors raised by distance / similarity metrics.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetricError {
    /// The two vectors have different lengths and cannot be compared.
    #[error("dimension mismatch: lhs={lhs}, rhs={rhs}")]
    DimensionMismatch {
        /// Length of the left-hand vector.
        lhs: usize,
        /// Length of the right-hand vector.
        rhs: usize,
    },
    /// At least one of the inputs has length zero.
    #[error("empty input vector — cannot compute metric on a 0-length slice")]
    EmptyVector,
}

#[inline]
fn check_compatible(a: &[f32], b: &[f32]) -> Result<(), MetricError> {
    if a.is_empty() || b.is_empty() {
        return Err(MetricError::EmptyVector);
    }
    if a.len() != b.len() {
        return Err(MetricError::DimensionMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    Ok(())
}

/// Cosine similarity between two real-valued vectors.
///
/// Returns a value in `[-1.0, 1.0]` — `1.0` is identical direction,
/// `0.0` is orthogonal, `-1.0` is opposite direction. Magnitude is
/// divided out, so `cos(a, 2*a) == cos(a, a) == 1.0`.
///
/// If either vector is the zero vector, returns `0.0` (orthogonal by
/// convention — avoids dividing by zero).
///
/// # Errors
///
/// * [`MetricError::EmptyVector`] if either input is empty.
/// * [`MetricError::DimensionMismatch`] if the lengths differ.
///
/// # Example
///
/// ```
/// use vec_core::cosine_similarity;
/// let a = [1.0_f32, 2.0, 3.0];
/// let b = [2.0_f32, 4.0, 6.0]; // same direction, double magnitude
/// let s = cosine_similarity(&a, &b).unwrap();
/// assert!((s - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> MetricResult {
    check_compatible(a, b)?;
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// Dot product of two real-valued vectors.
///
/// Returns the sum of element-wise products. Unlike cosine similarity
/// the result is unbounded — magnitude matters. Use this when the
/// embedder *encodes* importance into the vector norm.
///
/// # Errors
///
/// * [`MetricError::EmptyVector`] if either input is empty.
/// * [`MetricError::DimensionMismatch`] if the lengths differ.
pub fn dot_product(a: &[f32], b: &[f32]) -> MetricResult {
    check_compatible(a, b)?;
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

/// Euclidean (L2) distance between two real-valued vectors.
///
/// Returns `sqrt(sum((a_i - b_i)^2))`. Lower means closer — this is the
/// only one of the three metrics where smaller-is-better. Useful when
/// the embedding space genuinely has spatial meaning (image features,
/// raw geometric data).
///
/// # Errors
///
/// * [`MetricError::EmptyVector`] if either input is empty.
/// * [`MetricError::DimensionMismatch`] if the lengths differ.
pub fn l2_distance(a: &[f32], b: &[f32]) -> MetricResult {
    check_compatible(a, b)?;
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum();
    Ok(sum_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!(approx_eq(cosine_similarity(&v, &v).unwrap(), 1.0));
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(approx_eq(cosine_similarity(&a, &b).unwrap(), 0.0));
    }

    #[test]
    fn cosine_opposite_is_negative_one() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b).unwrap(), -1.0));
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let a = [1.0_f32, 1.0];
        let b = [10.0_f32, 10.0];
        assert!(approx_eq(cosine_similarity(&a, &b).unwrap(), 1.0));
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = [1.0_f32, 1.0];
        let b = [0.0_f32, 0.0];
        // Exact zero is the convention for cos(_, 0) so float_cmp is fine here.
        assert!(approx_eq(cosine_similarity(&a, &b).unwrap(), 0.0));
    }

    #[test]
    fn dot_product_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!(approx_eq(dot_product(&a, &b).unwrap(), 32.0));
    }

    #[test]
    fn dot_product_orthogonal_is_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(approx_eq(dot_product(&a, &b).unwrap(), 0.0));
    }

    #[test]
    fn l2_identical_is_zero() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!(approx_eq(l2_distance(&v, &v).unwrap(), 0.0));
    }

    #[test]
    fn l2_basic() {
        let a = [0.0_f32, 0.0];
        let b = [3.0_f32, 4.0];
        // sqrt(9 + 16) = 5
        assert!(approx_eq(l2_distance(&a, &b).unwrap(), 5.0));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32, 2.0, 3.0];
        assert_eq!(
            cosine_similarity(&a, &b).unwrap_err(),
            MetricError::DimensionMismatch { lhs: 2, rhs: 3 }
        );
        assert_eq!(
            dot_product(&a, &b).unwrap_err(),
            MetricError::DimensionMismatch { lhs: 2, rhs: 3 }
        );
        assert_eq!(
            l2_distance(&a, &b).unwrap_err(),
            MetricError::DimensionMismatch { lhs: 2, rhs: 3 }
        );
    }

    #[test]
    fn empty_vector_errors() {
        let empty: [f32; 0] = [];
        let v = [1.0_f32];
        assert_eq!(
            cosine_similarity(&empty, &v).unwrap_err(),
            MetricError::EmptyVector
        );
        assert_eq!(
            dot_product(&v, &empty).unwrap_err(),
            MetricError::EmptyVector
        );
        assert_eq!(
            l2_distance(&empty, &empty).unwrap_err(),
            MetricError::EmptyVector
        );
    }
}
