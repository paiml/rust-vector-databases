//! Embedder trait + a deterministic [`HashEmbedder`] used by both
//! workspace demos.
//!
//! # Why a hash embedder
//!
//! The course needs an embedding step that:
//!
//! 1. produces deterministic output (same input → same vector across runs),
//! 2. requires zero downloads (no model weights, no ONNX runtime),
//! 3. yields meaningfully different vectors for different inputs so
//!    nearest-neighbor demos rank distinct items differently,
//! 4. compiles on the MSRV without pulling in C / Python.
//!
//! `HashEmbedder` does this by seeding a `DefaultHasher` from the
//! input string, then drawing `dim` floats in `[-1, 1]`. The result is
//! L2-normalized so cosine similarity behaves predictably.
//!
//! # Production swap
//!
//! In a real pipeline you would replace `HashEmbedder` with one of:
//!
//! * [`fastembed`](https://crates.io/crates/fastembed) — ONNX-runtime,
//!   ships BGE-small / BGE-base / etc. as 384 / 768-dim sentence vectors.
//! * [`candle`](https://crates.io/crates/candle-core) +
//!   `candle-transformers` — pure-Rust BERT / sentence-transformers
//!   that load directly from Hugging Face.
//! * `aprender-rag`'s `FastEmbedder` (gated behind the `embeddings`
//!   feature) — same fastembed wrapping, integrated with the rest of
//!   the aprender stack.
//!
//! All three implement the same shape — `dim()` + `embed(&str) ->
//! Vec<f32>` — so the [`Embedder`] trait below is the single seam.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use thiserror::Error;

/// Errors produced by [`Embedder`] implementations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmbedError {
    /// The caller asked an embedder to encode an empty string. Most
    /// embedders treat this as user error rather than silently
    /// returning the zero vector.
    #[error("cannot embed an empty string")]
    EmptyText,
    /// The embedder configuration requested a 0-dimensional output.
    #[error("invalid dimension: {0} (must be >= 1)")]
    InvalidDimension(usize),
}

/// The single seam every backend implements.
///
/// `dim` is the per-vector length; the [`embed`](Self::embed) method
/// must always return a `Vec<f32>` of exactly that length, or an
/// [`EmbedError`].
pub trait Embedder {
    /// Number of components per emitted vector.
    fn dim(&self) -> usize;

    /// Encode a single string into a fixed-length vector.
    ///
    /// # Errors
    /// Returns [`EmbedError::EmptyText`] when called with `""`.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Convenience: encode a batch of strings. Default implementation
    /// just iterates `embed`; specialized backends (BGE, OpenAI) can
    /// override for batched throughput.
    ///
    /// # Errors
    /// Returns the first per-input [`EmbedError`].
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// Default vector dimension used by the workspace demos. 384 is the
/// same width as BGE-small-en, so swapping in a real fastembed model
/// later requires only changing the embedder, not the collection.
pub const DEFAULT_DIM: usize = 384;

/// Deterministic teaching / test embedder.
///
/// Seeded entirely from the input string — same input always yields
/// the same vector across runs, processes, and machines. The output
/// is L2-normalized, so [`vec_core::cosine_similarity`] is well-behaved.
///
/// **Not a production embedder.** It captures *no semantic meaning*;
/// the goal is reproducibility for demos and tests. Identical strings
/// produce identical vectors; different strings produce different
/// (but semantically arbitrary) vectors.
///
/// # Example
///
/// ```
/// use vec_embed::{Embedder, HashEmbedder, DEFAULT_DIM};
/// let e = HashEmbedder::new(DEFAULT_DIM);
/// let v = e.embed("a romantic comedy").unwrap();
/// assert_eq!(v.len(), DEFAULT_DIM);
/// // determinism — same input always produces the same vector
/// let v2 = e.embed("a romantic comedy").unwrap();
/// assert_eq!(v, v2);
/// ```
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// Construct a new `HashEmbedder` emitting `dim`-dimensional vectors.
    ///
    /// # Panics
    /// If `dim == 0` (every embedder must produce at least one component).
    #[must_use]
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "HashEmbedder dim must be >= 1, got {dim}");
        Self { dim }
    }

    fn hash_to_floats(text: &str, dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(dim);
        for i in 0..dim {
            // Re-seed per component so we don't just produce a stretched
            // copy of the same hash.
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Map [0, u64::MAX] → [-1.0, 1.0]
            #[allow(clippy::cast_precision_loss)]
            let normalized = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            #[allow(clippy::cast_possible_truncation)]
            out.push(normalized as f32);
        }
        Self::l2_normalize(&mut out);
        out
    }

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::EmptyText);
        }
        Ok(Self::hash_to_floats(text, self.dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vec_core::cosine_similarity;

    #[test]
    fn dim_is_what_we_asked_for() {
        let e = HashEmbedder::new(64);
        assert_eq!(e.dim(), 64);
        let v = e.embed("hello").unwrap();
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn deterministic_across_calls() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        let a = e.embed("a romantic comedy").unwrap();
        let b = e.embed("a romantic comedy").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_inputs_different_vectors() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        let a = e.embed("a romantic comedy").unwrap();
        let b = e.embed("a war epic").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn output_is_l2_normalized() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        let v = e.embed("anything").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn cosine_self_similarity_is_one() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        let v = e.embed("self-similarity check").unwrap();
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_text_errors() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        assert_eq!(e.embed("").unwrap_err(), EmbedError::EmptyText);
    }

    #[test]
    fn batch_matches_single() {
        let e = HashEmbedder::new(DEFAULT_DIM);
        let texts = ["one", "two", "three"];
        let batch = e.embed_batch(&texts).unwrap();
        assert_eq!(batch.len(), 3);
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(batch[i], e.embed(t).unwrap());
        }
    }

    #[test]
    #[should_panic(expected = "HashEmbedder dim must be >= 1")]
    fn zero_dim_panics() {
        let _ = HashEmbedder::new(0);
    }
}
