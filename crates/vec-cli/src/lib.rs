//! Library half of `vec-cli` — re-exported helpers shared by the
//! `vec-cli` binary and the two `examples/` end-to-end demos.
//!
//! The film corpus is loaded from `data/films.json` at the workspace
//! root. The path is resolved relative to `CARGO_MANIFEST_DIR` so the
//! same code works under `cargo run`, `cargo test`, and the examples.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use vec_qdrant::Film;

/// Default name of the on-disk film corpus.
pub const DEFAULT_CORPUS_FILENAME: &str = "films.json";

/// Default Qdrant gRPC URL — matches `compose.yml`.
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6334";

/// Default collection name used by every demo.
pub const DEFAULT_COLLECTION: &str = "sakila_films";

/// Default top-k for query results — every demo asserts this exact count.
pub const DEFAULT_TOP_K: usize = 5;

/// Default canned query — short, generic enough to land non-trivial
/// results against the Sakila corpus.
pub const DEFAULT_QUERY: &str = "a romantic comedy";

/// Resolve `data/films.json` from the workspace root, returning the
/// canonical [`PathBuf`] for the corpus.
///
/// # Panics
/// Panics if the `CARGO_MANIFEST_DIR` of `vec-cli` is at the
/// filesystem root (impossible in normal layouts — every crate sits
/// under `crates/<name>/`, so two `parent()` walks always succeed).
#[must_use]
pub fn default_corpus_path() -> PathBuf {
    // CARGO_MANIFEST_DIR points at crates/vec-cli, so step up two
    // levels to reach the workspace root, then into data/.
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .expect("crate dir has a parent (crates/)")
        .parent()
        .expect("crates dir has a parent (workspace root)")
        .join("data")
        .join(DEFAULT_CORPUS_FILENAME)
}

/// Load the JSON film corpus from `path`. Returns the deserialized
/// `Vec<Film>` or an error wrapping the IO / JSON failure.
///
/// # Errors
/// IO error reading the file, or `serde_json` error parsing it.
pub fn load_corpus(path: &std::path::Path) -> Result<Vec<Film>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read film corpus at {}", path.display()))?;
    let films: Vec<Film> = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse film corpus at {}", path.display()))?;
    Ok(films)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_path_lives_under_workspace_root() {
        let p = default_corpus_path();
        let s = p.to_string_lossy();
        assert!(s.ends_with("data/films.json"), "got {s}");
    }

    #[test]
    fn loading_real_corpus_yields_at_least_50_films() {
        // The committed corpus has ~50 films; if it ever drops below
        // that the demo loses statistical interest. Sentinel.
        let p = default_corpus_path();
        let films = load_corpus(&p).expect("loading committed corpus must succeed");
        assert!(
            films.len() >= 50,
            "corpus has {} films, expected >=50",
            films.len()
        );
        for f in &films {
            assert!(f.id > 0, "film {} has non-positive id", f.title);
            assert!(!f.title.is_empty(), "film id {} has empty title", f.id);
            assert!(
                !f.description.is_empty(),
                "film {} has empty description",
                f.title
            );
        }
    }
}
