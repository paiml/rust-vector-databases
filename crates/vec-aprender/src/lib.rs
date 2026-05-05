//! In-process Sakila-film RAG pipeline built on `aprender-rag`.
//!
//! This crate is the "no Docker required" half of the workspace. It
//! wraps [`aprender_rag::RagPipeline`] (the published `aprender-rag`
//! crate's library name) configured for the Sakila film corpus:
//!
//! * `RecursiveChunker(512, 50)` — film descriptions are short
//!   (1–3 sentences), so each film almost always lands in a single
//!   chunk.
//! * `MockEmbedder(384)` — deterministic, hash-based, zero downloads.
//!   Production swaps in `FastEmbedder` (BGE / MiniLM) but the API
//!   surface is identical.
//! * `NoOpReranker` — the demo doesn't need a cross-encoder; the
//!   pipeline returns the dense + sparse fusion ranking directly.
//! * `FusionStrategy::RRF { k: 60.0 }` — the literature-standard
//!   default for combining BM25 and dense scores.
//!
//! `vec-qdrant`'s [`Film`] / [`FilmHit`] structs are reused so the two
//! demos in `vec-cli` produce byte-identical JSON shapes regardless of
//! which backend ran.
//!
//! # Document → film mapping
//!
//! Each film becomes one [`aprender_rag::Document`] with the Sakila
//! `film.id` stored in `metadata.custom["film_id"]`. On query, we
//! reach into the returned chunk's metadata to recover the id, look
//! the film back up by id (cheap — a `HashMap` is built at index
//! time), and emit a [`FilmHit`] with the fused score.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;

use anyhow::Result;
use aprender_rag::{
    chunk::RecursiveChunker, embed::MockEmbedder, fusion::FusionStrategy,
    pipeline::RagPipelineBuilder, rerank::NoOpReranker, ChunkId, Document, RagPipeline,
};
pub use vec_qdrant::{Film, FilmHit};

/// 384 dims = same width as BGE-small. Lets the demo swap to a real
/// embedder later without re-creating the collection.
pub const PIPELINE_DIM: usize = 384;
/// Default RAG chunker chunk size, in characters. Sakila descriptions
/// are short, so each film almost always lands in a single chunk.
pub const CHUNK_SIZE: usize = 512;
/// Default RAG chunker overlap, in characters.
pub const CHUNK_OVERLAP: usize = 50;

/// Sakila-film RAG pipeline backed entirely by the `aprender-rag` library.
///
/// The pipeline owns the chunker, embedder, hybrid retriever, reranker,
/// and two lookup tables:
///
/// * `chunk_to_film: HashMap<ChunkId, u64>` — built at index time by
///   capturing the [`ChunkId`]s returned from `index_document`. The
///   chunker does not propagate `Document.metadata.custom` to
///   `Chunk.metadata.custom` in `aprender-rag` 0.32.0, so we keep the
///   id-mapping on our side instead.
/// * `by_id: HashMap<u64, Film>` — recovers the full film row from the
///   id at search time.
pub struct FilmRagPipeline {
    inner: RagPipeline<MockEmbedder, NoOpReranker>,
    chunk_to_film: HashMap<ChunkId, u64>,
    by_id: HashMap<u64, Film>,
}

impl FilmRagPipeline {
    /// Build a default pipeline with [`MockEmbedder`] at [`PIPELINE_DIM`]
    /// dimensions, RRF fusion, and the recursive chunker. Returns an
    /// error only if the underlying builder rejects the configuration
    /// — should never happen with these defaults.
    ///
    /// # Errors
    /// Surfaces any error from [`RagPipelineBuilder::build`].
    pub fn new() -> Result<Self> {
        let inner = RagPipelineBuilder::new()
            .chunker(RecursiveChunker::new(CHUNK_SIZE, CHUNK_OVERLAP))
            .embedder(MockEmbedder::new(PIPELINE_DIM))
            .reranker(NoOpReranker::new())
            .fusion(FusionStrategy::RRF { k: 60.0 })
            .build()
            .map_err(|e| anyhow::anyhow!("RagPipelineBuilder::build failed: {e}"))?;
        Ok(Self {
            inner,
            chunk_to_film: HashMap::new(),
            by_id: HashMap::new(),
        })
    }

    /// Index every film as a single [`Document`]. The chunker returns
    /// the produced [`aprender_rag::Chunk`]s; we capture each
    /// [`ChunkId`] and map it back to the originating Sakila id so
    /// queries can recover the full film.
    ///
    /// # Errors
    /// Returns the first underlying indexing error; partial indexing
    /// state is preserved (we don't roll back already-indexed docs).
    pub fn index_films(&mut self, films: &[Film]) -> Result<()> {
        for film in films {
            let doc = Document::new(film.description.clone()).with_title(film.title.clone());
            let chunks = self
                .inner
                .index_document(&doc)
                .map_err(|e| anyhow::anyhow!("index_document({}) failed: {e}", film.id))?;
            for c in &chunks {
                self.chunk_to_film.insert(c.id, film.id);
            }
            self.by_id.insert(film.id, film.clone());
        }
        Ok(())
    }

    /// Run a query against the indexed films and return the top-`top_k`
    /// hits. Hits are sorted descending by fused score (RRF rank → score).
    ///
    /// Multiple chunks of the same film are de-duplicated to one
    /// `FilmHit` (the highest-scoring chunk wins). For the Sakila demo
    /// this almost never triggers — every description is shorter than
    /// `CHUNK_SIZE` — but it's the right behaviour for any future
    /// long-document corpus.
    ///
    /// # Errors
    /// Surfaces any error from the underlying retriever.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<FilmHit>> {
        // Over-fetch by 4× to give us slack to dedupe chunks by film.
        // Capped at 4× and at least top_k to handle small corpora.
        let raw_k = (top_k * 4).max(top_k);
        let results = self
            .inner
            .query(query, raw_k)
            .map_err(|e| anyhow::anyhow!("RagPipeline::query failed: {e}"))?;

        let mut by_film: HashMap<u64, FilmHit> = HashMap::new();
        let mut order: Vec<u64> = Vec::new();

        for r in results {
            let Some(&film_id) = self.chunk_to_film.get(&r.chunk.id) else {
                continue;
            };
            let Some(film) = self.by_id.get(&film_id) else {
                continue;
            };
            let score = score_of(&r);
            by_film
                .entry(film_id)
                .and_modify(|h| {
                    if score > h.score {
                        h.score = score;
                    }
                })
                .or_insert_with(|| {
                    order.push(film_id);
                    FilmHit {
                        id: film.id,
                        title: film.title.clone(),
                        score,
                    }
                });
        }

        // Re-sort by best chunk score per film, descending. The retriever
        // already gave us roughly that ordering, but the dedupe step can
        // perturb it.
        let mut hits: Vec<FilmHit> = order
            .into_iter()
            .filter_map(|id| by_film.remove(&id))
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Number of indexed films (one document per film).
    #[must_use]
    pub fn film_count(&self) -> usize {
        self.by_id.len()
    }
}

/// Best-available score from a [`aprender_rag::RetrievalResult`].
///
/// Hybrid retrieval surfaces multiple scoring layers; we pick the
/// fused score if present, otherwise the dense score, otherwise the
/// sparse score, otherwise 0.0.
fn score_of(r: &aprender_rag::RetrievalResult) -> f32 {
    r.fused_score
        .or(r.dense_score)
        .or(r.sparse_score)
        .or(r.rerank_score)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_films() -> Vec<Film> {
        vec![
            Film {
                id: 1,
                title: "ACADEMY DINOSAUR".to_string(),
                description: "An epic comedy of a dinosaur and a teacher who must defeat a moose."
                    .to_string(),
            },
            Film {
                id: 2,
                title: "ROMANCE LIONS".to_string(),
                description: "A romantic comedy about two lions in love at the central park zoo."
                    .to_string(),
            },
            Film {
                id: 3,
                title: "WAR ENGINES".to_string(),
                description: "An epic war drama about a brave squadron of helicopter pilots."
                    .to_string(),
            },
        ]
    }

    #[test]
    fn build_pipeline_default_succeeds() {
        let p = FilmRagPipeline::new();
        assert!(p.is_ok(), "default builder should succeed: {:?}", p.err());
    }

    #[test]
    fn index_films_populates_lookup() {
        let mut p = FilmRagPipeline::new().unwrap();
        let films = three_films();
        p.index_films(&films).unwrap();
        assert_eq!(p.film_count(), 3);
    }

    #[test]
    fn search_returns_hits_with_known_ids() {
        let mut p = FilmRagPipeline::new().unwrap();
        let films = three_films();
        p.index_films(&films).unwrap();
        let hits = p.search("a romantic comedy", 3).unwrap();
        // We can't assert a specific ranking with the mock embedder,
        // but every hit must be one of the indexed ids.
        assert!(!hits.is_empty(), "should return at least one hit");
        let allowed: std::collections::HashSet<u64> = films.iter().map(|f| f.id).collect();
        for h in &hits {
            assert!(
                allowed.contains(&h.id),
                "hit id {} not in {:?}",
                h.id,
                allowed
            );
        }
    }

    #[test]
    fn search_top_k_limits_results() {
        let mut p = FilmRagPipeline::new().unwrap();
        p.index_films(&three_films()).unwrap();
        let hits = p.search("anything", 2).unwrap();
        assert!(
            hits.len() <= 2,
            "expected at most 2 hits, got {}",
            hits.len()
        );
    }
}
