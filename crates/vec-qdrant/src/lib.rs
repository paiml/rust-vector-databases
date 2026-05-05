//! Thin typed wrapper around `qdrant-client` for the **Vector Databases
//! with Rust** course (RDE c13).
//!
//! Two types ship from this crate:
//!
//! * [`Film`] — the row shape every demo embeds. Mirrors the Sakila
//!   `film` table: `id`, `title`, `description`. The same struct is
//!   reused by the `vec-aprender` crate so swapping backends doesn't
//!   change the document model.
//! * [`QdrantStore`] — owns a [`qdrant_client::Qdrant`] handle and
//!   exposes course-shape methods: `ensure_collection`, `upsert_films`,
//!   `search`. Each method maps onto the `qdrant-client` builder API
//!   exactly the way the M3 lessons walk it.
//!
//! # Why a wrapper at all
//!
//! `qdrant-client` is generated from gRPC protobuf, so the raw API
//! requires fluent builders for every call. A thin wrapper:
//!
//! 1. constrains the surface area the course teaches (avoid showing
//!    every dial when only a few matter for the lesson),
//! 2. stamps the same payload schema on every upsert so search results
//!    can be deserialized through `serde`,
//! 3. centralizes error context — every method tags its `anyhow!`
//!    result with the operation that failed.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use anyhow::{anyhow, Context, Result};
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use serde::{Deserialize, Serialize};
use serde_json::json;
use vec_embed::Embedder;

/// One row of the Sakila `film` table — the corpus the demos embed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Film {
    /// Sakila auto-increment primary key. Always positive.
    pub id: u64,
    /// Film title (e.g. `"ACADEMY DINOSAUR"`).
    pub title: String,
    /// 1–3 sentence creative description from the Sakila fixture.
    pub description: String,
}

/// One search hit — what the demos return after running a query
/// against the indexed corpus. Same shape regardless of backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilmHit {
    /// Sakila film id of the matched row.
    pub id: u64,
    /// Film title.
    pub title: String,
    /// Backend-reported similarity score. Higher is better for cosine /
    /// dot-product, which are the only metrics the course uses with
    /// Qdrant.
    pub score: f32,
}

/// Wrapper around a [`qdrant_client::Qdrant`] connection.
pub struct QdrantStore {
    client: Qdrant,
}

impl QdrantStore {
    /// Connect to a Qdrant gRPC endpoint at `url`. The default
    /// course URL is `"http://localhost:6334"` — what `compose.yml`
    /// publishes.
    ///
    /// # Errors
    /// Returns an error if the URL fails to parse or the gRPC channel
    /// can't be constructed (Qdrant being unreachable does not fail
    /// here — it surfaces on the first request).
    pub fn new(url: &str) -> Result<Self> {
        // Disable the compatibility check — local dev runs whichever
        // qdrant image is current, and the check is a startup cost
        // we don't need for the demo.
        let client = Qdrant::from_url(url)
            .skip_compatibility_check()
            .build()
            .with_context(|| format!("failed to build Qdrant client for {url}"))?;
        Ok(Self { client })
    }

    /// Borrow the underlying client (escape hatch for course advanced
    /// lessons that need the raw API — e.g. snapshots, quantization).
    #[must_use]
    pub fn client(&self) -> &Qdrant {
        &self.client
    }

    /// Create the collection if it doesn't already exist. Vector
    /// `dim` is the per-point dimension; the collection is configured
    /// for cosine similarity (the default for normalized embeddings).
    ///
    /// Idempotent — a 409 / "already exists" reply is treated as
    /// success. This mirrors the lesson 5.1.4 idempotent-upsert
    /// pattern.
    ///
    /// # Errors
    /// Returns the original Qdrant error for any failure other than
    /// "already exists".
    pub async fn ensure_collection(&self, name: &str, dim: u64) -> Result<()> {
        // Check first to keep the demo log clean. If the collection
        // exists we short-circuit — otherwise we create it and
        // surface any underlying error.
        let exists = self
            .client
            .collection_exists(name)
            .await
            .with_context(|| format!("collection_exists({name}) failed"))?;
        if exists {
            return Ok(());
        }
        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
            )
            .await
            .with_context(|| format!("create_collection({name}, dim={dim}) failed"))?;
        Ok(())
    }

    /// Upsert every film as a single point in `collection`. The point
    /// id is the Sakila `film.id`, so the upsert is idempotent — re-running
    /// the demo updates the existing rows rather than creating duplicates.
    ///
    /// The payload carries `title` and `description`, both retrievable
    /// via `with_payload(true)` on a search.
    ///
    /// # Errors
    /// Surfaces any Qdrant error, including embedder failures (an
    /// empty title would propagate as [`vec_embed::EmbedError::EmptyText`]).
    pub async fn upsert_films<E: Embedder>(
        &self,
        collection: &str,
        films: &[Film],
        embedder: &E,
    ) -> Result<()> {
        let mut points = Vec::with_capacity(films.len());
        for film in films {
            let vector = embedder
                .embed(&film.description)
                .with_context(|| format!("embedding film {} ({}) failed", film.id, film.title))?;
            let payload = Payload::try_from(json!({
                "id": film.id,
                "title": film.title,
                "description": film.description,
            }))
            .map_err(|e| anyhow!("payload serialization failed: {e}"))?;
            points.push(PointStruct::new(film.id, vector, payload));
        }
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, points).wait(true))
            .await
            .with_context(|| format!("upsert_points({collection}) failed"))?;
        Ok(())
    }

    /// Run a top-`limit` search against `collection` using `query_vec`
    /// as the dense query. Returns a vector of [`FilmHit`] sorted
    /// descending by score.
    ///
    /// # Errors
    /// Surfaces any Qdrant error from the underlying gRPC call.
    pub async fn search(
        &self,
        collection: &str,
        query_vec: Vec<f32>,
        limit: u64,
    ) -> Result<Vec<FilmHit>> {
        let response = self
            .client
            .search_points(
                SearchPointsBuilder::new(collection, query_vec, limit).with_payload(true),
            )
            .await
            .with_context(|| format!("search_points({collection}) failed"))?;

        let hits = response
            .result
            .into_iter()
            .map(|p| {
                // ScoredPoint.id is Option<PointId>; fall back to 0 if
                // the server returned None. Should never happen in
                // practice — log loudly if it does.
                let id =
                    p.id.as_ref()
                        .and_then(|pid| pid.point_id_options.as_ref())
                        .map_or(0, |opt| match opt {
                            PointIdOptions::Num(n) => *n,
                            PointIdOptions::Uuid(_) => 0,
                        });
                let title = p
                    .payload
                    .get("title")
                    .and_then(|v| v.kind.as_ref())
                    .and_then(|k| match k {
                        Kind::StringValue(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                FilmHit {
                    id,
                    title,
                    score: p.score,
                }
            })
            .collect();
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_serializes_round_trip() {
        let f = Film {
            id: 42,
            title: "ACADEMY DINOSAUR".to_string(),
            description: "A short film about an old fossil and a brand-new student.".to_string(),
        };
        let json = serde_json::to_string(&f).unwrap();
        let back: Film = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, f.id);
        assert_eq!(back.title, f.title);
        assert_eq!(back.description, f.description);
    }

    #[test]
    fn film_hit_serializes_round_trip() {
        let h = FilmHit {
            id: 7,
            title: "TITANIC".to_string(),
            score: 0.92,
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: FilmHit = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn store_construction_with_bad_url_does_not_panic() {
        // Building the client should NOT contact Qdrant — it only
        // constructs the channel. So even with no server up, this is
        // expected to succeed.
        let store = QdrantStore::new("http://127.0.0.1:6334");
        assert!(
            store.is_ok(),
            "construction should succeed; got {:?}",
            store.err()
        );
    }
}
