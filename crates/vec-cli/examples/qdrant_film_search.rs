//! Server-backed Qdrant film-search demo (Coursera Rust for Data Engineering c13, M3+M5).
//!
//! Same end-to-end story as `aprender_film_search` but against a
//! running Qdrant server (`make up` at the workspace root). Embedding
//! is the same deterministic [`vec_embed::HashEmbedder`] used by
//! `vec-aprender` so the two demos can be compared head-to-head
//! without a model download.
//!
//! Skip the network step (and exit 0) when env var
//! `VEC_SKIP_QDRANT=1` is set or Qdrant is unreachable. CI without
//! `make up` therefore still passes when this example is invoked.
//!
//! Run:  make up && cargo run -p vec-cli --example qdrant_film_search

use anyhow::Result;
use vec_cli::{
    default_corpus_path, load_corpus, DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, DEFAULT_QUERY,
    DEFAULT_TOP_K,
};
use vec_embed::{Embedder, HashEmbedder, DEFAULT_DIM};
use vec_qdrant::QdrantStore;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("VEC_SKIP_QDRANT").is_ok() {
        eprintln!("VEC_SKIP_QDRANT is set — exiting 0 without contacting Qdrant.");
        return Ok(());
    }

    let corpus = load_corpus(&default_corpus_path())?;
    let store = QdrantStore::new(DEFAULT_QDRANT_URL)?;
    let embedder = HashEmbedder::new(DEFAULT_DIM);

    // Surface a clear "Qdrant not reachable, skipping" message rather
    // than a stack trace if the server isn't up. CI's `integration`
    // job brings the server up; local `cargo test` doesn't, so this
    // is the path most non-Docker callers will take.
    if let Err(e) = store
        .ensure_collection(DEFAULT_COLLECTION, DEFAULT_DIM as u64)
        .await
    {
        eprintln!("Qdrant unreachable at {DEFAULT_QDRANT_URL} — skipping demo. ({e})");
        return Ok(());
    }

    store
        .upsert_films(DEFAULT_COLLECTION, &corpus, &embedder)
        .await?;

    let qvec = embedder.embed(DEFAULT_QUERY)?;
    let hits = store
        .search(DEFAULT_COLLECTION, qvec, DEFAULT_TOP_K as u64)
        .await?;

    println!("query: {DEFAULT_QUERY}");
    let json = serde_json::to_string_pretty(&hits)?;
    println!("{json}");

    // ---------------------------------------------------------------------
    // Provable contracts — runtime invariants the demo MUST hold.
    //   * row count exactly DEFAULT_TOP_K
    //   * top score is non-negative (cosine in Qdrant is in [-1, 1] but
    //     the demo's normalized hash vectors keep us in [0, 1])
    //   * every Sakila id is positive (AUTO_INCREMENT starts at 1)
    //   * results are sorted by score descending (Qdrant default order)
    // ---------------------------------------------------------------------
    assert!(
        hits.len() == DEFAULT_TOP_K,
        "Qdrant contract: expected exactly {} hits, got {}. \
         Either the upsert dropped rows or the corpus is smaller than top-k.",
        DEFAULT_TOP_K,
        hits.len(),
    );
    let top = hits
        .first()
        .expect("hits.len() == DEFAULT_TOP_K was just asserted");
    assert!(
        top.score >= 0.0,
        "Qdrant contract: top hit score must be non-negative, got {}",
        top.score
    );
    for h in &hits {
        assert!(
            h.id > 0,
            "Qdrant contract: every film id must be positive (Sakila AUTO_INCREMENT). \
             Got id={} for title={:?}",
            h.id,
            h.title
        );
    }
    let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    assert!(
        scores == sorted,
        "Qdrant contract: hits must be sorted by score descending. Got {scores:?}"
    );

    Ok(())
}
