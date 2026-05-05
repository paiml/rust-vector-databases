//! In-process film-search demo (Coursera Rust for Data Engineering c13, M5).
//!
//! Closes the loop the M5 capstone (5.1.4-shipping-vector-pipeline)
//! opens — load Sakila films, embed every description with a
//! deterministic mock embedder, run a query through the
//! `aprender-rag` pipeline, return top-5 hits as JSON.
//!
//! No Docker. No model download. Same hits every run.
//!
//! Run:  cargo run -p vec-cli --example aprender_film_search

use anyhow::Result;
use vec_aprender::FilmRagPipeline;
use vec_cli::{default_corpus_path, load_corpus, DEFAULT_QUERY, DEFAULT_TOP_K};

fn main() -> Result<()> {
    let corpus = load_corpus(&default_corpus_path())?;
    let mut pipeline = FilmRagPipeline::new()?;
    pipeline.index_films(&corpus)?;

    let hits = pipeline.search(DEFAULT_QUERY, DEFAULT_TOP_K)?;

    println!("query: {DEFAULT_QUERY}");
    let json = serde_json::to_string_pretty(&hits)?;
    println!("{json}");

    // ---------------------------------------------------------------------
    // Provable contracts — runtime invariants the demo MUST hold.
    //   * row count exactly DEFAULT_TOP_K
    //   * top score is non-negative (RRF returns non-negative fused scores)
    //   * every Sakila id is positive (AUTO_INCREMENT starts at 1)
    //   * results are sorted by score descending
    // ---------------------------------------------------------------------
    assert!(
        hits.len() == DEFAULT_TOP_K,
        "Aprender contract: expected exactly {} hits, got {}. \
         Either the pipeline returned fewer results than requested or \
         the corpus is too small.",
        DEFAULT_TOP_K,
        hits.len(),
    );
    let top = hits
        .first()
        .expect("hits.len() == DEFAULT_TOP_K was just asserted");
    assert!(
        top.score >= 0.0,
        "Aprender contract: top hit score must be non-negative, got {}",
        top.score
    );
    for h in &hits {
        assert!(
            h.id > 0,
            "Aprender contract: every film id must be positive (Sakila AUTO_INCREMENT). \
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
        "Aprender contract: hits must be sorted by score descending. Got {scores:?}"
    );

    Ok(())
}
