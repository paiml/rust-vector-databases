//! `vec-cli` — clap binary that wraps the two example demos so the
//! commands ship as one tool.
//!
//! ```text
//! vec-cli aprender ingest                       # build in-process index
//! vec-cli aprender search "a romantic comedy"   # query the in-process index
//! vec-cli qdrant   ingest                       # upsert into Qdrant
//! vec-cli qdrant   search "a romantic comedy"   # query Qdrant
//! ```
//!
//! All four subcommands share `--corpus`, `--collection`, and (for
//! `qdrant`) `--url`. Defaults match the values in `vec_cli::*`
//! constants.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use vec_aprender::FilmRagPipeline;
use vec_cli::{
    default_corpus_path, load_corpus, DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, DEFAULT_QUERY,
    DEFAULT_TOP_K,
};
use vec_embed::{Embedder, HashEmbedder, DEFAULT_DIM};
use vec_qdrant::QdrantStore;

#[derive(Debug, Parser)]
#[command(name = "vec-cli", about = "c13 vector demo CLI", version)]
struct Cli {
    /// Path to the JSON film corpus. Defaults to `data/films.json`
    /// at the workspace root.
    #[arg(long, global = true)]
    corpus: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// In-process aprender-rag pipeline (no Docker, no model download).
    Aprender {
        #[command(subcommand)]
        action: Action,
    },
    /// Server-backed Qdrant pipeline (requires `make up`).
    Qdrant {
        #[command(subcommand)]
        action: Action,
        /// Qdrant gRPC endpoint. Defaults to the compose URL.
        #[arg(long, default_value = DEFAULT_QDRANT_URL, global = true)]
        url: String,
        /// Collection name. Defaults to `sakila_films`.
        #[arg(long, default_value = DEFAULT_COLLECTION, global = true)]
        collection: String,
    },
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Index every film. Idempotent.
    Ingest,
    /// Run a search; defaults to `"a romantic comedy"`.
    Search {
        /// Query text.
        #[arg(default_value = DEFAULT_QUERY)]
        query: String,
        /// Number of hits to return.
        #[arg(long, default_value_t = DEFAULT_TOP_K)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let corpus_path = cli.corpus.clone().unwrap_or_else(default_corpus_path);

    match cli.command {
        Command::Aprender { action } => run_aprender(action, &corpus_path),
        Command::Qdrant {
            action,
            url,
            collection,
        } => run_qdrant(action, &corpus_path, &url, &collection).await,
    }
}

fn run_aprender(action: Action, corpus_path: &std::path::Path) -> Result<()> {
    let films = load_corpus(corpus_path)?;
    let mut pipeline = FilmRagPipeline::new()?;
    pipeline
        .index_films(&films)
        .context("indexing aprender films failed")?;

    match action {
        Action::Ingest => {
            println!(
                "indexed {} films into in-process aprender pipeline",
                pipeline.film_count()
            );
            Ok(())
        }
        Action::Search { query, top_k } => {
            let hits = pipeline.search(&query, top_k)?;
            print_hits(&query, &hits);
            Ok(())
        }
    }
}

async fn run_qdrant(
    action: Action,
    corpus_path: &std::path::Path,
    url: &str,
    collection: &str,
) -> Result<()> {
    let films = load_corpus(corpus_path)?;
    let store = QdrantStore::new(url)?;
    let embedder = HashEmbedder::new(DEFAULT_DIM);
    store
        .ensure_collection(collection, DEFAULT_DIM as u64)
        .await
        .context("ensure_collection failed; is Qdrant running?")?;
    store
        .upsert_films(collection, &films, &embedder)
        .await
        .context("upsert_films failed")?;

    match action {
        Action::Ingest => {
            println!(
                "upserted {} films into Qdrant collection {collection}",
                films.len()
            );
            Ok(())
        }
        Action::Search { query, top_k } => {
            let qvec = embedder.embed(&query)?;
            let hits = store.search(collection, qvec, top_k as u64).await?;
            print_hits(&query, &hits);
            Ok(())
        }
    }
}

fn print_hits(query: &str, hits: &[vec_qdrant::FilmHit]) {
    println!("query: {query}");
    println!("top-{} hits:", hits.len());
    let json = serde_json::to_string_pretty(&hits)
        .unwrap_or_else(|_| "[serialization failed]".to_string());
    println!("{json}");
}
