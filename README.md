# rust-vector-databases

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.95-orange.svg)](rust-toolchain.toml)

Reference Rust workspace for course **c13 — Vector Databases with Rust** in the Coursera
[Rust for Data Engineering](https://www.coursera.org/) specialization.

Production vector-search pipelines in Rust — embed with `candle`, index in Qdrant, query with filter-then-search and hybrid retrieval, scale with quantization. Anchored on the official `qdrant-client` Rust SDK.

## Workspace layout

- [`crates/vec-core`](crates/vec-core) — Distance metrics, vector types, ANN benchmarks
- [`crates/vec-embed`](crates/vec-embed) — Sentence and image encoders via candle
- [`crates/vec-qdrant`](crates/vec-qdrant) — qdrant-client wrappers — collections, points, search
- [`crates/vec-search`](crates/vec-search) — Filter-then-search, hybrid (BM25 + dense), rerankers
- [`crates/vec-cli`](crates/vec-cli) — clap binary for ingest, search, and recall benchmarking

## Quick start

```bash
git clone https://github.com/paiml/rust-vector-databases
cd rust-vector-databases
cargo test --workspace
```

## Status

Scaffold. Lessons land as recordings ship. Track companion config at
[`paiml/course-studio`](https://github.com/paiml/course-studio).

## License

Dual-licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
