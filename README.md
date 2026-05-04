<p align="center">
  <img src="assets/hero.svg" alt="rust-vector-databases — Sakila → embedded vectors → in-process aprender or server Qdrant" width="1280" />
</p>

[![CI](https://github.com/paiml/rust-vector-databases/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/rust-vector-databases/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.95-orange.svg)](rust-toolchain.toml)
[![pmat comply](https://img.shields.io/badge/pmat%20comply-COMPLIANT-brightgreen.svg)](Makefile)
[![pv lint](https://img.shields.io/badge/pv%20lint-PASS-brightgreen.svg)](contracts/)

# rust-vector-databases

Companion repo for the Coursera course **Vector Databases with Rust** (`c13` of the
[Rust for Data Engineering](https://www.coursera.org/) specialization).

Two end-to-end vector-search demos against the same 50-row Sakila film fixture:

* an **in-process** pipeline built on `aprender-rag` — no Docker, no model download,
  same hits every run.
* a **server-backed** pipeline against Qdrant 1.x via `qdrant-client` — `make up`
  brings the database online, the demo upserts, queries, and asserts the same
  invariants.

The Sakila fixture was chosen on purpose: the immediate prior course in the
specialization (`mysql-from-zero`) loads the same `film` table into MySQL. Same
data, two paradigms — relational then vector.

## Quick start

```bash
git clone https://github.com/paiml/rust-vector-databases
cd rust-vector-databases
cargo test --workspace          # unit tests (no Docker required)
make seed                       # in-process aprender demo (no Docker)
make up && make qdrant-demo     # server-backed Qdrant demo
```

Run `make help` for every target. `make verify` runs `cargo fmt --check`,
`cargo clippy -D warnings`, the workspace test suite, and `pv lint contracts/`
in one shot — match what `gate` runs in CI.

## Two backends

### Backend 1 — in-process via `aprender-rag` (no Docker)

```bash
cargo run -p vec-cli --example aprender_film_search
```

Boots a [`trueno_rag::RagPipeline`](https://crates.io/crates/aprender-rag) with
`RecursiveChunker(512, 50)` + `MockEmbedder(384)` + `NoOpReranker` + `RRF{k=60}`,
indexes every film, runs the query `"a romantic comedy"`, prints the top-5 JSON,
asserts four runtime contracts:

* `hits.len() == 5` — top-k honoured
* `hits[0].score >= 0.0` — non-negative top score
* every `id > 0` — Sakila AUTO_INCREMENT positive
* pairwise descending sort

The pipeline is fully deterministic — same input → same output across runs,
processes, and machines. CI runs this example on every gate-matrix entry
(stable + 1.95.0).

### Backend 2 — server-backed via Qdrant + `qdrant-client`

```bash
make up                                                       # docker compose up -d (Qdrant 1.x)
cargo run -p vec-cli --example qdrant_film_search             # upserts, queries, asserts contracts
```

Same fixture, same query, same four contracts. Connects on `localhost:6334`
(gRPC), creates the `sakila_films` collection (cosine, 384d), upserts every
film as a `PointStruct` with a `{title, description, id}` payload, then issues
a `SearchPointsBuilder::new(...).with_payload(true)` for the top 5 hits.

The example exits 0 (without contacting the server) when env var
`VEC_SKIP_QDRANT=1` is set or Qdrant is unreachable, so CI without `make up`
still passes.

## Workspace layout

```
crates/
├── vec-core/          distance metrics + shared error types
├── vec-embed/         Embedder trait + deterministic HashEmbedder
├── vec-search/        RRF fusion + filtered-query builder
├── vec-qdrant/        Qdrant wrapper (Film, FilmHit, QdrantStore)
├── vec-aprender/      in-process FilmRagPipeline (wraps aprender-rag)
└── vec-cli/           clap binary + 2 end-to-end examples
data/films.json        50 Sakila film rows the demos embed
contracts/             provable-contract YAML (linted by `pv lint`)
compose.yml            local Qdrant 1.x service
Makefile               same gates CI runs
```

Dependency tree:

```
vec-cli ─┬─► vec-aprender ─► vec-qdrant ─► vec-embed ─► vec-core
         ├─► vec-qdrant
         ├─► vec-embed
         └─► vec-search   (peer — used by hybrid examples)
```

## Provable contracts

Every demo binary asserts runtime invariants — see
[`contracts/vec-rust-v1.yaml`](contracts/vec-rust-v1.yaml) for the formal spec
(`pv lint contracts/` validates the schema and obligations). The contract
file holds 8 equations (4 per backend) that map 1:1 to `assert!` calls in:

* `crates/vec-cli/examples/aprender_film_search.rs`
* `crates/vec-cli/examples/qdrant_film_search.rs`

The contracts hold for *any* corpus that satisfies the preconditions
(non-empty fixture, ≥ top-k rows, positive `film.id` values). The committed
`data/films.json` is one such fixture; swap in `data/films.local.json` (added
to `.gitignore`) to test against a different corpus without disturbing the
default.

## Course outline

The 5-module / 20-lesson plan that this repo backs lives in
[`paiml/course-studio:config/rde_c13_vector_databases.lua`](https://github.com/paiml/course-studio/blob/main/config/rde_c13_vector_databases.lua):

* **M1** — Vector Search Foundations (cosine vs dot vs L2, HNSW, candle quickstart)
* **M2** — Embeddings in Rust (sentence transformers via fastembed / candle)
* **M3** — Qdrant — Collections, Points, Search (`qdrant-client` walkthrough)
* **M4** — Hybrid, Filtered, and Multi-Tenant Search (BM25 + dense, RRF)
* **M5** — Operations, Scaling, and Cost (snapshots, quantization, recall@k)

The capstone — `5.1.4-shipping-vector-pipeline` — is what the two demos in
this repo embody.

## License

Dual-licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
