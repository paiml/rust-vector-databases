SHELL := /bin/bash
.DELETE_ON_ERROR:
.SUFFIXES:
.ONESHELL:

# Local Qdrant container name (matches compose.yml services.qdrant.container_name)
QDRANT_CTR := rust-vector-databases-qdrant
QDRANT_URL := http://localhost:6334

CARGO ?= cargo

.PHONY: help up down nuke wait seed qdrant-demo clean test fmt lint verify

help: ## Print available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' "$(MAKEFILE_LIST)"

up: ## Start the Qdrant container via docker compose
	docker compose up -d

down: ## Stop the Qdrant container (keeps the named volume)
	docker compose down

nuke: ## Stop the container AND delete the named storage volume (full reset)
	docker compose down -v
	@echo "✓ container + rust-vector-databases-qdrant-data volume removed. 'make up' starts fresh."

wait: ## Block until Qdrant reports healthy (compose healthcheck)
	@printf "→ waiting for Qdrant to become healthy"
	@i=0; while [ $$i -lt 60 ]; do \
	  status="$$(docker inspect --format '{{.State.Health.Status}}' "$(QDRANT_CTR)" 2>/dev/null || echo "missing")"; \
	  if [ "$$status" = "healthy" ]; then echo " ✓"; exit 0; fi; \
	  printf "."; sleep 2; \
	  i=$$((i + 1)); \
	done; \
	echo " ✗ (timed out — check 'docker compose logs qdrant')"; exit 1

seed: ## Run the in-process aprender film-search demo (no Docker needed)
	$(CARGO) run -p vec-cli --example aprender_film_search

qdrant-demo: wait ## Run the server-backed Qdrant film-search demo (requires `make up`)
	$(CARGO) run -p vec-cli --example qdrant_film_search

clean: ## Cargo clean
	$(CARGO) clean

test: ## Run the workspace test suite
	$(CARGO) test --workspace

fmt: ## cargo fmt --all
	$(CARGO) fmt --all

lint: ## cargo clippy with -D warnings
	$(CARGO) clippy --workspace --all-targets -- -D warnings

verify: fmt lint test ## fmt + clippy + tests + pv lint contracts/
	$(CARGO) fmt --all -- --check
	pv lint contracts/
