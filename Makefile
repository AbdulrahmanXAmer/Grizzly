# Grizzly — end-to-end tasks.
#
# Everything here runs either on the host (if you have a Rust toolchain and a
# virtualenv) or inside the container (`make docker-*`), which needs only
# Docker. The container targets are the ones to use for benchmarks: they fix
# the software half of the environment — same Python, same compiler, same
# library versions, same core count — which the host cannot promise.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python
IMAGE ?= grizzly:local
DATA_DIR ?= data

# CPU and memory are pinned so a benchmark run is comparable across machines.
# Adjust for your host, but change them deliberately: they are part of the
# measurement, and results.json records what was used.
BENCH_CPUS ?= 4
BENCH_MEMORY ?= 4g

DOCKER_RUN = docker run --rm -v "$(PWD)/$(DATA_DIR):/app/$(DATA_DIR)"
DOCKER_RUN_PINNED = $(DOCKER_RUN) --cpus="$(BENCH_CPUS)" --memory="$(BENCH_MEMORY)"

.PHONY: help
help:  ## Show this help
	@echo "Grizzly — available targets"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Container targets need only Docker. Benchmarks are pinned to"
	@echo "$(BENCH_CPUS) CPUs / $(BENCH_MEMORY) memory; override with BENCH_CPUS= BENCH_MEMORY=."

# ---------------------------------------------------------------------------
# host targets
# ---------------------------------------------------------------------------

.PHONY: build
build:  ## Build the native extension in release mode
	maturin develop --release
	@$(PYTHON) -c "import grizzly, sys; sys.exit(0 if grizzly.is_native() else 'native extension did not load')"
	@echo "native extension OK"

.PHONY: test
test:  ## Run the Python test suite
	$(PYTHON) -m pytest tests -q

.PHONY: test-all
test-all: test  ## Run every test layer, including Rust and the panic hook
	cargo test
	maturin develop --release --features testing
	$(PYTHON) -m pytest tests/test_panic_propagation.py -q
	maturin develop --release

.PHONY: lint
lint:  ## Run every lint gate CI enforces
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy src/grizzly benches

.PHONY: fuzz
fuzz:  ## Fuzz the CSV parser for 60s per target (needs nightly + cargo-fuzz)
	cargo +nightly fuzz run parse_csv fuzz/corpus/parse_csv fuzz/seeds/parse_csv -- -max_total_time=60
	cargo +nightly fuzz run chunk_alignment fuzz/corpus/chunk_alignment fuzz/seeds/chunk_alignment -- -max_total_time=60

.PHONY: demo
demo:  ## Run the end-to-end pipeline on real NYC taxi data
	$(PYTHON) -m demo.pipeline --data-dir $(DATA_DIR)/demo

.PHONY: demo-offline
demo-offline:  ## Run the demo without downloading anything
	$(PYTHON) -m demo.pipeline --data-dir $(DATA_DIR)/demo --offline --rows 300000

.PHONY: bench
bench:  ## Run the benchmark suite (host — noisy, see docker-bench)
	$(PYTHON) -m benches.bench --strict

.PHONY: study
study:  ## Run the sampling accuracy-vs-speed study
	$(PYTHON) -m benches.study_sampling

.PHONY: render
render:  ## Regenerate the README's measured sections
	$(PYTHON) -m benches.render --write

.PHONY: check-render
check-render:  ## Fail if the README has drifted from the measured results
	$(PYTHON) -m benches.render --check

# ---------------------------------------------------------------------------
# container targets
# ---------------------------------------------------------------------------

# The commit is passed in because .dockerignore excludes .git; without it,
# results taken in the container would record no provenance.
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)

.PHONY: docker-build
docker-build:  ## Build the container image
	docker build --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE) .

.PHONY: docker-demo
docker-demo: docker-build  ## Run the demo in the container
	@mkdir -p $(DATA_DIR)
	$(DOCKER_RUN) $(IMAGE) make demo

.PHONY: docker-demo-offline
docker-demo-offline: docker-build  ## Run the demo in the container, no network
	@mkdir -p $(DATA_DIR)
	$(DOCKER_RUN) --network none $(IMAGE) make demo-offline

.PHONY: docker-test
docker-test: docker-build  ## Run the test suite in the container
	$(DOCKER_RUN) $(IMAGE) make test

# The reason the container exists. Pinned CPU and memory plus a fixed software
# stack removes the environment as a variable, which a laptop cannot do.
.PHONY: docker-bench
docker-bench: docker-build  ## Run benchmarks in the container (stable comparison)
	@mkdir -p $(DATA_DIR)
	$(DOCKER_RUN_PINNED) $(IMAGE) \
	  python -m benches.bench --strict --out $(DATA_DIR)/container-results.json
	@echo
	@echo "Results written to $(DATA_DIR)/container-results.json"
	@echo "To publish them: cp $(DATA_DIR)/container-results.json benches/results/results.json && make render"

.PHONY: docker-study
docker-study: docker-build  ## Run the sampling study in the container
	@mkdir -p $(DATA_DIR)
	$(DOCKER_RUN_PINNED) $(IMAGE) \
	  python -m benches.study_sampling --out $(DATA_DIR)/container-sampling-study.json

.PHONY: docker-shell
docker-shell: docker-build  ## Open a shell in the container
	$(DOCKER_RUN) -it $(IMAGE) bash

.PHONY: all
all: build lint test demo  ## Build, lint, test, and run the demo

.PHONY: clean
clean:  ## Remove build artefacts and generated data
	rm -rf target/ $(DATA_DIR)/demo $(DATA_DIR)/*.csv
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.so' -not -path './target/*' -delete
