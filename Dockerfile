# Reproducible environment for the demo, the benchmarks, and the studies.
#
# Two reasons this exists.
#
# First, the demo needs to be runnable by someone who has neither a Rust
# toolchain nor a Python environment set up: `make demo` should work from a
# clean checkout.
#
# Second, and more usefully, benchmark numbers taken on a developer laptop are
# noisy -- background work, thermal throttling, and whatever else the machine is
# doing all land in the measurement. Running the suite inside a pinned image
# with an explicit CPU and memory allocation does not make the host quiet, but
# it does fix the software half of the environment: same Python, same compiler,
# same library versions, same core count, every time and on every machine.

FROM python:3.12-slim-bookworm AS builder

# Pinned so the image is reproducible; bump deliberately, not incidentally.
ARG RUST_VERSION=1.90.0

# .dockerignore excludes .git to keep the build context small, so the commit is
# passed in instead. Without it, benchmark results taken in the container would
# record their provenance as "unknown", which defeats the point of pinning the
# environment in the first place.
ARG GIT_COMMIT=unknown
ENV GRIZZLY_COMMIT=$GIT_COMMIT

# maturin develop requires an active virtualenv, so the image gets one and puts
# it first on PATH. That also keeps the build isolated from the base image's
# system packages.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cargo/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
    && rustc --version

RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip

WORKDIR /app

# Dependency layer first, so source edits do not invalidate the wheel cache.
COPY benches/requirements.txt benches/requirements.txt
RUN python -m pip install -r benches/requirements.txt certifi

# Cargo manifests before sources, so dependency compilation is cached across
# source-only changes.
COPY Cargo.toml Cargo.lock ./
COPY pyproject.toml README.md ./
RUN mkdir -p src && echo "// placeholder for dependency caching" > src/lib.rs \
    && cargo fetch

COPY src/ src/
COPY tests/ tests/
COPY benches/ benches/
COPY demo/ demo/
COPY Makefile ./

# --release matters: the debug build is several times slower and would make any
# benchmark taken in this image meaningless.
RUN maturin develop --release \
    && python -c "import grizzly, sys; sys.exit(0 if grizzly.is_native() else 'native extension did not load')"

# Default to the demo. Override with `docker run ... make bench` etc.
CMD ["make", "demo"]
