# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   builder — installs uv, syncs deps with bench extras, no test/docs deps.
#   runtime — minimal, runs as non-root, healthcheck-friendly.
#
# Notes:
#   * libgomp1 is needed at runtime for LightGBM's OpenMP linkage.
#   * We do NOT bake the data files into the image — they live on a
#     mounted volume so the same image runs across machines/datasets.

# --- builder ---------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy

# CatBoost wheel needs libgomp1 at install time on slim images, and the
# build stage uses libgomp1 too for any from-source wheels (lightgbm).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single static binary).
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

WORKDIR /app

# Copy lock + project metadata first so changes to source don't bust
# the dep cache.
COPY pyproject.toml uv.lock README.md LICENSE ./

# Sync runtime + bench extras only — no dev / docs in the image.
RUN uv sync --frozen --extra bench --no-dev --no-install-project

# Now copy source and install the package itself (editable not needed in image).
COPY src/ ./src/
COPY configs/ ./configs/
RUN uv sync --frozen --extra bench --no-dev

# --- runtime ---------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    UPLIFT_BENCH_CONFIGS=/app/configs

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /app /app
RUN chown -R app:app /app

USER app

# Default entry — show help so `docker run image` is informative.
ENTRYPOINT ["uplift-bench"]
CMD ["--help"]
