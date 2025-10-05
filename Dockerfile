# syntax=docker/dockerfile:1.9

# ---------- builder ----------
FROM python:3.12-slim AS builder
WORKDIR /app

# Build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install uv system-wide so any user can run it
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# Docker/uv performance knobs
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Project sources
COPY ./pyproject.toml ./README.md ./
COPY ./graphiti_core ./graphiti_core

# Build wheel (BuildKit cache)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build

# Make wheel available to runtime
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install dist/*.whl


# ---------- runtime ----------
FROM python:3.12-slim

# Minimal runtime deps + uv in system path
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# Docker/uv runtime knobs
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Install core wheel first (system site)
COPY --from=builder /app/dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /tmp/*.whl

# Server app
WORKDIR /app
COPY ./server/pyproject.toml ./server/README.md ./server/uv.lock ./
COPY ./server/graph_service ./graph_service

# Resolve server env (creates /app/.venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Own app dir and prefer venv binaries
RUN chown -R app:app /app
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

USER app
ENV PORT=8000
EXPOSE 8000

# uv is now in /usr/local/bin (on PATH) and accessible to user app
CMD ["uv", "run", "uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
