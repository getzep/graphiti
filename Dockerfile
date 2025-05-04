# Build stage – create an isolated venv with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# uv performance toggles
ENV UV_COMPILE_BYTECODE=1  \
    UV_LINK_MODE=copy

# Copy all source code and manifests
COPY pyproject.toml uv.lock README.md ./
COPY graphiti_core ./graphiti_core
COPY server ./server

# Create and activate virtual environment
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv PATH="/app/.venv/bin:$PATH"

# Install graphiti-core (editable)
RUN uv pip install --no-cache --editable .

# Install the FastAPI server package and its specific dependencies
RUN uv pip install --no-cache ./server

# Runtime stage – copy the ready venv into a minimal image
FROM python:3.12-slim
ARG UID=10001
RUN adduser --disabled-password --gecos "" --uid "$UID" appuser
WORKDIR /app

# Copy the virtual‑env and the necessary server code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/server/graph_service /app/graph_service

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser
CMD ["uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
