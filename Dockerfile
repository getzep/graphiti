# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

# Inherit build arguments for labels
ARG GRAPHITI_VERSION
ARG BUILD_DATE
ARG VCS_REF

# OCI image annotations
LABEL org.opencontainers.image.title="Graphiti FastAPI Server"
LABEL org.opencontainers.image.description="FastAPI server for Graphiti temporal knowledge graphs"
LABEL org.opencontainers.image.version="${GRAPHITI_VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Zep AI"
LABEL org.opencontainers.image.source="https://github.com/getzep/graphiti"
LABEL org.opencontainers.image.documentation="https://github.com/getzep/graphiti/tree/main/server"
LABEL io.graphiti.core.version="${GRAPHITI_VERSION}"

# Install uv using the installer script
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Configure uv for runtime
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Set up the server application first
WORKDIR /app
COPY ./server/pyproject.toml ./server/README.md ./server/uv.lock ./
COPY ./server/graph_service ./graph_service

# Copy the local graphiti-core source so we can install from it
COPY ./pyproject.toml ./README.md /graphiti-core/
COPY ./graphiti_core /graphiti-core/graphiti_core

# Then install graphiti-core from local source with the appropriate extras.
ARG INSTALL_FALKORDB=false
ARG INSTALL_NEPTUNE=false
ARG INSTALL_KUZU=false
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    EXTRA="" && \
    if [ "$INSTALL_FALKORDB" = "true" ]; then EXTRA="[falkordb]"; \
    elif [ "$INSTALL_NEPTUNE" = "true" ]; then EXTRA="[neptune]"; \
    elif [ "$INSTALL_KUZU" = "true" ]; then EXTRA="[kuzu]"; \
    fi && \
    uv pip install --reinstall --no-cache "/graphiti-core${EXTRA}"

# Change ownership to app user
RUN chown -R app:app /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER app

# Set port
ENV PORT=8000
EXPOSE $PORT

# Use uv run with --no-sync to avoid re-syncing on startup
CMD ["uv", "run", "--no-sync", "uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
