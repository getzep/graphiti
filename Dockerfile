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

# Install server dependencies
# The lockfile has an old graphiti-core version, so we install it separately with pip
ARG INSTALL_FALKORDB=true
ARG GRAPHITI_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    # Install dependencies without graphiti-core first
    uv sync --no-dev --no-install-package graph-service && \
    # Force remove old graphiti-core if installed from lockfile
    uv pip uninstall -y graphiti-core 2>/dev/null || true && \
    # Install graphiti-core with falkordb support using uv pip (to install in venv)
    # Use --force-reinstall to ensure we get the latest version
    if [ "$INSTALL_FALKORDB" = "true" ]; then \
        if [ -n "$GRAPHITI_VERSION" ]; then \
            uv pip install --force-reinstall --no-deps "graphiti-core[falkordb]==$GRAPHITI_VERSION"; \
        else \
            uv pip install --force-reinstall --no-deps "graphiti-core[falkordb]"; \
        fi; \
    else \
        if [ -n "$GRAPHITI_VERSION" ]; then \
            uv pip install --force-reinstall --no-deps "graphiti-core==$GRAPHITI_VERSION"; \
        else \
            uv pip install --force-reinstall --no-deps "graphiti-core"; \
        fi; \
    fi && \
    # Finally install the graph-service package itself
    uv pip install -e .

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

# Use the venv Python directly to avoid uv run reinstalling from lockfile
CMD [".venv/bin/python", "-m", "uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
