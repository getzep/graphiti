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

# Install server dependencies (without graphiti-core from lockfile)
# Then install graphiti-core from PyPI at the desired version
# This prevents the stale lockfile from pinning an old graphiti-core version
ARG INSTALL_FALKORDB=false
ARG INSTALL_GOOGLE_GENAI=false
ARG INSTALL_ANTHROPIC=false
ARG INSTALL_GROQ=false
ARG INSTALL_VOYAGEAI=false
ARG INSTALL_SENTENCE_TRANSFORMERS=false
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    # Build extras list based on enabled providers
    EXTRAS=""; \
    if [ "$INSTALL_FALKORDB" = "true" ]; then EXTRAS="${EXTRAS},falkordb"; fi; \
    if [ "$INSTALL_GOOGLE_GENAI" = "true" ]; then EXTRAS="${EXTRAS},google-genai"; fi; \
    if [ "$INSTALL_ANTHROPIC" = "true" ]; then EXTRAS="${EXTRAS},anthropic"; fi; \
    if [ "$INSTALL_GROQ" = "true" ]; then EXTRAS="${EXTRAS},groq"; fi; \
    if [ "$INSTALL_VOYAGEAI" = "true" ]; then EXTRAS="${EXTRAS},voyageai"; fi; \
    if [ "$INSTALL_SENTENCE_TRANSFORMERS" = "true" ]; then EXTRAS="${EXTRAS},sentence-transformers"; fi; \
    # Remove leading comma if present
    EXTRAS=$(echo "$EXTRAS" | sed 's/^,//'); \
    # Install graphiti-core with selected extras
    if [ -n "$GRAPHITI_VERSION" ]; then \
        if [ -n "$EXTRAS" ]; then \
            echo "Installing graphiti-core==$GRAPHITI_VERSION with extras: $EXTRAS"; \
            uv pip install --system --upgrade "graphiti-core[$EXTRAS]==$GRAPHITI_VERSION"; \
        else \
            uv pip install --system --upgrade "graphiti-core==$GRAPHITI_VERSION"; \
        fi; \
    else \
        if [ -n "$EXTRAS" ]; then \
            echo "Installing graphiti-core with extras: $EXTRAS"; \
            uv pip install --system --upgrade "graphiti-core[$EXTRAS]"; \
        else \
            uv pip install --system --upgrade graphiti-core; \
        fi; \
    fi

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

# Use uv run for execution
CMD ["uv", "run", "uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
