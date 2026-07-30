# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

# The graphiti-core release installed from PyPI below, and the value of the version
# labels. This default is the single source of truth for the pin: render.yaml,
# docker-compose.yml, and local builds all inherit it, so a bump happens here and
# nowhere else. It is a PyPI release number, not this repo's own version.
#
# 0.29.2 in particular does not work on FalkorDB: it writes episodes fine but reads them
# back empty, so /search and /episodes return nothing. Verify a write-then-search round
# trip before bumping. Release CI overrides this with the version it is publishing.
ARG GRAPHITI_VERSION=0.29.3

# Inherit build arguments for labels
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
RUN --mount=type=cache,target=/root/.cache/uv \
    : "${GRAPHITI_VERSION:?must be a graphiti-core release from PyPI}" && \
    uv sync --frozen --no-dev && \
    if [ "$INSTALL_FALKORDB" = "true" ]; then \
        uv pip install --upgrade "graphiti-core[falkordb]==$GRAPHITI_VERSION"; \
    else \
        uv pip install --upgrade "graphiti-core==$GRAPHITI_VERSION"; \
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

# Shell form, so $PORT actually reaches uvicorn: the exec form does no variable expansion,
# which is why this used to hardcode 8000 and ignore the PORT it declares above. `:-8000`
# covers PORT set but empty, which a dashboard or a .env makes easy and which is not the
# same as unset. `exec` replaces the shell, so uvicorn is PID 1 and gets SIGTERM directly
# on deploy or shutdown, rather than the shell absorbing it until the runtime kills us.
#
# uvicorn straight from the venv, not `uv run --no-sync uvicorn`: uv is installed under
# /root/.local/bin, and /root is 0700, so the app user cannot resolve it through PATH. The
# exec form got away with it because the runtime resolves the binary as root before
# dropping privileges; a shell doing its own PATH lookup gets EACCES. PATH already puts
# /app/.venv/bin first, so this is the same interpreter uv run would have selected.
CMD ["sh", "-c", "exec uvicorn graph_service.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
