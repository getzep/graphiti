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

# Install Socket Firewall (sfw) for dependency fetches when SOCKET_API_KEY is
# provided as a BuildKit secret. Community builds without the secret fall back
# to direct installs (see the RUN --mount below).
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) SFW_ARCH=x86_64 ;; \
      arm64) SFW_ARCH=arm64 ;; \
      *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 --proto '=https' --tlsv1.2 \
      "https://github.com/SocketDev/firewall-release/releases/latest/download/sfw-linux-${SFW_ARCH}" \
      -o /usr/local/bin/sfw; \
    chmod +x /usr/local/bin/sfw

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
# When socket_api_key is mounted (official releases), route through sfw;
# otherwise install directly so public docker builds keep working.
#
# sfw can only inspect packages that are actually downloaded, so an enforced
# build must defeat both caches or a newly flagged package would ship unscanned:
# SOCKET_SCAN_ID (unique per release run) invalidates this layer, and
# UV_NO_CACHE stops uv serving wheels from the cache mount below.
ARG INSTALL_FALKORDB=false
ARG SOCKET_FIREWALL_ENABLED=false
ARG SOCKET_SCAN_ID=
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=socket_api_key \
    set -eu; \
    export SFW_TELEMETRY_DISABLED=true; \
    export SFW_CUSTOM_REGISTRIES="wrap:storage.googleapis.com,wrap:classic.yarnpkg.com,wrap:files.pythonhosted.org"; \
    if [ "$SOCKET_FIREWALL_ENABLED" = "true" ]; then \
      if [ ! -s /run/secrets/socket_api_key ]; then \
        echo "ERROR: SOCKET_FIREWALL_ENABLED=true but socket_api_key is missing" >&2; \
        exit 1; \
      fi; \
      export SOCKET_API_KEY="$(cat /run/secrets/socket_api_key)"; \
      export UV_NO_CACHE=1; \
      echo "Socket Firewall enforced (scan id: ${SOCKET_SCAN_ID})"; \
      UV_CMD="sfw uv"; \
    else \
      echo "socket_api_key absent; installing without Socket Firewall"; \
      UV_CMD="uv"; \
    fi; \
    $UV_CMD sync --frozen --no-dev; \
    if [ -n "$GRAPHITI_VERSION" ]; then \
        if [ "$INSTALL_FALKORDB" = "true" ]; then \
            $UV_CMD pip install --upgrade "graphiti-core[falkordb]==$GRAPHITI_VERSION"; \
        else \
            $UV_CMD pip install --upgrade "graphiti-core==$GRAPHITI_VERSION"; \
        fi; \
    else \
        if [ "$INSTALL_FALKORDB" = "true" ]; then \
            $UV_CMD pip install --upgrade "graphiti-core[falkordb]"; \
        else \
            $UV_CMD pip install --upgrade graphiti-core; \
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

# Use uv run with --no-sync to avoid re-syncing on startup
CMD ["uv", "run", "--no-sync", "uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
