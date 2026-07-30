# syntax=docker/dockerfile:1
# Combined FalkorDB + Graphiti MCP Server Image
# This extends the official FalkorDB image to include the MCP server

FROM falkordb/falkordb:latest AS falkordb-base

# Install Python and system dependencies
# Note: Debian 13 "trixie" (FalkorDB base) ships with Python 3.13.
# procps is omitted: its postinst runs update-rc.d (a perl script), and the
# FalkorDB base image no longer ships perl (purged upstream for CVE reduction).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install Socket Firewall (sfw) for dependency fetches when SOCKET_API_KEY is
# provided as a BuildKit secret. Community builds without the secret fall back
# to direct installs.
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

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Set up MCP server directory
WORKDIR /app/mcp

# Accept graphiti-core version as build argument
ARG GRAPHITI_CORE_VERSION=0.29.1
ARG SOCKET_FIREWALL_ENABLED=false
# Unique per release run so enforced builds re-resolve and re-download instead
# of reusing cached layers, which sfw would never see.
ARG SOCKET_SCAN_ID=

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Remove the local path override for graphiti-core in Docker builds
# and regenerate lock file to match the PyPI version.
# When socket_api_key is mounted (official releases), route through sfw.
RUN --mount=type=secret,id=socket_api_key \
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
    sed -i '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml; \
    if [ -n "${GRAPHITI_CORE_VERSION}" ]; then \
      sed -i "s/graphiti-core\[falkordb\][>=]\+[0-9]\+\.[0-9]\+\.[0-9]\+/graphiti-core[falkordb]==${GRAPHITI_CORE_VERSION}/" pyproject.toml; \
    fi; \
    echo "Regenerating lock file for PyPI graphiti-core..."; \
    rm -f uv.lock; \
    $UV_CMD lock

# Install Python dependencies (exclude dev dependency group)
# Include optional provider extras so the image matches the providers
# advertised by the MCP server configuration and docs.
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
    $UV_CMD sync --no-group dev --extra providers --extra azure

# Store graphiti-core version
RUN echo "${GRAPHITI_CORE_VERSION}" > /app/mcp/.graphiti-core-version

# Copy MCP server application code
COPY main.py ./
COPY src/ ./src/
COPY config/ ./config/

# Copy FalkorDB combined config (uses localhost since both services in same container)
COPY config/config-docker-falkordb-combined.yaml /app/mcp/config/config.yaml

# Create log and data directories
RUN mkdir -p /var/log/graphiti /var/lib/falkordb/data

# Create startup script that runs both services
RUN cat > /start-services.sh <<'EOF'
#!/bin/bash
set -e

# Start FalkorDB in background using the correct module path
echo "Starting FalkorDB..."
redis-server \
  --loadmodule /var/lib/falkordb/bin/falkordb.so \
  --protected-mode no \
  --bind 0.0.0.0 \
  --port 6379 \
  --dir /var/lib/falkordb/data \
  --daemonize yes

# Wait for FalkorDB to be ready
echo "Waiting for FalkorDB to be ready..."
until redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; do
  echo "FalkorDB not ready yet, waiting..."
  sleep 1
done
echo "FalkorDB is ready!"

# Start FalkorDB Browser if enabled (default: enabled)
if [ "${BROWSER:-1}" = "1" ]; then
  if [ -d "/var/lib/falkordb/browser" ] && [ -f "/var/lib/falkordb/browser/server.js" ]; then
    echo "Starting FalkorDB Browser on port 3000..."
    cd /var/lib/falkordb/browser
    HOSTNAME="0.0.0.0" node server.js > /var/log/graphiti/browser.log 2>&1 &
    echo "FalkorDB Browser started in background"
  else
    echo "Warning: FalkorDB Browser files not found, skipping browser startup"
  fi
else
  echo "FalkorDB Browser disabled (BROWSER=${BROWSER})"
fi

# Start MCP server in foreground
echo "Starting MCP server..."
cd /app/mcp
exec /root/.local/bin/uv run --no-sync main.py
EOF

RUN chmod +x /start-services.sh

# Add Docker labels with version information
ARG MCP_SERVER_VERSION=1.0.2
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.title="FalkorDB + Graphiti MCP Server" \
      org.opencontainers.image.description="Combined FalkorDB graph database with Graphiti MCP server" \
      org.opencontainers.image.version="${MCP_SERVER_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Zep AI" \
      org.opencontainers.image.source="https://github.com/zep-ai/graphiti" \
      graphiti.core.version="${GRAPHITI_CORE_VERSION}"

# Expose ports
EXPOSE 6379 3000 8000

# Health check - verify FalkorDB is responding
# MCP server startup is logged and visible in container output
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD redis-cli -p 6379 ping > /dev/null || exit 1

# Override the FalkorDB entrypoint and use our startup script
ENTRYPOINT ["/start-services.sh"]
CMD []
