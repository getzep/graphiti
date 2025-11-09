# syntax=docker/dockerfile:1
# Combined FalkorDB + Graphiti MCP Server Image
# This extends the official FalkorDB image to include the MCP server

FROM falkordb/falkordb:latest AS falkordb-base

# Install Python and system dependencies
# Note: Debian Bookworm (FalkorDB base) ships with Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Set up MCP server directory
WORKDIR /app/mcp

# Accept graphiti-core version as build argument
ARG GRAPHITI_CORE_VERSION=0.23.1

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Remove the local path override for graphiti-core in Docker builds
# and regenerate lock file to match the PyPI version
RUN sed -i '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml && \
    if [ -n "${GRAPHITI_CORE_VERSION}" ]; then \
      sed -i "s/graphiti-core\[falkordb\]>=[0-9]\+\.[0-9]\+\.[0-9]\+$/graphiti-core[falkordb]==${GRAPHITI_CORE_VERSION}/" pyproject.toml; \
    fi && \
    echo "Regenerating lock file for PyPI graphiti-core..." && \
    rm -f uv.lock && \
    uv lock

# Install Python dependencies (exclude dev dependency group)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-group dev

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
ARG MCP_SERVER_VERSION=1.0.1
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
