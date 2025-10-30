# syntax=docker/dockerfile:1.9
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the installer script
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

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# Copy project files for dependency installation (better caching)
COPY pyproject.toml uv.lock ./

# Remove the local path override for graphiti-core in Docker builds
# This allows uv to use the published package from PyPI
RUN sed -i '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml

# Install dependencies first (better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Extract graphiti-core version and store it in a file for runtime access
RUN uv pip freeze | grep graphiti-core | cut -d'=' -f3 > /app/.graphiti-core-version

# Copy application code and configuration
COPY main.py ./
COPY src/ ./src/
COPY config/ ./config/

# Set execute permissions on main.py and change ownership to app user
RUN chmod +x /app/main.py && chown -Rv app:app /app

# Add Docker labels with version information
ARG MCP_SERVER_VERSION=1.0.0rc0
ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.title="Graphiti MCP Server" \
      org.opencontainers.image.description="MCP server for Graphiti knowledge graph" \
      org.opencontainers.image.version="${MCP_SERVER_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Zep AI" \
      org.opencontainers.image.source="https://github.com/zep-ai/graphiti" \
      graphiti.core.version="$(cat /app/.graphiti-core-version)"

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "main.py"]
