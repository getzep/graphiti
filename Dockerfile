# Railway-optimized Dockerfile for Graphiti MCP Server
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the installer script
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Configure uv for optimal Docker usage without cache mounts
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MCP_SERVER_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app

# First, copy and install the core graphiti library
COPY ./pyproject.toml ./README.md ./
COPY ./graphiti_core ./graphiti_core

# Build and install graphiti-core (no cache mount for Railway compatibility)
RUN uv build && \
    pip install dist/*.whl

# Now set up the MCP server
COPY ./mcp_server/pyproject.toml ./mcp_server/uv.lock ./mcp_server/
COPY ./mcp_server/graphiti_mcp_server.py ./

# Install MCP server dependencies (no cache mount for Railway compatibility)
RUN uv sync --frozen --no-dev

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables for Railway
ENV PORT=8000
ENV MCP_SERVER_HOST=0.0.0.0

# Expose port (Railway will override with PORT env var)
EXPOSE $PORT

# Command to run the MCP server with SSE transport
# Railway will set PORT environment variable, host and port are configured via env vars
CMD ["uv", "run", "graphiti_mcp_server.py", "--transport", "sse"]