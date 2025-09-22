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

# Install dependencies first (better layer caching)
ARG INSTALL_FALKORDB=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_FALKORDB" = "true" ]; then \
        uv sync --frozen --no-dev --extra falkordb; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Copy application code
COPY graphiti_mcp_server.py ./

# Change ownership to app user
RUN chown -Rv app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE $PORT

# Command to run the application
CMD ["uv", "run", "graphiti_mcp_server.py"]
