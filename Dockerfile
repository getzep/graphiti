# syntax=docker/dockerfile:1.9
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the installer script
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Configure uv for optimal Docker usage
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy and build main graphiti-core project
COPY ./pyproject.toml ./README.md ./
COPY ./graphiti_core ./graphiti_core

# Build graphiti-core wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build

# Install the built wheel to make it available for server
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install dist/*.whl

# Runtime stage - build the server here
FROM python:3.12-slim

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

# Copy graphiti-core wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install graphiti-core wheel first
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /tmp/*.whl

# Set up the server application
WORKDIR /app
COPY ./server/pyproject.toml ./server/README.md ./server/uv.lock ./
COPY ./server/graph_service ./graph_service

# Install server dependencies and application
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

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
