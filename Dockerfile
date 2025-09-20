# Ultra-simple Railway-compatible Dockerfile for Graphiti MCP Server
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install graphiti-core from source using standard pip
RUN pip install --no-cache-dir .

# Install MCP server dependencies using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r app && useradd -r -d /app -g app app
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Change to MCP server directory and run
WORKDIR /app/mcp_server
CMD ["python", "graphiti_mcp_server.py", "--transport", "sse"]