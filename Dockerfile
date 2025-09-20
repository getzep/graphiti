FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install uv

# Copy the entire project
COPY . .

# Install dependencies for the MCP server
WORKDIR /app/mcp_server
RUN uv sync

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app:/app/mcp_server

# Expose port for Railway
EXPOSE 8080

# Run the MCP server
CMD ["uv", "run", "graphiti_mcp_server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8080"]