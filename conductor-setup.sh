#!/bin/bash
set -e

echo "🚀 Setting up Graphiti workspace..."

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv package manager not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Copy environment file if it exists in the root
if [ -f "$CONDUCTOR_ROOT_PATH/.env" ]; then
    echo "📋 Copying environment configuration..."
    cp "$CONDUCTOR_ROOT_PATH/.env" .env
elif [ -f "$CONDUCTOR_ROOT_PATH/.env.example" ]; then
    echo "📋 Copying example environment configuration..."
    cp "$CONDUCTOR_ROOT_PATH/.env.example" .env
    echo "⚠️  Please configure your API keys in .env file"
else
    echo "⚠️  No .env file found. You may need to configure environment variables."
fi

# Install main project dependencies
echo "📦 Installing core dependencies..."
uv sync --extra dev

# Install server dependencies
echo "📦 Installing server dependencies..."
cd server
uv sync --extra dev
cd ..

# Install MCP server dependencies if available
if [ -d "mcp_server" ]; then
    echo "📦 Installing MCP server dependencies..."
    cd mcp_server
    uv sync
    cd ..
fi

# Run initial checks to ensure everything is working
echo "🔍 Running initial checks..."
uv run ruff check --select I --fix
uv run ruff format
echo "✨ Graphiti workspace setup complete!"

# Display helpful information
echo ""
echo "📚 Quick Start Guide:"
echo "• Main project commands: make format, make lint, make test"
echo "• Server commands: cd server && make format, make lint, make test"
echo "• Run server: Click 'Run' button or use 'cd server && uv run uvicorn graph_service.main:app --reload'"
echo "• Configure API keys in .env file (OPENAI_API_KEY required)"
echo ""