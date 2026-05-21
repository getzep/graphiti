#!/bin/bash
set -e

echo "🚀 Setting up Graphiti workspace..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10 or higher is required (found: $python_version)"
    exit 1
fi

echo "✓ Prerequisites check passed"

# Copy necessary files from root repo
echo "📄 Copying project files from root repo..."
cp "$CONDUCTOR_ROOT_PATH/pyproject.toml" .
cp "$CONDUCTOR_ROOT_PATH/uv.lock" .
cp "$CONDUCTOR_ROOT_PATH/README.md" .
if [ -f "$CONDUCTOR_ROOT_PATH/pytest.ini" ]; then
    cp "$CONDUCTOR_ROOT_PATH/pytest.ini" .
fi
if [ -f "$CONDUCTOR_ROOT_PATH/conftest.py" ]; then
    cp "$CONDUCTOR_ROOT_PATH/conftest.py" .
fi
if [ -f "$CONDUCTOR_ROOT_PATH/py.typed" ]; then
    cp "$CONDUCTOR_ROOT_PATH/py.typed" .
fi

# Create symlink to source code instead of copying
echo "🔗 Creating symlinks to source code..."
ln -sf "$CONDUCTOR_ROOT_PATH/graphiti_core" graphiti_core
ln -sf "$CONDUCTOR_ROOT_PATH/tests" tests
ln -sf "$CONDUCTOR_ROOT_PATH/examples" examples

# Install dependencies
echo "📦 Installing dependencies with uv..."
uv sync --frozen --extra dev

# Create workspace-specific Makefile
echo "📝 Creating workspace Makefile..."
cat > Makefile << 'EOF'
.PHONY: install format lint test all check

# Define variables - using virtualenv directly instead of uv run
PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest
RUFF = .venv/bin/ruff
PYRIGHT = .venv/bin/pyright

# Default target
all: format lint test

# Install dependencies
install:
	@echo "Dependencies already installed via conductor-setup.sh"
	@echo "Run './conductor-setup.sh' to reinstall"

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) ./graphiti_core

# Run tests
test:
	DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 $(PYTEST) -m "not integration"

# Run format, lint, and test
check: format lint test
EOF

# Handle environment variables
if [ -f "$CONDUCTOR_ROOT_PATH/.env" ]; then
    echo "🔗 Linking .env file from root repo..."
    ln -sf "$CONDUCTOR_ROOT_PATH/.env" .env
    echo "✓ Environment file linked"
else
    echo "⚠️  No .env file found in root repo"
    echo "   Copy $CONDUCTOR_ROOT_PATH/.env.example to $CONDUCTOR_ROOT_PATH/.env"
    echo "   and add your API keys, then rerun setup"
    exit 1
fi

# Check for required environment variable
if ! grep -q "OPENAI_API_KEY=.*[^[:space:]]" .env 2>/dev/null; then
    echo "⚠️  Warning: OPENAI_API_KEY not set in .env file"
    echo "   This is required for most Graphiti functionality"
fi

echo "✅ Workspace setup complete!"
echo ""
echo "Available commands:"
echo "  make test    - Run unit tests"
echo "  make lint    - Lint and type check code"
echo "  make format  - Format code with ruff"
echo "  make check   - Run all checks (format, lint, test)"
