#!/bin/bash
# Script to build Docker image with proper version tagging
# Reads the graphiti-core version from the monorepo instead of PyPI.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Extract the top-level version from a pyproject.toml (stdlib only, any Python 3)
_pyproject_version() {
  python3 -c "import re,sys; m=re.search(r'^version\s*=\s*\"([^\"]+)\"',open(sys.argv[1]).read(),re.M); print(m.group(1) if m else '')" "$1"
}

MCP_VERSION=$(_pyproject_version "$SCRIPT_DIR/../pyproject.toml")
if [ -z "$MCP_VERSION" ]; then
  echo "Error: failed to parse MCP server version from mcp_server/pyproject.toml" >&2
  exit 1
fi

GRAPHITI_CORE_VERSION=$(_pyproject_version "$REPO_ROOT/pyproject.toml")
if [ -z "$GRAPHITI_CORE_VERSION" ]; then
  echo "Error: failed to parse graphiti-core version from pyproject.toml" >&2
  exit 1
fi
echo "Graphiti Core version (monorepo): ${GRAPHITI_CORE_VERSION}"

# Get build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build from repo root so the Dockerfile can COPY graphiti_core/
echo "Building Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${MCP_VERSION}" \
  -f "$SCRIPT_DIR/Dockerfile" \
  -t "zepai/graphiti-mcp:${MCP_VERSION}" \
  -t "zepai/graphiti-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}" \
  -t "zepai/graphiti-mcp:latest" \
  "$REPO_ROOT"

echo ""
echo "Build complete!"
echo "  MCP Server Version: ${MCP_VERSION}"
echo "  Graphiti Core Version: ${GRAPHITI_CORE_VERSION}"
echo "  Build Date: ${BUILD_DATE}"
echo ""
echo "Image tags:"
echo "  - zepai/graphiti-mcp:${MCP_VERSION}"
echo "  - zepai/graphiti-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}"
echo "  - zepai/graphiti-mcp:latest"
echo ""
echo "To inspect image metadata:"
echo "  docker inspect zepai/graphiti-mcp:${MCP_VERSION} | jq '.[0].Config.Labels'"
