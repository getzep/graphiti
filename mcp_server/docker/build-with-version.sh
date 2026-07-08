#!/bin/bash
# Script to build Docker image with proper version tagging
# This script queries PyPI for the latest graphiti-core version and includes it in the image tag

set -e

# Get MCP server version from pyproject.toml
MCP_VERSION=$(grep '^version = ' ../pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Get latest graphiti-core version from PyPI
echo "Querying PyPI for latest graphiti-core version..."
GRAPHITI_CORE_VERSION=$(curl -s https://pypi.org/pypi/graphiti-core/json | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")
echo "Latest graphiti-core version: ${GRAPHITI_CORE_VERSION}"

# Get build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build the image with explicit graphiti-core version
echo "Building Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${MCP_VERSION}" \
  -f Dockerfile \
  -t "zepai/graphiti-mcp:${MCP_VERSION}" \
  -t "zepai/graphiti-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}" \
  -t "zepai/graphiti-mcp:latest" \
  ..

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
