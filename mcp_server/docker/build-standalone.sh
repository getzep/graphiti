#!/bin/bash
# Script to build and push standalone Docker image with both Neo4j and FalkorDB drivers
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
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build the standalone image with explicit graphiti-core version
echo "Building standalone Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${VCS_REF}" \
  -f Dockerfile.standalone \
  -t "zepai/knowledge-graph-mcp:standalone" \
  -t "zepai/knowledge-graph-mcp:${MCP_VERSION}-standalone" \
  -t "zepai/knowledge-graph-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}-standalone" \
  ..

echo ""
echo "Build complete!"
echo "  MCP Server Version: ${MCP_VERSION}"
echo "  Graphiti Core Version: ${GRAPHITI_CORE_VERSION}"
echo "  Build Date: ${BUILD_DATE}"
echo "  VCS Ref: ${VCS_REF}"
echo ""
echo "Image tags:"
echo "  - zepai/knowledge-graph-mcp:standalone"
echo "  - zepai/knowledge-graph-mcp:${MCP_VERSION}-standalone"
echo "  - zepai/knowledge-graph-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}-standalone"
echo ""
echo "To push to DockerHub:"
echo "  docker push zepai/knowledge-graph-mcp:standalone"
echo "  docker push zepai/knowledge-graph-mcp:${MCP_VERSION}-standalone"
echo "  docker push zepai/knowledge-graph-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}-standalone"
echo ""
echo "Or push all tags:"
echo "  docker push --all-tags zepai/knowledge-graph-mcp"
