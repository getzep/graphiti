#!/bin/bash
# Script to build Docker image with proper version tagging
# This script extracts the graphiti-core version and includes it in the image tag

set -e

# Get MCP server version from pyproject.toml
MCP_VERSION=$(grep '^version = ' ../pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Get build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build the image
echo "Building Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${VCS_REF}" \
  -f Dockerfile \
  -t "zepai/graphiti-mcp:${MCP_VERSION}" \
  -t "zepai/graphiti-mcp:latest" \
  ..

# Extract graphiti-core version from the built image
GRAPHITI_CORE_VERSION=$(docker run --rm "zepai/graphiti-mcp:${MCP_VERSION}" cat /app/.graphiti-core-version)

echo ""
echo "Build complete!"
echo "  MCP Server Version: ${MCP_VERSION}"
echo "  Graphiti Core Version: ${GRAPHITI_CORE_VERSION}"
echo "  VCS Ref: ${VCS_REF}"
echo "  Build Date: ${BUILD_DATE}"
echo ""
echo "Image tags:"
echo "  - zepai/graphiti-mcp:${MCP_VERSION}"
echo "  - zepai/graphiti-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}"
echo "  - zepai/graphiti-mcp:latest"

# Tag with graphiti-core version
docker tag "zepai/graphiti-mcp:${MCP_VERSION}" "zepai/graphiti-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}"

echo ""
echo "To inspect image metadata:"
echo "  docker inspect zepai/graphiti-mcp:${MCP_VERSION} | jq '.[0].Config.Labels'"
