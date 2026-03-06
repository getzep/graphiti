#!/bin/bash
# Script to build and push standalone Docker image with both Neo4j and FalkorDB drivers
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
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build from repo root so the Dockerfile can COPY graphiti_core/
echo "Building standalone Docker image..."
docker build \
  --build-arg MCP_SERVER_VERSION="${MCP_VERSION}" \
  --build-arg GRAPHITI_CORE_VERSION="${GRAPHITI_CORE_VERSION}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  --build-arg VCS_REF="${VCS_REF}" \
  -f "$SCRIPT_DIR/Dockerfile.standalone" \
  -t "zepai/knowledge-graph-mcp:standalone" \
  -t "zepai/knowledge-graph-mcp:${MCP_VERSION}-standalone" \
  -t "zepai/knowledge-graph-mcp:${MCP_VERSION}-graphiti-${GRAPHITI_CORE_VERSION}-standalone" \
  "$REPO_ROOT"

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
