# PyPI Publishing Setup and Workflow

## Overview

The `graphiti-mcp-varming` package is published to PyPI for easy installation via `uvx` in stdio mode deployments (LibreChat, Claude Desktop, etc.).

**Package Name:** `graphiti-mcp-varming`  
**PyPI URL:** https://pypi.org/project/graphiti-mcp-varming/  
**GitHub Repo:** https://github.com/Varming73/graphiti

## Current Status (as of 2025-11-10)

### Version Information

**Current Version in Code:** 1.0.3 (in `mcp_server/pyproject.toml`)  
**Last Published Version:** 1.0.3 (tag: `mcp-v1.0.3`, commit: 1dd3f6b)  
**HEAD Commit:** 9d594c1 (2 commits ahead of last release)

### Unpublished Changes Since v1.0.3

**Commits not yet in PyPI:**

1. **ba938c9** - Add SEMAPHORE_LIMIT logging to startup configuration
   - Type: Enhancement
   - Files: `mcp_server/src/graphiti_mcp_server.py` (1 line added)
   - Impact: Logs SEMAPHORE_LIMIT value at startup for troubleshooting

2. **9d594c1** - Fix: Pass database parameter to Neo4j driver initialization
   - Type: Bug fix
   - Files: 
     - `mcp_server/src/graphiti_mcp_server.py` (11 lines changed)
     - `mcp_server/src/services/factories.py` (4 lines changed)
     - `mcp_server/tests/test_database_param.py` (74 lines added - test file)
   - Impact: Fixes NEO4J_DATABASE environment variable being ignored

**Total Changes:** 3 files modified, 85 insertions(+), 4 deletions(-)

### Version Bump Recommendation

**Recommended Next Version:** 1.0.4 (PATCH bump)

**Reasoning:**
- Database configuration fix is a bug fix (PATCH level)
- SEMAPHORE_LIMIT logging is minor enhancement (could be PATCH or MINOR, but grouped with bug fix)
- Both changes are backward compatible (no breaking changes)
- Follows Semantic Versioning 2.0.0

**Semantic Versioning Rules:**
- MAJOR (X.0.0): Breaking changes
- MINOR (0.X.0): New features, backward compatible
- PATCH (0.0.X): Bug fixes, backward compatible

## Publishing Workflow

### Automated Publishing (Recommended)

**Trigger:** Push a git tag matching `mcp-v*.*.*`

**Workflow File:** `.github/workflows/publish-mcp-pypi.yml`

**Steps:**
1. Update version in `mcp_server/pyproject.toml`
2. Commit and push changes
3. Create and push tag: `git tag mcp-v1.0.4 && git push origin mcp-v1.0.4`
4. GitHub Actions automatically:
   - Removes local graphiti-core override from pyproject.toml
   - Builds package with `uv build`
   - Publishes to PyPI with `uv publish`
   - Creates GitHub release with dist files

**Secrets Required:**
- `PYPI_API_TOKEN` - Must be configured in GitHub repository secrets

### Manual Publishing

```bash
cd mcp_server

# Remove local graphiti-core override
sed -i.bak '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml

# Build package
uv build

# Publish to PyPI
uv publish --token your-pypi-token-here

# Restore backup for local development
mv pyproject.toml.bak pyproject.toml
```

## Tag History

```
mcp-v1.0.3  (1dd3f6b) - Fix: Include config directory in PyPI package
mcp-v1.0.2  (cbaffa1) - Release v1.0.2: Add api-providers extra without sentence-transformers
mcp-v1.0.1  (f6be572) - Release v1.0.1: Enhanced config with custom entity types
mcp-v1.0.0  (eddeda6) - Fix graphiti-mcp-varming package for PyPI publication
```

## Package Features

### Installation Methods

**Basic (Neo4j support included):**
```bash
uvx graphiti-mcp-varming
```

**With FalkorDB support:**
```bash
uvx --with graphiti-mcp-varming[falkordb] graphiti-mcp-varming
```

**With additional LLM providers (Anthropic, Groq, Gemini, Voyage):**
```bash
uvx --with graphiti-mcp-varming[api-providers] graphiti-mcp-varming
```

**With all extras:**
```bash
uvx --with graphiti-mcp-varming[all] graphiti-mcp-varming
```

### Extras Available

Defined in `mcp_server/pyproject.toml`:

- `falkordb` - Adds FalkorDB (Redis-based graph database) support
- `api-providers` - Adds Anthropic, Groq, Gemini, Voyage embeddings support
- `all` - Includes all optional dependencies
- `dev` - Development dependencies (pytest, ruff, etc.)

## LibreChat Integration

The primary use case for this package is LibreChat stdio mode deployment:

```yaml
mcpServers:
  graphiti:
    type: stdio
    command: uvx
    args:
      - graphiti-mcp-varming
    env:
      GRAPHITI_GROUP_ID: "{{LIBRECHAT_USER_ID}}"
      NEO4J_URI: "bolt://neo4j:7687"
      NEO4J_USER: "neo4j"
      NEO4J_PASSWORD: "your_password"
      NEO4J_DATABASE: "graphiti"  # ← Now properly used after v1.0.4!
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
```

**Key Benefits:**
- ✅ No pre-installation needed in LibreChat container
- ✅ Automatic per-user process spawning
- ✅ Auto-downloads from PyPI on first use
- ✅ Easy updates (clear uvx cache to force latest version)

## Documentation Files

Located in `mcp_server/`:

1. **PYPI_SETUP_COMPLETE.md** - Overview of PyPI setup and usage examples
2. **PYPI_PUBLISHING.md** - Detailed publishing instructions and troubleshooting
3. **PUBLISHING_CHECKLIST.md** - Step-by-step checklist for first publish

## Important Notes

### Local Development vs PyPI Build

**Local Development:**
- Uses `[tool.uv.sources]` to override graphiti-core with local path
- Allows testing changes to both MCP server and graphiti-core together

**PyPI Build:**
- GitHub Actions removes `[tool.uv.sources]` section before building
- Uses official `graphiti-core` from PyPI
- Ensures published package doesn't depend on local files

### Package Structure

```
mcp_server/
├── src/
│   ├── graphiti_mcp_server.py  # Main MCP server
│   ├── config/                 # Configuration schemas
│   ├── models/                 # Response types
│   ├── services/               # Factories for LLM, embedder, database
│   └── utils/                  # Utilities
├── config/
│   └── config.yaml             # Default configuration
├── tests/                      # Test suite
├── pyproject.toml              # Package metadata and dependencies
└── README.md                   # Package documentation
```

### Version Management Best Practices

1. **Always update version in pyproject.toml** before creating tag
2. **Tag format must be `mcp-v*.*.*`** to trigger workflow
3. **Commit message should explain changes** (included in GitHub release notes)
4. **Test locally first** with `uv build` before tagging
5. **Monitor GitHub Actions** after pushing tag to ensure successful publish

## Next Steps for v1.0.4 Release

To publish the database configuration fix and SEMAPHORE_LIMIT logging:

1. Update version in `mcp_server/pyproject.toml`: `version = "1.0.4"`
2. Commit: `git commit -m "Bump version to 1.0.4 for database fix and logging enhancement"`
3. Push: `git push`
4. Tag: `git tag mcp-v1.0.4`
5. Push tag: `git push origin mcp-v1.0.4`
6. Monitor: https://github.com/Varming73/graphiti/actions
7. Verify: https://pypi.org/project/graphiti-mcp-varming/ shows v1.0.4

## References

- **Semantic Versioning:** https://semver.org/
- **uv Documentation:** https://docs.astral.sh/uv/
- **PyPI Publishing Guide:** https://packaging.python.org/en/latest/tutorials/packaging-projects/
- **GitHub Actions:** https://docs.github.com/en/actions
