# PyPI Publishing Setup Instructions

This guide explains how to publish the `graphiti-mcp-varming` package to PyPI.

## One-Time Setup

### 1. Add PyPI Token to GitHub Secrets

1. Go to your repository on GitHub: https://github.com/Varming73/graphiti
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token (starts with `pypi-`)
6. Click **Add secret**

## Publishing a New Version

### Option 1: Automatic Publishing (Recommended)

1. Update the version in `mcp_server/pyproject.toml`:
   ```toml
   version = "1.0.1"  # Increment version
   ```

2. Commit the change:
   ```bash
   cd mcp_server
   git add pyproject.toml
   git commit -m "Bump MCP server version to 1.0.1"
   git push
   ```

3. Create and push a tag:
   ```bash
   git tag mcp-v1.0.1
   git push origin mcp-v1.0.1
   ```

4. GitHub Actions will automatically:
   - Build the package
   - Publish to PyPI
   - Create a GitHub release

5. Monitor the workflow:
   - Go to **Actions** tab in GitHub
   - Watch the "Publish MCP Server to PyPI" workflow

### Option 2: Manual Publishing

If you prefer to publish manually:

```bash
cd mcp_server

# Remove local graphiti-core override
sed -i.bak '/\[tool\.uv\.sources\]/,/graphiti-core/d' pyproject.toml

# Build the package
uv build

# Publish to PyPI
uv publish --token your-pypi-token-here

# Restore the backup if needed for local development
mv pyproject.toml.bak pyproject.toml
```

## After Publishing

Users can install your package with:

```bash
# Basic installation (Neo4j support included)
uvx graphiti-mcp-varming

# With FalkorDB support
uvx --with graphiti-mcp-varming[falkordb] graphiti-mcp-varming

# With all LLM providers (Anthropic, Groq, Gemini, Voyage, etc.)
uvx --with graphiti-mcp-varming[providers] graphiti-mcp-varming

# With everything
uvx --with graphiti-mcp-varming[all] graphiti-mcp-varming
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features, backwards compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes

## Tag Naming Convention

Use `mcp-v{VERSION}` format for tags:
- `mcp-v1.0.0` - Initial release
- `mcp-v1.0.1` - Patch release
- `mcp-v1.1.0` - Minor release
- `mcp-v2.0.0` - Major release

This distinguishes MCP server releases from graphiti-core releases.

## Troubleshooting

### Publishing Fails with "File already exists"

You tried to publish a version that already exists on PyPI. Increment the version number in `pyproject.toml` and try again.

### "Invalid or missing authentication token"

The PyPI token in GitHub secrets is incorrect or expired:
1. Generate a new token at https://pypi.org/manage/account/token/
2. Update the `PYPI_API_TOKEN` secret in GitHub

### Workflow doesn't trigger

Make sure:
- Tag matches pattern `mcp-v*.*.*`
- Tag is pushed to GitHub: `git push origin mcp-v1.0.0`
- Workflow file is on the `main` branch

## Checking Published Package

After publishing, verify at:
- PyPI page: https://pypi.org/project/graphiti-mcp-varming/
- Test installation: `uvx graphiti-mcp-varming --help`
