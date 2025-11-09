# Publishing Checklist

Use this checklist for your first PyPI publication.

## Pre-Publishing Setup

- [ ] **Update email in `pyproject.toml`** (optional but recommended)
  - File: `mcp_server/pyproject.toml`
  - Line: `{name = "Varming", email = "your-email@example.com"}`

- [ ] **Add PyPI token to GitHub Secrets**
  - URL: https://github.com/Varming73/graphiti/settings/secrets/actions
  - Secret name: `PYPI_API_TOKEN`
  - Secret value: Your PyPI token from https://pypi.org/manage/account/token/

## Publishing Steps

- [ ] **Commit all changes**
  ```bash
  git add .
  git commit -m "Prepare graphiti-mcp-varming v1.0.0 for PyPI"
  git push
  ```

- [ ] **Create and push release tag**
  ```bash
  git tag mcp-v1.0.0
  git push origin mcp-v1.0.0
  ```

- [ ] **Monitor GitHub Actions workflow**
  - URL: https://github.com/Varming73/graphiti/actions
  - Workflow name: "Publish MCP Server to PyPI"
  - Expected duration: 2-3 minutes

## Post-Publishing Verification

- [ ] **Check PyPI page**
  - URL: https://pypi.org/project/graphiti-mcp-varming/
  - Verify version shows as `1.0.0`
  - Check description and links are correct

- [ ] **Test installation**
  ```bash
  uvx graphiti-mcp-varming --help
  ```

- [ ] **Test in LibreChat** (if applicable)
  - Update `librechat.yaml` with `uvx graphiti-mcp-varming`
  - Restart LibreChat
  - Verify tools appear in UI

## If Something Goes Wrong

Common issues and solutions in `PYPI_PUBLISHING.md`

- Authentication error → Check token in GitHub secrets
- File already exists → Version already published, bump version number
- Workflow doesn't trigger → Check tag format is `mcp-v*.*.*`
- Package not found → Wait a few minutes for PyPI to propagate

---

**After first successful publish, this checklist can be deleted!**
