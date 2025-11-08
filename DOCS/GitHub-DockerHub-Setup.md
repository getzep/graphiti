# GitHub Actions → Docker Hub Automated Build Setup

This guide explains how to automatically build your custom Graphiti MCP Docker image with your local changes and push it to Docker Hub using GitHub Actions.

## Why This Approach?

✅ **Automatic builds** - Every push to main triggers a new build
✅ **Reproducible** - Anyone can see exactly what was built
✅ **Multi-platform** - Builds for both AMD64 and ARM64
✅ **No local building** - GitHub does all the work
✅ **Version tracking** - Tied to git commits
✅ **Clean workflow** - Professional CI/CD pipeline

## Prerequisites

1. **GitHub account** with a fork of the graphiti repository
2. **Docker Hub account** (username: `lvarming`)
3. **Docker Hub Access Token** (for GitHub Actions to push images)

---

## Step 1: Create Docker Hub Access Token

1. Go to [Docker Hub](https://hub.docker.com/)
2. Click your username → **Account Settings**
3. Click **Security** → **New Access Token**
4. Give it a description: "GitHub Actions - Graphiti MCP"
5. **Copy the token** (you'll only see it once!)

---

## Step 2: Add Token to GitHub Repository Secrets

1. Go to your forked repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `DOCKERHUB_TOKEN`
5. Value: Paste the access token from Step 1
6. Click **Add secret**

---

## Step 3: Verify Workflow Files

The repository already includes the necessary workflow file at:
```
.github/workflows/build-custom-mcp.yml
```

And the custom Dockerfile at:
```
mcp_server/docker/Dockerfile.custom
```

These files are configured to:
- Build using YOUR local `graphiti-core` changes (not PyPI)
- Push to `lvarming/graphiti-mcp` on Docker Hub
- Tag with version numbers and `latest`

---

## Step 4: Trigger a Build

### Option A: Automatic Build (On Push)

The workflow automatically triggers when you:
- Push to the `main` branch
- Modify files in `graphiti_core/` or `mcp_server/`

Simply commit and push your changes:
```bash
git add .
git commit -m "Update graphiti-core with custom changes"
git push origin main
```

### Option B: Manual Build

1. Go to your repository on GitHub
2. Click **Actions** tab
3. Select **Build Custom MCP Server** workflow
4. Click **Run workflow** dropdown
5. (Optional) Specify a custom tag, or leave as `latest`
6. Click **Run workflow**

---

## Step 5: Monitor the Build

1. Click on the running workflow to see progress
2. The build takes about 5-10 minutes
3. You'll see:
   - Version extraction
   - Docker image build (for AMD64 and ARM64)
   - Push to Docker Hub
   - Build summary with tags

---

## Step 6: Verify Image on Docker Hub

1. Go to [Docker Hub](https://hub.docker.com/)
2. Navigate to your repository: `lvarming/graphiti-mcp`
3. Check the **Tags** tab
4. You should see tags like:
   - `latest`
   - `mcp-1.0.0`
   - `mcp-1.0.0-core-0.23.0`
   - `sha-abc1234`

---

## Step 7: Use Your Custom Image

### In Unraid

Update your Docker container to use:
```
Repository: lvarming/graphiti-mcp:latest
```

### In Docker Compose

```yaml
services:
  graphiti-mcp:
    image: lvarming/graphiti-mcp:latest
    container_name: graphiti-mcp
    restart: unless-stopped
    # ... rest of your config
```

### Pull Manually

```bash
docker pull lvarming/graphiti-mcp:latest
```

---

## Understanding the Build Process

### What Gets Built

The custom Dockerfile (`Dockerfile.custom`) does the following:

1. **Copies entire project** - Both `graphiti_core/` and `mcp_server/`
2. **Builds graphiti-core from local source** - Not from PyPI
3. **Installs MCP server** - Using the local graphiti-core
4. **Creates multi-platform image** - AMD64 and ARM64

### Version Tagging

Each build creates multiple tags:

| Tag | Description | Example |
|-----|-------------|---------|
| `latest` | Always points to most recent build | `lvarming/graphiti-mcp:latest` |
| `mcp-X.Y.Z` | MCP server version | `lvarming/graphiti-mcp:mcp-1.0.0` |
| `mcp-X.Y.Z-core-A.B.C` | Full version info | `lvarming/graphiti-mcp:mcp-1.0.0-core-0.23.0` |
| `sha-xxxxxxx` | Git commit SHA | `lvarming/graphiti-mcp:sha-abc1234` |

### Build Arguments

The workflow passes these build arguments:

```dockerfile
GRAPHITI_CORE_VERSION=0.23.0       # From pyproject.toml
MCP_SERVER_VERSION=1.0.0            # From mcp_server/pyproject.toml
BUILD_DATE=2025-11-08T12:00:00Z    # UTC timestamp
VCS_REF=abc1234                     # Git commit hash
```

---

## Workflow Customization

### Change Docker Hub Username

If you want to use a different Docker Hub account, edit `.github/workflows/build-custom-mcp.yml`:

```yaml
env:
  DOCKERHUB_USERNAME: your-username  # Change this
  IMAGE_NAME: graphiti-mcp
```

### Change Trigger Conditions

To only build on tags instead of every push:

```yaml
on:
  push:
    tags:
      - 'v*.*.*'
```

### Add Slack/Discord Notifications

Add a notification step at the end of the workflow:

```yaml
- name: Notify on Success
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "✅ New Graphiti MCP image built: lvarming/graphiti-mcp:latest"
      }
```

---

## Troubleshooting

### Build Fails - "Error: buildx failed"

**Cause**: Docker Buildx issue
**Solution**: Re-run the workflow (transient issue)

### Build Fails - "unauthorized: incorrect username or password"

**Cause**: Invalid Docker Hub credentials
**Solution**:
1. Verify `DOCKERHUB_TOKEN` secret is correct
2. Regenerate access token on Docker Hub
3. Update the secret in GitHub

### Build Fails - "No space left on device"

**Cause**: GitHub runner out of disk space
**Solution**: Add cleanup step before build:

```yaml
- name: Free up disk space
  run: |
    docker system prune -af
    df -h
```

### Image Not Found on Docker Hub

**Cause**: Image is private
**Solution**:
1. Go to Docker Hub → lvarming/graphiti-mcp
2. Click **Settings**
3. Make repository **Public**

### Workflow Doesn't Trigger

**Cause**: Branch protection or incorrect path filters
**Solution**:
1. Check you're pushing to `main` branch
2. Verify changes are in `graphiti_core/` or `mcp_server/`
3. Manually trigger from Actions tab

---

## Advanced: Multi-Repository Setup

If you want separate images for development and production:

### Development Image

Create `.github/workflows/build-dev-mcp.yml`:

```yaml
name: Build Dev MCP Server

on:
  push:
    branches:
      - dev
      - feature/*

env:
  DOCKERHUB_USERNAME: lvarming
  IMAGE_NAME: graphiti-mcp-dev  # Different image name
```

### Production Image

Keep the main workflow for production builds on `main` branch.

---

## Comparing with Official Builds

| Feature | Official (zepai) | Your Custom Build |
|---------|-----------------|-------------------|
| Source | PyPI graphiti-core | Local graphiti-core |
| Trigger | Manual tags only | Auto on push + manual |
| Docker Hub | zepai/knowledge-graph-mcp | lvarming/graphiti-mcp |
| Build Platform | Depot (paid) | GitHub Actions (free) |
| Customization | Limited | Full control |

---

## Best Practices

### 1. **Pin Versions for Production**

Instead of `latest`, use specific versions:
```yaml
image: lvarming/graphiti-mcp:mcp-1.0.0-core-0.23.0
```

### 2. **Test Before Deploying**

Add a test step in the workflow:
```yaml
- name: Test image
  run: |
    docker run --rm lvarming/graphiti-mcp:latest --version
```

### 3. **Keep Workflows Updated**

GitHub Actions updates frequently. Use Dependabot:

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 4. **Monitor Build Times**

If builds are slow, enable caching:
```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```
(Already enabled in the workflow!)

### 5. **Security Scanning**

Add Trivy security scanner:
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: lvarming/graphiti-mcp:latest
    format: 'table'
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```

---

## Next Steps

1. ✅ Set up Docker Hub access token
2. ✅ Add secret to GitHub repository
3. ✅ Push changes to trigger first build
4. ✅ Verify image appears on Docker Hub
5. ✅ Update your Unraid/LibreChat config to use new image
6. 📝 Document any custom changes in DOCS/

---

## Questions?

- **GitHub Actions Issues**: Check the Actions tab for detailed logs
- **Docker Hub Issues**: Verify your account and access token
- **Build Failures**: Review the workflow logs for specific errors

## Related Documentation

- [LibreChat Setup Guide](./Librechat.setup.md)
- [OpenAI Compatible Endpoints](./OpenAI-Compatible-Endpoints.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
