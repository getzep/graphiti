# Git Workflow for Graphiti Fork

## Repository Setup

This repository is a fork of the official Graphiti project with custom MCP server enhancements.

### Remote Configuration

```bash
origin    https://github.com/Varming73/graphiti.git (your fork)
upstream  https://github.com/getzep/graphiti.git (official Graphiti)
```

**Best Practice Convention:**
- `origin` = Your fork (where you push your changes)
- `upstream` = Official project (where you pull updates from)

## Common Workflows

### Push Your Changes
```bash
git add <files>
git commit -m "Your message"
git push origin main
```

### Pull Upstream Updates
```bash
# Fetch latest from upstream
git fetch upstream

# Merge upstream changes into your main branch
git merge upstream/main

# Or rebase if you prefer
git rebase upstream/main

# Push to your fork
git push origin main
```

### Check for Upstream Updates
```bash
git fetch upstream
git log HEAD..upstream/main --oneline  # See what's new
```

### Sync with Upstream (Full Update)
```bash
# Fetch upstream
git fetch upstream

# Switch to main
git checkout main

# Merge or rebase
git merge upstream/main  # or git rebase upstream/main

# Push to your fork
git push origin main
```

## Current Status

### Last Repository Replacement
- **Date**: 2025-11-08
- **Action**: Force-pushed clean code to replace "messed" project
- **Commit**: Added get_entities_by_type and compare_facts_over_time MCP tools
- **Result**: Successfully replaced entire fork history with clean implementation

### Upstream Tracking
- Upstream connection verified and working
- Can freely pull updates from official Graphiti project
- Your customizations remain in your fork

## MCP Server Customizations

Your fork contains these custom MCP tools (not in upstream):
1. `get_entities_by_type` - Retrieve entities by type classification
2. `compare_facts_over_time` - Compare facts between time periods
3. Enhanced `add_memory` UUID documentation

**Important**: When pulling upstream updates, these customizations are ONLY in `mcp_server/src/graphiti_mcp_server.py`. You may need to manually merge if upstream changes that file.

## Safety Notes

- **Never push to upstream** - You don't have permission and shouldn't try
- **Always test locally** before pushing to origin
- **Pull upstream regularly** to stay current with bug fixes and features
- **Document custom changes** in commit messages for future reference

## If You Need to Reset to Upstream

```bash
# Backup your current work first!
git checkout -b backup-branch

# Reset main to match upstream exactly
git checkout main
git reset --hard upstream/main
git push origin main --force

# Then cherry-pick your custom commits from backup-branch
```
