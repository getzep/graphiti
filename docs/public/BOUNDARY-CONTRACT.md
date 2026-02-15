# Public Foundation Boundary Contract + Allowlist Audit v1

This repository uses a default-closed boundary for public export.

- Files are **allowed** only when a path matches the allowlist policy.
- Files are **blocked** when a path matches denylist patterns.
- Matches in denylist always win over allowlist matches.
- Files that match neither are **ambiguous** and must be resolved before export.

## Default policy

- `config/public_export_allowlist.yaml`
- `config/public_export_denylist.yaml`

## Contract examples

- Allowlist foundations: `graphiti_core/**`, `mcp_server/**`, `server/**`, `tests/**`, `docs/public/**`.
- Denylist protections: `.env`, `.env.*`, `*.key`, `*.pem`, `*.p12`, and explicit secret folders (`**/secrets/**`).
- Sensitive folders: `state/**`, `backup/**`, `exports/**`, `reports/**`.
- Private overlays: workflow/content/private pack paths.

## Usage

```bash
python3 scripts/public_repo_boundary_audit.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml \
  --report /tmp/boundary-audit.md
```

Use `--strict` to make the script return a non-zero exit code when any path is BLOCK or AMBIGUOUS.

Use `--include-untracked` to include `git status --porcelain` untracked files in the scan.
