# Public Foundation Boundary Contract + Allowlist Audit v1

This repository uses a **default-closed** public export boundary.

1. A path matching the denylist is **BLOCK**.
2. Otherwise, a path matching the allowlist is **ALLOW**.
3. Otherwise, the path is **AMBIGUOUS** (must be resolved before export).

Denylist matches always win over allowlist matches.

## Policy files

- `config/public_export_allowlist.yaml`
- `config/public_export_denylist.yaml`

## Example policy intent

- Allowlisted foundation paths: `graphiti_core/**`, `mcp_server/**`, `server/**`, `tests/**`, `docs/public/**`.
- Denylisted sensitive patterns: `.env*`, key/cert files, `**/secrets/**`, and private state/export folders.

## Run audit

```bash
python3 scripts/public_repo_boundary_audit.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml \
  --report /tmp/boundary-audit.md
```

### Fail closed for CI

```bash
python3 scripts/public_repo_boundary_audit.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml \
  --report /tmp/boundary-audit.md \
  --strict
```

`--strict` exits non-zero when any path is `BLOCK` or `AMBIGUOUS`.

### Include untracked files

Use `--include-untracked` to include local untracked paths via:
`git ls-files --others --exclude-standard`.
