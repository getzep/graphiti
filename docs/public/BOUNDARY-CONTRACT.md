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
  --report /tmp/boundary-audit.md \
  --summary-json /tmp/boundary-audit-summary.json
```

`--summary-json` is optional and emits machine-readable counts + offending path lists for CI/report tooling.

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

## Lint policy files

```bash
python3 scripts/public_boundary_policy_lint.py \
  --manifest config/public_export_allowlist.yaml \
  --denylist config/public_export_denylist.yaml
```

This fails when policy rules are duplicated or contradictory (same pattern in both allowlist and denylist).

## Related migration/sync toolchain

Boundary policy now also governs delta-layer migration/sync tooling (`scripts/state_migration_*`,
`scripts/upstream_sync_doctor.py`, `extensions/**`, and migration workflow wiring).

See:
- `docs/public/MIGRATION-SYNC-TOOLKIT.md`
- `scripts/extension_contract_check.py`
