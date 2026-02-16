# Public Release Checklist

Use this checklist before any public cutover, merge, or upstream sync publish.

## 0) Precondition

- [ ] I have compared `docs/public/README.md`, `SECURITY-BOUNDARIES.md`, and `WHAT-NOT-TO-MIGRATE.md` for consistency.
- [ ] All intended changes are foundation-only and do not include private overlays.

## 1) Boundary controls

- [ ] `python3 scripts/public_repo_boundary_audit.py --manifest config/public_export_allowlist.yaml --denylist config/public_export_denylist.yaml --strict --report /tmp/boundary-audit.md`
- [ ] `python3 scripts/public_boundary_policy_lint.py --manifest config/public_export_allowlist.yaml --denylist config/public_export_denylist.yaml`
- [ ] Confirm no `BLOCK`/`AMBIGUOUS` findings remain outside approved remediation paths.

## 2) Tooling and migration safety

- [ ] `python3 scripts/delta_tool.py contracts-check -- --policy config/migration_sync_policy.json --state-manifest config/state_migration_manifest.json --contract-policy config/delta_contract_policy.json --extensions-dir extensions --strict`
- [ ] `python3 scripts/upstream_sync_doctor.py --repo . --dry-run --check-sync-button-safety`
- [ ] `python3 scripts/state_migration_check.py --dry-run`

## 3) Runbook and docs gates

- [ ] `test -s docs/public/README.md`
- [ ] `test -s docs/public/SECURITY-BOUNDARIES.md`
- [ ] `test -s docs/public/RELEASE-CHECKLIST.md`
- [ ] `test -s docs/public/WHAT-NOT-TO-MIGRATE.md`

## Content-marketing / public write-up gate

The public write-up is deferred until all readiness criteria are true:

- [ ] Ingest backlog is stable (no unresolved data-dependent blockers in current intake).
- [ ] Content workflow and state export readiness checklist is green.
- [ ] External-facing messaging owner explicitly signs off that no private workflow or content-pack assumptions remain.

Until these gates are met: **content marketing stays deferred**.
