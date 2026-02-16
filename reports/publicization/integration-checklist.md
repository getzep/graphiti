# Publicization Integration Checklist

Generated: 2026-02-16
Gate: **NO-GO**

## Mission outcome
- Final integrated gate recommendation: **NO-GO** until boundary strictness and sync-button preconditions are remediated.

## Evidence checklist by child area

| Area | Evidence command | Result | Key artifact(s) |
| --- | --- | --- | --- |
| boundary contract checks | `python3 scripts/public_repo_boundary_audit.py --strict --manifest config/public_export_allowlist.yaml --denylist config/public_export_denylist.yaml --report /tmp/boundary-audit.md` | **FAIL (1)** | `/tmp/boundary-audit.md` (ALLOW 245 / BLOCK 2 / AMBIGUOUS 81) |
| history migration decision/scorecard | `test -s reports/publicization/history-scorecard.md`, `rg -n "Winner: \`clean-foundation\`" reports/publicization/history-scorecard.md` | **PASS** | `reports/publicization/history-scorecard.md` |
| upstream sync doctor | `python3 scripts/upstream_sync_doctor.py --repo . --dry-run` | **PASS** | doctor summary (origin-only 25, upstream-only 5, clean worktree: true) |
| upstream sync safety policy | `python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety` | **FAIL (1)** | sync button decision: **DENY** (exceeds max_origin_only_commits=0) |
| extension adapter contract | `python3 scripts/extension_contract_check.py --strict` | **PASS** | extension contract summary: 1 extension, 6 commands |
| state migration dry-run flow | `python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export-clean`, `python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-clean --dry-run`, `python3 scripts/state_migration_import.py --dry-run --allow-overwrite --in /tmp/graphiti-state-export-clean` | **PASS** | package manifest + dry-run import plan (0 blocking conflicts) |
| public docs/release rails presence | `test -s docs/public/README.md` etc. (security boundaries/release checklist/what-not-to-migrate + runbooks) | **PASS** | docs files are present and release checklist includes content-marketing gate |

## Cross-repo baseline review (executed)
- Source baseline (`/Users/archibald/clawd/projects/graphiti`) and target (`/Users/archibald/clawd/projects/graphiti-openclaw`) paths reviewed.
- Concrete inventory includes workflow/runbook/script/doc coverage around boundary contract, upstream sync lane, migration tooling, and docs rails.
- Simplification loop: 4 candidates reviewed; 3 selected (command reuse, artifact-first reporting, command-sequence determinism), 1 deferred (dedicated integration report generator script).
