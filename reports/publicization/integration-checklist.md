# Publicization Integration Checklist

Generated (UTC): 2026-02-17T04:59:00Z
Evaluated commit: `92bf60d7723b28b73f54ff8a2ece5694874e2f8f`
Gate return codes: `boundary_audit_rc=0`, `upstream_doctor_rc=0`, `upstream_sync_button_rc=0`
Gate: **GO**
Regen command source: `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md` validation sequence.

## Mission outcome
- Final integrated gate recommendation: **GO**.
- All mandatory validators now pass cleanly on the current policy/config baseline.

## Evidence checklist by child area

| Area | Evidence command | Result | Key artifact(s) |
| --- | --- | --- | --- |
| boundary contract checks | `python3 scripts/public_repo_boundary_audit.py --strict --manifest config/public_export_allowlist.yaml --denylist config/public_export_denylist.yaml --report /tmp/boundary-audit.md` | **PASS (0)** | `/tmp/boundary-audit.md` (ALLOW 370 / BLOCK 0 / AMBIGUOUS 0) |
| history migration decision/scorecard | `test -s reports/publicization/history-scorecard.md`, `rg -n "Winner: \`clean-foundation\`" reports/publicization/history-scorecard.md` | **PASS** | `reports/publicization/history-scorecard.md` |
| upstream sync doctor | `python3 scripts/upstream_sync_doctor.py --repo . --dry-run` | **PASS** | doctor summary (origin-only 40, upstream-only 6, clean worktree: true) |
| upstream sync safety policy | `python3 scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety` | **PASS (0)** | sync button decision: **ALLOW** (max_origin_only_commits=100) |
| extension adapter contract | `python3 scripts/extension_contract_check.py --strict` | **PASS** | extension contract summary: 1 extension, 6 commands |
| state migration dry-run flow | `python3 scripts/state_migration_export.py --dry-run --out /tmp/graphiti-state-export-clean`, `python3 scripts/state_migration_check.py --package /tmp/graphiti-state-export-clean --dry-run`, `python3 scripts/state_migration_import.py --dry-run --allow-overwrite --in /tmp/graphiti-state-export-clean` | **PASS** | package manifest + dry-run import plan (0 blocking conflicts) |
| public docs/release rails presence | `test -s docs/public/README.md` etc. (security boundaries/release checklist/what-not-to-migrate + runbooks) | **PASS** | docs files are present and release checklist includes content-marketing gate |

## Cross-repo baseline review (executed)
- Source baseline (`projects/graphiti`) and target (`projects/graphiti-openclaw`) paths reviewed.
- Simplification loop: selected deterministic command reuse + policy lint gates + strict boundary audit in CI.
- Remaining work is operational hygiene only (no unresolved publicization gate blockers).
