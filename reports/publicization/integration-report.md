# Publicization Integration Cutover Report

Generated (UTC): 2026-02-17T14:22:08Z
Evaluated commit: `7174af94c1eebea61bc3792d2cb4553d2f820616`
Gate return codes: `boundary_audit_rc=0`, `upstream_doctor_rc=0`, `upstream_sync_button_rc=0`
Decision: **GO**
Regen command source: `prd/EXEC-PUBLICIZATION-INTEGRATION-E2E-v1.md` validation sequence.

## Gate table

| Child area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Boundary contract checks | ✅ PASS | `scripts/public_repo_boundary_audit.py --strict ... --report /tmp/boundary-audit.md` | `BLOCK: 0`, `AMBIGUOUS: 0` |
| History migration decision/scorecard | ✅ PASS | `reports/publicization/history-scorecard.md` | Winner: `clean-foundation` |
| Upstream sync doctor | ✅ PASS | `scripts/upstream_sync_doctor.py --repo . --dry-run` | origin-only 43 / upstream-only 6 |
| Upstream sync safety policy | ✅ PASS | `scripts/upstream_sync_doctor.py --repo . --check-sync-button-safety` | sync button decision: **ALLOW** |
| Extension adapter contract | ✅ PASS | `scripts/extension_contract_check.py --strict` | Extension contract valid; 1 extension, 6 commands |
| State migration dry-run flow | ✅ PASS | `state_migration_export/check/import --dry-run` | 0 blocked conflicts |
| Public docs/release rails presence | ✅ PASS | `docs/public/*` + `docs/runbooks/*` checks | Content-marketing gate still explicit |

## Hardening deltas included in this gate pass
- Boundary policy rails tightened and expanded:
  - strict boundary audit now resolves all paths (`ALLOW=370`).
  - denylist now targets real secrets/runtime artifacts without blocking public-safe docs/examples.
- Added `scripts/public_repo_hardening_lint.py` to block:
  - legacy hard-path dependencies (`projects/graphiti`, `clawd-graphiti`) in active surfaces.
  - private workflow identifiers leaking into public files.
- CI toolkit now enforces:
  - boundary policy lint
  - strict boundary audit
  - hardening lint
  before migration/sync scorecard checks.
- Upstream sync policy updated (`max_origin_only_commits=100`) to match intentional fork divergence while preserving explicit policy guardrails.

## Post-hardening deltas (PR #34)
- Surgical security/review fixes from CLR verify passes on PR #32 and #33:
  - Denylist broadened to `.env*` with allowlist-overrides-denylist precedence fix
  - PII annotations on `parse_telegram_context` (medium finding logged as issue #35)
  - Input validation, KV store warnings, failure_reason sanitization
  - stdlib `queue` shadow resolved in `registry_status.py`
  - Remaining nits logged as issue #36
- Boundary audit: `classify_path` now checks allowlist before denylist (explicit allow > general deny)
- All 5 `.env.example` BLOCKs resolved via `**/.env.example` allowlist entry

## Recommendation
- **GO** for post-publicization integration gate.
- All CLR verify passes (PR #6, #32, #33, #34) completed with no unresolved CRITICAL/HIGH blockers.
- Remaining medium security finding (issue #35) and nits (issue #36) are tracked and non-blocking.
