# Public Security Boundaries

## Boundary intent

This repo is public by design, but only for generalized foundation artifacts. The boundary exists to prevent accidental migration of private execution, private data, and person-specific logic.

## Threat-oriented rationale

- **Data exfiltration risk**: private prompts, credentials, exports, and local state.
- **Operational dependency risk**: private packs and scripts that break portability.
- **Reputation and IP leakage risk**: personal workflows or private evaluation data.
- **Maintenance burden risk**: one-off integrations that increase support cost and reduce extensibility.

## Exclusion classes (must not migrate)

- **Secrets and credentials**: `.env*`, API keys, certs, tokens, and signed identities.
- **Private context and workflow packs**: packs that encode organization-specific playbooks.
- **Personal/private data**: raw source artifacts, logs, private reports, local snapshots.
- **Overlay-specific extensions**: adapters tied to one operator workflow.
- **Stateful runtime artifacts**: generated state files, backfills, backups, and local persistence caches.
- **Content strategy pack** items not intended as generic library docs.

## Inclusion baseline

Allowed foundations should be generally reusable with limited assumptions about environment or operator identity.

If an asset is hard to describe in general terms and would require personal/process-specific onboarding, treat it as private and keep it out of public release.

## Review requirements

Any migration or release PR must pass boundary policy checks in `docs/public/RELEASE-CHECKLIST.md` before publication.
